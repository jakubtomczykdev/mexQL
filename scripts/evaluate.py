#!/usr/bin/env python
"""Evaluate one model on one split with greedy decoding and print EX by difficulty.

Examples::

    python scripts/evaluate.py --model configs/models/qwen3_4b_instruct.yaml --split dev
    python scripts/evaluate.py --model configs/models/qwen3_4b_instruct.yaml --split val_internal
    python scripts/evaluate.py --model configs/models/qwen25_coder_0.5b.yaml --backend hf --smoke
    python scripts/evaluate.py --backend dummy --smoke          # pipeline check without a model

Outputs go to ``outputs/runs/<run_name>/``: ``predictions.jsonl`` (one row per question with
prompt, response, extracted SQL, status and correctness), ``metrics.json`` and ``metrics.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from mexql.config import load_config
from mexql.data.bird import BirdPaths, load_split
from mexql.inference.base import create_backend
from mexql.metrics import format_markdown, format_table
from mexql.pipeline import (append_jsonl, chunked, done_uids, generate_and_score, load_tokenizer, read_jsonl,
                            summarize_records, sync_outputs)
from mexql.prompt import WhitespaceTokenizer

log = logging.getLogger("evaluate")


def add_common_args(ap: argparse.ArgumentParser, default_config: str) -> None:
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--model", default=None, help="model YAML from configs/models/")
    ap.add_argument("--split", default=None, help="dev | val_internal | train_rl | train")
    ap.add_argument("--backend", default=None, choices=["vllm", "hf", "dummy"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--smoke", action="store_true", help="smoke test on cfg.smoke_limit questions spread over the split")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="generate and score in chunks of N questions, appending results after each chunk (0 = all at once)")
    ap.add_argument("--resume", action="store_true", help="skip questions already present in the output file")
    ap.add_argument("--sync-dir", type=Path, default=None,
                    help="mirror the run directory here after every chunk (e.g. a mounted Google Drive); "
                         "with --resume an existing mirror is restored first")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")


def load_cfg(args: argparse.Namespace):
    files = [args.base_config, args.config] + ([args.model] if args.model else [])
    overrides = list(args.overrides)
    for key in ("split", "backend", "limit", "run_name"):
        value = getattr(args, key)
        if value is not None:
            overrides.append(f"{key}={value}")
    return load_config(files, overrides)


def select_examples(examples: list, cfg, smoke: bool) -> list:
    if smoke:
        n = min(cfg.smoke_limit, len(examples))
        step = max(1, len(examples) // n)
        return examples[::step][:n]
    if cfg.limit:
        return examples[: cfg.limit]
    return examples


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, "configs/evaluate.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_cfg(args)
    if cfg.backend != "dummy":
        cfg.require_model()
    model_name = cfg.model.name if cfg.model else "dummy"
    run_name = cfg.run_name or f"{model_name}_{cfg.split}" + ("_smoke" if args.smoke else "")
    out_dir = Path(cfg.paths.outputs_dir) / "runs" / run_name

    paths = BirdPaths(cfg)
    examples = select_examples(load_split(cfg.split, cfg, paths), cfg, args.smoke)
    pred_file = out_dir / "predictions.jsonl"
    if args.resume:
        sync_outputs(out_dir, args.sync_dir, Path(cfg.paths.outputs_dir), restore=True)
    elif pred_file.exists():
        pred_file.unlink()
    done = done_uids(pred_file) if args.resume else set()
    todo = [ex for ex in examples if ex.uid not in done]
    log.info("split=%s questions=%d (done=%d, todo=%d) backend=%s model=%s", cfg.split, len(examples), len(done),
             len(todo), cfg.backend, model_name)

    tokenizer = load_tokenizer(cfg.model) if cfg.model else WhitespaceTokenizer()
    timing = {"prompts_s": 0.0, "generation_s": 0.0, "execution_s": 0.0, "model_load_s": 0.0}
    if todo:
        t0 = time.monotonic()
        backend = create_backend(cfg)
        timing["model_load_s"] = time.monotonic() - t0
        n_done = len(done)
        for i, chunk in enumerate(chunked(todo, args.chunk_size)):
            records, t = generate_and_score(cfg, chunk, backend, tokenizer, paths,
                                            chat_template_kwargs=cfg.model.chat_template_kwargs if cfg.model else None)
            for k, v in t.items():
                timing[k] += v
            append_jsonl(pred_file, records)
            n_done += len(chunk)
            sync_outputs(out_dir, args.sync_dir, Path(cfg.paths.outputs_dir))
            log.info("chunk %d done: %d questions appended (%d/%d)", i + 1, len(chunk), n_done, len(examples))

    records = read_jsonl(pred_file)
    summary = summarize_records(records)
    label = f"{model_name} / {cfg.split}" + (" (smoke)" if args.smoke else "")
    metrics = {"run_name": run_name, "model": cfg.model.model_dump() if cfg.model else None, "split": cfg.split,
               "backend": cfg.backend, "n_questions": len(records), "n_generated_this_session": len(todo),
               "sampling": cfg.sampling.model_dump(), "prompt": cfg.prompt.model_dump(),
               "execution": cfg.execution.model_dump(), "seed": cfg.seed,
               "timing": {k: round(v, 2) for k, v in timing.items()}, "summary": summary}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out_dir / "metrics.md").write_text(format_markdown(summary, label) + "\n")
    sync_outputs(out_dir, args.sync_dir, Path(cfg.paths.outputs_dir))
    print()
    print(format_table(summary, f"Execution accuracy: {label}"))
    print("timing (this session): " + ", ".join(f"{k}={v:.1f}s" for k, v in timing.items()))
    print(f"outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
