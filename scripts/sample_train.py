#!/usr/bin/env python
"""Stage-2 preparation: sample several completions per train question and record the
fraction that executes to the gold result (pass-rate statistics for difficulty filtering).

Examples::

    python scripts/sample_train.py --model configs/models/qwen3_4b_instruct.yaml            # first 200 questions
    python scripts/sample_train.py --model configs/models/qwen3_4b_instruct.yaml --limit 0  # whole train_rl split
    python scripts/sample_train.py --backend dummy --smoke

Outputs under ``outputs/sampling/<run_name>/``: ``samples.jsonl`` (one row per sample, without
the prompt), ``prompts.jsonl`` (one row per question), ``pass_rates.jsonl`` and ``summary.json``
with a time estimate for the full split.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from mexql.config import load_config  # noqa: F401  (re-exported for symmetry with evaluate.py)
from mexql.data.bird import BirdPaths, load_split
from mexql.inference.base import create_backend
from mexql.pipeline import (append_jsonl, chunked, done_uids, generate_and_score, load_tokenizer, read_jsonl,
                            sync_outputs, write_jsonl)
from mexql.prompt import WhitespaceTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import add_common_args, load_cfg, select_examples  # noqa: E402

log = logging.getLogger("sample_train")


def pass_rate_rows(examples: list, records: list[dict]) -> list[dict]:
    by_uid: dict[str, list[dict]] = {}
    for r in records:
        by_uid.setdefault(r["uid"], []).append(r)
    rows = []
    for ex in examples:
        samples = by_uid[ex.uid]
        n_correct = sum(1 for r in samples if r["status"] == "correct")
        gold_ok = samples[0]["gold_status"] == "ok"
        rows.append({
            "uid": ex.uid, "question_id": ex.question_id, "db_id": ex.db_id, "n": len(samples),
            "n_correct": n_correct, "pass_rate": n_correct / len(samples) if gold_ok else None,
            "gold_status": samples[0]["gold_status"], "prompt_tokens": samples[0]["prompt_tokens"],
            "statuses": dict(Counter(r["status"] for r in samples)),
            "sqls": [r["sql"] for r in samples],
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, "configs/sample_train.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.limit == 0:  # --limit 0 means the whole split
        args.limit = None
        args.overrides.append("limit=null")
    cfg = load_cfg(args)
    if cfg.backend != "dummy":
        cfg.require_model()
    model_name = cfg.model.name if cfg.model else "dummy"
    run_name = cfg.run_name or f"{model_name}_{cfg.split}_n{cfg.sampling.n}" + ("_smoke" if args.smoke else "")
    out_dir = Path(cfg.paths.outputs_dir) / "sampling" / run_name

    paths = BirdPaths(cfg)
    all_examples = load_split(cfg.split, cfg, paths)
    examples = select_examples(all_examples, cfg, args.smoke)
    rates_file = out_dir / "pass_rates.jsonl"
    if args.resume:
        sync_outputs(out_dir, args.sync_dir, Path(cfg.paths.outputs_dir), restore=True)
    else:
        for name in ("pass_rates.jsonl", "samples.jsonl"):
            if (out_dir / name).exists():
                (out_dir / name).unlink()
    done = done_uids(rates_file) if args.resume else set()
    todo = [ex for ex in examples if ex.uid not in done]
    log.info("split=%s questions=%d/%d (done=%d, todo=%d) samples per question=%d temperature=%.2f",
             cfg.split, len(examples), len(all_examples), len(done), len(todo), cfg.sampling.n, cfg.sampling.temperature)

    tokenizer = load_tokenizer(cfg.model) if cfg.model else WhitespaceTokenizer()
    timing = {"prompts_s": 0.0, "generation_s": 0.0, "execution_s": 0.0, "model_load_s": 0.0}
    session_records: list[dict] = []
    if todo:
        t0 = time.monotonic()
        backend = create_backend(cfg)
        timing["model_load_s"] = time.monotonic() - t0
        n_done = len(done)
        for i, chunk in enumerate(chunked(todo, args.chunk_size)):
            records, t = generate_and_score(cfg, chunk, backend, tokenizer, paths, include_prompt=False,
                                            chat_template_kwargs=cfg.model.chat_template_kwargs if cfg.model else None)
            for k, v in t.items():
                timing[k] += v
            session_records.extend(records)
            append_jsonl(out_dir / "samples.jsonl", records)
            append_jsonl(rates_file, pass_rate_rows(chunk, records))
            n_done += len(chunk)
            sync_outputs(out_dir, args.sync_dir, Path(cfg.paths.outputs_dir))
            log.info("chunk %d done: %d questions (%d/%d)", i + 1, len(chunk), n_done, len(examples))

    rows = read_jsonl(rates_file) if rates_file.is_file() else []
    scored = [p for p in rows if p["pass_rate"] is not None]
    hist = Counter(p["n_correct"] for p in scored)
    mean_pass = sum(p["pass_rate"] for p in scored) / len(scored) if scored else 0.0
    n_total = len(all_examples)
    n_session = max(1, len(todo))
    per_q_gen = timing["generation_s"] / n_session
    per_q_exec = timing["execution_s"] / n_session
    status_counts = Counter(st for p in rows for st, c in p["statuses"].items() for _ in range(c))
    summary = {
        "run_name": run_name, "model": model_name, "split": cfg.split, "backend": cfg.backend,
        "n_questions": len(rows), "n_generated_this_session": len(todo), "n_total_in_split": n_total,
        "n_samples": cfg.sampling.n, "temperature": cfg.sampling.temperature, "seed": cfg.seed,
        "n_scored": len(scored), "n_gold_invalid": len(rows) - len(scored),
        "mean_pass_rate": mean_pass,
        "pass_at_n": sum(1 for p in scored if p["n_correct"] > 0) / len(scored) if scored else 0.0,
        "all_correct": sum(1 for p in scored if p["n_correct"] == p["n"]) / len(scored) if scored else 0.0,
        "n_correct_histogram": {str(k): hist.get(k, 0) for k in range(cfg.sampling.n + 1)},
        "sample_status_counts": dict(status_counts),
        "timing": {k: round(v, 2) for k, v in timing.items()},
        "estimate_full_split": {
            "n_questions": n_total,
            "based_on_questions": len(todo),
            "generation_s": round(per_q_gen * n_total, 1) if todo else None,
            "execution_s": round(per_q_exec * n_total, 1) if todo else None,
            "total_hours": round((per_q_gen + per_q_exec) * n_total / 3600, 2) if todo else None,
        },
    }
    write_jsonl(out_dir / "prompts.jsonl", [{"uid": ex.uid, "question_id": ex.question_id, "db_id": ex.db_id}
                                             for ex in examples])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    sync_outputs(out_dir, args.sync_dir, Path(cfg.paths.outputs_dir))

    print()
    print(f"Sampling: {model_name} / {cfg.split}: {len(rows)} questions x {cfg.sampling.n} samples "
          f"(T={cfg.sampling.temperature}); {len(todo)} generated in this session")
    print(f"mean pass rate {mean_pass:.3f}; pass@{cfg.sampling.n} {summary['pass_at_n']:.3f}; "
          f"all-correct {summary['all_correct']:.3f}; gold invalid {summary['n_gold_invalid']}")
    print("n_correct histogram: " + ", ".join(f"{k}: {v}" for k, v in summary["n_correct_histogram"].items()))
    print("sample statuses: " + ", ".join(f"{k}={v}" for k, v in summary["sample_status_counts"].items()))
    print("timing (this session): " + ", ".join(f"{k}={v:.1f}s" for k, v in timing.items()))
    est = summary["estimate_full_split"]
    if est["total_hours"] is not None:
        print(f"estimate for all {n_total} questions of {cfg.split} (from {est['based_on_questions']} measured): "
              f"generation {est['generation_s'] / 3600:.2f} h, execution {est['execution_s'] / 3600:.2f} h, "
              f"total {est['total_hours']:.2f} h")
    print(f"outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
