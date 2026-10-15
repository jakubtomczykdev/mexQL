#!/usr/bin/env python
"""Prepare caches and reports after the data has been downloaded:

1. build the deterministic val_internal / train_rl split of the train set,
2. extract and cache the schema (DDL + example values) of every available database,
3. execute and cache every gold query (reports non-executable gold queries),
4. measure prompt lengths in tokens for a model and list databases over the context budget.

Examples::

    python scripts/prepare_data.py --model configs/models/qwen3_4b_instruct.yaml
    python scripts/prepare_data.py --sources dev --skip-lengths
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mexql.config import load_config
from mexql.data.bird import BirdPaths, load_examples, load_or_build_splits
from mexql.pipeline import build_prompt_items, ensure_gold_results, gold_cache, load_tokenizer
from mexql.schema import load_or_extract_schema

log = logging.getLogger("prepare_data")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default=None, help="model YAML; needed for the prompt-length report")
    ap.add_argument("--sources", nargs="+", default=["dev", "train"], choices=["dev", "train"])
    ap.add_argument("--skip-gold", action="store_true")
    ap.add_argument("--retry-gold-failures", action="store_true",
                    help="re-execute cached gold queries that timed out or failed")
    ap.add_argument("--skip-lengths", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    files = [args.config] + ([args.model] if args.model else [])
    cfg = load_config(files, args.overrides)
    paths = BirdPaths(cfg)
    t_start = time.monotonic()

    # 1. splits
    if paths.questions_file("train").is_file():
        splits = load_or_build_splits(cfg, paths)
        print(f"splits: val_internal={len(splits['val_internal'])} train_rl={len(splits['train_rl'])}")
    else:
        print("splits: skipped (train.json missing)")

    available = [s for s in args.sources if paths.questions_file(s).is_file() and paths.has_databases(s)]
    for s in args.sources:
        if s not in available:
            print(f"{s}: questions or databases missing under {paths.raw_dir}, skipping")

    # 2. schema cache
    examples = {s: load_examples(s, paths) for s in available}
    for s in available:
        db_ids = sorted({ex.db_id for ex in examples[s]})
        t0 = time.monotonic()
        for db_id in db_ids:
            load_or_extract_schema(cfg, s, db_id, paths.db_path(s, db_id))
        print(f"{s}: schema cache ready for {len(db_ids)} databases ({time.monotonic() - t0:.1f}s)")

    # 3. gold cache
    if not args.skip_gold:
        cache = gold_cache(cfg)
        for s in available:
            t0 = time.monotonic()
            gold = ensure_gold_results(cfg, examples[s], paths, cache, retry_failed=args.retry_gold_failures)
            counts = Counter(r.status for r in gold.values())
            rows = [r.n_rows for r in gold.values() if r.ok]
            print(f"{s}: gold executed for {len(gold)} questions in {time.monotonic() - t0:.1f}s: "
                  f"{dict(counts)}; max rows={max(rows) if rows else 0}, "
                  f"slowest={max(r.elapsed_s for r in gold.values()):.2f}s")
            bad = [(ex.uid, gold[ex.uid].status, (gold[ex.uid].error or '')[:80]) for ex in examples[s] if not gold[ex.uid].ok]
            if bad:
                out = Path(cfg.paths.outputs_dir) / "reports" / f"gold_invalid_{s}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(bad, indent=1))
                print(f"{s}: {len(bad)} gold queries are not executable; listed in {out}")
        cache.close()

    # 4. prompt lengths
    if not args.skip_lengths:
        if args.model is None:
            print("prompt lengths: skipped (pass --model to tokenize)")
        else:
            model = cfg.require_model()
            tokenizer = load_tokenizer(model)
            budget = cfg.prompt.prompt_budget
            report = {"model": model.hf_id, "max_model_len": cfg.prompt.max_model_len,
                      "max_new_tokens": cfg.prompt.max_new_tokens, "prompt_budget": budget, "sources": {}}
            md = [f"# Prompt lengths for {model.hf_id}", "",
                  f"Context window {cfg.prompt.max_model_len} tokens, generation budget {cfg.prompt.max_new_tokens}, "
                  f"prompt budget {budget} tokens.", ""]
            for s in available:
                items = build_prompt_items(cfg, examples[s], tokenizer, paths, model.chat_template_kwargs)
                lens = np.array([it.n_tokens for it in items])
                per_db: dict[str, list[int]] = defaultdict(list)
                for ex, it in zip(examples[s], items):
                    per_db[ex.db_id].append(it.n_tokens)
                db_stats = {db: {"n": len(v), "min": int(min(v)), "max": int(max(v)),
                                 "over_budget": int(sum(x > budget for x in v))} for db, v in per_db.items()}
                over = {db: st for db, st in db_stats.items() if st["over_budget"]}
                pct = {f"p{p}": int(np.percentile(lens, p)) for p in (50, 90, 95, 99)}
                report["sources"][s] = {"n": int(len(lens)), "min": int(lens.min()), "max": int(lens.max()),
                                        "mean": float(lens.mean()), **pct,
                                        "n_over_budget": int((lens > budget).sum()), "per_db": db_stats}
                md += [f"## {s}: {len(lens)} prompts", "",
                       f"min {lens.min()}, mean {lens.mean():.0f}, p50 {pct['p50']}, p90 {pct['p90']}, "
                       f"p95 {pct['p95']}, p99 {pct['p99']}, max {lens.max()}; over budget: {int((lens > budget).sum())}", ""]
                if over:
                    md += ["Databases with prompts over the budget:", "", "| db | questions | max tokens | over budget |", "|---|---|---|---|"]
                    md += [f"| {db} | {st['n']} | {st['max']} | {st['over_budget']} |" for db, st in sorted(over.items(), key=lambda kv: -kv[1]['max'])]
                    md.append("")
                md += ["Longest databases:", "", "| db | questions | min tokens | max tokens |", "|---|---|---|---|"]
                md += [f"| {db} | {st['n']} | {st['min']} | {st['max']} |" for db, st in sorted(db_stats.items(), key=lambda kv: -kv[1]['max'])[:15]]
                md.append("")
                print(f"{s}: prompt tokens min={lens.min()} p50={pct['p50']} p90={pct['p90']} p99={pct['p99']} "
                      f"max={lens.max()}; over budget={int((lens > budget).sum())} ({len(over)} databases)")
            out_dir = Path(cfg.paths.outputs_dir) / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"prompt_lengths_{model.name}.json").write_text(json.dumps(report, indent=1))
            (out_dir / f"prompt_lengths_{model.name}.md").write_text("\n".join(md))
            print(f"prompt length report written to {out_dir / f'prompt_lengths_{model.name}.md'}")
    print(f"done in {time.monotonic() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
