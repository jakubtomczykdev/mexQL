#!/usr/bin/env python
"""Re-parse and re-execute the responses of an existing run without generating again
(useful after changes to the parser or the execution settings).

Example::

    python scripts/rescore.py --run outputs/runs/qwen3-4b-instruct-2507_dev_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mexql.config import load_config
from mexql.data.bird import BirdPaths, load_split
from mexql.execution.runner import results_match
from mexql.metrics import final_status, format_markdown, format_table
from mexql.parsing import parse_response
from mexql.pipeline import ensure_gold_results, execute_predictions, read_jsonl, summarize_records, write_jsonl


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config([args.base_config], args.overrides)
    metrics_file = args.run / "metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.is_file() else {}
    split = metrics.get("split") or "dev"
    paths = BirdPaths(cfg)
    examples = {ex.uid: ex for ex in load_split(split, cfg, paths)}

    records = read_jsonl(args.run / "predictions.jsonl")
    used = [examples[r["uid"]] for r in records if r["uid"] in examples]
    gold = ensure_gold_results(cfg, used, paths)
    parsed = {}
    to_execute = []
    for r in records:
        p = parse_response(r.get("response"))
        parsed[(r["uid"], r["sample_idx"])] = p
        if p.ok and gold[r["uid"]].ok and not r.get("prompt_too_long"):
            to_execute.append((r["uid"], r["sample_idx"], p.sql))
    exec_results = execute_predictions(cfg, examples, to_execute, paths) if to_execute else {}

    for r in records:
        p = parsed[(r["uid"], r["sample_idx"])]
        g = gold[r["uid"]]
        e = exec_results.get((r["uid"], r["sample_idx"]))
        match = results_match(e, g) if e is not None else False
        status = final_status(p.status, e.status if e else None, g.ok, match, prompt_too_long=bool(r.get("prompt_too_long")))
        r.update({
            "parse_status": p.status, "parse_reason": p.reason, "format_ok": p.format_ok, "format_issue": p.format_issue,
            "sql": p.sql, "exec_status": e.status if e else None, "exec_error": e.error if e else None,
            "n_rows": e.n_rows if e else None, "exec_elapsed_s": round(e.elapsed_s, 4) if e else None,
            "gold_status": g.status, "gold_error": g.error,
            "correct": (status == "correct") if status != "gold_invalid" else None, "status": status,
        })
    summary = summarize_records(records)
    write_jsonl(args.run / "predictions.jsonl", records)
    metrics["summary"] = summary
    metrics["rescored"] = True
    metrics_file.write_text(json.dumps(metrics, indent=1))
    label = f"{(metrics.get('model') or {}).get('name', '?')} / {split} (rescored)"
    (args.run / "metrics.md").write_text(format_markdown(summary, label) + "\n")
    print(format_table(summary, f"Execution accuracy: {label}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
