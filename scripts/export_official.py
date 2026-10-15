#!/usr/bin/env python
"""Convert a predictions.jsonl of a dev run into the input format of the official BIRD
evaluation script, so the EX number can be cross-checked with the upstream code.

Example::

    python scripts/export_official.py --run outputs/runs/qwen3-4b-instruct-2507_dev
    # then, inside a checkout of https://github.com/AlibabaResearch/DAMO-ConvAI (bird/llm):
    python -u src/evaluation.py --predicted_sql_path <out>/ --ground_truth_path <out>/ \
        --data_mode dev --db_root_path data/raw/dev_20240627/dev_databases/ \
        --diff_json_path data/raw/dev_20240627/dev.json --num_cpus 8 --meta_time_out 30
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from mexql.pipeline import read_jsonl

SEPARATOR = "\t----- bird -----\t"
# Questions without parsable SQL are exported as a query that always fails, which the
# official script scores as 0 (the same verdict our pipeline gives to format errors).
FORMAT_ERROR_SQL = "SELECT * FROM __mexql_format_error__"


def export(predictions: list[dict], out_dir: Path, gold_sql_file: Path | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    predict = {}
    for r in sorted(predictions, key=lambda r: r["question_id"]):
        sql = r.get("sql") or FORMAT_ERROR_SQL
        predict[str(r["question_id"])] = f"{sql}{SEPARATOR}{r['db_id']}"
    out = out_dir / "predict_dev.json"
    out.write_text(json.dumps(predict, indent=1, ensure_ascii=False))
    if gold_sql_file and gold_sql_file.is_file():
        shutil.copy(gold_sql_file, out_dir / "dev_gold.sql")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="run directory containing predictions.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <run>/official/")
    ap.add_argument("--gold-sql", type=Path, default=Path("data/raw/dev_20240627/dev.sql"))
    args = ap.parse_args(argv)
    predictions = read_jsonl(args.run / "predictions.jsonl")
    if any(r["source"] != "dev" for r in predictions):
        ap.error("the official script needs difficulty labels; only dev runs can be exported")
    out_dir = args.out_dir or (args.run / "official")
    out = export(predictions, out_dir, args.gold_sql)
    print(f"wrote {out} ({len(predictions)} predictions) and dev_gold.sql")
    print("run the official script with --predicted_sql_path and --ground_truth_path both set to "
          f"'{out_dir}/' (trailing slash required by the upstream code)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
