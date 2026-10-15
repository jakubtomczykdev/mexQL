#!/usr/bin/env python
"""Print a Markdown summary of all evaluation runs and sampling runs under outputs/.

Example::

    python scripts/collect_results.py                 # all runs
    python scripts/collect_results.py --outputs /content/drive/MyDrive/mexql/outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mexql.metrics import DIFFICULTIES


def eval_rows(outputs: Path) -> list[str]:
    rows = []
    for metrics_file in sorted(outputs.glob("runs/*/metrics.json")):
        m = json.loads(metrics_file.read_text())
        s = m["summary"]
        by = s["by_difficulty"]
        cells = [f"{by[lv]['ex']:.1f}" if lv in by else "-" for lv in DIFFICULTIES]
        fmt = s.get("format_ok_rate")
        model = (m.get("model") or {}).get("name", "dummy")
        gen = m["timing"].get("generation_s", 0.0)
        rows.append(f"| {model} | {m['split']} | {m['backend']} | {s['n_scored']} | " + " | ".join(cells)
                    + f" | **{s['ex']:.1f}** | {s['ex_all']:.1f} | {fmt:.0f}% | {gen / 60:.1f} min |"
                    if fmt is not None else
                    f"| {model} | {m['split']} | {m['backend']} | {s['n_scored']} | " + " | ".join(cells)
                    + f" | **{s['ex']:.1f}** | {s['ex_all']:.1f} | - | {gen / 60:.1f} min |")
    return rows


def sampling_rows(outputs: Path) -> list[str]:
    rows = []
    for summary_file in sorted(outputs.glob("sampling/*/summary.json")):
        s = json.loads(summary_file.read_text())
        est = s["estimate_full_split"]
        hist = ", ".join(f"{k}:{v}" for k, v in s["n_correct_histogram"].items())
        rows.append(f"| {s['model']} | {s['split']} | {s['n_questions']} / {s['n_total_in_split']} | {s['n_samples']} | "
                    f"{s['temperature']} | {s['mean_pass_rate']:.3f} | {s['pass_at_n']:.3f} | {s['all_correct']:.3f} | "
                    f"{hist} | {s['timing']['generation_s'] / 60:.1f} min | {est['total_hours']:.1f} h |")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = ap.parse_args(argv)
    ev = eval_rows(args.outputs)
    print("### Execution accuracy (greedy)\n")
    print("| model | split | backend | n | simple | moderate | challenging | EX | EX (all) | format ok | generation |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(ev) if ev else "| (no runs) |")
    sm = sampling_rows(args.outputs)
    print("\n### Train sampling (stage-2 preparation)\n")
    print("| model | split | questions | n | T | mean pass rate | pass@n | all correct | n_correct histogram | generation | est. full split |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(sm) if sm else "| (no sampling runs) |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
