"""Execution accuracy (EX) by difficulty and a breakdown of failure modes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

FINAL_STATUSES = (
    "correct",          # executed and row set equals the gold row set
    "wrong_result",     # executed, different rows
    "format_error",     # no parsable SQL in the response
    "prompt_too_long",  # prompt exceeded the context budget; not generated
    "sql_error",        # SQLite raised an error
    "timeout",          # exceeded the per-query time limit
    "too_large",        # result set exceeded max_rows
    "gold_invalid",     # the gold query itself fails; excluded from EX
)
DIFFICULTIES = ("simple", "moderate", "challenging")


@dataclass
class ScoredItem:
    uid: str
    difficulty: str | None
    status: str
    format_ok: bool | None = None  # strict output-format compliance, independent of correctness

    @property
    def correct(self) -> bool:
        return self.status == "correct"


def final_status(parse_status: str, exec_status: str | None, gold_ok: bool, match: bool,
                 prompt_too_long: bool = False) -> str:
    if not gold_ok:
        return "gold_invalid"
    if prompt_too_long:
        return "prompt_too_long"
    if parse_status != "ok":
        return "format_error"
    if exec_status != "ok":
        return exec_status or "sql_error"
    return "correct" if match else "wrong_result"


def summarize(items: list[ScoredItem]) -> dict:
    """Aggregate EX overall and per difficulty; ``ex`` excludes gold_invalid, ``ex_all`` does not."""
    counts = Counter(it.status for it in items)
    scored = [it for it in items if it.status != "gold_invalid"]
    n_correct = sum(it.correct for it in scored)
    by_diff: dict[str, dict] = {}
    groups: dict[str, list[ScoredItem]] = defaultdict(list)
    for it in scored:
        groups[it.difficulty or "unlabeled"].append(it)
    for level in list(DIFFICULTIES) + sorted(k for k in groups if k not in DIFFICULTIES):
        g = groups.get(level)
        if not g:
            continue
        c = sum(it.correct for it in g)
        by_diff[level] = {"n": len(g), "correct": c, "ex": 100.0 * c / len(g)}
    with_format = [it for it in scored if it.format_ok is not None]
    return {
        "n": len(items),
        "n_scored": len(scored),
        "correct": n_correct,
        "ex": 100.0 * n_correct / len(scored) if scored else 0.0,
        "ex_all": 100.0 * n_correct / len(items) if items else 0.0,
        "format_ok_rate": 100.0 * sum(it.format_ok for it in with_format) / len(with_format) if with_format else None,
        "by_difficulty": by_diff,
        "status_counts": {s: counts.get(s, 0) for s in FINAL_STATUSES if counts.get(s, 0)},
    }


def format_table(summary: dict, title: str = "") -> str:
    """Plain-text report in the spirit of the official script's output."""
    lines = []
    if title:
        lines.append(title)
    levels = list(summary["by_difficulty"].keys()) + ["total"]
    lines.append("{:12}".format("") + "".join("{:>14}".format(lv) for lv in levels))
    counts = [summary["by_difficulty"][lv]["n"] for lv in levels[:-1]] + [summary["n_scored"]]
    lines.append("{:12}".format("count") + "".join("{:>14}".format(c) for c in counts))
    accs = [summary["by_difficulty"][lv]["ex"] for lv in levels[:-1]] + [summary["ex"]]
    lines.append("{:12}".format("EX (%)") + "".join("{:>14.2f}".format(a) for a in accs))
    lines.append("")
    lines.append("status breakdown: " + ", ".join(f"{k}={v}" for k, v in summary["status_counts"].items()))
    if summary.get("format_ok_rate") is not None:
        lines.append(f"strict format compliance (<think> + one sql block): {summary['format_ok_rate']:.1f}%")
    if summary["n_scored"] != summary["n"]:
        lines.append(f"EX over all {summary['n']} questions (gold_invalid counted as wrong): {summary['ex_all']:.2f}")
    return "\n".join(lines)


def format_markdown(summary: dict, label: str) -> str:
    levels = list(summary["by_difficulty"].keys())
    header = "| model / split | " + " | ".join(levels) + " | total | n | format ok |"
    sep = "|" + "---|" * (len(levels) + 4)
    fmt = summary.get("format_ok_rate")
    cells = [f"{summary['by_difficulty'][lv]['ex']:.1f}" for lv in levels] + [
        f"{summary['ex']:.1f}", str(summary["n_scored"]), f"{fmt:.1f}%" if fmt is not None else "-"]
    row = f"| {label} | " + " | ".join(cells) + " |"
    breakdown = ", ".join(f"{k} {v}" for k, v in summary["status_counts"].items())
    return "\n".join([header, sep, row, "", f"Status breakdown: {breakdown}"])
