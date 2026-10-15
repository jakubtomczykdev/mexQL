"""Parsing model responses of the form ``<think>...</think>`` followed by a ```sql block.

Two things are reported separately:

* ``status`` / ``sql``: whether a SQL query could be extracted at all (lenient: the last
  fenced ``sql`` block anywhere in the response, even if the ``<think>`` block is missing or
  never closed). This drives execution accuracy.
* ``format_ok`` / ``format_issue``: whether the response follows the required format exactly
  (one closed ``<think>`` block, then exactly one ``sql`` block). This is what a format
  reward can use during RL.

The parser never raises on malformed input; it reports ``status="format_error"`` with a
machine-readable ``reason`` instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ParseStatus = Literal["ok", "format_error"]

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_SQL_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


@dataclass
class ParsedResponse:
    status: ParseStatus
    sql: str | None
    think: str | None
    reason: str | None = None  # why no SQL could be extracted
    format_ok: bool = False  # strict format compliance
    format_issue: str | None = None  # first violation found, if any
    n_sql_blocks: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _clean_sql(sql: str) -> str:
    sql = sql.strip()
    while sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return sql


def parse_response(text: str | None) -> ParsedResponse:
    """Extract the reasoning and the final SQL query from a model response."""
    if not text or not text.strip():
        return ParsedResponse("format_error", None, None, "empty_response", format_issue="empty_response")

    think: str | None = None
    body = text
    format_issue: str | None = None
    m = _THINK_RE.search(text)
    if m:
        think = m.group(1).strip()
        body = text[m.end():]
        if text[: m.start()].strip():
            format_issue = "text_before_think"
    elif "<think>" in text.lower():
        format_issue = "unterminated_think"  # search the whole text for SQL anyway
    else:
        format_issue = "missing_think"

    fences = _FENCE_RE.findall(body)
    sql_blocks = [code for lang, code in fences if lang.lower() in ("sql", "sqlite")]
    if not sql_blocks:
        untagged = [code for lang, code in fences if lang == "" and _SQL_START.match(code)]
        if untagged:
            sql_blocks = untagged
            format_issue = format_issue or "untagged_sql_block"
    if not sql_blocks:
        reason = "unterminated_think" if format_issue == "unterminated_think" else "no_sql_block"
        return ParsedResponse("format_error", None, think, reason, format_issue=format_issue or "no_sql_block")
    if len(sql_blocks) > 1:
        format_issue = format_issue or "multiple_sql_blocks"

    sql = _clean_sql(sql_blocks[-1])
    if not sql:
        return ParsedResponse("format_error", None, think, "empty_sql", format_issue=format_issue or "empty_sql",
                              n_sql_blocks=len(sql_blocks))
    if not _SQL_START.match(sql):
        return ParsedResponse("format_error", None, think, "not_a_select", format_issue=format_issue or "not_a_select",
                              n_sql_blocks=len(sql_blocks))
    return ParsedResponse("ok", sql, think, None, format_ok=format_issue is None, format_issue=format_issue,
                          n_sql_blocks=len(sql_blocks))
