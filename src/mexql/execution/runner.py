"""Read-only SQL execution with timeouts, result-size limits and order-insensitive digests.

Result sets are compared the way the official BIRD script does (``set(cursor.fetchall())``
equality), but instead of keeping the rows we keep a 16-byte digest per row. The digest
encoding preserves Python's tuple-equality semantics for SQLite values (``1 == 1.0``,
``'1' != 1``, ``None`` only equals ``None``), so two queries match if and only if their
row *sets* are equal.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ExecStatus = Literal["ok", "sql_error", "timeout", "too_large"]

DIGEST_SIZE = 16
_FETCH_SIZE = 2000
_PROGRESS_EVERY_N_OPS = 5000


@dataclass
class ExecResult:
    status: ExecStatus
    n_rows: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    digests: frozenset[bytes] | None = None  # row digests when status == "ok"
    columns: list[str] = field(default_factory=list)
    head: list[list] = field(default_factory=list)  # first few rows, JSON-safe, for error analysis

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _canon(value: object) -> bytes:
    """Type-tagged canonical encoding of one cell, consistent with Python equality."""
    if value is None:
        return b"n"
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return b"i" + repr(int(value)).encode()
        return b"f" + repr(value).encode()
    if isinstance(value, int):
        return b"i" + repr(value).encode()
    if isinstance(value, str):
        return b"s" + value.encode("utf-8", "surrogatepass")
    if isinstance(value, (bytes, memoryview)):
        return b"b" + bytes(value)
    return b"o" + repr(value).encode()


def row_digest(row: tuple) -> bytes:
    h = hashlib.blake2b(digest_size=DIGEST_SIZE)
    for cell in row:
        enc = _canon(cell)
        h.update(struct.pack("<I", len(enc)))
        h.update(enc)
    return h.digest()


def _json_safe(value: object) -> object:
    if isinstance(value, (bytes, memoryview)):
        return f"<blob {len(bytes(value))} bytes>"
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def _authorizer(action: int, *_args) -> int:
    """Allow reads only: no ATTACH/DETACH/PRAGMA, no writes (mode=ro already blocks writes)."""
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    conn.set_authorizer(_authorizer)
    return conn


def execute_query(db_path: str | Path, sql: str, timeout_s: float = 30.0, max_rows: int = 500_000,
                  head_rows: int = 3) -> ExecResult:
    """Execute one read-only query and digest its rows; never raises for SQL problems."""
    start = time.monotonic()
    deadline = start + timeout_s
    try:
        conn = open_readonly(db_path)
    except Exception as exc:  # missing file, unreadable
        return ExecResult("sql_error", error=f"{type(exc).__name__}: {exc}", elapsed_s=time.monotonic() - start)
    timed_out = False

    def _progress() -> int:
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1  # non-zero aborts the running statement
        return 0

    conn.set_progress_handler(_progress, _PROGRESS_EVERY_N_OPS)
    digests: set[bytes] = set()
    head: list[list] = []
    columns: list[str] = []
    n_rows = 0
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        while True:
            rows = cur.fetchmany(_FETCH_SIZE)
            if not rows:
                break
            for row in rows:
                n_rows += 1
                if n_rows > max_rows:
                    return ExecResult("too_large", n_rows=n_rows, elapsed_s=time.monotonic() - start,
                                      error=f"result exceeds max_rows={max_rows}", columns=columns, head=head)
                digests.add(row_digest(row))
                if len(head) < head_rows:
                    head.append([_json_safe(v) for v in row])
            if time.monotonic() > deadline:
                timed_out = True
                raise sqlite3.OperationalError("interrupted")
        return ExecResult("ok", n_rows=n_rows, elapsed_s=time.monotonic() - start, digests=frozenset(digests),
                          columns=columns, head=head)
    except sqlite3.Error as exc:
        elapsed = time.monotonic() - start
        if timed_out or elapsed >= timeout_s:
            return ExecResult("timeout", n_rows=n_rows, elapsed_s=elapsed, error=f"timeout after {timeout_s}s",
                              columns=columns, head=head)
        return ExecResult("sql_error", elapsed_s=elapsed, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        return ExecResult("sql_error", elapsed_s=time.monotonic() - start, error=f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()


def results_match(pred: ExecResult, gold: ExecResult) -> bool:
    """Official semantics: both executed and their row sets are equal."""
    return pred.ok and gold.ok and pred.digests == gold.digests
