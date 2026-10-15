"""Persistent cache of execution results (used for gold queries) stored in a SQLite file."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable

from mexql.execution.runner import DIGEST_SIZE, ExecResult


def cache_key(db_rel_path: str, sql: str) -> str:
    return hashlib.sha256(f"{db_rel_path}\n{sql}".encode("utf-8")).hexdigest()


class ResultCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS results (
                   key TEXT PRIMARY KEY, status TEXT NOT NULL, n_rows INTEGER, elapsed_s REAL,
                   error TEXT, columns TEXT, head TEXT, digests BLOB)"""
        )
        self.conn.commit()

    def get(self, key: str) -> ExecResult | None:
        row = self.conn.execute(
            "SELECT status, n_rows, elapsed_s, error, columns, head, digests FROM results WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        status, n_rows, elapsed, error, columns, head, blob = row
        digests = None
        if status == "ok" and blob is not None:
            digests = frozenset(bytes(blob[i:i + DIGEST_SIZE]) for i in range(0, len(blob), DIGEST_SIZE))
        return ExecResult(status=status, n_rows=n_rows or 0, elapsed_s=elapsed or 0.0, error=error,
                          digests=digests, columns=json.loads(columns or "[]"), head=json.loads(head or "[]"))

    def get_many(self, keys: Iterable[str]) -> dict[str, ExecResult]:
        out: dict[str, ExecResult] = {}
        for key in keys:
            res = self.get(key)
            if res is not None:
                out[key] = res
        return out

    def put_many(self, items: Iterable[tuple[str, ExecResult]]) -> None:
        rows = []
        for key, res in items:
            blob = b"".join(sorted(res.digests)) if res.digests is not None else None
            rows.append((key, res.status, res.n_rows, res.elapsed_s, res.error, json.dumps(res.columns),
                         json.dumps(res.head, default=str), blob))
        self.conn.executemany("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
