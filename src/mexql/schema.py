"""Schema extraction from SQLite files and rendering as CREATE TABLE statements.

The DDL is rebuilt from ``PRAGMA table_info`` / ``PRAGMA foreign_key_list`` instead of the
raw ``sqlite_master.sql`` text so that every database is rendered in the same compact,
normalised style. Up to ``max_examples`` distinct example values per column are appended
as comments; they are taken from the first ``scan_rows`` non-null rows, which keeps the
extraction cheap even on multi-gigabyte databases.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mexql.config import Config

_BARE_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Identifiers that collide with SQL keywords are quoted even when they look bare.
_KEYWORDS = {
    "ABORT", "ACTION", "ADD", "AFTER", "ALL", "ALTER", "ALWAYS", "ANALYZE", "AND", "AS", "ASC", "ATTACH",
    "AUTOINCREMENT", "BEFORE", "BEGIN", "BETWEEN", "BY", "CASCADE", "CASE", "CAST", "CHECK", "COLLATE",
    "COLUMN", "COMMIT", "CONFLICT", "CONSTRAINT", "CREATE", "CROSS", "CURRENT", "CURRENT_DATE",
    "CURRENT_TIME", "CURRENT_TIMESTAMP", "DATABASE", "DEFAULT", "DEFERRABLE", "DEFERRED", "DELETE", "DESC",
    "DETACH", "DISTINCT", "DO", "DROP", "EACH", "ELSE", "END", "ESCAPE", "EXCEPT", "EXCLUDE", "EXCLUSIVE",
    "EXISTS", "EXPLAIN", "FAIL", "FILTER", "FIRST", "FOLLOWING", "FOR", "FOREIGN", "FROM", "FULL",
    "GENERATED", "GLOB", "GROUP", "GROUPS", "HAVING", "IF", "IGNORE", "IMMEDIATE", "IN", "INDEX", "INDEXED",
    "INITIALLY", "INNER", "INSERT", "INSTEAD", "INTERSECT", "INTO", "IS", "ISNULL", "JOIN", "KEY", "LAST",
    "LEFT", "LIKE", "LIMIT", "MATCH", "MATERIALIZED", "NATURAL", "NO", "NOT", "NOTHING", "NOTNULL", "NULL",
    "NULLS", "OF", "OFFSET", "ON", "OR", "ORDER", "OTHERS", "OUTER", "OVER", "PARTITION", "PLAN", "PRAGMA",
    "PRECEDING", "PRIMARY", "QUERY", "RAISE", "RANGE", "RECURSIVE", "REFERENCES", "REGEXP", "REINDEX",
    "RELEASE", "RENAME", "REPLACE", "RESTRICT", "RETURNING", "RIGHT", "ROLLBACK", "ROW", "ROWS", "SAVEPOINT",
    "SELECT", "SET", "TABLE", "TEMP", "TEMPORARY", "THEN", "TIES", "TO", "TRANSACTION", "TRIGGER", "UNBOUNDED",
    "UNION", "UNIQUE", "UPDATE", "USING", "VACUUM", "VALUES", "VIEW", "VIRTUAL", "WHEN", "WHERE", "WINDOW",
    "WITH", "WITHOUT",
}


@dataclass
class Column:
    name: str
    type: str
    notnull: bool
    pk: int  # 1-based position in the primary key, 0 if not part of it
    examples: list[str] = field(default_factory=list)


@dataclass
class ForeignKey:
    from_columns: list[str]
    to_table: str
    to_columns: list[str]


@dataclass
class Table:
    name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey]


@dataclass
class DbSchema:
    db_id: str
    tables: list[Table]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DbSchema":
        tables = [
            Table(
                name=t["name"],
                columns=[Column(**c) for c in t["columns"]],
                foreign_keys=[ForeignKey(**fk) for fk in t["foreign_keys"]],
            )
            for t in data["tables"]
        ]
        return cls(db_id=data["db_id"], tables=tables)


def quote_ident(name: str) -> str:
    """Quote an identifier the way BIRD gold queries do (backticks) when it is not a bare word."""
    if _BARE_IDENT.fullmatch(name) and name.upper() not in _KEYWORDS:
        return name
    if "`" in name:
        return '"' + name.replace('"', '""') + '"'
    return f"`{name}`"


def format_example(value: object, max_chars: int) -> str | None:
    """Render a cell value as a SQL-ish literal for the schema comment; None if not renderable."""
    if value is None:
        return None
    if isinstance(value, (bytes, memoryview)):
        return None
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if text == "":
        return None
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return "'" + text.replace("'", "''") + "'"


def _column_examples(conn: sqlite3.Connection, table: str, column: str, max_examples: int, max_chars: int,
                     scan_rows: int) -> list[str]:
    q_table, q_col = quote_ident(table), quote_ident(column)
    try:
        rows = conn.execute(
            f"SELECT {q_col} FROM {q_table} WHERE {q_col} IS NOT NULL LIMIT {int(scan_rows)}"
        ).fetchall()
    except sqlite3.Error:
        return []
    seen: list[str] = []
    for (value,) in rows:
        rendered = format_example(value, max_chars)
        if rendered is None or rendered in seen:
            continue
        seen.append(rendered)
        if len(seen) >= max_examples:
            break
    return seen


def extract_schema(db_path: str | Path, db_id: str, max_examples: int = 3, max_chars: int = 40,
                   scan_rows: int = 200) -> DbSchema:
    """Read tables, columns, keys and example values from a SQLite file (read-only)."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(
            f"SQLite database not found: {db_path}. Download and unpack the matching BIRD split first "
            "(scripts/download_data.py --dev / --train)."
        )
    uri = f"file:{db_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
            )
        ]
        tables: list[Table] = []
        for name in names:
            cols: list[Column] = []
            for _cid, col_name, col_type, notnull, _default, pk in conn.execute(
                f"PRAGMA table_info({quote_ident(name)})"
            ):
                examples = _column_examples(conn, name, col_name, max_examples, max_chars, scan_rows) if max_examples else []
                cols.append(Column(name=col_name, type=(col_type or "").strip(), notnull=bool(notnull), pk=int(pk),
                                   examples=examples))
            fks: dict[int, ForeignKey] = {}
            for fk_id, _seq, to_table, from_col, to_col, *_ in conn.execute(
                f"PRAGMA foreign_key_list({quote_ident(name)})"
            ):
                fk = fks.setdefault(fk_id, ForeignKey(from_columns=[], to_table=to_table, to_columns=[]))
                fk.from_columns.append(from_col)
                fk.to_columns.append(to_col if to_col is not None else "")
            tables.append(Table(name=name, columns=cols, foreign_keys=[fks[k] for k in sorted(fks)]))
    finally:
        conn.close()
    return DbSchema(db_id=db_id, tables=tables)


def render_ddl(schema: DbSchema) -> str:
    """Render the schema as CREATE TABLE statements with example values as comments."""
    statements: list[str] = []
    for table in schema.tables:
        lines: list[str] = []
        pk_cols = [c.name for c in sorted((c for c in table.columns if c.pk), key=lambda c: c.pk)]
        for col in table.columns:
            parts = [quote_ident(col.name)]
            if col.type:
                parts.append(col.type)
            if len(pk_cols) == 1 and col.pk:
                parts.append("PRIMARY KEY")
            if col.notnull and not col.pk:
                parts.append("NOT NULL")
            line = " ".join(parts)
            if col.examples:
                line += f", -- examples: {', '.join(col.examples)}"
            else:
                line += ","
            lines.append("  " + line)
        if len(pk_cols) > 1:
            lines.append("  PRIMARY KEY (" + ", ".join(quote_ident(c) for c in pk_cols) + "),")
        for fk in table.foreign_keys:
            to_cols = ", ".join(quote_ident(c) for c in fk.to_columns if c)
            ref = quote_ident(fk.to_table) + (f"({to_cols})" if to_cols else "")
            lines.append(
                "  FOREIGN KEY (" + ", ".join(quote_ident(c) for c in fk.from_columns) + f") REFERENCES {ref},"
            )
        if lines:
            # drop the trailing comma of the last line (keep an inline comment intact)
            last = lines[-1]
            lines[-1] = last.replace(", -- examples:", " -- examples:", 1) if ", -- examples:" in last else last.rstrip(",")
        statements.append(f"CREATE TABLE {quote_ident(table.name)} (\n" + "\n".join(lines) + "\n);")
    return "\n\n".join(statements)


def schema_cache_path(cfg: Config, source: str, db_id: str) -> Path:
    return Path(cfg.paths.processed_dir) / "schema_cache" / source / f"{db_id}.json"


def load_or_extract_schema(cfg: Config, source: str, db_id: str, db_path: Path) -> DbSchema:
    """Return the cached schema for a database, extracting and caching it on first use."""
    cache = schema_cache_path(cfg, source, db_id)
    if cache.is_file():
        with open(cache, "r", encoding="utf-8") as fh:
            return DbSchema.from_dict(json.load(fh))
    p = cfg.prompt
    schema = extract_schema(db_path, db_id, p.max_examples_per_column, p.max_example_chars, p.example_scan_rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(schema.to_dict(), fh, ensure_ascii=False)
    return schema
