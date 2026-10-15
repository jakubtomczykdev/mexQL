import sqlite3

from mexql.schema import DbSchema, extract_schema, quote_ident, render_ddl


def test_quote_ident():
    assert quote_ident("age") == "age"
    assert quote_ident("full name") == "`full name`"
    assert quote_ident("Order") == "`Order`"
    assert quote_ident("3d") == "`3d`"


def test_extract_and_render(tiny_db):
    schema = extract_schema(tiny_db, "tiny", max_examples=3, max_chars=9, scan_rows=100)
    assert [t.name for t in schema.tables] == ["people", "orders"]
    people = schema.tables[0]
    by_name = {c.name: c for c in people.columns}
    assert by_name["id"].pk == 1 and by_name["full name"].notnull
    assert by_name["age"].examples == ["30", "41"]  # distinct values only
    assert by_name["full name"].examples == ["'Ann Lee'", "'Bob O''Ne…'", "'Cé Dé'"]  # escaped and truncated
    assert by_name["photo"].examples == []  # blobs are not rendered
    assert by_name["nickname"].examples == ["'annie'", "'cd'"]
    orders = schema.tables[1]
    assert orders.foreign_keys[0].from_columns == ["person_id"] and orders.foreign_keys[0].to_table == "people"
    assert [c.name for c in orders.columns if c.pk] == ["order_id", "person_id"]

    ddl = render_ddl(schema)
    assert "CREATE TABLE people (" in ddl
    assert "  id INTEGER PRIMARY KEY, -- examples: 1, 2, 3" in ddl
    assert "  `full name` TEXT NOT NULL, -- examples: 'Ann Lee', 'Bob O''Ne…', 'Cé Dé'" in ddl
    assert "  photo BLOB" in ddl
    assert "  PRIMARY KEY (order_id, person_id)," in ddl
    assert "  FOREIGN KEY (person_id) REFERENCES people(id)" in ddl
    assert "`Order` TEXT, -- examples: 'first', 'second'" in ddl
    assert ddl.count(";") == 2  # two statements
    # the rendered DDL is itself valid SQLite
    conn = sqlite3.connect(":memory:")
    conn.executescript(ddl)
    assert {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"people", "orders"}


def test_roundtrip_dict(tiny_db):
    schema = extract_schema(tiny_db, "tiny")
    again = DbSchema.from_dict(schema.to_dict())
    assert render_ddl(again) == render_ddl(schema)


def test_no_examples(tiny_db):
    ddl = render_ddl(extract_schema(tiny_db, "tiny", max_examples=0))
    assert "examples" not in ddl
