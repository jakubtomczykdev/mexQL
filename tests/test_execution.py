import time

import pytest

from mexql.execution.cache import ResultCache, cache_key
from mexql.execution.pool import ExecTask, run_parallel
from mexql.execution.runner import execute_query, results_match, row_digest

INFINITE = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
MANY_ROWS = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 5000) SELECT x FROM c"


def test_digest_equality_semantics():
    assert row_digest((1,)) == row_digest((1.0,))
    assert row_digest((1,)) != row_digest(("1",))
    assert row_digest((None,)) == row_digest((None,))
    assert row_digest((None,)) != row_digest((0,))
    assert row_digest((2.5,)) != row_digest((2,))
    assert row_digest(("a", "b")) != row_digest(("ab",))  # length-prefixed cells
    assert row_digest((b"x",)) != row_digest(("x",))


def test_ok_query(tiny_db):
    r = execute_query(tiny_db, "SELECT id, age FROM people ORDER BY id")
    assert r.ok and r.n_rows == 4 and r.columns == ["id", "age"] and r.head[0] == [1, 30]
    r2 = execute_query(tiny_db, "SELECT id, age FROM people ORDER BY id DESC")
    assert results_match(r, r2)  # order-insensitive
    r3 = execute_query(tiny_db, "SELECT age FROM people WHERE age IS NOT NULL")
    r4 = execute_query(tiny_db, "SELECT DISTINCT age FROM people WHERE age IS NOT NULL")
    assert results_match(r3, r4)  # set semantics, duplicates do not matter
    assert not results_match(r, execute_query(tiny_db, "SELECT age, id FROM people"))


def test_sql_error(tiny_db):
    r = execute_query(tiny_db, "SELECT nope FROM people")
    assert r.status == "sql_error" and "no such column" in r.error
    assert execute_query(tiny_db, "SELECT 1; SELECT 2").status == "sql_error"
    assert execute_query("/nonexistent/x.sqlite", "SELECT 1").status == "sql_error"


def test_read_only_and_authorizer(tiny_db):
    assert execute_query(tiny_db, "DELETE FROM people").status == "sql_error"
    assert execute_query(tiny_db, "INSERT INTO people(id, `full name`) VALUES (9, 'x')").status == "sql_error"
    assert execute_query(tiny_db, "PRAGMA writable_schema = 1").status == "sql_error"
    assert execute_query(tiny_db, "ATTACH ':memory:' AS m").status == "sql_error"
    assert execute_query(tiny_db, "SELECT COUNT(*) FROM people").ok  # still intact
    assert execute_query(tiny_db, "SELECT COUNT(*) FROM people").head == [[4]]


def test_timeout(tiny_db):
    t0 = time.monotonic()
    r = execute_query(tiny_db, INFINITE, timeout_s=0.5)
    assert r.status == "timeout" and time.monotonic() - t0 < 5


def test_too_large(tiny_db):
    r = execute_query(tiny_db, MANY_ROWS, max_rows=1000)
    assert r.status == "too_large" and r.n_rows == 1001
    assert execute_query(tiny_db, MANY_ROWS, max_rows=5000).ok


def test_pool_mixed(tiny_db):
    tasks = [
        ExecTask("ok", str(tiny_db), "SELECT id FROM people"),
        ExecTask("err", str(tiny_db), "SELECT nope"),
        ExecTask("slow", str(tiny_db), INFINITE),
        ExecTask("big", str(tiny_db), MANY_ROWS),
    ] + [ExecTask(f"dup{i}", str(tiny_db), "SELECT COUNT(*) FROM orders") for i in range(10)]
    res = run_parallel(tasks, timeout_s=0.5, max_rows=100, num_workers=3, grace_s=1.0)
    assert set(res) == {t.key for t in tasks}
    assert res["ok"].ok and res["ok"].n_rows == 4
    assert res["err"].status == "sql_error"
    assert res["slow"].status == "timeout"
    assert res["big"].status == "too_large"
    assert all(res[f"dup{i}"].head == [[3]] for i in range(10))


def test_pool_empty():
    assert run_parallel([], num_workers=2) == {}


def test_cache_roundtrip(tiny_db, tmp_path):
    cache = ResultCache(tmp_path / "c.sqlite")
    r = execute_query(tiny_db, "SELECT id, `full name` FROM people")
    bad = execute_query(tiny_db, "SELECT nope")
    k1, k2 = cache_key("dev/tiny", "q1"), cache_key("dev/tiny", "q2")
    assert k1 != k2
    cache.put_many([(k1, r), (k2, bad)])
    back = cache.get(k1)
    assert back.ok and back.digests == r.digests and back.columns == r.columns and back.head == r.head
    assert cache.get(k2).status == "sql_error" and cache.get(k2).digests is None
    assert cache.get("missing") is None
    assert set(cache.get_many([k1, k2, "missing"])) == {k1, k2}
    cache.close()
