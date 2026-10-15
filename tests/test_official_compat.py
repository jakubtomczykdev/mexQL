"""Agreement between our execution-accuracy implementation and the official BIRD script.

The vendored ``tests/vendor/bird_official_evaluation.py`` is imported unchanged; for every
(predicted, gold) pair both implementations must return the same 0/1 verdict.
"""

import json

import pytest

from mexql.execution.runner import execute_query, results_match
from tests.conftest import BIRD_DEV_DBS, REPO
from tests.vendor import bird_official_evaluation as official

INFINITE = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"


def official_verdict(pred: str, gold: str, db, timeout: float) -> int:
    return official.execute_model(pred, gold, str(db), 0, timeout)["res"]


def our_verdict(pred: str, gold: str, db, timeout: float) -> int:
    p = execute_query(db, pred, timeout_s=timeout)
    g = execute_query(db, gold, timeout_s=timeout)
    return int(results_match(p, g))


# (predicted, gold, expected verdict) on the tiny database; all read-only.
SYNTHETIC = [
    ("SELECT id FROM people", "SELECT id FROM people", 1),
    ("SELECT id FROM people ORDER BY id DESC", "SELECT id FROM people ORDER BY id", 1),  # order ignored
    ("SELECT age FROM people WHERE age IS NOT NULL", "SELECT DISTINCT age FROM people WHERE age IS NOT NULL", 1),  # duplicates ignored
    ("SELECT 1", "SELECT 1.0", 1),  # int == float
    ("SELECT 1000", "SELECT 1e3", 1),
    ("SELECT 1", "SELECT '1'", 0),  # int != str
    ("SELECT 2", "SELECT 2.5", 0),
    ("SELECT NULL", "SELECT NULL", 1),
    ("SELECT NULL", "SELECT 0", 0),
    ("SELECT NULL", "SELECT ''", 0),
    ("SELECT id, age FROM people", "SELECT age, id FROM people", 0),  # column order matters
    ("SELECT id, age, height FROM people", "SELECT id, age FROM people", 0),  # extra column
    ("SELECT id FROM people WHERE 1 = 0", "SELECT id FROM people WHERE age > 1000", 1),  # both empty
    ("SELECT * FROM people", "SELECT id, `full name`, age, height, nickname, photo FROM people", 1),
    ("SELECT `full name` FROM people WHERE id = 2", "SELECT 'Bob O''Neil'", 1),
    ("SELECT `full name` FROM people WHERE id = 3", "SELECT 'Cé Dé'", 1),
    ("SELECT age > 30 FROM people", "SELECT CASE WHEN age > 30 THEN 1 ELSE 0 END FROM people", 0),  # NULL age differs
    ("SELECT age > 30 FROM people WHERE age IS NOT NULL", "SELECT CASE WHEN age > 30 THEN 1 ELSE 0 END FROM people WHERE age IS NOT NULL", 1),
    ("SELECT COUNT(*) FROM people", "SELECT COUNT(id) FROM people", 1),
    ("SELECT COUNT(*) FROM people", "SELECT COUNT(age) FROM people", 0),
    ("SELECT AVG(amount) FROM orders", "SELECT SUM(amount) / COUNT(*) FROM orders", 1),
    ("SELECT age / 2 FROM people", "SELECT age / 2.0 FROM people", 0),  # integer division vs real
    ("SELECT 0.1 + 0.2", "SELECT 0.3", 0),
    ("SELECT CAST('12' AS INTEGER)", "SELECT 12", 1),
    ("SELECT CAST(12 AS TEXT)", "SELECT 12", 0),
    ("SELECT person_id, SUM(amount) FROM orders GROUP BY person_id", "SELECT person_id, SUM(amount) FROM orders GROUP BY person_id HAVING SUM(amount) > 0", 1),
    ("SELECT person_id FROM orders GROUP BY person_id HAVING COUNT(*) > 1", "SELECT 1", 1),
    ("SELECT p.`full name` FROM people p JOIN orders o ON o.person_id = p.id", "SELECT DISTINCT p.`full name` FROM people p JOIN orders o ON o.person_id = p.id", 1),
    ("SELECT `full name` FROM people LIMIT 1", "SELECT `full name` FROM people ORDER BY id LIMIT 1", 1),
    ("SELECT `full name` FROM people ORDER BY id DESC LIMIT 1", "SELECT `full name` FROM people ORDER BY id LIMIT 1", 0),
    ("SELECT photo FROM people WHERE id = 1", "SELECT X'00ff'", 1),  # blobs
    ("SELECT photo FROM people WHERE id = 1", "SELECT '00ff'", 0),
    ("SELECT nope FROM people", "SELECT id FROM people", 0),  # predicted error
    ("SELEC id FROM people", "SELECT id FROM people", 0),  # syntax error
    ("SELECT id FROM people", "SELECT nope FROM people", 0),  # gold error
    ("SELECT 1; SELECT 2", "SELECT 1", 0),  # multiple statements
    ("SELECT id FROM people WHERE nickname = 'annie'", "SELECT 1", 1),
    ("SELECT id FROM people WHERE nickname IS NULL", "SELECT 2 UNION SELECT 4", 1),
    ("SELECT id FROM people WHERE nickname IS NULL", "SELECT 2 UNION ALL SELECT 4 UNION ALL SELECT 4", 1),
    ("SELECT `Order` FROM orders", "SELECT 'first' UNION SELECT 'second'", 1),
    ("SELECT 'a', 1", "SELECT 'a', 1.0", 1),
    ("SELECT MAX(height) FROM people", "SELECT 1.82", 1),
    ("SELECT MAX(height) FROM people", "SELECT 1.8", 0),
    ("SELECT id FROM people WHERE age = (SELECT MIN(age) FROM people)", "SELECT 1 UNION SELECT 3", 1),
    ("WITH a AS (SELECT id FROM people) SELECT id FROM a", "SELECT id FROM people", 1),
    (INFINITE, "SELECT 1", 0),  # predicted query times out
    ("SELECT 1", INFINITE, 0),  # gold query times out
]


@pytest.mark.parametrize("pred, gold, expected", SYNTHETIC)
def test_agreement_on_synthetic_pairs(tiny_db, pred, gold, expected):
    timeout = 1.5
    ours = our_verdict(pred, gold, tiny_db, timeout)
    theirs = official_verdict(pred, gold, tiny_db, timeout)
    assert ours == theirs == expected


def _perturbations(gold: str) -> list[str]:
    return [
        gold,
        f"SELECT * FROM ({gold}) AS sub ORDER BY 1 DESC",
        f"SELECT * FROM ({gold}) AS sub LIMIT 1",
        f"SELECT COUNT(*) FROM ({gold}) AS sub",
        f"SELECT sub.*, 0 FROM ({gold}) AS sub",
        gold.replace("SELECT", "SELEC", 1),
    ]


@pytest.mark.bird_data
@pytest.mark.slow
def test_agreement_on_bird_dev_examples():
    dev_json = REPO / "data" / "raw" / "dev_20240627" / "dev.json"
    if not (dev_json.is_file() and BIRD_DEV_DBS.is_dir()):
        pytest.skip("BIRD dev data not available")
    with open(dev_json, "r", encoding="utf-8") as fh:
        dev = json.load(fh)
    chosen = dev[::70][:22]  # spread across databases and difficulties
    assert len({ex["db_id"] for ex in chosen}) >= 8
    disagreements = []
    verdict_counts = {0: 0, 1: 0}
    for ex in chosen:
        db = BIRD_DEV_DBS / ex["db_id"] / f"{ex['db_id']}.sqlite"
        for pred in _perturbations(ex["SQL"]):
            ours = our_verdict(pred, ex["SQL"], db, 30.0)
            theirs = official_verdict(pred, ex["SQL"], db, 30.0)
            verdict_counts[theirs] += 1
            if ours != theirs:
                disagreements.append((ex["question_id"], pred, ours, theirs))
    assert not disagreements, disagreements
    assert verdict_counts[0] > 0 and verdict_counts[1] >= len(chosen)  # both verdicts exercised
