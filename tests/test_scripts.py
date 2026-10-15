import json
import sys
from pathlib import Path
from types import SimpleNamespace

from mexql.pipeline import read_jsonl, write_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import select_examples  # noqa: E402
from export_official import FORMAT_ERROR_SQL, SEPARATOR, export  # noqa: E402
from sample_train import pass_rate_rows  # noqa: E402


def test_select_examples_smoke_spreads_over_split():
    cfg = SimpleNamespace(smoke_limit=4, limit=None)
    picked = select_examples(list(range(100)), cfg, smoke=True)
    assert picked == [0, 25, 50, 75]
    assert select_examples(list(range(3)), cfg, smoke=True) == [0, 1, 2]
    cfg.limit = 10
    assert select_examples(list(range(100)), cfg, smoke=False) == list(range(10))
    cfg.limit = None
    assert len(select_examples(list(range(100)), cfg, smoke=False)) == 100


def test_export_official_format(tmp_path):
    preds = [
        {"question_id": 1, "db_id": "financial", "source": "dev", "sql": "SELECT 1"},
        {"question_id": 0, "db_id": "card_games", "source": "dev", "sql": None},
    ]
    gold = tmp_path / "dev.sql"
    gold.write_text("SELECT 1\tcard_games\nSELECT 1\tfinancial\n")
    out = export(preds, tmp_path / "official", gold)
    data = json.loads(out.read_text())
    assert list(data) == ["0", "1"]  # sorted by question id
    assert data["1"] == f"SELECT 1{SEPARATOR}financial"
    assert data["0"] == f"{FORMAT_ERROR_SQL}{SEPARATOR}card_games"
    assert (tmp_path / "official" / "dev_gold.sql").read_text() == gold.read_text()
    # the vendored official loader must accept the file
    from tests.vendor import bird_official_evaluation as official

    sqls, dbs = official.package_sqls(str(tmp_path / "official") + "/", "/dbs/", mode="gpt", data_mode="dev")
    assert sqls == [FORMAT_ERROR_SQL, "SELECT 1"] and dbs == ["/dbs/card_games/card_games.sqlite", "/dbs/financial/financial.sqlite"]


def test_pass_rate_rows():
    ex = [SimpleNamespace(uid="train:1", question_id=1, db_id="x"), SimpleNamespace(uid="train:2", question_id=2, db_id="x")]
    rec = lambda uid, status, gold="ok": {"uid": uid, "status": status, "gold_status": gold, "prompt_tokens": 10, "sql": "S"}
    records = [rec("train:1", "correct"), rec("train:1", "wrong_result"), rec("train:1", "correct"),
               rec("train:2", "gold_invalid", "sql_error")] * 1 + [rec("train:2", "gold_invalid", "sql_error")] * 2
    rows = pass_rate_rows(ex, records)
    assert rows[0]["n"] == 3 and rows[0]["n_correct"] == 2 and abs(rows[0]["pass_rate"] - 2 / 3) < 1e-9
    assert rows[0]["statuses"] == {"correct": 2, "wrong_result": 1}
    assert rows[1]["pass_rate"] is None and rows[1]["gold_status"] == "sql_error"


def test_jsonl_roundtrip(tmp_path):
    rows = [{"a": 1, "b": "zażółć"}, {"a": None}]
    write_jsonl(tmp_path / "x.jsonl", rows)
    assert read_jsonl(tmp_path / "x.jsonl") == rows
