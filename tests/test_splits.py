from collections import Counter

import pytest

from mexql.data.bird import BirdPaths, Example, load_examples, load_or_build_splits, load_split, make_splits
from tests.conftest import REPO


def _fake_train(n_per_db):
    examples, qid = [], 0
    for db, n in n_per_db.items():
        for _ in range(n):
            examples.append(Example(f"train:{qid}", "train", qid, db, "q", "", "SELECT 1", None))
            qid += 1
    return examples


def test_make_splits_stratified_and_deterministic():
    train = _fake_train({"a": 100, "b": 50, "c": 7, "d": 3})
    s1 = make_splits(train, val_size=16, seed=1)
    s2 = make_splits(train, val_size=16, seed=1)
    s3 = make_splits(train, val_size=16, seed=2)
    assert s1 == s2 and s1 != s3
    assert len(s1["val_internal"]) == 16 and len(s1["train_rl"]) == 144
    assert not set(s1["val_internal"]) & set(s1["train_rl"])
    by_db = {ex.question_id: ex.db_id for ex in train}
    counts = Counter(by_db[q] for q in s1["val_internal"])
    assert counts["a"] == 10 and counts["b"] == 5 and counts["c"] == 1 and counts["d"] == 0
    assert s1["val_internal"] == sorted(s1["val_internal"])


def test_make_splits_rejects_bad_sizes():
    train = _fake_train({"a": 5})
    with pytest.raises(ValueError):
        make_splits(train, val_size=0, seed=0)
    with pytest.raises(ValueError):
        make_splits(train, val_size=5, seed=0)


@pytest.mark.bird_data
def test_real_train_split(cfg):
    paths = BirdPaths(cfg)
    if not paths.questions_file("train").is_file():
        pytest.skip("train.json not available")
    train = load_examples("train", paths)
    assert len(train) == 9428 and train[0].difficulty is None
    splits = load_or_build_splits(cfg, paths)
    assert len(splits["val_internal"]) == 500
    assert len(splits["val_internal"]) + len(splits["train_rl"]) == len(train)
    again = load_or_build_splits(cfg, paths)  # read back from disk
    assert again == splits
    val = load_split("val_internal", cfg, paths)
    assert len(val) == 500 and all(ex.source == "train" for ex in val)
    dev_file = REPO / "data" / "raw" / cfg.data.dev_dirname / "dev.json"
    if dev_file.is_file():
        dev = load_split("dev", cfg, paths)
        assert len(dev) == 1534 and Counter(ex.difficulty for ex in dev)["simple"] == 925
