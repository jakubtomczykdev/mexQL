"""Loading the BIRD benchmark files and building the internal train/val split.

Layout expected under ``paths.raw_dir`` (as shipped in the official archives)::

    dev_20240627/dev.json, dev.sql, dev_tables.json, dev_databases/<db_id>/<db_id>.sqlite
    train/train.json, train_gold.sql, train_tables.json, train_databases/<db_id>/<db_id>.sqlite

``train.json`` has no ``question_id`` and no ``difficulty``; we use the position in the
file as a stable id.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from mexql.config import Config

Source = Literal["dev", "train"]
SPLITS = ("dev", "val_internal", "train_rl", "train")


@dataclass(frozen=True)
class Example:
    uid: str  # "<source>:<question_id>", unique across sources
    source: str  # "dev" | "train"
    question_id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str | None  # only the dev set is annotated

    def to_dict(self) -> dict:
        return asdict(self)


class BirdPaths:
    """Resolves file locations for a given raw directory."""

    def __init__(self, cfg: Config):
        self.raw_dir = Path(cfg.paths.raw_dir)
        self.dev_dir = self.raw_dir / cfg.data.dev_dirname
        self.train_dir = self.raw_dir / cfg.data.train_dirname

    def questions_file(self, source: Source) -> Path:
        return self.dev_dir / "dev.json" if source == "dev" else self.train_dir / "train.json"

    def databases_dir(self, source: Source) -> Path:
        return self.dev_dir / "dev_databases" if source == "dev" else self.train_dir / "train_databases"

    def db_path(self, source: Source, db_id: str) -> Path:
        return self.databases_dir(source) / db_id / f"{db_id}.sqlite"

    def db_rel_path(self, source: Source, db_id: str) -> str:
        """Stable identifier of a database, independent of the machine's raw_dir."""
        return f"{source}/{db_id}"

    def has_databases(self, source: Source) -> bool:
        return self.databases_dir(source).is_dir()


def load_examples(source: Source, paths: BirdPaths) -> list[Example]:
    file = paths.questions_file(source)
    if not file.is_file():
        raise FileNotFoundError(f"Missing {file}. Run scripts/download_data.py first.")
    with open(file, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    examples: list[Example] = []
    for idx, row in enumerate(rows):
        qid = int(row.get("question_id", idx))
        examples.append(
            Example(
                uid=f"{source}:{qid}",
                source=source,
                question_id=qid,
                db_id=row["db_id"],
                question=row["question"].strip(),
                evidence=(row.get("evidence") or "").strip(),
                gold_sql=row["SQL"].strip(),
                difficulty=row.get("difficulty"),
            )
        )
    ids = [e.question_id for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{file}: question ids are not unique")
    return examples


def make_splits(train_examples: list[Example], val_size: int, seed: int) -> dict[str, list[int]]:
    """Hold out ``val_size`` train questions, stratified by database.

    Per-database quotas are proportional to database size (largest-remainder rounding),
    the choice inside each database is a seeded shuffle. Returns sorted question ids.
    """
    if val_size <= 0 or val_size >= len(train_examples):
        raise ValueError(f"val_size must be in (0, {len(train_examples)}), got {val_size}")
    by_db: dict[str, list[int]] = defaultdict(list)
    for ex in train_examples:
        by_db[ex.db_id].append(ex.question_id)
    total = len(train_examples)
    exact = {db: val_size * len(ids) / total for db, ids in by_db.items()}
    quotas = {db: int(v) for db, v in exact.items()}
    remainder = val_size - sum(quotas.values())
    # Largest fractional parts win the remaining slots; ties broken by db name for determinism.
    for db, _ in sorted(exact.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))[:remainder]:
        quotas[db] += 1
    rng = random.Random(seed)
    val: list[int] = []
    for db in sorted(by_db):
        ids = sorted(by_db[db])
        rng.shuffle(ids)
        val.extend(ids[: quotas[db]])
    val_set = set(val)
    train_rl = [ex.question_id for ex in train_examples if ex.question_id not in val_set]
    return {"val_internal": sorted(val), "train_rl": sorted(train_rl)}


def splits_file(cfg: Config) -> Path:
    return Path(cfg.paths.processed_dir) / "splits" / "splits.json"


def load_or_build_splits(cfg: Config, paths: BirdPaths | None = None) -> dict[str, list[int]]:
    file = splits_file(cfg)
    if file.is_file():
        with open(file, "r", encoding="utf-8") as fh:
            return json.load(fh)["splits"]
    paths = paths or BirdPaths(cfg)
    train = load_examples("train", paths)
    splits = make_splits(train, cfg.data.val_internal_size, cfg.seed)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", encoding="utf-8") as fh:
        json.dump(
            {"seed": cfg.seed, "val_internal_size": cfg.data.val_internal_size, "n_train": len(train), "splits": splits},
            fh,
        )
    return splits


def load_split(name: str, cfg: Config, paths: BirdPaths | None = None) -> list[Example]:
    """Return the examples of a split: dev, val_internal, train_rl or train (all)."""
    paths = paths or BirdPaths(cfg)
    if name == "dev":
        return load_examples("dev", paths)
    if name not in SPLITS:
        raise ValueError(f"Unknown split {name!r}; choose from {SPLITS}")
    train = load_examples("train", paths)
    if name == "train":
        return train
    ids = set(load_or_build_splits(cfg, paths)[name])
    return [ex for ex in train if ex.question_id in ids]
