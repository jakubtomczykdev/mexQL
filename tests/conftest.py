import sqlite3
from pathlib import Path

import pytest

from mexql.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]
BIRD_DEV_DBS = REPO / "data" / "raw" / "dev_20240627" / "dev_databases"


def create_tiny_db(path: Path) -> Path:
    """A small database with the quirks that matter: spaces in names, keyword names,
    NULLs, floats, blobs, composite keys and a foreign key."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE people (
            id INTEGER PRIMARY KEY,
            `full name` TEXT NOT NULL,
            age INTEGER,
            height REAL,
            nickname TEXT,
            photo BLOB
        );
        CREATE TABLE orders (
            order_id INTEGER,
            person_id INTEGER,
            amount REAL,
            `Order` TEXT,
            PRIMARY KEY (order_id, person_id),
            FOREIGN KEY (person_id) REFERENCES people(id)
        );
        INSERT INTO people VALUES (1, 'Ann Lee', 30, 1.70, 'annie', X'00ff');
        INSERT INTO people VALUES (2, 'Bob O''Neil', 41, 1.82, NULL, NULL);
        INSERT INTO people VALUES (3, 'Cé Dé', 30, 1.60, 'cd', NULL);
        INSERT INTO people VALUES (4, 'Dan Who', NULL, NULL, NULL, NULL);
        INSERT INTO orders VALUES (10, 1, 9.99, 'first');
        INSERT INTO orders VALUES (11, 1, 20.0, 'second');
        INSERT INTO orders VALUES (12, 3, 5.5, 'first');
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope="session")
def tiny_db(tmp_path_factory) -> Path:
    return create_tiny_db(tmp_path_factory.mktemp("db") / "tiny.sqlite")


@pytest.fixture()
def tiny_raw_dir(tmp_path) -> Path:
    """A raw_dir that mimics the BIRD layout with the tiny database as a dev database."""
    raw = tmp_path / "raw"
    create_tiny_db_at = raw / "dev_20240627" / "dev_databases" / "tiny"
    create_tiny_db_at.mkdir(parents=True)
    create_tiny_db(create_tiny_db_at / "tiny.sqlite")
    return raw


@pytest.fixture()
def cfg(tmp_path) -> Config:
    return load_config(
        [REPO / "configs" / "base.yaml"],
        overrides=[
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.outputs_dir={tmp_path / 'outputs'}",
            "execution.num_workers=2",
        ],
    )


@pytest.fixture()
def bird_cfg(tmp_path) -> Config:
    if not BIRD_DEV_DBS.is_dir():
        pytest.skip("BIRD dev databases not available under data/raw")
    return load_config(
        [REPO / "configs" / "base.yaml"],
        overrides=[
            f"paths.raw_dir={REPO / 'data' / 'raw'}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.outputs_dir={tmp_path / 'outputs'}",
            "execution.num_workers=4",
        ],
    )
