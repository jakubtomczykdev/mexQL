#!/usr/bin/env python
"""Download and unpack the BIRD benchmark.

Examples::

    python scripts/download_data.py --dev                 # dev set (~2 GB on disk)
    python scripts/download_data.py --train-meta-only     # train.json/gold/tables only (~7 MB)
    python scripts/download_data.py --train               # full train set (~50 GB peak, ~32 GB after)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mexql.data.download import BIRD_URLS, download, fetch_train_metadata, unpack


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw", type=Path)
    ap.add_argument("--dev", action="store_true", help="download and unpack dev.zip")
    ap.add_argument("--train", action="store_true", help="download and unpack train.zip (large)")
    ap.add_argument("--train-meta-only", action="store_true",
                    help="fetch only train.json, train_gold.sql and train_tables.json via HTTP range requests")
    ap.add_argument("--keep-zips", action="store_true", help="keep the downloaded and nested zip files")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not (args.dev or args.train or args.train_meta_only):
        ap.error("nothing to do: pass --dev, --train and/or --train-meta-only")

    for name in ("dev", "train"):
        if not getattr(args, name):
            continue
        zip_path = args.raw_dir / f"{name}.zip"
        download(BIRD_URLS[name], zip_path)
        split_dir = unpack(zip_path, args.raw_dir, keep_nested_zip=args.keep_zips)
        if not args.keep_zips:
            zip_path.unlink()
        dbs = sorted(p.name for p in (split_dir / f"{name}_databases").iterdir() if p.is_dir())
        print(f"{name}: unpacked to {split_dir} with {len(dbs)} databases")
    if args.train_meta_only:
        for p in fetch_train_metadata(args.raw_dir):
            print(f"fetched {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
