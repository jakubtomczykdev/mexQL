"""Downloading and unpacking the official BIRD archives.

* ``dev.zip`` (~346 MB) unpacks to ``dev_20240627/`` with a nested ``dev_databases.zip``.
* ``train.zip`` (~8.9 GB) unpacks to ``train/`` with a nested ``train_databases.zip`` (~9.3 GB,
  ~32 GB extracted).
* ``fetch_train_metadata`` reads only ``train.json``, ``train_gold.sql`` and ``train_tables.json``
  out of the remote ``train.zip`` with HTTP range requests, for machines that cannot hold the
  databases (the three files together are about 7 MB).
"""

from __future__ import annotations

import io
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

BIRD_URLS = {
    "dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
    "train": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
}
TRAIN_META_MEMBERS = ("train/train.json", "train/train_gold.sql", "train/train_tables.json")


def download(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest``; resumes a partial download if the server supports ranges."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))
    existing = dest.stat().st_size if dest.exists() else 0
    if total and existing == total:
        log.info("%s already downloaded (%d bytes)", dest.name, total)
        return dest
    headers = {"Range": f"bytes={existing}-"} if existing and total else {}
    mode = "ab" if headers else "wb"
    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        if headers and resp.status_code != 206:
            existing, mode = 0, "wb"
        with open(dest, mode) as fh, tqdm(total=total or None, initial=existing, unit="B", unit_scale=True,
                                          desc=dest.name) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                bar.update(len(chunk))
    if total and dest.stat().st_size != total:
        raise IOError(f"{dest}: expected {total} bytes, got {dest.stat().st_size}")
    return dest


def _extract(zf: zipfile.ZipFile, target: Path, desc: str) -> None:
    members = [m for m in zf.infolist() if not m.filename.startswith("__MACOSX/")]
    for m in tqdm(members, desc=desc, unit="file", leave=False):
        zf.extract(m, target)


def unpack(zip_path: Path, raw_dir: Path, keep_nested_zip: bool = False) -> Path:
    """Extract an official archive and the ``*_databases.zip`` nested inside it."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        top = sorted({m.filename.split("/")[0] for m in zf.infolist() if not m.filename.startswith("__MACOSX")})
        _extract(zf, raw_dir, f"unpack {zip_path.name}")
    if len(top) != 1:
        raise ValueError(f"{zip_path}: expected one top-level directory, found {top}")
    split_dir = raw_dir / top[0]
    for nested in sorted(split_dir.glob("*_databases.zip")):
        with zipfile.ZipFile(nested) as zf:
            _extract(zf, split_dir, f"unpack {nested.name}")
        if not keep_nested_zip:
            nested.unlink()
    macosx = raw_dir / "__MACOSX"
    if macosx.is_dir():
        shutil.rmtree(macosx)
    return split_dir


class _HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file using HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_train_metadata(raw_dir: Path, url: str = BIRD_URLS["train"]) -> list[Path]:
    """Fetch the small train files without downloading the databases."""
    target = raw_dir / "train"
    target.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(io.BufferedReader(_HttpRangeFile(url), buffer_size=4 << 20)) as zf:
        for member in TRAIN_META_MEMBERS:
            data = zf.read(member)
            out = target / Path(member).name
            out.write_bytes(data)
            written.append(out)
            log.info("fetched %s (%d bytes)", out, len(data))
    return written
