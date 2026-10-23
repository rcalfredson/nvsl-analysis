"""CSV and immutable bundle I/O shared by tracking-gap review commands.

Episodes use zero-based half-open frame intervals [start, stop). Reading a bundle
verifies its episode hash; callers may add display metadata to returned copies,
but must never rewrite the source episodes or their stable review IDs.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def read_csv(path):
    """Read records as strings, matching the on-disk bundle schema."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    """Write records, requiring explicit field names when the list is empty."""
    if not rows and fields is None:
        raise ValueError("Empty CSV output requires explicit fields")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_episodes(bundle):
    """Verify immutable intervals before returning their CSV records."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    episodes_hash = hashlib.sha256((bundle / "episodes.csv").read_bytes()).hexdigest()
    if episodes_hash != manifest["episodes_sha256"]:
        raise ValueError("episodes.csv changed; edit annotations.csv only")
    return read_csv(bundle / "episodes.csv")


def verify_sources(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    for source in manifest["source_files"]:
        stat = Path(source["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
            raise ValueError(f"Source file changed since review bundle creation: {source['path']}")
