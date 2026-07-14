"""Restricted metadata edits for NumPy ``.npz`` data bundles.

This module deliberately works at the ZIP-member level: data arrays are copied as
opaque ``.npy`` bytes and only an allowlisted metadata member is decoded/replaced.
"""

from __future__ import annotations

import io
import os
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np


EDITABLE_FIELDS = frozenset({"group_label", "group_labels"})


def _member_name(field: str) -> str:
    if field not in EDITABLE_FIELDS:
        raise ValueError(
            f"Metadata field {field!r} is not editable; allowed fields: "
            f"{', '.join(sorted(EDITABLE_FIELDS))}"
        )
    return f"{field}.npy"


def _load_metadata_member(bundle: Path, field: str) -> np.ndarray:
    member = _member_name(field)
    with zipfile.ZipFile(bundle, "r") as archive:
        names = archive.namelist()
        if names.count(member) != 1:
            detail = "is missing" if member not in names else "appears more than once"
            raise ValueError(f"Bundle {bundle} {detail}: {member}")
        return np.load(io.BytesIO(archive.read(member)), allow_pickle=True)


def read_group_labels(bundle: str | os.PathLike[str]) -> dict[str, list[str]]:
    """Return only the editable label metadata present in *bundle*."""
    path = Path(bundle)
    found: dict[str, list[str]] = {}
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
    for field in sorted(EDITABLE_FIELDS):
        if _member_name(field) in names:
            values = _load_metadata_member(path, field).reshape(-1)
            found[field] = [str(value) for value in values]
    return found


def edit_group_labels(
    bundle: str | os.PathLike[str],
    labels: Sequence[str],
    *,
    field: str = "group_labels",
    output: str | os.PathLike[str] | None = None,
) -> Path:
    """Replace an allowlisted label field, atomically when editing in place.

    All non-target members are copied without loading or interpreting their arrays.
    The number of labels must remain unchanged, preventing accidental changes to
    the relationship between group indices and their display labels.
    """
    source = Path(bundle)
    target = Path(output) if output is not None else source
    member = _member_name(field)
    old = _load_metadata_member(source, field)
    new_labels = [str(label) for label in labels]
    if any(not label.strip() for label in new_labels):
        raise ValueError("Group labels must not be empty or whitespace-only")
    if old.size != len(new_labels):
        raise ValueError(
            f"{field} contains {old.size} label(s), but {len(new_labels)} were provided"
        )

    replacement = np.asarray(new_labels, dtype=str)
    if old.ndim == 0:
        replacement = replacement.reshape(())
    else:
        replacement = replacement.reshape(old.shape)
    buffer = io.BytesIO()
    np.save(buffer, replacement, allow_pickle=False)

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(
            temporary, "w", allowZip64=True
        ) as dst:
            dst.comment = src.comment
            for info in src.infolist():
                payload = buffer.getvalue() if info.filename == member else src.read(info)
                dst.writestr(info, payload)
        if target == source:
            os.chmod(temporary, stat.S_IMODE(source.stat().st_mode))
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
