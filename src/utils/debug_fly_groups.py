# src/utils/debug_fly_groups.py

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("fly_group_debug")
logger.setLevel(logging.INFO)

_initialized = False


def init_fly_group_logging(log_path="debug_fly_groups.log"):
    """
    Initialize the file handler for fly group debug logging.
    This must be called explicitly by the main script when logging is desired.
    """
    global _initialized
    if _initialized:
        return

    handler = logging.FileHandler(str(log_path))
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    _initialized = True


def log_fly_group(group_name: str, indices, vas):
    """
    Logs each fly in a given group.
    Each index refers to a row in vas (one fly).
    """
    if not _initialized:
        return  # logging disabled
    if indices is None:
        logger.info(f"[{group_name}] No flies")
        return

    logger.info(f"[{group_name}] {len(indices)} flies")
    for idx in indices:
        try:
            va = vas[idx]
            logger.info(f"  idx={idx}, video='{va.fn}', fly={va.f}")
        except Exception as e:
            logger.info(f"  idx={idx} - ERROR retrieving fly info: {e}")


def fly_list_label(va) -> str:
    """Return the stable, human-readable identifier used in cohort dumps."""
    video = os.path.basename(str(getattr(va, "fn", "unknown_video")))
    fly = getattr(va, "f", None)
    return f"{video}\tfly={fly if fly is not None else 'unknown'}"


def write_sorted_fly_list(path, included, vas) -> Path:
    """Write one sorted ``video<TAB>fly=...`` identifier per included VA."""
    included_arr = np.asarray(included)
    if included_arr.dtype == bool:
        if included_arr.ndim != 1 or included_arr.size != len(vas):
            raise ValueError("fly-list boolean mask must have one entry per VA")
        indices = np.flatnonzero(included_arr)
    else:
        indices = np.asarray(included, dtype=int).reshape(-1)

    labels = sorted(fly_list_label(vas[int(idx)]) for idx in indices)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{label}\n" for label in labels)
    out_path.write_text(text, encoding="utf-8")
    print(f"[SLI cohort debug] wrote {len(labels)} flies to {out_path}")
    return out_path
