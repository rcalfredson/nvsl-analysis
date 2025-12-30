from __future__ import annotations

from typing import Optional

import numpy as np


def build_wall_contact_mask_for_window(
    va,
    f: int,
    *,
    fi: int,
    n_frames: int,
    enabled: bool,
    warned_missing_wc: list[bool],
    log_tag: str,
) -> Optional[np.ndarray]:
    """
    Build a per-frame wall-contact boolean mask for an arbitrary frame window.

    Args:
        va: VideoAnalysis instance
        f: fly index
        fi: start frame (absolute index)
        n_frames: window length in frames
        enabled: if False, returns None
        warned_missing_wc: 1-item list used as a "warn once" mutable flag
        log_tag: tag used in warning messages

    Returns:
        wc: np.ndarray shape (n_frames,) dtype=bool, or None if unavailable/disabled.

    Notes:
        - Prefers boundary_contact_regions if present (interval representation).
        - Falls back to boundary_contact per-frame boolean array if present.
        - Mirrors the wall-contact access pattern used in plotting code.
    """
    if not enabled:
        return None
    if n_frames <= 0:
        return None

    try:
        leaf = va.trx[f].boundary_event_stats["wall"]["all"]["edge"]
        regions = leaf.get("boundary_contact_regions", None)
        if regions is not None:
            wc = np.zeros(int(n_frames), dtype=bool)
            win_start = int(fi)
            win_end = int(fi + n_frames)
            for a, b in regions:
                s = max(int(a), win_start)
                e = min(int(b), win_end)
                if e > s:
                    wc[s - win_start : e - win_start] = True
            return wc

        bc = leaf.get("boundary_contact", None)
        if bc is not None:
            # bc is typically full-length per-frame; slice the requested window
            return np.asarray(bc[int(fi) : int(fi + n_frames)], dtype=bool)

    except Exception:
        if not warned_missing_wc[0]:
            print(
                f"[{log_tag}] warning: can't load wall-contact data; "
                "wall-contact exclusion will be ignored for some videos."
            )
            warned_missing_wc[0] = True

    return None
