from __future__ import annotations

from typing import Any, Optional, Tuple

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

    def _region_to_start_stop(r: Any) -> Optional[Tuple[int, int]]:
        """
        Convert a region representation to (start, stop) absolute frame indices.

        Supported:
          - slice(start, stop[, step]) (step ignored)
          - (start, stop) tuple/list
          - np.ndarray row-like of length 2

        Notes:
          - None start => 0
          - None stop => treated as open-ended (caller will clamp to window end)
          - Returned interval is half-open: [start, stop)
        """
        if r is None:
            return None

        # slice objects
        if isinstance(r, slice):
            a = 0 if r.start is None else int(r.start)
            # stop may be None (open-ended)
            b = None if r.stop is None else int(r.stop)
            return (a, b)  # b may be None, handled later

        # tuple/list/row-like
        if isinstance(r, (tuple, list, np.ndarray)):
            try:
                if len(r) != 2:
                    return None
                a = r[0]
                b = r[1]
                a_i = 0 if a is None else int(a)
                b_i = None if b is None else int(b)
                return (a_i, b_i)
            except Exception:
                return None

        # slice-like objects with start/stop attrs
        if hasattr(r, "start") and hasattr(r, "stop"):
            try:
                a = 0 if getattr(r, "start") is None else int(getattr(r, "start"))
                b = None if getattr(r, "stop") is None else int(getattr(r, "stop"))
                return (a, b)
            except Exception:
                return None

        return None

    try:
        leaf = va.trx[f].boundary_event_stats["wall"]["all"]["edge"]
        regions = leaf.get("boundary_contact_regions", None)
        if regions is not None:
            wc = np.zeros(int(n_frames), dtype=bool)
            win_start = int(fi)
            win_end = int(fi + n_frames)
            for r in regions:
                ab = _region_to_start_stop(r)
                if ab is None:
                    continue
                a, b = ab
                # allow open-ended stop
                b = win_end if b is None else b

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
