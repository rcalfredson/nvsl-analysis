# src/plotting/between_reward_segment_metrics.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def seg_keep_frames(
    *,
    traj,
    s: int,
    e: int,
    fi: int,
    nonwalk_mask,
    exclude_nonwalk: bool,
    min_keep_frames: int = 2,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Return (keep_frames, L) for the effective segment window [s, s+L).

    keep_frames is length L aligned to absolute frames s..s+L-1.
    L may be less than (e-s) if we clamp to available nonwalk_mask window,
    mirroring iterator behavior.
    """
    s = int(s)
    e = int(e)
    L0 = int(max(0, e - s))
    if L0 <= 0:
        return None, 0

    L = L0
    keep = np.ones((L0,), dtype=bool)

    if exclude_nonwalk and nonwalk_mask is not None:
        s2 = max(0, min(s - fi, len(nonwalk_mask)))
        e2 = max(0, min(e - fi, len(nonwalk_mask)))
        if e2 <= s2:
            return None, 0
        L = int(min(L0, e2 - s2))
        if L <= 0:
            return None, 0
        keep = keep[:L] & (~np.asarray(nonwalk_mask[s2 : s2 + L], dtype=bool))

    # Clamp L to available trajectory data
    max_L = int(min(len(traj.x) - s, len(traj.y) - s))
    if max_L <= 0:
        return None, 0
    if L > max_L:
        L = max_L
        keep = keep[:L]

    # finite xy filter on the same effective window
    xs = np.asarray(traj.x[s : s + L], dtype=float)
    ys = np.asarray(traj.y[s : s + L], dtype=float)
    fin = np.isfinite(xs) & np.isfinite(ys)
    keep &= fin

    min_keep_frames = int(max(2, min_keep_frames))
    if int(np.sum(keep)) < min_keep_frames:
        return None, L
    return keep, L


def dist_traveled_mm_masked(
    *,
    traj,
    s: int,
    e: int,
    fi: int,
    nonwalk_mask,
    exclude_nonwalk: bool,
    px_per_mm: float,
    start_override: int | None = None,
    min_keep_frames: int = 2,
) -> float:
    """
    Distance traveled in mm within [s, e) frames, using masked frames (walking+finite).

    Step semantics:
      - include step i if both frames i and i+1 are kept
      - steps range is within [s, s+L) effective window, so step indices [s, s+L-1)

    If start_override is provided, compute within [start_override, e) (still clamped to window)
    """
    keep, L = seg_keep_frames(
        traj=traj,
        s=s,
        e=e,
        fi=fi,
        nonwalk_mask=nonwalk_mask,
        exclude_nonwalk=exclude_nonwalk,
        min_keep_frames=min_keep_frames,
    )
    if keep is None or L < 2:
        return np.nan

    px_per_mm = float(px_per_mm)
    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
        return np.nan

    s = int(s)

    # Determine start offset within the effective [s, s+L) window
    if start_override is None:
        off = 0
    else:
        start_override = int(start_override)
        if start_override <= s:
            off = 0
        elif start_override >= s + L:
            return 0.0
        else:
            off = int(start_override - s)

    # Ensure start frame is kept; if not, advance to next kept frame
    if not keep[off]:
        nxt = np.where(keep[off:])[0]
        if nxt.size == 0:
            return np.nan
        off = int(off + nxt[0])
        if off >= L - 1:
            return 0.0

    # Steps exist for indices 0..L-2 within this effective window
    keep_steps = keep[:-1] & keep[1:]  # length L-1
    if off > 0:
        keep_steps = keep_steps[off:]
        step_px = np.asarray(traj.d[(s + off) : (s + L - 1)], dtype=float)
    else:
        step_px = np.asarray(traj.d[s : (s + L - 1)], dtype=float)

    if step_px.size == 0:
        return 0.0

    n = int(min(keep_steps.size, step_px.size))
    if n <= 0:
        return 0.0
    keep_steps = keep_steps[:n]
    step_px = step_px[:n]

    dpx = float(np.sum(step_px[keep_steps]))
    return float(dpx / px_per_mm)
