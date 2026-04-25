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


def net_displacement_mm_masked(
    *,
    traj,
    s: int,
    e: int,
    fi: int,
    nonwalk_mask,
    exclude_nonwalk: bool,
    px_per_mm: float,
    min_keep_frames: int = 2,
) -> float:
    """
    Net displacement in mm across the kept frames of [s, e).

    Uses the same effective window + keep-mask semantics as
    dist_traveled_mm_masked(), then measures the chord length between the first
    and last kept finite positions.
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

    kept = np.flatnonzero(keep)
    if kept.size < 2:
        return np.nan

    s = int(s)
    i0 = int(kept[0])
    i1 = int(kept[-1])

    x0 = float(traj.x[s + i0])
    y0 = float(traj.y[s + i0])
    x1 = float(traj.x[s + i1])
    y1 = float(traj.y[s + i1])
    if not np.all(np.isfinite([x0, y0, x1, y1])):
        return np.nan

    dpx = float(np.hypot(x1 - x0, y1 - y0))
    return float(dpx / px_per_mm)


def max_radial_distance_mm_masked(
    *,
    traj,
    s: int,
    e: int,
    fi: int,
    nonwalk_mask,
    exclude_nonwalk: bool,
    px_per_mm: float,
    center_xy: tuple[float, float],
    min_keep_frames: int = 2,
) -> float:
    """
    Maximum radial distance from center_xy in mm across kept frames of [s, e).
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

    cx, cy = center_xy
    cx = float(cx)
    cy = float(cy)
    if not np.all(np.isfinite([cx, cy])):
        return np.nan

    s = int(s)
    xs = np.asarray(traj.x[s : s + L], dtype=float)
    ys = np.asarray(traj.y[s : s + L], dtype=float)
    if xs.size == 0 or ys.size == 0:
        return np.nan

    d = np.hypot(xs - cx, ys - cy)
    d = d[keep & np.isfinite(d)]
    if d.size == 0:
        return np.nan

    return float(np.nanmax(d) / px_per_mm)


def path_length_and_max_radius_mm_masked(
    *,
    traj,
    s: int,
    e: int,
    fi: int,
    nonwalk_mask,
    exclude_nonwalk: bool,
    px_per_mm: float,
    center_xy: tuple[float, float],
    min_keep_frames: int = 2,
) -> tuple[float, float]:
    """
    Return (path_length_mm, max_radius_mm) for [s, e) under shared masking.
    """
    path_mm = dist_traveled_mm_masked(
        traj=traj,
        s=s,
        e=e,
        fi=fi,
        nonwalk_mask=nonwalk_mask,
        exclude_nonwalk=exclude_nonwalk,
        px_per_mm=px_per_mm,
        min_keep_frames=min_keep_frames,
    )
    if not np.isfinite(path_mm) or path_mm <= 0:
        return np.nan, np.nan

    radius_mm = max_radial_distance_mm_masked(
        traj=traj,
        s=s,
        e=e,
        fi=fi,
        nonwalk_mask=nonwalk_mask,
        exclude_nonwalk=exclude_nonwalk,
        px_per_mm=px_per_mm,
        center_xy=center_xy,
        min_keep_frames=min_keep_frames,
    )
    if not np.isfinite(radius_mm):
        return np.nan, np.nan
    return float(path_mm), float(radius_mm)


def tortuosity_metric_masked(
    *,
    traj,
    s: int,
    e: int,
    fi: int,
    nonwalk_mask,
    exclude_nonwalk: bool,
    px_per_mm: float,
    mode: str = "path_over_max_radius",
    reward_center_xy: tuple[float, float] | None = None,
    min_keep_frames: int = 2,
    min_displacement_mm: float = 0.0,
    min_radius_mm: float = 0.0,
) -> float:
    """
    Segment tortuosity/straightness metric on [s, e) under masked-frame semantics.

    Supported modes:
      - "path_over_max_radius": path length / max distance from reward center
      - "path_over_displacement": path length / net displacement (>= 1 when finite)
      - "straightness": net displacement / path length (in [0, 1] when finite)
      - "excess_path": (path length / net displacement) - 1
    """
    mode = str(mode or "path_over_max_radius").strip().lower()

    path_mm = dist_traveled_mm_masked(
        traj=traj,
        s=s,
        e=e,
        fi=fi,
        nonwalk_mask=nonwalk_mask,
        exclude_nonwalk=exclude_nonwalk,
        px_per_mm=px_per_mm,
        min_keep_frames=min_keep_frames,
    )
    if not np.isfinite(path_mm) or path_mm <= 0:
        return np.nan

    if mode == "path_over_max_radius":
        if reward_center_xy is None:
            return np.nan
        radius_mm = max_radial_distance_mm_masked(
            traj=traj,
            s=s,
            e=e,
            fi=fi,
            nonwalk_mask=nonwalk_mask,
            exclude_nonwalk=exclude_nonwalk,
            px_per_mm=px_per_mm,
            center_xy=reward_center_xy,
            min_keep_frames=min_keep_frames,
        )
        if not np.isfinite(radius_mm):
            return np.nan
        min_radius = float(min_radius_mm or 0.0)
        if not np.isfinite(min_radius):
            min_radius = 0.0
        if radius_mm <= max(0.0, min_radius):
            return np.nan
        return float(path_mm / radius_mm)

    disp_mm = net_displacement_mm_masked(
        traj=traj,
        s=s,
        e=e,
        fi=fi,
        nonwalk_mask=nonwalk_mask,
        exclude_nonwalk=exclude_nonwalk,
        px_per_mm=px_per_mm,
        min_keep_frames=min_keep_frames,
    )
    if not np.isfinite(disp_mm):
        return np.nan

    min_disp = float(min_displacement_mm or 0.0)
    if not np.isfinite(min_disp):
        min_disp = 0.0
    if disp_mm <= max(0.0, min_disp):
        return np.nan

    if mode == "straightness":
        return float(disp_mm / path_mm)

    ratio = float(path_mm / disp_mm)
    if mode == "path_over_displacement":
        return ratio
    if mode == "excess_path":
        return float(ratio - 1.0)

    raise ValueError(f"Unknown tortuosity mode: {mode!r}")
