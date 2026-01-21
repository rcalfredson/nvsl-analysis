from __future__ import annotations

import os
import numpy as np

from src.plotting.wall_contact_utils import build_wall_contact_mask_for_window


def video_base(va: "VideoAnalysis") -> str:
    fn = getattr(va, "fn", None)
    if fn:
        try:
            return os.path.splitext(os.path.basename(str(fn)))[0]
        except Exception:
            pass
    return f"va_{id(va)}"


def fly_role_name(role_idx: int) -> str:
    return "exp" if int(role_idx) == 0 else "yok"


def x_edges(*, x_bin_width_mm: float, x_min_mm: float, x_max_mm: float) -> np.ndarray:
    w = float(x_bin_width_mm)
    if not np.isfinite(w) or w <= 0:
        w = 2.0
    x0 = float(x_min_mm)
    x1 = float(x_max_mm)
    if not np.isfinite(x0):
        x0 = 0.0
    if not np.isfinite(x1) or x1 <= x0:
        x1 = x0 + 10.0

    edges = np.arange(x0, x1 + 0.5 * w, w, dtype=float)
    if edges.size < 2:
        edges = np.array([x0, x0 + w], dtype=float)
    return edges


def sync_bucket_window(
    va: "VideoAnalysis",
    trn,
    *,
    t_idx: int,
    f: int,
    skip_first: int,
    use_exclusion_mask: bool,
) -> tuple[int, int, int, list[bool]]:
    """
    Return (fi, df, n_buckets, complete) describing included sync buckets.

    Fallback:
      - single bucket spanning [trn.start, trn.stop)
    """
    ranges = getattr(va, "sync_bucket_ranges", None)
    if not ranges or t_idx >= len(ranges):
        fi0 = int(trn.start)
        df0 = int(max(1, trn.stop - trn.start))
        return (fi0, df0, 1, [True])

    rr = ranges[t_idx]
    if not rr:
        fi0 = int(trn.start)
        df0 = int(max(1, trn.stop - trn.start))
        return (fi0, df0, 1, [True])

    if skip_first < 0:
        skip_first = 0
    if skip_first >= len(rr):
        return (0, 1, 0, [])

    rr2 = rr[skip_first:]
    fi = int(rr2[0][0])

    df_list = [int(b - a) for (a, b) in rr2 if b > a]
    if not df_list:
        return (0, 1, 0, [])

    df = int(df_list[0])
    if any(int(d) != df for d in df_list):
        la = int(rr2[-1][1])
        df_fallback = int(max(1, la - fi))
        return (fi, df_fallback, 1, [True])

    n_buckets = int(len(rr2))

    if use_exclusion_mask and hasattr(va, "reward_exclusion_mask"):
        try:
            mask = va.reward_exclusion_mask[trn.n - 1][f]
        except Exception:
            mask = []
        complete = []
        for j in range(n_buckets):
            b_idx_orig = j + skip_first
            is_excl = bool(mask[b_idx_orig]) if b_idx_orig < len(mask) else False
            complete.append(not is_excl)
    else:
        complete = [True] * n_buckets

    return (fi, df, n_buckets, complete)


def build_nonwalk_mask(opts, va, trx_idx: int, fi: int, n_frames: int):
    exclude_nonwalk = bool(
        getattr(opts, "btw_rwd_conditioned_exclude_nonwalking_frames", False)
    )
    if not exclude_nonwalk:
        return None

    traj = va.trx[trx_idx]
    walking = getattr(traj, "walking", None)
    if walking is None:
        return None

    s0 = max(0, min(int(fi), len(walking)))
    e0 = max(0, min(int(fi + n_frames), len(walking)))

    wwin = np.zeros((n_frames,), dtype=bool)
    if e0 > s0:
        wseg = np.asarray(walking[s0:e0], dtype=float)
        wseg = np.where(np.isfinite(wseg), wseg, 0.0)
        wwin[: len(wseg)] = wseg > 0

    return ~wwin


def wall_contact_mask(
    opts, va, trx_idx: int, *, fi: int, n_frames: int, log_tag: str, warned_missing_wc
):
    exclude_wall = bool(getattr(opts, "com_exclude_wall_contact", False))
    return build_wall_contact_mask_for_window(
        va,
        trx_idx,
        fi=fi,
        n_frames=n_frames,
        enabled=exclude_wall,
        warned_missing_wc=warned_missing_wc,
        log_tag=log_tag,
    )
