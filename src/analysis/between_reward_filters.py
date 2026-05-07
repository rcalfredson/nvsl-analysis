from __future__ import annotations

import numpy as np


DEFAULT_MIN_BETWEEN_REWARD_SYNC_BUCKET_TRAJECTORIES = 5
BTW_RWD_MIN_TRAJECTORIES_OPT = "btw_rwd_sync_bucket_min_trajectories"


def min_between_reward_sync_bucket_trajectories(opts) -> int:
    """Return the per-fly, per-sync-bucket between-reward trajectory threshold."""
    raw = getattr(
        opts,
        BTW_RWD_MIN_TRAJECTORIES_OPT,
        DEFAULT_MIN_BETWEEN_REWARD_SYNC_BUCKET_TRAJECTORIES,
    )
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return DEFAULT_MIN_BETWEEN_REWARD_SYNC_BUCKET_TRAJECTORIES


def mask_metric_by_min_between_reward_trajectories(
    values,
    counts,
    min_trajectories: int,
):
    """
    Mask metric values whose contributing between-reward trajectory count is too low.

    The filter is per fly/condition and per sync bucket. Counts are returned separately
    by callers so they remain the raw number of contributing trajectories.
    """
    arr = np.asarray(values, dtype=float)
    n = np.asarray(counts, dtype=int)
    min_n = max(0, int(min_trajectories or 0))
    if min_n <= 0:
        return arr
    return np.where(n >= min_n, arr, np.nan)
