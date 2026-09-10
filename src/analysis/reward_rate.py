"""Reward rates over complete selected sync-bucket windows."""

import numpy as np


def pooled_rewards_per_minute(
    va,
    *,
    training_idx: int,
    skip_first_sync_buckets: int = 0,
    keep_first_sync_buckets: int = 0,
) -> float:
    """Count target entries over the complete selected window, ignoring PI masks.

    Missing requested buckets invalidate the result rather than shortening the
    denominator. This is independent of the RPD/speed aggregation option.
    """
    if getattr(va, "_skipped", False):
        return np.nan
    trx = getattr(va, "trx", [])
    if not trx or trx[0].bad():
        return np.nan
    t_idx = int(training_idx)
    ranges = getattr(va, "sync_bucket_ranges", [])
    if t_idx < 0 or t_idx >= len(ranges) or t_idx >= len(va.trns):
        return np.nan
    skip = max(0, int(skip_first_sync_buckets or 0))
    keep = max(0, int(keep_first_sync_buckets or 0))
    available = ranges[t_idx]
    end = skip + keep if keep else len(available)
    if skip >= end or end > len(available):
        return np.nan
    selected = np.asarray(available[skip:end], dtype=float)
    if (
        selected.ndim != 2 or selected.shape[1] != 2
        or not np.all(np.isfinite(selected))
        or np.any(selected[:, 1] <= selected[:, 0])
        or np.any(selected[1:, 0] != selected[:-1, 1])
    ):
        return np.nan
    start, stop = int(selected[0, 0]), int(selected[-1, 1])
    trn = va.trns[t_idx]
    if start < trn.start or stop > trn.stop:
        return np.nan
    fps = float(getattr(va, "fps", np.nan))
    if not np.isfinite(fps) or fps <= 0:
        return np.nan
    rewards = va._countOn(start, stop, calc=True, ctrl=False, f=0)
    if not np.isfinite(rewards):
        return np.nan
    return float(rewards / ((stop - start) / fps / 60.0))

