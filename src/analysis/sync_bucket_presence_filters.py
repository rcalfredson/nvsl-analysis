from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_TRAINING = 2
DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_SYNC_BUCKET = 5
EXP_FLY_INDEX = 0


@dataclass(frozen=True)
class ExpTargetSyncBucketFilterResult:
    enabled: bool
    eligible: bool
    reason: str
    training: int = DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_TRAINING
    sync_bucket: int = DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_SYNC_BUCKET
    fly_index: int = EXP_FLY_INDEX
    available_sync_buckets: int = 0
    target_bucket_start: float = np.nan
    target_bucket_stop: float = np.nan


def exp_target_sync_bucket_filter_enabled(opts) -> bool:
    return bool(
        getattr(
            opts,
            "require_exp_target_sync_bucket",
            getattr(opts, "require_exp_pi_threshold_bucket", False),
        )
    )


def exp_target_sync_bucket_filter_target(opts) -> tuple[int, int]:
    training = getattr(
        opts,
        "exp_target_sync_bucket_filter_training",
        getattr(
            opts,
            "exp_pi_threshold_filter_training",
            DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_TRAINING,
        ),
    )
    sync_bucket = getattr(
        opts,
        "exp_target_sync_bucket_filter_sync_bucket",
        getattr(
            opts,
            "exp_pi_threshold_filter_sync_bucket",
            DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_SYNC_BUCKET,
        ),
    )
    return max(1, int(training or 1)), max(1, int(sync_bucket or 1))


def _finite_float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _bucket_presence_from_sync_ranges(va, training_idx: int, bucket_idx: int):
    ranges = getattr(va, "sync_bucket_ranges", None)
    if ranges is None:
        return None

    try:
        training_ranges = ranges[training_idx]
    except (IndexError, TypeError):
        return 0, np.nan, np.nan, "target_training_missing"

    if training_ranges is None:
        return 0, np.nan, np.nan, "target_training_missing"

    try:
        available = len(training_ranges)
    except TypeError:
        return 0, np.nan, np.nan, "missing_sync_bucket_data"

    if available <= bucket_idx:
        return available, np.nan, np.nan, "target_sync_bucket_missing"

    try:
        start, stop = training_ranges[bucket_idx]
    except (TypeError, ValueError):
        return available, np.nan, np.nan, "target_sync_bucket_invalid"

    start = _finite_float(start)
    stop = _finite_float(stop)
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        return available, start, stop, "target_sync_bucket_invalid"
    return available, start, stop, None


def _bucket_presence_from_buckets(va, training_idx: int, bucket_idx: int):
    buckets = getattr(va, "buckets", None)
    if buckets is None:
        return None

    try:
        training_buckets = buckets[training_idx]
    except (IndexError, TypeError):
        return 0, np.nan, np.nan, "target_training_missing"

    if training_buckets is None:
        return 0, np.nan, np.nan, "target_training_missing"

    try:
        bucket_edges = list(training_buckets)
    except TypeError:
        return 0, np.nan, np.nan, "missing_sync_bucket_data"

    if not bucket_edges:
        return 0, np.nan, np.nan, "target_sync_bucket_missing"

    start = _finite_float(bucket_edges[0])
    if not np.isfinite(start):
        return 0, np.nan, np.nan, "target_sync_bucket_missing"

    ranges = []
    for edge in bucket_edges[1:]:
        stop = _finite_float(edge)
        if not np.isfinite(stop):
            break
        if stop > start:
            ranges.append((start, stop))
        start = stop

    available = len(ranges)
    if available <= bucket_idx:
        return available, np.nan, np.nan, "target_sync_bucket_missing"

    start, stop = ranges[bucket_idx]
    return available, start, stop, None


def exp_target_sync_bucket_filter_result(va, opts) -> ExpTargetSyncBucketFilterResult:
    training, sync_bucket = exp_target_sync_bucket_filter_target(opts)
    result_kwargs = {
        "enabled": exp_target_sync_bucket_filter_enabled(opts),
        "training": training,
        "sync_bucket": sync_bucket,
        "fly_index": EXP_FLY_INDEX,
    }

    if not result_kwargs["enabled"]:
        return ExpTargetSyncBucketFilterResult(
            eligible=True,
            reason="disabled",
            **result_kwargs,
        )

    training_idx = training - 1
    bucket_idx = sync_bucket - 1
    presence = _bucket_presence_from_sync_ranges(va, training_idx, bucket_idx)
    if presence is None:
        presence = _bucket_presence_from_buckets(va, training_idx, bucket_idx)

    if presence is None:
        return ExpTargetSyncBucketFilterResult(
            eligible=False,
            reason="missing_sync_bucket_data",
            **result_kwargs,
        )

    available, start, stop, reason = presence
    presence_kwargs = {
        "available_sync_buckets": int(available),
        "target_bucket_start": start,
        "target_bucket_stop": stop,
    }
    if reason is not None:
        return ExpTargetSyncBucketFilterResult(
            eligible=False,
            **result_kwargs,
            reason=reason,
            **presence_kwargs,
        )

    return ExpTargetSyncBucketFilterResult(
        eligible=True,
        reason="passes",
        **result_kwargs,
        **presence_kwargs,
    )


def exp_target_sync_bucket_eligibility_mask(vas, opts) -> np.ndarray:
    return np.asarray(
        [exp_target_sync_bucket_filter_result(va, opts).eligible for va in vas],
        dtype=bool,
    )


def exp_target_sync_bucket_filter_payload(vas, opts, *, prefix: str) -> dict:
    results = [exp_target_sync_bucket_filter_result(va, opts) for va in vas]
    eligible = np.asarray([r.eligible for r in results], dtype=bool)
    reasons = np.asarray([r.reason for r in results], dtype=object)
    enabled = exp_target_sync_bucket_filter_enabled(opts)
    training, sync_bucket = exp_target_sync_bucket_filter_target(opts)
    return {
        f"{prefix}_enabled": np.array(enabled, dtype=bool),
        f"{prefix}_training": np.array(training, dtype=int),
        f"{prefix}_sync_bucket": np.array(sync_bucket, dtype=int),
        f"{prefix}_fly_index": np.array(EXP_FLY_INDEX, dtype=int),
        f"{prefix}_eligible": eligible,
        f"{prefix}_reason": reasons,
        f"{prefix}_available_sync_buckets": np.asarray(
            [r.available_sync_buckets for r in results], dtype=int
        ),
        f"{prefix}_target_bucket_start": np.asarray(
            [r.target_bucket_start for r in results], dtype=float
        ),
        f"{prefix}_target_bucket_stop": np.asarray(
            [r.target_bucket_stop for r in results], dtype=float
        ),
    }


def mask_by_exp_target_sync_bucket_filter(values, eligible):
    arr = np.asarray(values, dtype=float)
    keep = np.asarray(eligible, dtype=bool).reshape(-1)
    if arr.shape[:1] != keep.shape:
        raise ValueError(
            "target sync-bucket eligibility mask length must match first array dimension"
        )
    shape = (keep.size,) + (1,) * (arr.ndim - 1)
    return np.where(keep.reshape(shape), arr, np.nan)
