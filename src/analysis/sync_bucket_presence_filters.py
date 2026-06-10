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
    pi_threshold: int = 0
    target_exp_count: float = np.nan
    target_ctrl_count: float = np.nan
    target_count_sum: float = np.nan


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


def exp_target_sync_bucket_pi_threshold_value(opts) -> int:
    return max(0, int(getattr(opts, "piTh", 0) or 0))


def _target_bucket_counts(va, training_idx: int, bucket_idx: int, fly_index: int):
    flies = getattr(va, "flies", None)
    if flies is None:
        return None
    try:
        n_flies = len(flies)
    except TypeError:
        n_flies = 1
    n_flies = max(1, int(n_flies or 1))
    row_idx = training_idx * n_flies + fly_index

    try:
        num_rewards_tot = getattr(va, "numRewardsTot")
    except AttributeError:
        return None
    try:
        exp_counts = num_rewards_tot[1][0][row_idx]
        ctrl_counts = num_rewards_tot[1][1][row_idx]
    except (IndexError, TypeError):
        return np.nan, np.nan, np.nan, "target_training_or_fly_missing"

    try:
        exp_count = float(exp_counts[bucket_idx])
        ctrl_count = float(ctrl_counts[bucket_idx])
    except IndexError:
        return np.nan, np.nan, np.nan, "target_sync_bucket_missing"
    except (TypeError, ValueError):
        return np.nan, np.nan, np.nan, "target_sync_bucket_nan"

    count_sum = exp_count + ctrl_count
    return exp_count, ctrl_count, count_sum, None


def exp_target_sync_bucket_filter_result(va, opts) -> ExpTargetSyncBucketFilterResult:
    training, sync_bucket = exp_target_sync_bucket_filter_target(opts)
    result_kwargs = {
        "enabled": exp_target_sync_bucket_filter_enabled(opts),
        "training": training,
        "sync_bucket": sync_bucket,
        "fly_index": EXP_FLY_INDEX,
        "pi_threshold": exp_target_sync_bucket_pi_threshold_value(opts),
    }

    if not result_kwargs["enabled"]:
        return ExpTargetSyncBucketFilterResult(
            eligible=True,
            reason="disabled",
            **result_kwargs,
        )

    training_idx = training - 1
    bucket_idx = sync_bucket - 1
    counts = _target_bucket_counts(va, training_idx, bucket_idx, EXP_FLY_INDEX)
    if counts is not None:
        exp_count, ctrl_count, count_sum, count_reason = counts
        count_kwargs = {
            "target_exp_count": exp_count,
            "target_ctrl_count": ctrl_count,
            "target_count_sum": count_sum,
        }
        if count_reason is not None:
            return ExpTargetSyncBucketFilterResult(
                eligible=False,
                reason=count_reason,
                **result_kwargs,
                **count_kwargs,
            )
        if not np.isfinite(count_sum):
            return ExpTargetSyncBucketFilterResult(
                eligible=False,
                reason="target_sync_bucket_nan",
                **result_kwargs,
                **count_kwargs,
            )
        excluded = count_sum < result_kwargs["pi_threshold"]
        return ExpTargetSyncBucketFilterResult(
            eligible=not excluded,
            reason="passes" if not excluded else "pi_threshold_failed",
            **result_kwargs,
            **count_kwargs,
        )

    mask = getattr(va, "reward_exclusion_mask", None)
    if mask is None:
        return ExpTargetSyncBucketFilterResult(
            eligible=False,
            reason="missing_reward_exclusion_mask",
            **result_kwargs,
        )

    try:
        fly_mask = mask[training_idx][EXP_FLY_INDEX]
    except (IndexError, TypeError):
        return ExpTargetSyncBucketFilterResult(
            eligible=False,
            reason="target_training_or_fly_missing",
            **result_kwargs,
        )

    try:
        excluded = bool(fly_mask[bucket_idx])
    except (IndexError, TypeError):
        return ExpTargetSyncBucketFilterResult(
            eligible=False,
            reason="target_sync_bucket_missing",
            **result_kwargs,
        )

    return ExpTargetSyncBucketFilterResult(
        eligible=not excluded,
        reason="passes" if not excluded else "pi_threshold_failed",
        **result_kwargs,
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
        f"{prefix}_pi_threshold": np.array(
            exp_target_sync_bucket_pi_threshold_value(opts), dtype=int
        ),
        f"{prefix}_eligible": eligible,
        f"{prefix}_reason": reasons,
        f"{prefix}_target_exp_count": np.asarray(
            [r.target_exp_count for r in results], dtype=float
        ),
        f"{prefix}_target_ctrl_count": np.asarray(
            [r.target_ctrl_count for r in results], dtype=float
        ),
        f"{prefix}_target_count_sum": np.asarray(
            [r.target_count_sum for r in results], dtype=float
        ),
    }


def mask_by_exp_target_sync_bucket_filter(values, eligible):
    arr = np.asarray(values, dtype=float)
    keep = np.asarray(eligible, dtype=bool).reshape(-1)
    if arr.shape[:1] != keep.shape:
        raise ValueError(
            "PI-threshold eligibility mask length must match first array dimension"
        )
    shape = (keep.size,) + (1,) * (arr.ndim - 1)
    return np.where(keep.reshape(shape), arr, np.nan)
