from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_EXP_PI_FILTER_TRAINING = 2
DEFAULT_EXP_PI_FILTER_SYNC_BUCKET = 5
EXP_FLY_INDEX = 0


@dataclass(frozen=True)
class ExpPIThresholdFilterResult:
    enabled: bool
    eligible: bool
    reason: str
    training: int = DEFAULT_EXP_PI_FILTER_TRAINING
    sync_bucket: int = DEFAULT_EXP_PI_FILTER_SYNC_BUCKET
    fly_index: int = EXP_FLY_INDEX
    pi_threshold: int = 0


def exp_pi_threshold_filter_enabled(opts) -> bool:
    return bool(getattr(opts, "require_exp_pi_threshold_bucket", False))


def exp_pi_threshold_filter_target(opts) -> tuple[int, int]:
    training = getattr(
        opts,
        "exp_pi_threshold_filter_training",
        DEFAULT_EXP_PI_FILTER_TRAINING,
    )
    sync_bucket = getattr(
        opts,
        "exp_pi_threshold_filter_sync_bucket",
        DEFAULT_EXP_PI_FILTER_SYNC_BUCKET,
    )
    return max(1, int(training or 1)), max(1, int(sync_bucket or 1))


def exp_pi_threshold_value(opts) -> int:
    return max(0, int(getattr(opts, "piTh", 0) or 0))


def exp_pi_threshold_filter_result(va, opts) -> ExpPIThresholdFilterResult:
    training, sync_bucket = exp_pi_threshold_filter_target(opts)
    result_kwargs = {
        "enabled": exp_pi_threshold_filter_enabled(opts),
        "training": training,
        "sync_bucket": sync_bucket,
        "fly_index": EXP_FLY_INDEX,
        "pi_threshold": exp_pi_threshold_value(opts),
    }

    if not result_kwargs["enabled"]:
        return ExpPIThresholdFilterResult(
            eligible=True,
            reason="disabled",
            **result_kwargs,
        )

    mask = getattr(va, "reward_exclusion_mask", None)
    if mask is None:
        return ExpPIThresholdFilterResult(
            eligible=False,
            reason="missing_reward_exclusion_mask",
            **result_kwargs,
        )

    training_idx = training - 1
    bucket_idx = sync_bucket - 1
    try:
        fly_mask = mask[training_idx][EXP_FLY_INDEX]
    except (IndexError, TypeError):
        return ExpPIThresholdFilterResult(
            eligible=False,
            reason="target_training_or_fly_missing",
            **result_kwargs,
        )

    try:
        excluded = bool(fly_mask[bucket_idx])
    except (IndexError, TypeError):
        return ExpPIThresholdFilterResult(
            eligible=False,
            reason="target_sync_bucket_missing",
            **result_kwargs,
        )

    return ExpPIThresholdFilterResult(
        eligible=not excluded,
        reason="passes" if not excluded else "pi_threshold_failed",
        **result_kwargs,
    )


def exp_pi_threshold_eligibility_mask(vas, opts) -> np.ndarray:
    return np.asarray(
        [exp_pi_threshold_filter_result(va, opts).eligible for va in vas],
        dtype=bool,
    )


def exp_pi_threshold_filter_payload(vas, opts, *, prefix: str) -> dict:
    results = [exp_pi_threshold_filter_result(va, opts) for va in vas]
    eligible = np.asarray([r.eligible for r in results], dtype=bool)
    reasons = np.asarray([r.reason for r in results], dtype=object)
    enabled = exp_pi_threshold_filter_enabled(opts)
    training, sync_bucket = exp_pi_threshold_filter_target(opts)
    return {
        f"{prefix}_enabled": np.array(enabled, dtype=bool),
        f"{prefix}_training": np.array(training, dtype=int),
        f"{prefix}_sync_bucket": np.array(sync_bucket, dtype=int),
        f"{prefix}_fly_index": np.array(EXP_FLY_INDEX, dtype=int),
        f"{prefix}_pi_threshold": np.array(exp_pi_threshold_value(opts), dtype=int),
        f"{prefix}_eligible": eligible,
        f"{prefix}_reason": reasons,
    }


def mask_by_exp_pi_threshold_filter(values, eligible):
    arr = np.asarray(values, dtype=float)
    keep = np.asarray(eligible, dtype=bool).reshape(-1)
    if arr.shape[:1] != keep.shape:
        raise ValueError(
            "PI-threshold eligibility mask length must match first array dimension"
        )
    shape = (keep.size,) + (1,) * (arr.ndim - 1)
    return np.where(keep.reshape(shape), arr, np.nan)
