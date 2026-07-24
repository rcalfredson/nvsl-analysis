"""Audit export for reward-PI-difference AUC calculations."""

from __future__ import annotations

import csv
import os

import numpy as np

from src.utils.common import areaUnderCurve
import src.utils.util as util


DEFAULT_RPID_AUC_AUDIT_CSV = "exports/rpid_auc_audit.csv"

_FIELDS = (
    "video",
    "fly",
    "training",
    "bucket",
    "sync_bucket_minutes",
    "auc_x_spacing",
    "auc_units",
    "exp_reward_pi",
    "yoked_reward_pi",
    "difference",
    "included_in_auc",
    "next_difference",
    "trapezoid_term_to_next",
    "cumulative_auc_through_next",
    "effective_bucket_count",
    "trailing_all_nan_bucket_dropped",
    "equation",
    "recomputed_auc",
    "learning_stats_auc",
    "matches_learning_stats",
)


def _format_number(value) -> str:
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return format(value, ".12g")


def _auc_equation(differences: np.ndarray, auc: float) -> str:
    if len(differences) < 2:
        terms = "0 (fewer than two effective buckets)"
    else:
        terms = " + ".join(
            f"({_format_number(left)} + {_format_number(right)}) / 2"
            for left, right in zip(differences[:-1], differences[1:])
        )
    return f"AUC = {terms} = {_format_number(auc)}"


def _stored_rpid_auc(va, training_idx: int):
    entries = getattr(va, "saved_auc", {}).get("rpid", ())
    for entry in entries:
        if int(entry["training"]) == training_idx:
            return float(entry["exp"])
    return None


def write_rpid_auc_audit_csv(
    vas,
    trns,
    raw_reward_pi,
    *,
    out_csv: str = DEFAULT_RPID_AUC_AUDIT_CSV,
    sync_bucket_minutes: float,
) -> tuple[int, int]:
    """
    Write the exact inputs and trapezoid terms used for reward-PI-difference AUC.

    Returns ``(row_count, mismatch_count)``. A mismatch means the independently
    recomputed AUC does not equal the value staged for ``learning_stats.csv``.
    """
    raw_reward_pi = np.asarray(raw_reward_pi, dtype=float)
    if raw_reward_pi.ndim != 4:
        raise ValueError(
            "expected reward PI shaped video x training x fly x bucket, "
            f"got {raw_reward_pi.shape}"
        )
    if raw_reward_pi.shape[0] != len(vas):
        raise ValueError(
            "reward-PI audit row count does not match video analyses "
            f"({raw_reward_pi.shape[0]} vs. {len(vas)})"
        )
    if raw_reward_pi.shape[1] != len(trns):
        raise ValueError(
            "reward-PI audit training count does not match trainings "
            f"({raw_reward_pi.shape[1]} vs. {len(trns)})"
        )
    if raw_reward_pi.shape[2] < 2:
        raise ValueError("reward-PI AUC audit requires experimental and yoked values")
    if raw_reward_pi.shape[3] < 1:
        raise ValueError("reward-PI AUC audit requires at least one sync bucket")

    differences = raw_reward_pi[:, :, 0, :] - raw_reward_pi[:, :, 1, :]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    row_count = 0
    mismatch_count = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()

        for training_idx, trn in enumerate(trns):
            training_differences = differences[:, training_idx, :]
            recomputed_aucs = areaUnderCurve(training_differences)
            trailing_dropped = bool(
                np.all(np.isnan(training_differences[:, -1]))
            )
            effective_bucket_count = (
                training_differences.shape[1] - int(trailing_dropped)
            )

            for video_idx, va in enumerate(vas):
                effective_differences = training_differences[
                    video_idx, :effective_bucket_count
                ]
                recomputed_auc = float(recomputed_aucs[video_idx])
                stored_auc = _stored_rpid_auc(va, training_idx)
                matches = stored_auc is not None and bool(
                    np.isclose(
                        recomputed_auc,
                        stored_auc,
                        rtol=1e-12,
                        atol=1e-12,
                        equal_nan=True,
                    )
                )
                mismatch_count += int(not matches)
                equation = _auc_equation(effective_differences, recomputed_auc)

                cumulative_auc = 0.0
                for bucket_idx in range(training_differences.shape[1]):
                    included = bucket_idx < effective_bucket_count
                    has_next = included and bucket_idx + 1 < effective_bucket_count
                    if has_next:
                        next_difference = effective_differences[bucket_idx + 1]
                        trapezoid_term = (
                            effective_differences[bucket_idx] + next_difference
                        ) / 2
                        cumulative_auc += trapezoid_term
                    else:
                        next_difference = None
                        trapezoid_term = None

                    writer.writerow(
                        {
                            "video": util.basename(va.fn),
                            "fly": "" if va.f is None else va.f,
                            "training": getattr(trn, "n", training_idx + 1),
                            "bucket": bucket_idx + 1,
                            "sync_bucket_minutes": sync_bucket_minutes,
                            "auc_x_spacing": 1,
                            "auc_units": "PI*bucket_interval",
                            "exp_reward_pi": raw_reward_pi[
                                video_idx, training_idx, 0, bucket_idx
                            ],
                            "yoked_reward_pi": raw_reward_pi[
                                video_idx, training_idx, 1, bucket_idx
                            ],
                            "difference": training_differences[
                                video_idx, bucket_idx
                            ],
                            "included_in_auc": included,
                            "next_difference": (
                                "" if next_difference is None else next_difference
                            ),
                            "trapezoid_term_to_next": (
                                "" if trapezoid_term is None else trapezoid_term
                            ),
                            "cumulative_auc_through_next": (
                                "" if not has_next else cumulative_auc
                            ),
                            "effective_bucket_count": effective_bucket_count,
                            "trailing_all_nan_bucket_dropped": trailing_dropped,
                            "equation": equation if bucket_idx == 0 else "",
                            "recomputed_auc": recomputed_auc,
                            "learning_stats_auc": (
                                "" if stored_auc is None else stored_auc
                            ),
                            "matches_learning_stats": matches,
                        }
                    )
                    row_count += 1

    return row_count, mismatch_count
