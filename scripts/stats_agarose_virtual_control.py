#!/usr/bin/env python3
"""Paired physical-versus-virtual agarose dual-circle comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.stats import sem, t, ttest_rel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_index(size: int, index: int, label: str) -> int:
    resolved = index if index >= 0 else size + index
    if resolved < 0 or resolved >= size:
        raise IndexError(f"{label} index {index} is out of range for size {size}")
    return resolved


def _resolve_bucket_window(n_buckets, bucket_idx, bucket_start_idx, bucket_end_idx):
    if bucket_start_idx is None and bucket_end_idx is None:
        idx = _resolve_index(n_buckets, bucket_idx, "sync bucket")
        return idx, idx
    if bucket_start_idx is None or bucket_end_idx is None:
        raise ValueError("bucket start and end indices must be provided together")
    start = _resolve_index(n_buckets, bucket_start_idx, "start sync bucket")
    end = _resolve_index(n_buckets, bucket_end_idx, "end sync bucket")
    if end < start:
        raise ValueError("sync-bucket window end precedes its start")
    return start, end


def _pooled_role_ratio(bundle, prefix, role, training_idx, start_idx, end_idx):
    avoid_key = f"{prefix}_avoid_{role}"
    total_key = f"{prefix}_total_{role}"
    if avoid_key not in bundle or total_key not in bundle:
        raise ValueError(f"bundle is missing {avoid_key!r} or {total_key!r}")
    avoid = np.asarray(bundle[avoid_key], dtype=float)
    total = np.asarray(bundle[total_key], dtype=float)
    if avoid.shape != total.shape or avoid.ndim != 3:
        raise ValueError(
            f"{prefix} {role} counts must share a (video, training, bucket) shape"
        )
    pooled_avoid = np.sum(
        avoid[:, training_idx, start_idx : end_idx + 1], axis=1
    )
    pooled_total = np.sum(
        total[:, training_idx, start_idx : end_idx + 1], axis=1
    )
    ratio = np.full(pooled_total.shape, np.nan, dtype=float)
    np.divide(pooled_avoid, pooled_total, out=ratio, where=pooled_total > 0)
    min_total = int(np.asarray(bundle.get("min_agarose_episodes", 5)).item())
    ratio[pooled_total < min_total] = np.nan
    if role == "exp" and "exp_target_sync_bucket_filter_eligible" in bundle:
        eligible = np.asarray(
            bundle["exp_target_sync_bucket_filter_eligible"], dtype=bool
        )
        if eligible.shape == ratio.shape:
            ratio[~eligible] = np.nan
    return ratio


def _pooled_series(bundle, prefix, mode, training_idx, start_idx, end_idx):
    exp = _pooled_role_ratio(
        bundle, prefix, "exp", training_idx, start_idx, end_idx
    )
    if mode == "exp":
        return exp
    ctrl = _pooled_role_ratio(
        bundle, prefix, "ctrl", training_idx, start_idx, end_idx
    )
    return ctrl if mode == "ctrl" else exp - ctrl


def select_paired_values(
    bundle,
    *,
    mode: str,
    training_idx: int,
    bucket_idx: int = -1,
    bucket_start_idx: int | None = None,
    bucket_end_idx: int | None = None,
):
    shape = np.asarray(bundle["agarose_total_exp"]).shape
    virtual_shape = np.asarray(bundle["agarose_virtual_total_exp"]).shape
    if shape != virtual_shape or len(shape) != 3:
        raise ValueError(
            "actual and virtual counts must have the same (video, training, bucket) shape"
        )
    training_idx = _resolve_index(shape[1], training_idx, "training")
    start_idx, end_idx = _resolve_bucket_window(
        shape[2], bucket_idx, bucket_start_idx, bucket_end_idx
    )
    actual = _pooled_series(
        bundle, "agarose", mode, training_idx, start_idx, end_idx
    )
    virtual = _pooled_series(
        bundle, "agarose_virtual", mode, training_idx, start_idx, end_idx
    )
    paired = np.isfinite(actual) & np.isfinite(virtual)
    return actual, virtual, paired, training_idx, start_idx, end_idx


def paired_summary(actual: np.ndarray, virtual: np.ndarray, paired: np.ndarray):
    a = np.asarray(actual, dtype=float)[paired]
    v = np.asarray(virtual, dtype=float)[paired]
    delta = a - v
    n = int(delta.size)
    mean_delta = float(np.mean(delta)) if n else np.nan
    if n >= 2:
        test = ttest_rel(a, v, nan_policy="omit")
        half_width = float(t.ppf(0.975, n - 1) * sem(delta))
        ci = (mean_delta - half_width, mean_delta + half_width)
        t_stat, p_value = float(test.statistic), float(test.pvalue)
    else:
        ci = (np.nan, np.nan)
        t_stat = p_value = np.nan
    return {
        "n": n,
        "actual_mean": float(np.mean(a)) if n else np.nan,
        "virtual_mean": float(np.mean(v)) if n else np.nan,
        "mean_actual_minus_virtual": mean_delta,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "t_stat": t_stat,
        "p_value": p_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Within-video paired comparison of physical-well and rotated virtual-site "
            "dual-circle avoidance ratios. Positive differences favor agarose-specific avoidance."
        )
    )
    parser.add_argument("--bundle", required=True, help="Agarose+SLI .npz bundle.")
    parser.add_argument(
        "--mode", choices=("exp", "ctrl", "exp_minus_ctrl"), default="exp"
    )
    parser.add_argument(
        "--training-index", type=int, default=2, help="1-based training index (default: 2)."
    )
    parser.add_argument(
        "--sync-bucket-index",
        type=int,
        default=-1,
        help="1-based bucket index; negative values count from the end (default: -1).",
    )
    parser.add_argument(
        "--sync-bucket-start-index",
        type=int,
        help="Optional 1-based first bucket of a contiguous analysis window.",
    )
    parser.add_argument(
        "--sync-bucket-end-index",
        type=int,
        help="Optional 1-based last bucket of a contiguous analysis window.",
    )
    parser.add_argument("--csv-out", help="Optional per-video paired-value CSV.")
    args = parser.parse_args()

    supplied_indices = (
        args.training_index,
        args.sync_bucket_index,
        args.sync_bucket_start_index,
        args.sync_bucket_end_index,
    )
    if any(index == 0 for index in supplied_indices if index is not None):
        parser.error("indices are 1-based; zero is invalid")
    if (args.sync_bucket_start_index is None) != (
        args.sync_bucket_end_index is None
    ):
        parser.error(
            "--sync-bucket-start-index and --sync-bucket-end-index must be used together"
        )
    requested_training = (
        args.training_index - 1 if args.training_index > 0 else args.training_index
    )
    requested_bucket = (
        args.sync_bucket_index - 1
        if args.sync_bucket_index > 0
        else args.sync_bucket_index
    )
    requested_start = (
        args.sync_bucket_start_index - 1
        if args.sync_bucket_start_index is not None
        and args.sync_bucket_start_index > 0
        else args.sync_bucket_start_index
    )
    requested_end = (
        args.sync_bucket_end_index - 1
        if args.sync_bucket_end_index is not None
        and args.sync_bucket_end_index > 0
        else args.sync_bucket_end_index
    )

    with np.load(args.bundle, allow_pickle=True) as bundle:
        actual, virtual, paired, training_idx, start_idx, end_idx = select_paired_values(
            bundle,
            mode=args.mode,
            training_idx=requested_training,
            bucket_idx=requested_bucket,
            bucket_start_idx=requested_start,
            bucket_end_idx=requested_end,
        )
        stats = paired_summary(actual, virtual, paired)
        video_ids = np.asarray(
            bundle.get("video_ids", np.arange(actual.size)), dtype=object
        )
        rotation = float(np.asarray(bundle["agarose_virtual_rotation_deg"]).item())
        farthest_only = bool(
            np.asarray(bundle.get("agarose_farthest_from_reward_only", False)).item()
        )
        wall_facing_only = bool(
            np.asarray(bundle.get("agarose_wall_facing_entry_only", False)).item()
        )
        center_shift_mm = float(
            np.asarray(
                bundle.get("agarose_dual_circle_center_shift_mm", 0.0)
            ).item()
        )
        wall_reference = str(
            np.asarray(bundle.get("agarose_wall_facing_reference", "arena")).item()
        )

    entry_selection = "entries=all"
    if wall_facing_only:
        entry_selection = (
            "entries=reward-away"
            if wall_reference == "reward"
            else "entries=arena-outward"
        )
    print(
        f"Selection: mode={args.mode}, training={training_idx + 1}, "
        "sync_bucket="
        + (
            f"{start_idx + 1}"
            if start_idx == end_idx
            else f"{start_idx + 1}-{end_idx + 1}"
        )
        + f", virtual_rotation={rotation:g} deg, "
        + ("sites=farthest-from-reward" if farthest_only else "sites=all")
        + f", {entry_selection}"
        + f", center_shift={center_shift_mm:g} mm"
        + f", wall_reference={wall_reference}"
    )
    print(f"Paired fly/video observations: {stats['n']}")
    print(f"Physical mean: {stats['actual_mean']:.6g}")
    print(f"Virtual mean: {stats['virtual_mean']:.6g}")
    print(
        "Physical - virtual: "
        f"{stats['mean_actual_minus_virtual']:.6g} "
        f"(95% CI {stats['ci_low']:.6g}, {stats['ci_high']:.6g})"
    )
    print(f"Paired t-test: t={stats['t_stat']:.6g}, p={stats['p_value']:.6g}")

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "video_id",
                    "physical_ratio",
                    "virtual_ratio",
                    "physical_minus_virtual",
                    "paired_in_test",
                ),
            )
            writer.writeheader()
            for video_id, a, v, keep in zip(video_ids, actual, virtual, paired):
                writer.writerow(
                    {
                        "video_id": video_id,
                        "physical_ratio": a,
                        "virtual_ratio": v,
                        "physical_minus_virtual": a - v,
                        "paired_in_test": bool(keep),
                    }
                )
        print(f"Wrote paired values: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
