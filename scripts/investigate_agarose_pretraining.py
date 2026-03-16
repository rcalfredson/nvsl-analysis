#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.agarose_preinvestigation import (
    add_leave_one_out_shifts,
    attach_reconstruction,
    load_bundle_rows,
    reconstruct_pretraining_diagnostics,
    summarize_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Investigate pre-training dual-circle agarose avoidance differences "
            "between two chamber datasets."
        )
    )
    p.add_argument("--bundle-a", required=True, help="First agarose avoidance bundle (.npz).")
    p.add_argument("--bundle-b", required=True, help="Second agarose avoidance bundle (.npz).")
    p.add_argument("--label-a", default="Bundle A", help="Display label for bundle A.")
    p.add_argument("--label-b", default="Bundle B", help="Display label for bundle B.")
    p.add_argument(
        "--training-index",
        type=int,
        default=2,
        help="1-based training index used for the post-training reference (default: 2).",
    )
    p.add_argument(
        "--sync-bucket-index",
        type=int,
        default=-1,
        help="1-based sync-bucket index for the post-training reference; negative values count from the end (default: -1).",
    )
    p.add_argument(
        "--sync-bucket-start-index",
        type=int,
        default=None,
        help="Optional 1-based start sync-bucket index for a post window.",
    )
    p.add_argument(
        "--sync-bucket-end-index",
        type=int,
        default=None,
        help="Optional 1-based end sync-bucket index for a post window.",
    )
    p.add_argument(
        "--delta-mm",
        type=float,
        default=1.0,
        help="Outer-circle padding in mm used for the raw reconstruction (default: 1.0).",
    )
    p.add_argument(
        "--pre-window-min",
        type=float,
        default=10.0,
        help="Pre-training window length in minutes for the raw reconstruction (default: 10).",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to exports/agarose_preinvestigation_<date>.",
    )
    return p.parse_args()


def to_zero_based(x: int | None) -> int | None:
    if x is None:
        return None
    if x == 0:
        raise SystemExit("Sync bucket indices are 1-based; 0 is invalid.")
    return x - 1 if x > 0 else x


def build_bundle_artifacts(
    bundle_path: str,
    *,
    label: str,
    training_index_1based: int,
    bucket_index: int,
    bucket_start_index: int | None,
    bucket_end_index: int | None,
    delta_mm: float,
    pre_window_min: float,
) -> tuple[list[dict], list[dict], list[dict], object]:
    bundle_rows, spec = load_bundle_rows(
        bundle_path,
        label=label,
        training_index_1based=training_index_1based,
        bucket_index=bucket_index,
        bucket_start_index=bucket_start_index,
        bucket_end_index=bucket_end_index,
    )
    unique_videos = sorted({str(row["video_id"]) for row in bundle_rows})
    recon_rows: list[dict] = []
    for video_id in unique_videos:
        try:
            recon_rows.extend(
                reconstruct_pretraining_diagnostics(
                    video_id,
                    delta_mm=delta_mm,
                    pre_window_min=pre_window_min,
                )
            )
        except FileNotFoundError:
            continue
    merged_rows = attach_reconstruction(bundle_rows, recon_rows)
    add_leave_one_out_shifts(merged_rows)
    video_rows = summarize_rows(merged_rows, chamber_label=label, group_key="video_id")
    summary_rows = summarize_rows(merged_rows, chamber_label=label)
    return merged_rows, video_rows, summary_rows, spec


def plot_bundle_comparison(rows_a: list[dict], rows_b: list[dict], spec, out_path: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    bundles = [rows_a, rows_b]
    labels = [rows_a[0]["bundle_label"], rows_b[0]["bundle_label"]]
    colors = ["#1b5e20", "#b23a48"]

    for ax, key, title in (
        (axs[0, 0], "bundle_pre_ratio", "Pre-training Ratio"),
        (axs[0, 1], "bundle_pre_total_count", "Pre-training Episode Count"),
        (axs[1, 0], "bundle_delta", "Selected Post - Pre"),
    ):
        xs = np.arange(len(bundles))
        for i, (rows, color) in enumerate(zip(bundles, colors)):
            vals = np.array([row[key] for row in rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            jitter = np.linspace(-0.12, 0.12, max(1, len(vals)))
            ax.scatter(np.full(len(vals), xs[i]) + jitter, vals, s=18, alpha=0.7, color=color)
            ax.hlines(np.nanmean(vals), xs[i] - 0.2, xs[i] + 0.2, color="black", linewidth=2)
        ax.set_xticks(xs, labels)
        ax.set_title(title)
        ax.grid(alpha=0.2, axis="y")

    ax = axs[1, 1]
    for rows, label, color in zip(bundles, labels, colors):
        x = np.array([row["bundle_pre_total_count"] for row in rows], dtype=float)
        y = np.array([row["bundle_pre_ratio"] for row in rows], dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[keep], y[keep], s=22, alpha=0.7, label=label, color=color)
    ax.set_xlabel("Pre-training total episodes")
    ax.set_ylabel("Pre-training ratio")
    ax.set_title("Ratio vs Episode Count")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle(
        "Agarose Pre-training Investigation\n"
        f"Post reference: {spec.training_name}, sync bucket {spec.bucket_start_idx + 1}"
        + (
            ""
            if spec.bucket_start_idx == spec.bucket_end_idx
            else f"-{spec.bucket_end_idx + 1}"
        )
        + f" ({spec.bucket_start_min:.1f}-{spec.bucket_end_min:.1f} min)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir or f"exports/agarose_preinvestigation_{date.today().isoformat()}")
    outdir.mkdir(parents=True, exist_ok=True)

    bucket_index = to_zero_based(args.sync_bucket_index)
    bucket_start_index = to_zero_based(args.sync_bucket_start_index)
    bucket_end_index = to_zero_based(args.sync_bucket_end_index)

    rows_a, videos_a, summary_a, spec_a = build_bundle_artifacts(
        args.bundle_a,
        label=args.label_a,
        training_index_1based=args.training_index,
        bucket_index=bucket_index,
        bucket_start_index=bucket_start_index,
        bucket_end_index=bucket_end_index,
        delta_mm=float(args.delta_mm),
        pre_window_min=float(args.pre_window_min),
    )
    rows_b, videos_b, summary_b, spec_b = build_bundle_artifacts(
        args.bundle_b,
        label=args.label_b,
        training_index_1based=args.training_index,
        bucket_index=bucket_index,
        bucket_start_index=bucket_start_index,
        bucket_end_index=bucket_end_index,
        delta_mm=float(args.delta_mm),
        pre_window_min=float(args.pre_window_min),
    )

    if spec_a != spec_b:
        raise RuntimeError("Bundle selections do not match.")

    comparison_summary = summary_a + summary_b
    comparison_video_rows = videos_a + videos_b
    comparison_rows = rows_a + rows_b

    write_csv(outdir / "per_row.csv", comparison_rows)
    write_csv(outdir / "video_summary.csv", comparison_video_rows)
    write_csv(outdir / "summary.csv", comparison_summary)
    plot_bundle_comparison(rows_a, rows_b, spec_a, outdir / "comparison.png")

    print(
        "Selection:"
        f" training={spec_a.training_idx + 1} ({spec_a.training_name}),"
        f" sync_bucket={spec_a.bucket_start_idx + 1}"
        + (
            ""
            if spec_a.bucket_start_idx == spec_a.bucket_end_idx
            else f"-{spec_a.bucket_end_idx + 1}"
        )
        + f", bucket_window_min={spec_a.bucket_start_min:.1f}-{spec_a.bucket_end_min:.1f}"
    )
    for summary in comparison_summary:
        print(f"\n{summary['chamber']}")
        print(
            f"  pre ratio mean={summary['pre_ratio_mean']:.6g}"
            f" median={summary['pre_ratio_median']:.6g}"
            f" weighted={summary['pre_ratio_weighted']:.6g}"
        )
        print(
            f"  pre counts mean avoid/contact/total="
            f"{summary['pre_avoid_mean']:.6g}/{summary['pre_contact_mean']:.6g}/{summary['pre_total_mean']:.6g}"
        )
        print(
            f"  selected post mean={summary['post_ratio_mean']:.6g}"
            f" delta mean={summary['delta_mean']:.6g}"
        )
        print(
            f"  reconstructed occupancy outer/inner="
            f"{summary['recon_outer_frame_frac_mean']:.6g}/{summary['recon_inner_frame_frac_mean']:.6g}"
            f" bad_frac={summary['recon_bad_frac']:.6g}"
        )

    print(f"\nWrote: {outdir / 'per_row.csv'}")
    print(f"Wrote: {outdir / 'video_summary.csv'}")
    print(f"Wrote: {outdir / 'summary.csv'}")
    print(f"Wrote: {outdir / 'comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
