#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.agarose_bundle_stats import (
    compare_agarose_bundles,
    long_rows,
    summary_rows,
)


def _fmt(x: float) -> str:
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def _to_zero_based(x: int | None) -> int | None:
    if x is None:
        return None
    if x == 0:
        raise SystemExit("Sync bucket indices are now 1-based; 0 is invalid.")
    return x - 1 if x > 0 else x


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Compare two agarose+SLI bundles at pre-training versus one selected "
            "training/sync-bucket point, and test whether the change differs between bundles."
        )
    )
    p.add_argument("--bundle-a", required=True, help="First agarose .npz bundle.")
    p.add_argument("--bundle-b", required=True, help="Second agarose .npz bundle.")
    p.add_argument("--label-a", default=None, help="Optional display label for bundle A.")
    p.add_argument("--label-b", default=None, help="Optional display label for bundle B.")
    p.add_argument(
        "--mode",
        default="exp",
        choices=["exp", "ctrl", "exp_minus_ctrl"],
        help="Which agarose series to compare.",
    )
    p.add_argument(
        "--training-index",
        type=int,
        default=2,
        help="1-based training index to compare against pre-training (default: 2).",
    )
    p.add_argument(
        "--sync-bucket-index",
        type=int,
        default=-1,
        help="1-based sync bucket index within the selected training; negative values count from the end (default: -1).",
    )
    p.add_argument(
        "--sync-bucket-start-index",
        type=int,
        default=None,
        help="Optional 1-based start sync bucket index for a contiguous post window.",
    )
    p.add_argument(
        "--sync-bucket-end-index",
        type=int,
        default=None,
        help="Optional 1-based end sync bucket index for a contiguous post window.",
    )
    p.add_argument(
        "--csv-out",
        default=None,
        help="Optional summary CSV output path.",
    )
    p.add_argument(
        "--long-csv-out",
        default=None,
        help="Optional per-video CSV output path with pre/post/delta values.",
    )
    args = p.parse_args()

    result = compare_agarose_bundles(
        args.bundle_a,
        args.bundle_b,
        mode=args.mode,
        training_index_1based=args.training_index,
        bucket_index=_to_zero_based(args.sync_bucket_index),
        bucket_start_index=_to_zero_based(args.sync_bucket_start_index),
        bucket_end_index=_to_zero_based(args.sync_bucket_end_index),
        label_a=args.label_a,
        label_b=args.label_b,
    )
    sel_a = result["bundle_a"]
    sel_b = result["bundle_b"]
    within_a = result["within_a"]
    within_b = result["within_b"]
    interaction = result["interaction"]

    print(
        "Selection:"
        f" mode={sel_a.mode}, training={sel_a.training_idx + 1} ({sel_a.training_name}),"
        f" sync_bucket={sel_a.bucket_start_idx + 1}"
        + (
            ""
            if sel_a.bucket_start_idx == sel_a.bucket_end_idx
            else f"-{sel_a.bucket_end_idx + 1}"
        )
        + ","
        f" bucket_window_min={_fmt(sel_a.bucket_start_min)}-{_fmt(sel_a.bucket_end_min)}"
    )
    for sel, stats in ((sel_a, within_a), (sel_b, within_b)):
        print(f"\n{sel.bundle_label}")
        print(f"  n paired: {_fmt(stats['n_pairs'])}")
        print(
            "  pre (paired to selected post):"
            f" mean={_fmt(stats['pre'].mean)}"
            f" CI=[{_fmt(stats['pre'].ci_low)}, {_fmt(stats['pre'].ci_high)}]"
        )
        print(
            "  post    :"
            f" mean={_fmt(stats['post'].mean)}"
            f" CI=[{_fmt(stats['post'].ci_low)}, {_fmt(stats['post'].ci_high)}]"
        )
        print(
            "  delta   :"
            f" mean={_fmt(stats['delta'].mean)}"
            f" CI=[{_fmt(stats['delta'].ci_low)}, {_fmt(stats['delta'].ci_high)}]"
        )
        print(
            "  paired t:"
            f" t={_fmt(stats['t_stat'])}"
            f" p={_fmt(stats['p_value'])}"
        )

    print("\nBetween-Bundle Delta Comparison")
    print(
        "  test    : two-sided Welch t-test on the two unpaired samples of per-video deltas"
    )
    print(
        "  null    : mean(delta "
        f"{sel_a.bundle_label}) = mean(delta {sel_b.bundle_label})"
    )
    print(
        "  deltas  :"
        f" {sel_a.bundle_label}={_fmt(within_a['delta'].mean)}"
        f" vs {sel_b.bundle_label}={_fmt(within_b['delta'].mean)}"
    )
    print(
        "  mean difference:"
        f" mean={_fmt(interaction['mean_difference'])}"
        f" t={_fmt(interaction['t_stat'])}"
        f" p={_fmt(interaction['p_value'])}"
        f" n=({_fmt(interaction['n_a'])}, {_fmt(interaction['n_b'])})"
    )
    print(
        "  note    : for this 2-bundle / 2-timepoint design, this is the test of whether"
        " the mean pre->post change differs between bundles."
    )

    if args.csv_out:
        _write_csv(args.csv_out, summary_rows(result))
        print(f"\nWrote summary CSV: {args.csv_out}")
    if args.long_csv_out:
        _write_csv(args.long_csv_out, long_rows(result))
        print(f"Wrote long CSV: {args.long_csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
