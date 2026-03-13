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
    build_delta_table,
    read_bundle_manifest,
    selective_followups,
    two_way_anova_from_rows,
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
            "Run a two-factor ANOVA on agarose bundle deltas across phenotype and chamber, "
            "then run Welch simple-effects follow-ups on relevant slices."
        )
    )
    p.add_argument(
        "--manifest",
        required=True,
        help=(
            "CSV with columns: bundle_path, phenotype, chamber, and optional bundle_label."
        ),
    )
    p.add_argument(
        "--mode",
        default="exp",
        choices=["exp", "ctrl", "exp_minus_ctrl"],
        help="Which agarose series to analyze.",
    )
    p.add_argument(
        "--pre-scope",
        default="experiment",
        choices=["experiment", "training"],
        help=(
            "Which pre-training baseline to use: the experiment-wide last 10 minutes "
            "or the selected training's own last 10 minutes."
        ),
    )
    p.add_argument(
        "--training-index",
        type=int,
        default=2,
        help="1-based training index to compare against the selected pre baseline (default: 2).",
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
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for deciding which follow-up families to run.",
    )
    p.add_argument(
        "--always-run-followups",
        action="store_true",
        help="Run the simple-effects Welch follow-ups even if the omnibus terms are not significant.",
    )
    p.add_argument(
        "--anova-csv-out",
        default=None,
        help="Optional CSV path for the omnibus ANOVA table.",
    )
    p.add_argument(
        "--followup-csv-out",
        default=None,
        help="Optional CSV path for simple-effects Welch follow-ups.",
    )
    p.add_argument(
        "--bundle-summary-csv-out",
        default=None,
        help="Optional CSV path for per-bundle pre/post/delta summaries.",
    )
    p.add_argument(
        "--long-csv-out",
        default=None,
        help="Optional CSV path for the per-video delta table used by the ANOVA.",
    )
    args = p.parse_args()

    entries = read_bundle_manifest(args.manifest)
    long_rows, bundle_rows, meta = build_delta_table(
        entries,
        mode=args.mode,
        pre_scope=args.pre_scope,
        training_index_1based=args.training_index,
        bucket_index=_to_zero_based(args.sync_bucket_index),
        bucket_start_index=_to_zero_based(args.sync_bucket_start_index),
        bucket_end_index=_to_zero_based(args.sync_bucket_end_index),
    )
    anova = two_way_anova_from_rows(long_rows)
    followups = selective_followups(
        long_rows,
        anova,
        alpha=float(args.alpha),
        always_run=bool(args.always_run_followups),
    )

    print(
        "Selection:"
        f" mode={meta['mode']}, pre_scope={meta['pre_scope']},"
        f" training={meta['training_index']} ({meta['training_name']}),"
        f" sync_bucket={meta['bucket_start_index']}"
        + (
            ""
            if meta["bucket_start_index"] == meta["bucket_end_index"]
            else f"-{meta['bucket_end_index']}"
        )
        + ","
        f" bucket_window_min={_fmt(meta['bucket_start_min'])}-{_fmt(meta['bucket_end_min'])}"
    )
    print(f"Rows: n={len(long_rows)} observations across {len(entries)} bundles")

    print("\nOmnibus ANOVA on delta")
    for row in anova["terms"]:
        if row["term"] == "total":
            continue
        print(
            f"  {row['term']}:"
            f" df={row['df']}"
            f" SS={_fmt(row['sum_sq'])}"
            f" MS={_fmt(row['mean_sq'])}"
            f" F={_fmt(row['F'])}"
            f" p={_fmt(row['p_value'])}"
        )

    if followups:
        print("\nFollow-up Welch tests")
        for row in followups:
            print(
                f"  {row['compare_factor']} within {row['slice_factor']}={row['slice_level']}:"
                f" {row['level_a']} vs {row['level_b']}"
                f" mean_diff={_fmt(row['mean_diff'])}"
                f" t={_fmt(row['t_stat'])}"
                f" p={_fmt(row['p_value'])}"
                f" p_holm={_fmt(row['p_value_holm'])}"
                f" trigger={row['triggered_by']}"
            )
    else:
        print("\nFollow-up Welch tests")
        print("  none run under the current alpha threshold")

    if args.anova_csv_out:
        _write_csv(args.anova_csv_out, anova["terms"])
        print(f"\nWrote ANOVA CSV: {args.anova_csv_out}")
    if args.followup_csv_out:
        _write_csv(args.followup_csv_out, followups)
        print(f"Wrote follow-up CSV: {args.followup_csv_out}")
    if args.bundle_summary_csv_out:
        _write_csv(args.bundle_summary_csv_out, bundle_rows)
        print(f"Wrote bundle summary CSV: {args.bundle_summary_csv_out}")
    if args.long_csv_out:
        _write_csv(args.long_csv_out, long_rows)
        print(f"Wrote long CSV: {args.long_csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
