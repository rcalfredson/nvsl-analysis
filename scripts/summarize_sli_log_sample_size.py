#!/usr/bin/env python3
"""Summarize why reward-PI/SLI sample sizes are lost in analyze.py logs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

from scipy import stats

BLOCK_RE = re.compile(r"^=== analyzing (.*), fly (\d+) ===$")
TRAIN_RE = re.compile(r"^training (\d+)")
FIELD_RE = re.compile(r"^  (actual|calc\. exp|ctrl\. exp|calc\. yok|ctrl\. yok): (.*)$")


def _parse_values(raw: str) -> list[int | float] | str:
    raw = raw.strip()
    if raw in {"trajectory bad", "no full bucket", "skipped"}:
        return raw

    values: list[int | float] = []
    for part in raw.split(","):
        token = part.strip()
        values.append(math.nan if token == "nan" else int(token))
    return values


def parse_log(path: Path) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_rewards = False
    current_training: int | None = None

    for line in path.read_text().splitlines():
        block_match = BLOCK_RE.match(line)
        if block_match:
            current = {
                "block": len(blocks),
                "video": block_match.group(1),
                "fly": int(block_match.group(2)),
                "trainings": {},
            }
            blocks.append(current)
            in_rewards = False
            current_training = None
            continue

        if current is None:
            continue

        if line.startswith("number rewards by sync bucket"):
            in_rewards = True
            current_training = None
            continue

        if in_rewards and line.startswith("average distance traveled"):
            in_rewards = False
            current_training = None
            continue

        if not in_rewards:
            continue

        training_match = TRAIN_RE.match(line)
        if training_match:
            current_training = int(training_match.group(1))
            current["trainings"].setdefault(current_training, {})
            continue

        field_match = FIELD_RE.match(line)
        if field_match and current_training is not None:
            current["trainings"][current_training][field_match.group(1)] = (
                _parse_values(field_match.group(2))
            )

    return blocks


def _bucket_value(
    training: dict[str, Any], key: str, bucket: int
) -> int | float | None:
    values = training.get(key)
    if isinstance(values, list) and bucket <= len(values):
        return values[bucket - 1]
    return None


def build_rows(
    blocks: list[dict[str, Any]],
    *,
    pi_threshold: int,
    trainings: range,
    buckets: range,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        for training_idx in trainings:
            training = block["trainings"].get(training_idx, {})
            for bucket in buckets:
                calc_exp = _bucket_value(training, "calc. exp", bucket)
                ctrl_exp = _bucket_value(training, "ctrl. exp", bucket)
                calc_yok = _bucket_value(training, "calc. yok", bucket)
                ctrl_yok = _bucket_value(training, "ctrl. yok", bucket)
                actual = _bucket_value(training, "actual", bucket)

                exp_exists = calc_exp is not None and ctrl_exp is not None
                yok_exists = calc_yok is not None and ctrl_yok is not None
                bucket_exists = exp_exists and yok_exists

                exp_total = calc_exp + ctrl_exp if exp_exists else None
                yok_total = calc_yok + ctrl_yok if yok_exists else None
                exp_pi_valid = exp_total is not None and exp_total >= pi_threshold
                yok_pi_valid = yok_total is not None and yok_total >= pi_threshold
                sli_valid = exp_pi_valid and yok_pi_valid

                if exp_pi_valid:
                    exp_pi = (calc_exp - ctrl_exp) / exp_total
                else:
                    exp_pi = math.nan
                if yok_pi_valid:
                    yok_pi = (calc_yok - ctrl_yok) / yok_total
                else:
                    yok_pi = math.nan

                if sli_valid:
                    loss_reason = "valid"
                    sli = exp_pi - yok_pi
                elif not bucket_exists:
                    loss_reason = "missing_bucket_or_bad_trajectory"
                    sli = math.nan
                elif not exp_pi_valid and not yok_pi_valid:
                    loss_reason = "both_below_piTh"
                    sli = math.nan
                elif not exp_pi_valid:
                    loss_reason = "exp_below_piTh"
                    sli = math.nan
                else:
                    loss_reason = "yok_below_piTh"
                    sli = math.nan

                rows.append(
                    {
                        "block": block["block"],
                        "video": block["video"],
                        "fly": block["fly"],
                        "training": training_idx,
                        "bucket": bucket,
                        "actual": actual,
                        "calc_exp": calc_exp,
                        "ctrl_exp": ctrl_exp,
                        "calc_yok": calc_yok,
                        "ctrl_yok": ctrl_yok,
                        "exp_total": exp_total,
                        "yok_total": yok_total,
                        "exp_bucket_exists": exp_exists,
                        "yok_bucket_exists": yok_exists,
                        "bucket_exists_for_sli": bucket_exists,
                        "exp_pi_valid": exp_pi_valid,
                        "yok_pi_valid": yok_pi_valid,
                        "sli_valid": sli_valid,
                        "exp_pi": exp_pi,
                        "yok_pi": yok_pi,
                        "sli": sli,
                        "loss_reason": loss_reason,
                    }
                )
    return rows


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row[key]) for row in rows)


def build_reason_counts(
    rows: list[dict[str, Any]],
    *,
    group: str,
    trainings: range,
    buckets: range,
) -> list[dict[str, Any]]:
    counts: list[dict[str, Any]] = []
    for training_idx in trainings:
        for bucket in buckets:
            subset = [
                row
                for row in rows
                if row["training"] == training_idx and row["bucket"] == bucket
            ]
            reason_counts = {
                "missing_bucket_or_bad_trajectory": 0,
                "exp_below_piTh": 0,
                "yok_below_piTh": 0,
                "both_below_piTh": 0,
            }
            for row in subset:
                reason = row["loss_reason"]
                if reason in reason_counts:
                    reason_counts[reason] += 1

            counts.append(
                {
                    "group": group,
                    "training": training_idx,
                    "bucket": bucket,
                    "denominator": len(subset),
                    "bucket_exists": _count(subset, "bucket_exists_for_sli"),
                    "exp_valid": _count(subset, "exp_pi_valid"),
                    "yok_valid": _count(subset, "yok_pi_valid"),
                    "sli_valid": _count(subset, "sli_valid"),
                    **reason_counts,
                }
            )
    return counts


def _one_sided_ttest_greater(values: list[float]) -> tuple[float, float]:
    result = stats.ttest_1samp(values, popmean=0.0, alternative="greater")
    return float(result.statistic), float(result.pvalue)


def _one_sided_slope_test_greater(
    buckets: list[float], values: list[float]
) -> tuple[float, float, float]:
    result = stats.linregress(buckets, values)
    pvalue = result.pvalue / 2.0 if result.slope > 0 else 1.0 - result.pvalue / 2.0
    return float(result.slope), float(result.stderr), float(pvalue)


def build_summary_tests(
    reason_counts: list[dict[str, Any]],
    *,
    group: str,
    trainings: range,
) -> list[dict[str, Any]]:
    test_rows: list[dict[str, Any]] = []
    groups = sorted({str(row.get("group") or group) for row in reason_counts})
    for group_name in groups:
        group_counts = [
            row for row in reason_counts if str(row.get("group") or group) == group_name
        ]
        for training_idx in trainings:
            subset = [
                row for row in group_counts if int(row["training"]) == training_idx
            ]
            subset.sort(key=lambda row: int(row["bucket"]))
            if not subset:
                continue

            yok_minus_exp = [
                float(row["yok_below_piTh"] - row["exp_below_piTh"]) for row in subset
            ]
            t_stat, pvalue = _one_sided_ttest_greater(yok_minus_exp)
            test_rows.append(
                {
                    "group": group_name,
                    "training": training_idx,
                    "test": "yok_minus_exp_below_piTh",
                    "estimate": sum(yok_minus_exp) / len(yok_minus_exp),
                    "statistic": t_stat,
                    "pvalue": pvalue,
                    "n_buckets": len(yok_minus_exp),
                }
            )

            bucket_indices = [float(row["bucket"]) for row in subset]
            missing_counts = [
                float(row["missing_bucket_or_bad_trajectory"]) for row in subset
            ]
            slope, stderr, slope_pvalue = _one_sided_slope_test_greater(
                bucket_indices,
                missing_counts,
            )
            statistic = slope / stderr if stderr and math.isfinite(stderr) else math.nan
            test_rows.append(
                {
                    "group": group_name,
                    "training": training_idx,
                    "test": "missing_count_slope",
                    "estimate": slope,
                    "statistic": statistic,
                    "pvalue": slope_pvalue,
                    "n_buckets": len(missing_counts),
                }
            )
    return test_rows


def print_summary_tests(test_rows: list[dict[str, Any]]) -> None:
    print("\nSummary tests")
    print("test group training estimate statistic one_sided_p n_buckets")
    for row in test_rows:
        print(
            f"{row['test']} {row['group']} T{row['training']} "
            f"{row['estimate']:.6g} {row['statistic']:.6g} "
            f"{row['pvalue']:.6g} {row['n_buckets']}"
        )


def print_summary(
    blocks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    trainings: range,
    buckets: range,
) -> None:
    print(f"fly blocks parsed: {len(blocks)}")
    print("\nSLI finite sample size by training and sync bucket")
    print("training bucket denominator bucket_exists exp_valid yok_valid sli_valid")
    for training_idx in trainings:
        for bucket in buckets:
            subset = [
                row
                for row in rows
                if row["training"] == training_idx and row["bucket"] == bucket
            ]
            print(
                f"T{training_idx} SB{bucket}: "
                f"{len(subset):2d} "
                f"{_count(subset, 'bucket_exists_for_sli'):2d} "
                f"{_count(subset, 'exp_pi_valid'):2d} "
                f"{_count(subset, 'yok_pi_valid'):2d} "
                f"{_count(subset, 'sli_valid'):2d}"
            )

    print("\nLoss reason counts")
    for training_idx in trainings:
        for bucket in buckets:
            counts: dict[str, int] = {}
            for row in rows:
                if (
                    row["training"] == training_idx
                    and row["bucket"] == bucket
                    and not row["sli_valid"]
                ):
                    reason = row["loss_reason"]
                    counts[reason] = counts.get(reason, 0) + 1
            print(f"T{training_idx} SB{bucket}: {counts}")

    print("\nTraining-level finite SLI rollups")
    print(
        "definition: any finite bucket / all requested buckets finite / all buckets exist"
    )
    for training_idx in trainings:
        any_finite = 0
        all_finite = 0
        all_exist = 0
        finite_counts: list[int] = []
        existing_counts: list[int] = []
        for block in blocks:
            subset = [
                row
                for row in rows
                if row["block"] == block["block"] and row["training"] == training_idx
            ]
            finite_count = _count(subset, "sli_valid")
            existing_count = _count(subset, "bucket_exists_for_sli")
            finite_counts.append(finite_count)
            existing_counts.append(existing_count)
            any_finite += finite_count > 0
            all_finite += finite_count == len(buckets)
            all_exist += existing_count == len(buckets)

        finite_hist = {idx: finite_counts.count(idx) for idx in range(len(buckets) + 1)}
        existing_hist = {
            idx: existing_counts.count(idx) for idx in range(len(buckets) + 1)
        }
        print(
            f"T{training_idx}: any={any_finite}, all={all_finite}, "
            f"all_exist={all_exist}, finite_bucket_hist={finite_hist}, "
            f"existing_bucket_hist={existing_hist}"
        )

    print("\nMedian total entries among existing buckets (exp_total/yok_total)")
    for training_idx in trainings:
        for bucket in buckets:
            subset = [
                row
                for row in rows
                if row["training"] == training_idx and row["bucket"] == bucket
            ]
            exp_totals = [
                row["exp_total"] for row in subset if row["exp_total"] is not None
            ]
            yok_totals = [
                row["yok_total"] for row in subset if row["yok_total"] is not None
            ]
            exp_med = median(exp_totals) if exp_totals else None
            yok_med = median(yok_totals) if yok_totals else None
            print(f"T{training_idx} SB{bucket}: exp med {exp_med}, yok med {yok_med}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_reason_counts_csv(path: Path) -> list[dict[str, Any]]:
    int_columns = {
        "training",
        "bucket",
        "denominator",
        "bucket_exists",
        "exp_valid",
        "yok_valid",
        "sli_valid",
        "missing_bucket_or_bad_trajectory",
        "exp_below_piTh",
        "yok_below_piTh",
        "both_below_piTh",
    }
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for column in int_columns:
            if column in row and row[column] != "":
                row[column] = int(row[column])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, nargs="?", help="analyze.py log file")
    parser.add_argument(
        "--from-reason-counts",
        type=Path,
        help="run summary tests from a bucket-wise reason-count CSV",
    )
    parser.add_argument("--pi-threshold", type=int, default=10)
    parser.add_argument("--max-training", type=int, default=3)
    parser.add_argument("--max-bucket", type=int, default=5)
    parser.add_argument("--group", default="group", help="label for CSV/test outputs")
    parser.add_argument("--csv", type=Path, help="optional per-bucket row export")
    parser.add_argument(
        "--reason-counts-csv",
        type=Path,
        help="optional bucket-wise exclusion-count export",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        help="optional summary-test export",
    )
    args = parser.parse_args()

    trainings = range(1, args.max_training + 1)
    buckets = range(1, args.max_bucket + 1)

    if args.from_reason_counts:
        reason_counts = read_reason_counts_csv(args.from_reason_counts)
        test_rows = build_summary_tests(
            reason_counts,
            group=args.group,
            trainings=trainings,
        )
        print_summary_tests(test_rows)
        if args.stats_csv:
            write_csv(args.stats_csv, test_rows)
        return

    if args.log is None:
        parser.error("provide a log file or --from-reason-counts")

    blocks = parse_log(args.log)
    rows = build_rows(
        blocks,
        pi_threshold=args.pi_threshold,
        trainings=trainings,
        buckets=buckets,
    )
    print_summary(blocks, rows, trainings=trainings, buckets=buckets)
    reason_counts = build_reason_counts(
        rows,
        group=args.group,
        trainings=trainings,
        buckets=buckets,
    )
    test_rows = build_summary_tests(
        reason_counts,
        group=args.group,
        trainings=trainings,
    )
    print_summary_tests(test_rows)
    if args.csv:
        write_csv(args.csv, rows)
    if args.reason_counts_csv:
        write_csv(args.reason_counts_csv, reason_counts)
    if args.stats_csv:
        write_csv(args.stats_csv, test_rows)


if __name__ == "__main__":
    main()
