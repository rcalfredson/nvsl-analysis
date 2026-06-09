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


BLOCK_RE = re.compile(r"^=== analyzing (.*), fly (\d+) ===$")
TRAIN_RE = re.compile(r"^training (\d+)")
FIELD_RE = re.compile(
    r"^  (actual|calc\. exp|ctrl\. exp|calc\. yok|ctrl\. yok): (.*)$"
)


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


def _bucket_value(training: dict[str, Any], key: str, bucket: int) -> int | float | None:
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
    print("definition: any finite bucket / all requested buckets finite / all buckets exist")
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
            exp_totals = [row["exp_total"] for row in subset if row["exp_total"] is not None]
            yok_totals = [row["yok_total"] for row in subset if row["yok_total"] is not None]
            exp_med = median(exp_totals) if exp_totals else None
            yok_med = median(yok_totals) if yok_totals else None
            print(f"T{training_idx} SB{bucket}: exp med {exp_med}, yok med {yok_med}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="analyze.py log file")
    parser.add_argument("--pi-threshold", type=int, default=10)
    parser.add_argument("--max-training", type=int, default=3)
    parser.add_argument("--max-bucket", type=int, default=5)
    parser.add_argument("--csv", type=Path, help="optional per-bucket row export")
    args = parser.parse_args()

    blocks = parse_log(args.log)
    trainings = range(1, args.max_training + 1)
    buckets = range(1, args.max_bucket + 1)
    rows = build_rows(
        blocks,
        pi_threshold=args.pi_threshold,
        trainings=trainings,
        buckets=buckets,
    )
    print_summary(blocks, rows, trainings=trainings, buckets=buckets)
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
