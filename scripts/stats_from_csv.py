#!/usr/bin/env python3
"""
Compute mean and 95% CI for per-row sums of specific columns in a CSV section.

This script looks for a marker row whose first column contains:
    "number __calculated__ rewards by sync bucket:"

Immediately after that marker, it expects a CSV "table" that continues until the
next empty line. In that table, it sums (per row) columns from a named start
column through a fixed number of columns (default: 6 columns total).

- NaNs are ignored in the row-wise sum.
- Rows that are entirely NaN across the selected columns contribute 0.0
  (this matches pandas' sum(skipna=True) behavior and the earlier chat results).

Outputs per-file:
- n rows
- mean
- 95% CI (t-based)
- delta (half-width of CI)
Optionally also outputs a pooled summary across all input files.

Usage examples:
    python rewards_ci.py learning_stats_intact_ctrlKir_lgC.csv learning_stats_intact_ctrlKir_htl.csv

    python rewards_ci.py *.csv --start-col "training 2 exp b1" --ncols 6

    python rewards_ci.py file.csv --no-pooled
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


MARKER_DEFAULT = "number __calculated__ rewards by sync bucket:"
START_COL_DEFAULT = "training 2 exp b1"
NCOLS_DEFAULT = 6


@dataclass(frozen=True)
class Summary:
    path: str
    n: int
    mean: float
    ci_low: float
    ci_high: float
    delta: float


def _read_section_table_lines(path: Path, marker: str) -> list[str]:
    """
    Return list of lines belonging to the section table following `marker`,
    stopping at the first empty line.
    """
    lines = path.read_text().splitlines(keepends=True)

    start_idx = None
    for i, line in enumerate(lines):
        # marker is said to be in first column, but we accept "contains" for robustness
        if marker in line:
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError(f"Marker not found in {path}: {marker!r}")

    table_lines: list[str] = []
    for line in lines[start_idx:]:
        if line.strip() == "":
            break
        table_lines.append(line)

    if not table_lines:
        raise ValueError(f"Found marker but no table lines after it in {path}")

    return table_lines


def extract_row_sums(
    path: Path,
    *,
    marker: str = MARKER_DEFAULT,
    start_col: str = START_COL_DEFAULT,
    ncols: int = NCOLS_DEFAULT,
) -> np.ndarray:
    """
    Extract per-row sums over [start_col, start_col + ncols).

    NaNs are ignored in the sum; rows with all-NaN in the selected slice contribute 0.0.
    """
    table_lines = _read_section_table_lines(path, marker)
    df = pd.read_csv(StringIO("".join(table_lines)))

    if start_col not in df.columns:
        raise ValueError(
            f"Start column {start_col!r} not found in table columns for {path}.\n"
            f"Columns are: {list(df.columns)}"
        )

    col_start = df.columns.get_loc(start_col)
    cols = df.columns[col_start : col_start + ncols]

    if len(cols) != ncols:
        raise ValueError(
            f"Requested {ncols} columns starting at {start_col!r}, "
            f"but only found {len(cols)} columns available from that position in {path}."
        )

    data = df[cols].apply(pd.to_numeric, errors="coerce")
    row_sums = data.sum(axis=1, skipna=True).astype(float).to_numpy()
    return row_sums


def mean_ci_95(x: np.ndarray) -> tuple[float, float, float]:
    """
    Return (mean, ci_low, ci_high) using a t-based 95% CI.
    """
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    if n == 1:
        return (mean, np.nan, np.nan)

    s = float(np.std(x, ddof=1))
    sem = s / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    half = tcrit * sem
    return (mean, mean - half, mean + half)


def summarize_file(
    path: Path,
    *,
    marker: str,
    start_col: str,
    ncols: int,
) -> tuple[Summary, np.ndarray]:
    row_sums = extract_row_sums(path, marker=marker, start_col=start_col, ncols=ncols)
    mean, lo, hi = mean_ci_95(row_sums)
    delta = float(hi - mean) if np.isfinite(hi) and np.isfinite(mean) else np.nan
    return (
        Summary(
            path=str(path),
            n=int(row_sums.size),
            mean=float(mean),
            ci_low=float(lo),
            ci_high=float(hi),
            delta=float(delta),
        ),
        row_sums,
    )


def _fmt(x: float, nd: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compute per-row sums over selected columns and report mean + 95% CI."
    )
    p.add_argument("csvs", nargs="+", help="Input CSV files.")
    p.add_argument("--marker", default=MARKER_DEFAULT, help="Marker row text to find.")
    p.add_argument("--start-col", default=START_COL_DEFAULT, help="Start column name.")
    p.add_argument(
        "--ncols",
        type=int,
        default=NCOLS_DEFAULT,
        help="Number of consecutive columns to sum (default: 6 for b1..b6).",
    )
    p.add_argument(
        "--no-pooled",
        action="store_true",
        help="Do not print pooled summary across all input files.",
    )
    args = p.parse_args(argv)

    all_sums: list[np.ndarray] = []
    summaries: list[Summary] = []

    for f in args.csvs:
        path = Path(f)
        summary, row_sums = summarize_file(
            path, marker=args.marker, start_col=args.start_col, ncols=args.ncols
        )
        summaries.append(summary)
        all_sums.append(row_sums)

    # Per-file output
    print("Per-file summaries")
    print("-" * 80)
    for s in summaries:
        print(f"{s.path}")
        print(f"  n      : {s.n}")
        print(f"  mean   : {_fmt(s.mean, 4)}")
        print(f"  95% CI : [{_fmt(s.ci_low, 4)}, {_fmt(s.ci_high, 4)}]")
        print(f"  delta  : {_fmt(s.delta, 4)}")
        print()

    # Pooled output
    if not args.no_pooled:
        pooled = np.concatenate(all_sums) if all_sums else np.array([], float)
        mean, lo, hi = mean_ci_95(pooled)
        delta = hi - mean if np.isfinite(hi) and np.isfinite(mean) else np.nan
        print("Pooled summary (all rows across all files)")
        print("-" * 80)
        print(f"  n      : {pooled.size}")
        print(f"  mean   : {_fmt(mean, 4)}")
        print(f"  95% CI : [{_fmt(lo, 4)}, {_fmt(hi, 4)}]")
        print(f"  delta  : {_fmt(delta, 4)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
