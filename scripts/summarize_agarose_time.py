#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.agarose_time_summary import (
    DEFAULT_POST_COL,
    DEFAULT_PRE_COL,
    DEFAULT_SECTION,
    DEFAULT_SUMMARY_COLS,
    between_group_rows,
    descriptive_rows,
    paired_test,
    paired_test_rows,
    parse_group,
    welch_reduction_test,
    write_dict_csv,
    write_wiki_summary_html,
)


SUMMARY_FIELDS = ["group", "column", "n", "mean", "sd", "sem", "ci95_low", "ci95_high"]
PAIRED_FIELDS = [
    "group",
    "pre_col",
    "post_col",
    "n_pairs",
    "mean_pre",
    "mean_post",
    "mean_difference",
    "ci95_low",
    "ci95_high",
    "t_stat",
    "df",
    "p_value",
]
BETWEEN_FIELDS = [
    "group_a",
    "group_b",
    "n_a",
    "n_b",
    "mean_reduction_a",
    "mean_reduction_b",
    "mean_difference_a_minus_b",
    "ci95_low",
    "ci95_high",
    "t_stat",
    "df",
    "p_value",
    "test",
]


def _parse_group_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--group must be formatted as LABEL=CSV_PATH, for example Control=learning_stats.csv"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "--group must include both a non-empty label and a CSV path."
        )
    return label, path


def _fmt(x: float) -> str:
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Summarize agarose percent-time metrics from named sections in learning_stats.csv, "
            "including within-group paired t-tests and a two-group Welch test on reductions."
        )
    )
    p.add_argument(
        "--group",
        action="append",
        type=_parse_group_arg,
        required=True,
        metavar="LABEL=CSV_PATH",
        help="Group label and learning_stats.csv path. Repeat once per group.",
    )
    p.add_argument(
        "--section", default=DEFAULT_SECTION, help="Exact section title to parse."
    )
    p.add_argument(
        "--summary-cols",
        nargs="+",
        default=list(DEFAULT_SUMMARY_COLS),
        help="Numeric columns to summarize.",
    )
    p.add_argument(
        "--pre-col", default=DEFAULT_PRE_COL, help="Column used as paired pre value."
    )
    p.add_argument(
        "--post-col", default=DEFAULT_POST_COL, help="Column used as paired post value."
    )
    p.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix. The script writes *_summary.csv, *_paired_tests.csv, *_between_group_tests.csv, and *_wiki_summary.html.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print parsed section sizes and numeric coercion warnings.",
    )
    args = p.parse_args(argv)

    needed_cols = list(dict.fromkeys([*args.summary_cols, args.pre_col, args.post_col]))
    groups = [
        parse_group(label, path, section=args.section, numeric_cols=needed_cols)
        for label, path in args.group
    ]

    if args.verbose:
        for group in groups:
            print(f"{group.label}: parsed {len(group.data)} rows from {group.path}")
            for col in needed_cols:
                missing = int(group.data[col].isna().sum())
                print(f"  {col}: missing/non-finite values={missing}")
            for col, count in group.coercion_warnings.items():
                print(
                    f"  warning: coerced {count} non-numeric value(s) "
                    f"to NaN in {col!r}"
                )

    summary = descriptive_rows(groups, args.summary_cols)
    paired = [
        paired_test(group, pre_col=args.pre_col, post_col=args.post_col)
        for group in groups
    ]
    paired_rows_out = paired_test_rows(paired)
    between = welch_reduction_test(paired[0], paired[1]) if len(paired) == 2 else None
    between_rows_out = between_group_rows(between)

    prefix = Path(args.out_prefix)
    write_dict_csv(f"{prefix}_summary.csv", summary, SUMMARY_FIELDS)
    write_dict_csv(f"{prefix}_paired_tests.csv", paired_rows_out, PAIRED_FIELDS)
    write_dict_csv(f"{prefix}_between_group_tests.csv", between_rows_out, BETWEEN_FIELDS)
    write_wiki_summary_html(
        f"{prefix}_wiki_summary.html",
        section=args.section,
        summary_rows=summary,
        paired_rows=paired_rows_out,
        between_rows=between_rows_out,
    )

    print("Agarose time summary")
    print(f"  section: {args.section}")
    for row in paired_rows_out:
        print(
            f"  {row['group']}: n={row['n_pairs']}, "
            f"mean reduction={_fmt(row['mean_difference'])}, "
            f"95% CI=[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}], "
            f"t({_fmt(row['df'])})={_fmt(row['t_stat'])}, p={_fmt(row['p_value'])}"
        )
    if between_rows_out:
        row = between_rows_out[0]
        print(
            f"  Welch {row['group_a']} - {row['group_b']}: "
            f"diff={_fmt(row['mean_difference_a_minus_b'])}, "
            f"95% CI=[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}], "
            f"t({_fmt(row['df'])})={_fmt(row['t_stat'])}, p={_fmt(row['p_value'])}"
        )
    else:
        print("  Between-group Welch test skipped; supply exactly two groups to run it.")
    print(f"Wrote outputs with prefix: {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
