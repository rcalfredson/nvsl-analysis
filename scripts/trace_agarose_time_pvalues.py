#!/usr/bin/env python3
"""Trace agarose-time p-values across frame and statistical policies."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.agarose_time_summary import (  # noqa: E402
    DEFAULT_POST_COL,
    DEFAULT_PRE_COL,
    DEFAULT_SECTION,
    paired_test,
    parse_group,
    reduction_anova_and_posthoc,
)
from src.analysis.posthoc_tests import (  # noqa: E402
    games_howell_all_pairs,
    holm_adjust,
    welch_anova,
    welch_t_pair,
)
from src.utils.parsers import parse_labeled_path_arg  # noqa: E402


FIELDS = [
    "ratio_calculation",
    "statistical_method",
    "posthoc_scope",
    "groups_in_adjustment_family",
    "comparisons_in_adjustment_family",
    "comparisons_reported",
    "control_group",
    "comparison_group",
    "n_control",
    "n_comparison",
    "mean_reduction_control",
    "mean_reduction_comparison",
    "mean_difference_control_minus_comparison",
    "statistic_name",
    "statistic",
    "df",
    "p_value_raw",
    "p_value_reported",
    "omnibus_f",
    "omnibus_df_between",
    "omnibus_df_within",
    "omnibus_p_value",
    "input_files",
]

DETAIL_FIELDS = [
    "ratio_calculation",
    "test_method",
    "family_definition",
    "groups_in_family",
    "comparisons_in_family",
    "target_p_rank",
    "bonferroni_multiplier",
    "control_group",
    "comparison_group",
    "n_control",
    "n_comparison",
    "mean_reduction_control",
    "mean_reduction_comparison",
    "mean_difference_control_minus_comparison",
    "statistic_name",
    "statistic",
    "df",
    "raw_welch_p_value",
    "p_value_reported",
    "effective_p_multiplier",
    "omnibus_f",
    "omnibus_df_between",
    "omnibus_df_within",
    "omnibus_p_value",
    "input_files",
]


def _parse_group_arg(value: str) -> tuple[str, str]:
    return parse_labeled_path_arg(value, option_name="group input")


def _group_map(
    specs: list[tuple[str, str]], *, option_name: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for label, path in specs:
        if label in result:
            raise ValueError(f"Duplicate group {label!r} supplied to {option_name}")
        result[label] = path
    return result


def _matching_posthoc(posthoc, group_a: str, group_b: str):
    target = frozenset((group_a, group_b))
    matches = [
        result
        for result in posthoc
        if frozenset((result.group_a, result.group_b)) == target
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one post-hoc result for {group_a!r} versus {group_b!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _target_sign(result, control_group: str) -> float:
    return 1.0 if result.group_a == control_group else -1.0


def _effective_multiplier(raw_p: float, reported_p: float) -> float:
    if not np.isfinite(raw_p) or raw_p <= 0 or not np.isfinite(reported_p):
        return np.nan
    return float(reported_p / raw_p)


def _detail_row(
    *,
    ratio_policy: str,
    test_method: str,
    family_definition: str,
    groups_in_family: int,
    comparisons_in_family: int,
    target_p_rank: float,
    bonferroni_multiplier: float,
    control_test,
    comparison_test,
    result,
    raw_welch_p: float,
    reported_p: float,
    statistic_name: str,
    group_paths: dict[str, str],
    omnibus=None,
) -> dict[str, object]:
    sign = _target_sign(result, control_test.group)
    return {
        "ratio_calculation": ratio_policy,
        "test_method": test_method,
        "family_definition": family_definition,
        "groups_in_family": groups_in_family,
        "comparisons_in_family": comparisons_in_family,
        "target_p_rank": target_p_rank,
        "bonferroni_multiplier": bonferroni_multiplier,
        "control_group": control_test.group,
        "comparison_group": comparison_test.group,
        "n_control": control_test.n_pairs,
        "n_comparison": comparison_test.n_pairs,
        "mean_reduction_control": control_test.mean_difference,
        "mean_reduction_comparison": comparison_test.mean_difference,
        "mean_difference_control_minus_comparison": sign
        * result.mean_difference_a_minus_b,
        "statistic_name": statistic_name,
        "statistic": result.statistic,
        "df": result.df,
        "raw_welch_p_value": raw_welch_p,
        "p_value_reported": reported_p,
        "effective_p_multiplier": _effective_multiplier(raw_welch_p, reported_p),
        "omnibus_f": omnibus.statistic if omnibus is not None else np.nan,
        "omnibus_df_between": omnibus.df_numerator
        if omnibus is not None
        else np.nan,
        "omnibus_df_within": omnibus.df_denominator
        if omnibus is not None
        else np.nan,
        "omnibus_p_value": omnibus.p_value if omnibus is not None else np.nan,
        "input_files": json.dumps(group_paths, sort_keys=True),
    }


def build_multiple_comparison_rows(
    inputs_by_policy: dict[str, dict[str, str]],
    *,
    control_group: str,
    comparison_group: str,
    section: str,
    pre_col: str,
    post_col: str,
) -> list[dict[str, object]]:
    """Expose target-pair results under each defensible comparison family."""
    rows: list[dict[str, object]] = []
    for ratio_policy, group_paths in inputs_by_policy.items():
        labels = list(group_paths)
        if control_group not in group_paths or comparison_group not in group_paths:
            raise ValueError(
                f"{ratio_policy} inputs must contain control {control_group!r} and "
                f"comparison {comparison_group!r}; available groups: {labels}"
            )
        parsed = [
            parse_group(
                label,
                path,
                section=section,
                numeric_cols=[pre_col, post_col],
            )
            for label, path in group_paths.items()
        ]
        tests = [
            paired_test(group, pre_col=pre_col, post_col=post_col)
            for group in parsed
        ]
        by_name = {test.group: test for test in tests}
        control_test = by_name[control_group]
        comparison_test = by_name[comparison_group]
        samples = [test.reductions for test in tests]

        control_idx = labels.index(control_group)
        target_idx = labels.index(comparison_group)
        target_pair = tuple(sorted((control_idx, target_idx)))
        control_pairs = [
            tuple(sorted((control_idx, idx)))
            for idx in range(len(tests))
            if idx != control_idx
        ]
        all_pairs = list(combinations(range(len(tests)), 2))
        target_result = welch_t_pair(
            control_group,
            control_test.reductions,
            comparison_group,
            comparison_test.reductions,
        )
        raw_target_p = target_result.p_value

        rows.append(
            _detail_row(
                ratio_policy=ratio_policy,
                test_method="welch_t_unadjusted",
                family_definition="target_pair_only",
                groups_in_family=2,
                comparisons_in_family=1,
                target_p_rank=1,
                bonferroni_multiplier=1,
                control_test=control_test,
                comparison_test=comparison_test,
                result=target_result,
                raw_welch_p=raw_target_p,
                reported_p=raw_target_p,
                statistic_name="t",
                group_paths=group_paths,
            )
        )

        for family_definition, pairs in (
            ("planned_control_contrasts", control_pairs),
            ("all_pairwise_comparisons", all_pairs),
        ):
            family_results = [
                welch_t_pair(
                    labels[i], tests[i].reductions, labels[j], tests[j].reductions
                )
                for i, j in pairs
            ]
            target_position = pairs.index(target_pair)
            target_family_result = family_results[target_position]
            raw_p_values = [result.p_value for result in family_results]
            rank = int(np.argsort(np.argsort(raw_p_values))[target_position]) + 1
            bonferroni_p = min(1.0, raw_target_p * len(pairs))
            rows.append(
                _detail_row(
                    ratio_policy=ratio_policy,
                    test_method="welch_t_bonferroni",
                    family_definition=family_definition,
                    groups_in_family=len(tests),
                    comparisons_in_family=len(pairs),
                    target_p_rank=rank,
                    bonferroni_multiplier=len(pairs),
                    control_test=control_test,
                    comparison_test=comparison_test,
                    result=target_family_result,
                    raw_welch_p=raw_target_p,
                    reported_p=bonferroni_p,
                    statistic_name="t",
                    group_paths=group_paths,
                )
            )
            holm_p = holm_adjust(raw_p_values)[target_position]
            rows.append(
                _detail_row(
                    ratio_policy=ratio_policy,
                    test_method="welch_t_holm_bonferroni",
                    family_definition=family_definition,
                    groups_in_family=len(tests),
                    comparisons_in_family=len(pairs),
                    target_p_rank=rank,
                    bonferroni_multiplier=np.nan,
                    control_test=control_test,
                    comparison_test=comparison_test,
                    result=target_family_result,
                    raw_welch_p=raw_target_p,
                    reported_p=holm_p,
                    statistic_name="t",
                    group_paths=group_paths,
                )
            )

        for family_definition, gh_tests in (
            ("target_groups_only", [control_test, comparison_test]),
            ("all_groups", tests),
        ):
            gh_samples = [test.reductions for test in gh_tests]
            gh_labels = [test.group for test in gh_tests]
            gh_results = games_howell_all_pairs(
                gh_samples,
                group_names=gh_labels,
            )
            gh_target = _matching_posthoc(
                gh_results, control_group, comparison_group
            )
            omnibus = welch_anova(gh_samples, group_names=gh_labels)
            rows.append(
                _detail_row(
                    ratio_policy=ratio_policy,
                    test_method="games_howell",
                    family_definition=family_definition,
                    groups_in_family=len(gh_tests),
                    comparisons_in_family=len(gh_results),
                    target_p_rank=np.nan,
                    bonferroni_multiplier=np.nan,
                    control_test=control_test,
                    comparison_test=comparison_test,
                    result=gh_target,
                    raw_welch_p=raw_target_p,
                    reported_p=gh_target.p_value,
                    statistic_name="q",
                    group_paths=group_paths,
                    omnibus=omnibus,
                )
            )
    return rows


def build_trace_rows(
    inputs_by_policy: dict[str, dict[str, str]],
    *,
    control_group: str,
    comparison_group: str,
    section: str,
    pre_col: str,
    post_col: str,
    posthoc_scope: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ratio_policy, group_paths in inputs_by_policy.items():
        labels = list(group_paths)
        if control_group not in group_paths or comparison_group not in group_paths:
            raise ValueError(
                f"{ratio_policy} inputs must contain control {control_group!r} and "
                f"comparison {comparison_group!r}; available groups: {labels}"
            )
        parsed = [
            parse_group(
                label,
                path,
                section=section,
                numeric_cols=[pre_col, post_col],
            )
            for label, path in group_paths.items()
        ]
        tests = [
            paired_test(group, pre_col=pre_col, post_col=post_col)
            for group in parsed
        ]

        for method, method_label in (
            ("holm-welch", "welch_t_holm_bonferroni"),
            ("games-howell", "welch_anova_games_howell"),
        ):
            anova, posthoc = reduction_anova_and_posthoc(
                tests,
                control_group=control_group,
                posthoc_scope=posthoc_scope,
                posthoc_method=method,
            )
            result = _matching_posthoc(posthoc, control_group, comparison_group)
            sign = 1.0 if result.group_a == control_group else -1.0
            control_test = next(test for test in tests if test.group == control_group)
            comparison_test = next(
                test for test in tests if test.group == comparison_group
            )
            use_anova = method == "games-howell"
            adjustment_family_size = (
                len(tests) * (len(tests) - 1) // 2
                if use_anova
                else len(posthoc)
            )
            rows.append(
                {
                    "ratio_calculation": ratio_policy,
                    "statistical_method": method_label,
                    "posthoc_scope": posthoc_scope,
                    "groups_in_adjustment_family": len(tests),
                    "comparisons_in_adjustment_family": adjustment_family_size,
                    "comparisons_reported": len(posthoc),
                    "control_group": control_group,
                    "comparison_group": comparison_group,
                    "n_control": control_test.n_pairs,
                    "n_comparison": comparison_test.n_pairs,
                    "mean_reduction_control": control_test.mean_difference,
                    "mean_reduction_comparison": comparison_test.mean_difference,
                    "mean_difference_control_minus_comparison": sign
                    * result.mean_difference_a_minus_b,
                    "statistic_name": "q" if use_anova else "t",
                    "statistic": result.t_stat,
                    "df": result.df,
                    "p_value_raw": result.p_value,
                    "p_value_reported": result.p_value_holm,
                    "omnibus_f": anova.f_stat if use_anova and anova else np.nan,
                    "omnibus_df_between": anova.df_between
                    if use_anova and anova
                    else np.nan,
                    "omnibus_df_within": anova.df_within
                    if use_anova and anova
                    else np.nan,
                    "omnibus_p_value": anova.p_value
                    if use_anova and anova
                    else np.nan,
                    "input_files": json.dumps(group_paths, sort_keys=True),
                }
            )
    return rows


def write_trace_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _default_details_path(summary_path: str | Path) -> Path:
    path = Path(summary_path)
    return path.with_name(f"{path.stem}_multiple_comparisons{path.suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Trace p-values across agarose-area avoidance frame policies and "
            "Holm-Welch versus Games-Howell tests."
        )
    )
    parser.add_argument(
        "--legacy-group",
        action="append",
        type=_parse_group_arg,
        required=True,
        metavar="LABEL=CSV_PATH",
        help="Group computed with --agarose-time-lost-frame-policy legacy.",
    )
    parser.add_argument(
        "--corrected-group",
        action="append",
        type=_parse_group_arg,
        required=True,
        metavar="LABEL=CSV_PATH",
        help="Group computed with the corrected lost-frame policy.",
    )
    parser.add_argument(
        "--interpolated-inclusive-group",
        action="append",
        type=_parse_group_arg,
        default=None,
        metavar="LABEL=CSV_PATH",
        help=(
            "Optional group computed with interpolated frames included in both "
            "numerator and denominator. Repeat once per group."
        ),
    )
    parser.add_argument("--control-group", required=True)
    parser.add_argument("--comparison-group", required=True)
    parser.add_argument(
        "--posthoc-scope",
        choices=("control", "all"),
        default="control",
        help=(
            "For Holm, choose the adjusted comparison family. For Games-Howell, "
            "choose the pairs reported; its calculation still includes all supplied "
            "groups. Default: control."
        ),
    )
    parser.add_argument("--section", default=DEFAULT_SECTION)
    parser.add_argument("--pre-col", default=DEFAULT_PRE_COL)
    parser.add_argument("--post-col", default=DEFAULT_POST_COL)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--details-out",
        default=None,
        help=(
            "Detailed multiple-comparisons audit CSV. Defaults to OUT with "
            "'_multiple_comparisons' appended."
        ),
    )
    args = parser.parse_args(argv)

    legacy = _group_map(args.legacy_group, option_name="--legacy-group")
    corrected = _group_map(args.corrected_group, option_name="--corrected-group")
    if list(legacy) != list(corrected):
        raise ValueError(
            "Legacy and corrected inputs must contain the same groups in the same "
            f"order; got {list(legacy)} and {list(corrected)}"
        )
    inputs_by_policy = {"legacy": legacy, "corrected": corrected}
    if args.interpolated_inclusive_group:
        inclusive = _group_map(
            args.interpolated_inclusive_group,
            option_name="--interpolated-inclusive-group",
        )
        if list(inclusive) != list(corrected):
            raise ValueError(
                "Interpolated-inclusive and corrected inputs must contain the "
                "same groups in the same order; got "
                f"{list(inclusive)} and {list(corrected)}"
            )
        inputs_by_policy["interpolated-inclusive"] = inclusive

    rows = build_trace_rows(
        inputs_by_policy,
        control_group=args.control_group,
        comparison_group=args.comparison_group,
        section=args.section,
        pre_col=args.pre_col,
        post_col=args.post_col,
        posthoc_scope=args.posthoc_scope,
    )
    write_trace_csv(args.out, rows)
    detail_rows = build_multiple_comparison_rows(
        inputs_by_policy,
        control_group=args.control_group,
        comparison_group=args.comparison_group,
        section=args.section,
        pre_col=args.pre_col,
        post_col=args.post_col,
    )
    details_out = args.details_out or _default_details_path(args.out)
    out_path = Path(details_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(detail_rows)

    print(
        f"Agarose-time p-value trace: "
        f"{args.control_group} vs {args.comparison_group}"
    )
    for row in rows:
        print(
            f"  {row['ratio_calculation']}, {row['statistical_method']}: "
            f"p={float(row['p_value_reported']):.8g}"
        )
    print(f"Wrote: {args.out}")
    print(f"Wrote multiple-comparisons audit: {details_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
