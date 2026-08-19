import math
from pathlib import Path

import pytest

from scripts.trace_agarose_time_pvalues import (
    build_multiple_comparison_rows,
    build_trace_rows,
)
from src.analysis.agarose_time_summary import (
    DEFAULT_POST_COL,
    DEFAULT_PRE_COL,
    DEFAULT_SECTION,
)


def _write_learning_stats(path: Path, reductions: list[float]) -> None:
    rows = [
        f"video{i},1,{20 + reduction},{20}"
        for i, reduction in enumerate(reductions, start=1)
    ]
    path.write_text(
        "\n".join(
            [
                "# command: fake analyze.py",
                "",
                DEFAULT_SECTION,
                f"video,fly,{DEFAULT_PRE_COL},{DEFAULT_POST_COL}",
                *rows,
                "",
            ]
        )
    )


def test_build_trace_rows_crosses_ratio_and_statistical_policies(tmp_path):
    reductions = {
        "legacy": {
            "Control": [1, 2, 4, 5],
            "upd3 KO": [3, 7, 8, 10],
            "upd2+3 KO": [2, 3, 8, 12],
        },
        "corrected": {
            "Control": [1, 2, 3, 4],
            "upd3 KO": [5, 7, 9, 11],
            "upd2+3 KO": [2, 4, 8, 13],
        },
    }
    inputs = {}
    for policy, groups in reductions.items():
        inputs[policy] = {}
        for label, values in groups.items():
            path = tmp_path / f"{policy}_{label.replace(' ', '_')}.csv"
            _write_learning_stats(path, values)
            inputs[policy][label] = str(path)

    rows = build_trace_rows(
        inputs,
        control_group="Control",
        comparison_group="upd3 KO",
        section=DEFAULT_SECTION,
        pre_col=DEFAULT_PRE_COL,
        post_col=DEFAULT_POST_COL,
        posthoc_scope="control",
    )

    assert [
        (row["ratio_calculation"], row["statistical_method"]) for row in rows
    ] == [
        ("legacy", "welch_t_holm_bonferroni"),
        ("legacy", "welch_anova_games_howell"),
        ("corrected", "welch_t_holm_bonferroni"),
        ("corrected", "welch_anova_games_howell"),
    ]
    holm_rows = [row for row in rows if "holm" in row["statistical_method"]]
    games_howell_rows = [
        row for row in rows if "games_howell" in row["statistical_method"]
    ]
    assert all(row["comparisons_in_adjustment_family"] == 2 for row in holm_rows)
    assert all(
        row["comparisons_in_adjustment_family"] == 3
        for row in games_howell_rows
    )
    assert all(row["comparisons_reported"] == 2 for row in rows)
    assert all(row["n_control"] == 4 for row in rows)
    assert all(row["n_comparison"] == 4 for row in rows)
    assert all(float(row["p_value_reported"]) <= 1 for row in rows)
    assert rows[0]["omnibus_p_value"] != rows[0]["omnibus_p_value"]
    assert math.isfinite(rows[1]["omnibus_p_value"])
    assert rows[2]["mean_reduction_control"] == pytest.approx(2.5)
    assert rows[2]["mean_reduction_comparison"] == pytest.approx(8.0)


def test_build_trace_rows_requires_requested_pair_in_both_policies(tmp_path):
    path = tmp_path / "control.csv"
    _write_learning_stats(path, [1, 2, 3])

    with pytest.raises(ValueError, match="must contain control"):
        build_trace_rows(
            {"legacy": {"Control": str(path)}},
            control_group="Control",
            comparison_group="upd3 KO",
            section=DEFAULT_SECTION,
            pre_col=DEFAULT_PRE_COL,
            post_col=DEFAULT_POST_COL,
            posthoc_scope="control",
        )


def test_multiple_comparison_rows_expose_raw_and_family_adjusted_p_values(tmp_path):
    reductions = {
        "Control": [1, 2, 4, 5, 7],
        "upd3 KO": [3, 7, 8, 10, 12],
        "upd2+3 KO": [2, 3, 8, 12, 18],
    }
    inputs = {"corrected": {}}
    for label, values in reductions.items():
        path = tmp_path / f"{label.replace(' ', '_')}.csv"
        _write_learning_stats(path, values)
        inputs["corrected"][label] = str(path)

    rows = build_multiple_comparison_rows(
        inputs,
        control_group="Control",
        comparison_group="upd3 KO",
        section=DEFAULT_SECTION,
        pre_col=DEFAULT_PRE_COL,
        post_col=DEFAULT_POST_COL,
    )

    assert len(rows) == 7
    by_key = {
        (row["test_method"], row["family_definition"]): row for row in rows
    }
    raw = by_key[("welch_t_unadjusted", "target_pair_only")][
        "p_value_reported"
    ]
    bonf_control = by_key[
        ("welch_t_bonferroni", "planned_control_contrasts")
    ]
    bonf_all = by_key[
        ("welch_t_bonferroni", "all_pairwise_comparisons")
    ]
    assert bonf_control["comparisons_in_family"] == 2
    assert bonf_control["bonferroni_multiplier"] == 2
    assert bonf_control["raw_welch_p_value"] == pytest.approx(raw)
    assert bonf_control["p_value_reported"] == pytest.approx(min(1, 2 * raw))
    assert bonf_all["comparisons_in_family"] == 3
    assert bonf_all["bonferroni_multiplier"] == 3
    assert bonf_all["p_value_reported"] == pytest.approx(min(1, 3 * raw))

    holm_control = by_key[
        ("welch_t_holm_bonferroni", "planned_control_contrasts")
    ]
    holm_all = by_key[
        ("welch_t_holm_bonferroni", "all_pairwise_comparisons")
    ]
    assert holm_control["comparisons_in_family"] == 2
    assert holm_all["comparisons_in_family"] == 3
    assert holm_control["raw_welch_p_value"] == pytest.approx(raw)
    assert holm_all["raw_welch_p_value"] == pytest.approx(raw)

    gh_two = by_key[("games_howell", "target_groups_only")]
    gh_three = by_key[("games_howell", "all_groups")]
    assert gh_two["groups_in_family"] == 2
    assert gh_two["comparisons_in_family"] == 1
    assert gh_two["p_value_reported"] == pytest.approx(raw)
    assert gh_three["groups_in_family"] == 3
    assert gh_three["comparisons_in_family"] == 3
    assert gh_three["p_value_reported"] != pytest.approx(raw)
