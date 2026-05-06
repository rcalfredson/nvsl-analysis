from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src.analysis.agarose_time_summary import (
    DEFAULT_SECTION,
    describe_values,
    paired_test,
    parse_group,
    read_learning_stats_section,
    reduction_anova_and_posthoc,
    welch_reduction_test,
)


PRE = "% time spent on agarose- pre last 10m (exp)"
T1 = "% time spent on agarose- T1 start (exp)"
T3_END = "% time spent on agarose- T3 end (exp)"
POST = "% time spent on agarose- T3 post last 10m (exp)"


def _write_learning_stats(path: Path, rows: list[tuple[object, ...]]) -> None:
    path.write_text(
        "\n".join(
            [
                "# command: fake analyze.py",
                "",
                DEFAULT_SECTION,
                f"video,fly,{PRE},{T1},{T3_END},{POST}",
                *(",".join(map(str, row)) for row in rows),
                "",
                "other section:",
                "video,fly,x",
                "v1,1,1",
                "",
            ]
        )
    )


def test_read_learning_stats_section_by_exact_title(tmp_path):
    csv_path = tmp_path / "learning_stats.csv"
    _write_learning_stats(csv_path, [("v1", 1, 10, 7, 5, 4), ("v2", 2, "nan", 8, 6, 5)])

    df = read_learning_stats_section(csv_path, DEFAULT_SECTION)

    assert list(df.columns) == ["video", "fly", PRE, T1, T3_END, POST]
    assert len(df) == 2


def test_descriptive_and_paired_stats_drop_missing_pairs(tmp_path):
    csv_path = tmp_path / "learning_stats.csv"
    _write_learning_stats(
        csv_path,
        [
            ("v1", 1, 10, 7, 5, 4),
            ("v2", 2, 12, 8, 6, 6),
            ("v3", 3, "nan", 9, 7, 5),
            ("v4", 4, 18, 10, 8, "nan"),
        ],
    )
    group = parse_group("A", csv_path, numeric_cols=[PRE, T1, T3_END, POST])

    desc = describe_values(group.data[PRE])
    assert desc.n == 3
    assert desc.mean == pytest.approx(np.mean([10, 12, 18]))

    paired = paired_test(group, pre_col=PRE, post_col=POST)
    expected_reductions = np.asarray([6.0, 6.0])
    assert paired.n_pairs == 2
    assert paired.mean_difference == pytest.approx(expected_reductions.mean())
    assert paired.df == 1
    assert paired.t_stat == pytest.approx(stats.ttest_rel([10, 12], [4, 6]).statistic)


def test_welch_reduction_test_uses_satterthwaite_df(tmp_path):
    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.csv"
    _write_learning_stats(a_path, [("a1", 1, 10, 1, 1, 5), ("a2", 2, 14, 1, 1, 7), ("a3", 3, 20, 1, 1, 8)])
    _write_learning_stats(b_path, [("b1", 1, 11, 1, 1, 8), ("b2", 2, 17, 1, 1, 10), ("b3", 3, 19, 1, 1, 12)])
    a = paired_test(parse_group("A", a_path, numeric_cols=[PRE, POST]), pre_col=PRE, post_col=POST)
    b = paired_test(parse_group("B", b_path, numeric_cols=[PRE, POST]), pre_col=PRE, post_col=POST)

    result = welch_reduction_test(a, b)
    xa = np.asarray([5.0, 7.0, 12.0])
    xb = np.asarray([3.0, 7.0, 7.0])
    va = np.var(xa, ddof=1)
    vb = np.var(xb, ddof=1)
    se2 = va / len(xa) + vb / len(xb)
    expected_df = se2**2 / ((va / len(xa)) ** 2 / (len(xa) - 1) + (vb / len(xb)) ** 2 / (len(xb) - 1))

    assert result.n_a == 3
    assert result.n_b == 3
    assert result.mean_difference_a_minus_b == pytest.approx(xa.mean() - xb.mean())
    assert result.df == pytest.approx(expected_df)
    assert result.t_stat == pytest.approx(stats.ttest_ind(xa, xb, equal_var=False).statistic)


def test_reduction_anova_control_posthocs_holm_correct_only_control_family(tmp_path):
    paths = [tmp_path / f"{label}.csv" for label in ["control", "exp1", "exp2"]]
    rows_by_group = [
        [("c1", 1, 12, 1, 1, 8), ("c2", 2, 13, 1, 1, 8), ("c3", 3, 14, 1, 1, 8)],
        [("e1", 1, 18, 1, 1, 6), ("e2", 2, 19, 1, 1, 6), ("e3", 3, 20, 1, 1, 6)],
        [("f1", 1, 20, 1, 1, 5), ("f2", 2, 21, 1, 1, 5), ("f3", 3, 22, 1, 1, 5)],
    ]
    for path, rows in zip(paths, rows_by_group):
        _write_learning_stats(path, rows)

    tests = [
        paired_test(
            parse_group(label, path, numeric_cols=[PRE, POST]),
            pre_col=PRE,
            post_col=POST,
        )
        for label, path in zip(["Control", "Exp1", "Exp2"], paths)
    ]

    anova, posthoc = reduction_anova_and_posthoc(tests, posthoc_scope="control")

    assert anova is not None
    assert anova.df_between == 2
    assert anova.df_within == 6
    assert [(p.group_a, p.group_b) for p in posthoc] == [
        ("Control", "Exp1"),
        ("Control", "Exp2"),
    ]
    raw_pvals = [p.p_value for p in posthoc]
    expected_holm = sorted(raw_pvals)
    expected_holm = [
        min(1.0, expected_holm[0] * 2),
        min(1.0, max(expected_holm[0] * 2, expected_holm[1])),
    ]
    assert sorted(p.p_value_holm for p in posthoc) == pytest.approx(expected_holm)


def test_reference_learning_stats_fixture_can_be_summarized_with_older_post_column():
    csv_path = Path("test/reference/csv/learning_stats_on_agarose.csv")
    group = parse_group("fixture", csv_path, numeric_cols=[PRE, T1, T3_END])
    paired = paired_test(group, pre_col=PRE, post_col=T3_END)

    assert len(group.data) == 15
    assert paired.n_pairs == 15
    assert paired.mean_pre > paired.mean_post
