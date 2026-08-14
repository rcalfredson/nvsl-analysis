import numpy as np
import pytest
from scipy import stats

from src.analysis.posthoc_tests import games_howell_all_pairs, welch_anova
from src.plotting.stats_bars import (
    StatAnnotConfig,
    anova_and_posthoc,
    format_sig_label,
)


def test_format_sig_label_can_append_p_value():
    assert format_sig_label(0.00321, include_p_value=True) == "**\np=0.00321"


def test_format_sig_label_uses_mathtext_exponent_for_small_p_value():
    assert format_sig_label(1.234e-5, include_p_value=True) == (
        "****\n" + r"p=$1.23 \times 10^{-5}$"
    )


def test_format_sig_label_preserves_existing_star_only_default():
    assert format_sig_label(0.00321) == "**"


def test_format_sig_label_does_not_append_to_non_significant_or_nan_labels():
    assert format_sig_label(0.9, include_p_value=True) == "ns"
    assert format_sig_label(np.nan, include_p_value=True) == ""


def test_welch_anova_matches_statsmodels_reference_values():
    samples = [
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([2.0, 4.0, 8.0, 16.0, 32.0]),
        np.asarray([10.0, 11.0, 12.0]),
    ]

    result = welch_anova(samples, group_names=["A", "B", "C"])

    assert result.groups == ("A", "B", "C")
    assert result.ns == (4, 5, 3)
    assert result.df_numerator == 2.0
    assert result.statistic == pytest.approx(43.6872888781)
    assert result.df_denominator == pytest.approx(5.91125399665)
    assert result.p_value == pytest.approx(0.000287578684)
    assert result.test == "Welch's one-way ANOVA"


def test_three_group_bar_stats_use_welch_anova_and_games_howell():
    samples = [
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([2.0, 4.0, 8.0, 16.0, 32.0]),
        np.asarray([10.0, 11.0, 12.0]),
    ]
    names = ["A", "B", "C"]

    p_omnibus, pairs = anova_and_posthoc(
        samples,
        cfg=StatAnnotConfig(min_n_per_group=3),
        group_names=names,
    )

    assert p_omnibus == pytest.approx(
        welch_anova(samples, group_names=names, min_n_per_group=3).p_value
    )
    assert set(pairs) == {("A", "B"), ("A", "C"), ("B", "C")}
    expected = {
        (result.group_a, result.group_b): result.p_value
        for result in games_howell_all_pairs(samples, group_names=names)
    }
    assert pairs == pytest.approx(expected)


def test_two_group_bar_stats_use_unadjusted_welch_t_test():
    a = np.asarray([1.0, 2.0, 3.0, 4.0])
    b = np.asarray([4.0, 7.0, 9.0, 15.0, 20.0])

    p_test, pairs = anova_and_posthoc(
        [a, b],
        cfg=StatAnnotConfig(min_n_per_group=3),
        group_names=["A", "B"],
    )

    expected = stats.ttest_ind(a, b, equal_var=False).pvalue
    assert p_test == pytest.approx(expected)
    assert pairs == {("A", "B"): pytest.approx(expected)}
