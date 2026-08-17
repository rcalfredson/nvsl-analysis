import numpy as np
import pytest

from src.plotting.time_series_auc import (
    auc_or_abc_test_values,
    auc_samples,
    compute_auc_test,
    compute_single_group_auc_test,
    format_auc_annotation,
    format_auc_label,
    format_auc_stars,
    signed_auc_values,
)


def test_auc_samples_match_rowwise_trapezoids():
    samples = auc_samples(
        [
            np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]),
            np.array([[2.0, 2.0, 2.0], [np.nan, np.nan, np.nan]]),
        ]
    )

    assert samples[0] == pytest.approx([2.0, 2.0])
    assert samples[1] == pytest.approx([4.0])


def test_signed_auc_values_preserve_opposite_directions():
    values = signed_auc_values(
        np.array(
            [
                [-0.3, -0.3],
                [0.3, 0.3],
            ]
        )
    )

    assert values == pytest.approx([-0.3, 0.3])


def test_auc_comparisons_stay_signed_while_abc_remains_absolute():
    values = np.array([-0.3, 0.3])

    assert auc_or_abc_test_values(
        values, between_curves=False
    ) == pytest.approx([-0.3, 0.3])
    assert auc_or_abc_test_values(
        values, between_curves=True
    ) == pytest.approx([0.3, 0.3])


def test_compute_auc_test_formats_legacy_style_label():
    result = compute_auc_test(
        [
            np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.0, 0.1]]),
            np.array([[3.0, 3.0, 3.0], [3.0, 3.2, 3.0], [3.2, 3.0, 3.2]]),
        ]
    )

    assert result is not None
    assert result.ns == (3, 3)
    assert result.test == "Welch t-test"
    assert format_auc_label(result, include_p_value=True).startswith(
        "AUC (n = 3,3): **** (p = "
    )


def test_auc_annotation_uses_consistent_sample_size_spacing():
    assert format_auc_annotation(
        "ABC", (50, 21), 0.00321, include_p_value=False
    ) == "ABC (n = 50,21): **"


def test_single_group_auc_test_uses_paired_exp_yoked_differences():
    exp = np.array(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    )
    yoked = np.zeros_like(exp)

    result = compute_single_group_auc_test(exp, reference_series=yoked)

    assert result is not None
    assert result.ns == (3,)
    assert result.test == "paired t-test"


def test_single_group_auc_test_handles_exp_only_or_precomputed_difference():
    difference = np.array(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
    )

    result = compute_single_group_auc_test(difference)

    assert result is not None
    assert result.ns == (3,)
    assert result.test == "one-sample t-test"


def test_single_group_auc_test_gracefully_skips_insufficient_data():
    assert compute_single_group_auc_test(np.array([[0.1, 0.2]])) is None
    assert compute_single_group_auc_test(np.empty((0, 3))) is None


def test_format_auc_stars_can_hide_or_show_p_values():
    assert format_auc_stars(0.00321, include_p_value=True) == "** (p = 0.00321)"
    assert format_auc_stars(0.00321, include_p_value=False) == "**"
    assert format_auc_stars(0.42, include_p_value=True) == "ns (p = 0.42)"
    assert format_auc_stars(1.234e-5, include_p_value=True) == (
        "**** (" + r"p = $\mathregular{1.23 \times 10^{-5}}$" + ")"
    )
