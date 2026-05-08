import numpy as np
import pytest

from src.plotting.time_series_auc import (
    auc_samples,
    compute_auc_test,
    format_auc_label,
    format_auc_stars,
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
        "AUC (n=3,3): **** (p="
    )


def test_format_auc_stars_can_hide_or_show_p_values():
    assert format_auc_stars(0.00321, include_p_value=True) == "** (p=0.00321)"
    assert format_auc_stars(0.00321, include_p_value=False) == "**"
    assert format_auc_stars(0.42, include_p_value=True) == "ns (p=0.42)"
