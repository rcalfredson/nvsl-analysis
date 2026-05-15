import numpy as np
import pytest

from src.analysis.correlation_stats import (
    fisher_independent_correlation_test,
    pearson_correlation_summary,
)


def test_pearson_correlation_summary_filters_nan_pairs():
    x = np.asarray([1.0, 2.0, 3.0, np.nan, 4.0])
    y = np.asarray([2.0, 4.0, 6.0, 8.0, 8.0])

    out = pearson_correlation_summary(x, y)

    assert out.n == 4
    assert out.r == pytest.approx(1.0)
    assert out.p == pytest.approx(0.0)


def test_fisher_independent_correlation_test_matches_formula():
    out = fisher_independent_correlation_test(0.60, 30, 0.10, 28)
    z1 = np.arctanh(0.60)
    z2 = np.arctanh(0.10)
    se = np.sqrt(1 / (30 - 3) + 1 / (28 - 3))

    assert out.z1 == pytest.approx(z1)
    assert out.z2 == pytest.approx(z2)
    assert out.se == pytest.approx(se)
    assert out.z_stat == pytest.approx((z1 - z2) / se)
    assert 0.0 <= out.p_two_sided <= 1.0


def test_fisher_independent_correlation_test_requires_n_above_three():
    with pytest.raises(ValueError, match="n > 3"):
        fisher_independent_correlation_test(0.5, 3, 0.1, 10)
