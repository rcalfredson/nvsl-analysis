import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from src.analysis.multiple_comparisons import holm_adjust


def test_holm_adjust_matches_statsmodels_and_preserves_nonfinite_positions():
    raw = np.asarray([0.04, np.nan, 0.01, 0.03, np.inf])
    finite = np.isfinite(raw)
    expected = np.full_like(raw, np.nan)
    expected[finite] = multipletests(raw[finite], method="holm")[1]

    actual = np.asarray(holm_adjust(raw.tolist()))

    assert actual[finite] == pytest.approx(expected[finite])
    assert np.isnan(actual[~finite]).all()


def test_holm_adjust_handles_empty_and_all_nonfinite_families():
    assert holm_adjust([]) == []
    assert np.isnan(holm_adjust([np.nan, np.inf])).all()
