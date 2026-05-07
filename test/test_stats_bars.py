import numpy as np

from src.plotting.stats_bars import format_sig_label


def test_format_sig_label_can_append_p_value():
    assert format_sig_label(0.00321, include_p_value=True) == "**\np=0.00321"


def test_format_sig_label_preserves_existing_star_only_default():
    assert format_sig_label(0.00321) == "**"


def test_format_sig_label_does_not_append_to_non_significant_or_nan_labels():
    assert format_sig_label(0.9, include_p_value=True) == "ns"
    assert format_sig_label(np.nan, include_p_value=True) == ""
