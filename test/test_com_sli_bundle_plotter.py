import pytest
import numpy as np

from src.plotting.com_sli_bundle_plotter import (
    _agarose_placement_series,
    _resolve_metric_condition,
    _sample_size_label_kwargs,
    _select_sli_extremes,
)


def test_agarose_sample_size_labels_get_translucent_white_background():
    kwargs = _sample_size_label_kwargs("agarose")

    assert kwargs["zorder"] == 5
    assert kwargs["bbox"] == {
        "boxstyle": "round,pad=0.12",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.78,
    }


def test_other_metric_sample_size_labels_keep_existing_style():
    assert _sample_size_label_kwargs("commag") == {}


def test_compare_yoked_selects_exp_minus_ctrl_condition():
    assert _resolve_metric_condition("exp", compare_ctrl=True) == "exp_minus_ctrl"


def test_compare_yoked_rejects_ctrl_only_condition():
    with pytest.raises(ValueError, match="metric-condition ctrl"):
        _resolve_metric_condition("ctrl", compare_ctrl=True)


def test_agarose_physical_minus_virtual_uses_paired_exp_values():
    bundle = {
        "agarose_ratio_exp": np.array([[[0.8, 0.5]]]),
        "agarose_virtual_ratio_exp": np.array([[[0.3, 0.4]]]),
    }

    values = _agarose_placement_series(
        bundle, "exp", "physical_minus_virtual"
    )

    np.testing.assert_allclose(values, [[[0.5, 0.1]]])


def test_agarose_placement_delta_can_also_use_exp_minus_yoked():
    bundle = {
        "agarose_ratio_exp": np.array([[[0.8]]]),
        "agarose_virtual_ratio_exp": np.array([[[0.3]]]),
        "agarose_ratio_ctrl": np.array([[[0.6]]]),
        "agarose_virtual_ratio_ctrl": np.array([[[0.2]]]),
    }

    values = _agarose_placement_series(
        bundle, "exp_minus_ctrl", "physical_minus_virtual"
    )

    np.testing.assert_allclose(values, [[[0.1]]])


def test_both_sli_extremes_select_top_before_bottom():
    indices, labels, group_ids = _select_sli_extremes(
        np.array([-2.0, -1.0, 1.0, 2.0]),
        top_fraction=0.5,
        bottom_fraction=0.5,
        which="both",
    )

    np.testing.assert_array_equal(indices, [2, 3, 0, 1])
    assert labels == ["Top 50% learners", "Bottom 50% learners"]
    np.testing.assert_array_equal(group_ids, [0, 0, 1, 1])
