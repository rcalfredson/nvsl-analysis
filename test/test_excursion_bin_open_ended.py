from types import SimpleNamespace

import numpy as np

from src.exporting.return_prob_excursion_bin_sli_bundle import (
    _compute_return_prob_curves,
)
from src.exporting.turnback_excursion_bin_sli_bundle import _compute_turnback_curves
from src.plotting.return_prob_excursion_bin_sli_bundle_plotter import (
    _panel_labels as return_panel_labels,
)


class _Training:
    start = 0
    stop = 100

    def isCircle(self):
        return True

    def name(self):
        return "training"


class _ReturnTrajectory:
    _bad = False

    def reward_return_excursion_episodes_for_training(self, **_kwargs):
        return [
            {"stop": 10, "max_excursion_mm": 4.0, "returns": True},
            {"stop": 20, "max_excursion_mm": 10.0, "returns": False},
            {"stop": 30, "max_excursion_mm": 20.0, "returns": True},
            {"stop": 40, "max_excursion_mm": 30.0, "returns": False},
        ]


class _TurnbackTrajectory:
    _bad = False

    def reward_turnback_excursion_episodes_for_training(self, **_kwargs):
        return [
            {
                "stop": 10,
                "max_outer_delta_mm": 4.0,
                "effective_inner_delta_mm": 2.0,
                "turns_back": True,
            },
            {
                "stop": 20,
                "max_outer_delta_mm": 10.0,
                "effective_inner_delta_mm": 2.0,
                "turns_back": False,
            },
            {
                "stop": 30,
                "max_outer_delta_mm": 20.0,
                "effective_inner_delta_mm": 2.0,
                "turns_back": True,
            },
            {
                "stop": 40,
                "max_outer_delta_mm": 30.0,
                "effective_inner_delta_mm": 2.0,
                "turns_back": False,
            },
        ]


def _va_with_trx(trx):
    return SimpleNamespace(
        fn="fake-video",
        trns=[_Training()],
        trx=[trx],
        sync_bucket_ranges=None,
        noyc=True,
    )


def test_return_probability_open_ended_bin_uses_dynamic_finite_threshold_edge():
    ratio_exp, _, ret_exp, _, total_exp, _, _ = _compute_return_prob_curves(
        [_va_with_trx(_ReturnTrajectory())],
        bin_edges_mm=np.asarray([2.0, 8.0, 16.0, np.inf]),
        reward_delta_mm=0.0,
        border_width_mm=0.1,
        selected_trainings=[0],
        skip_first=0,
        keep_first=0,
        last_sync_buckets=0,
        debug=False,
    )

    np.testing.assert_allclose(ratio_exp, [[1.0 / 3.0, 0.5, 6.0 / 7.0]])
    np.testing.assert_allclose(ret_exp, [[2.0 / 3.0, 1.0, 12.0 / 7.0]])
    np.testing.assert_array_equal(total_exp, [[2, 2, 2]])


def test_turnback_open_ended_bin_uses_dynamic_finite_threshold_edge():
    ratio_exp, _, turn_exp, _, total_exp, _, _ = _compute_turnback_curves(
        [_va_with_trx(_TurnbackTrajectory())],
        bin_edges_mm=np.asarray([2.0, 8.0, 16.0, np.inf]),
        inner_delta_mm=2.0,
        border_width_mm=0.1,
        radius_offset_px=0.0,
        selected_trainings=[0],
        skip_first=0,
        keep_first=0,
        last_sync_buckets=0,
        debug=False,
    )

    np.testing.assert_allclose(ratio_exp, [[1.0 / 3.0, 0.5, 6.0 / 7.0]])
    np.testing.assert_allclose(turn_exp, [[2.0 / 3.0, 1.0, 12.0 / 7.0]])
    np.testing.assert_array_equal(total_exp, [[2, 2, 2]])


def test_open_ended_bin_label_survives_resolved_finite_edge():
    labels = return_panel_labels(
        {
            "return_prob_excursion_bin_edges_mm": np.asarray([2.0, 8.0, 16.0, 30.0]),
            "return_prob_excursion_bin_open_ended_upper_bin": np.asarray(True),
        }
    )

    assert labels == ["[2, 8) mm", "[8, 16) mm", "16+ mm"]
