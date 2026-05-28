from types import SimpleNamespace

import numpy as np
import pytest

from src.exporting.turnback_excursion_bin_sli_bundle import (
    _bin_edges_mm,
    _compute_turnback_curves,
    _effective_windowing,
    _integrated_bin_contribution,
    _selected_training_indices,
    _selected_windows_for_va,
)


class _Training:
    def __init__(self, start=0, stop=100, *, circle=True, name="training"):
        self.start = start
        self.stop = stop
        self._circle = circle
        self._name = name

    def isCircle(self):
        return self._circle

    def name(self):
        return self._name


class _Trajectory:
    def __init__(self, episodes, *, bad=False):
        self._episodes = list(episodes)
        self._bad = bad
        self.calls = []

    def reward_turnback_excursion_episodes_for_training(self, **kwargs):
        self.calls.append(kwargs)
        return list(self._episodes)


def _episode(stop, max_outer_delta_mm, turns_back, *, inner=2.0):
    return {
        "stop": stop,
        "max_outer_delta_mm": max_outer_delta_mm,
        "effective_inner_delta_mm": inner,
        "turns_back": turns_back,
    }


def _va(
    *,
    trns=None,
    trx=None,
    sync_bucket_ranges=None,
    noyc=False,
    skipped=False,
    fn="fake-video"
):
    return SimpleNamespace(
        fn=fn,
        trns=list(trns if trns is not None else [_Training()]),
        trx=list(trx if trx is not None else []),
        sync_bucket_ranges=sync_bucket_ranges,
        noyc=noyc,
        _skipped=skipped,
    )


def _curves(vas, edges=(2.0, 8.0, 16.0), **overrides):
    params = {
        "bin_edges_mm": np.asarray(edges, dtype=float),
        "inner_delta_mm": 2.0,
        "border_width_mm": 0.1,
        "radius_offset_px": 0.0,
        "selected_trainings": [0],
        "skip_first": 0,
        "keep_first": 0,
        "last_sync_buckets": 0,
        "debug": False,
    }
    params.update(overrides)
    return _compute_turnback_curves(vas, **params)


def test_integrated_bin_contribution_is_exact_threshold_average_above_inner_radius():
    assert (
        _integrated_bin_contribution(1.0, 2.0, 8.0, min_valid_outer_delta_mm=2.0) == 1.0
    )
    assert _integrated_bin_contribution(
        5.0, 2.0, 8.0, min_valid_outer_delta_mm=2.0
    ) == pytest.approx(0.5)
    assert (
        _integrated_bin_contribution(8.0, 2.0, 8.0, min_valid_outer_delta_mm=2.0) == 0.0
    )
    assert np.isnan(
        _integrated_bin_contribution(5.0, 2.0, 8.0, min_valid_outer_delta_mm=8.0)
    )


def test_bin_edges_accept_panel_edges_and_reject_bad_inputs():
    opts = SimpleNamespace(turnback_excursion_bin_edges_mm="2,8,16,inf")
    np.testing.assert_allclose(_bin_edges_mm(opts), [2.0, 8.0, 16.0, np.inf])

    with pytest.raises(ValueError, match="required"):
        _bin_edges_mm(SimpleNamespace(turnback_excursion_bin_edges_mm=""))

    with pytest.raises(ValueError, match="at least two"):
        _bin_edges_mm(SimpleNamespace(turnback_excursion_bin_edges_mm="2"))

    with pytest.raises(ValueError, match="monotonically increasing|inf.*last edge"):
        _bin_edges_mm(SimpleNamespace(turnback_excursion_bin_edges_mm="2,inf,16"))

    with pytest.raises(
        ValueError, match="monotonically increasing|strictly increasing"
    ):
        _bin_edges_mm(SimpleNamespace(turnback_excursion_bin_edges_mm="2,8,8"))


def test_training_selection_and_windowing_match_panel_metadata_conventions():
    vas = [_va(trns=[_Training(name="t1"), _Training(name="T2"), _Training(name="T3")])]

    opts = SimpleNamespace(turnback_excursion_bin_trainings="3,1-2,2")
    assert _selected_training_indices(vas, opts) == [0, 1, 2]

    opts = SimpleNamespace(turnback_excursion_bin_trainings="9")
    assert _selected_training_indices(vas, opts) == [0, 1, 2]

    assert _selected_training_indices([_va(skipped=True)], SimpleNamespace()) == []

    va = _va(
        trns=[_Training(name="T1")],
        sync_bucket_ranges=[[(0, 10), (10, 20), (20, 30), (30, 40)]],
    )
    windows = _selected_windows_for_va(
        va, [0], skip_first=1, keep_first=3, last_sync_buckets=2
    )

    assert windows == [
        {
            "training_idx": 0,
            "training_name": "T1",
            "start": 20,
            "stop": 40,
            "bucket_ranges": [(20, 30), (30, 40)],
        }
    ]


def test_effective_windowing_uses_metric_specific_overrides_and_clamps_negative_values():
    opts = SimpleNamespace(
        skip_first_sync_buckets=5,
        keep_first_sync_buckets=6,
        turnback_excursion_bin_skip_first_sync_buckets=-2,
        turnback_excursion_bin_keep_first_sync_buckets=None,
        turnback_excursion_bin_last_sync_buckets=-3,
    )

    assert _effective_windowing(opts) == (0, 6, 0)


def test_turnback_binned_counts_only_successful_observed_turnbacks():
    exp = _Trajectory(
        [
            _episode(10, 4.0, True),
            _episode(20, 10.0, False),
            _episode(30, np.nan, True),
            _episode(80, 4.0, True),
        ]
    )
    ctrl = _Trajectory([_episode(10, 10.0, True), _episode(20, 4.0, False)])
    va = _va(trx=[exp, ctrl], sync_bucket_ranges=[[(0, 50)]])

    ratio_exp, ratio_ctrl, turn_exp, turn_ctrl, total_exp, total_ctrl, _ = _curves([va])

    np.testing.assert_allclose(turn_exp, [[2.0 / 3.0, 1.0]])
    np.testing.assert_array_equal(total_exp, [[1, 1]])
    np.testing.assert_allclose(ratio_exp, [[2.0 / 3.0, 1.0]])

    np.testing.assert_allclose(turn_ctrl, [[0.0, 0.75]])
    np.testing.assert_array_equal(total_ctrl, [[1, 1]])
    np.testing.assert_allclose(ratio_ctrl, [[0.0, 0.75]])


def test_turnback_binned_average_is_not_bin_membership():
    exp = _Trajectory(
        [
            _episode(10, 1.0, True),
            _episode(20, 5.0, True),
            _episode(30, 8.0, True),
            _episode(40, 20.0, True),
        ]
    )

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _curves([_va(trx=[exp], noyc=True)])

    np.testing.assert_allclose(turn_exp, [[1.5, 3.0]])
    np.testing.assert_array_equal(total_exp, [[4, 4]])
    np.testing.assert_allclose(ratio_exp, [[0.375, 0.75]])


def test_open_ended_bin_resolves_from_nonturning_excursions_but_does_not_count_them():
    exp = _Trajectory([_episode(10, 20.0, False)])

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _curves(
        [_va(trx=[exp], noyc=True)], edges=(2.0, 8.0, np.inf)
    )

    assert np.isnan(ratio_exp).all()
    np.testing.assert_allclose(turn_exp, [[0.0, 0.0]])
    np.testing.assert_array_equal(total_exp, [[0, 0]])


def test_open_ended_bin_rejects_when_no_observed_excursion_exceeds_lower_edge():
    exp = _Trajectory([_episode(10, 8.0, True)])

    with pytest.raises(ValueError, match="no observed excursion exceeded it"):
        _curves([_va(trx=[exp], noyc=True)], edges=(2.0, 16.0, np.inf))


def test_bins_at_or_below_effective_inner_radius_are_not_counted():
    exp = _Trajectory([_episode(10, 8.0, True, inner=8.0)])

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _curves([_va(trx=[exp], noyc=True)])

    assert np.isnan(ratio_exp[0, 0])
    np.testing.assert_array_equal(total_exp, [[0, 1]])
    np.testing.assert_allclose(turn_exp, [[0.0, 1.0]])
