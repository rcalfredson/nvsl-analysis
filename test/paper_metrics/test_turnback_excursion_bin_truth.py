import csv
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.sli_bundle_utils import load_sli_bundle
from src.exporting.turnback_excursion_bin_sli_bundle import (
    _bin_edges_mm,
    _compute_pair_curves,
    _compute_turnback_curves,
    _effective_windowing,
    export_turnback_excursion_bin_sli_bundle,
    _integrated_bin_contribution,
    _pair_deltas_mm,
    _selected_training_indices,
    _selected_windows_for_va,
    _write_turnback_pair_debug_episodes_csv,
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
    def __init__(
        self,
        episodes,
        *,
        bad=False,
        x=None,
        y=None,
        theta=None,
        walking=None,
        f=0,
    ):
        self._episodes = list(episodes)
        self._bad = bad
        self.calls = []
        self.boundary_event_stats = {}
        self.x = np.asarray([] if x is None else x, dtype=float)
        self.y = np.asarray([] if y is None else y, dtype=float)
        self.theta = None if theta is None else np.asarray(theta, dtype=float)
        self.walking = None if walking is None else np.asarray(walking, dtype=bool)
        self.f = int(f)

    def reward_turnback_excursion_episodes_for_training(self, **kwargs):
        self.calls.append(kwargs)
        return list(self._episodes)

    def reward_turnback_dual_circle_episodes_for_training(self, **kwargs):
        self.calls.append(kwargs)
        return list(self._episodes)


def _episode(stop, max_outer_delta_mm, turns_back, *, inner=2.0):
    return {
        "stop": stop,
        "max_outer_delta_mm": max_outer_delta_mm,
        "effective_inner_delta_mm": inner,
        "turns_back": turns_back,
    }


def _wall_regions(*regions):
    return {"wall": {"all": {"edge": {"boundary_contact_regions": list(regions)}}}}


def _va(
    *,
    trns=None,
    trx=None,
    sync_bucket_ranges=None,
    noyc=False,
    skipped=False,
    fn="fake-video",
    f=None,
):
    return SimpleNamespace(
        fn=fn,
        f=f,
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


def _pair_curves(vas, pairs=((2.0, 4.0), (6.0, 8.0)), **overrides):
    params = {
        "inner_deltas_mm": np.asarray([p[0] for p in pairs], dtype=float),
        "outer_deltas_mm": np.asarray([p[1] for p in pairs], dtype=float),
        "border_width_mm": 0.1,
        "radius_offset_px": 0.0,
        "selected_trainings": [0],
        "skip_first": 0,
        "keep_first": 0,
        "last_sync_buckets": 0,
        "debug": False,
    }
    params.update(overrides)
    return _compute_pair_curves(vas, **params)


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


def test_pair_deltas_accept_independent_inner_outer_pairs_and_reject_bad_inputs():
    opts = SimpleNamespace(turnback_excursion_bin_pairs_mm="2:4,6:8,14:16")
    inner, outer = _pair_deltas_mm(opts)
    np.testing.assert_allclose(inner, [2.0, 6.0, 14.0])
    np.testing.assert_allclose(outer, [4.0, 8.0, 16.0])

    assert _pair_deltas_mm(SimpleNamespace(turnback_excursion_bin_pairs_mm=None)) is None

    with pytest.raises(ValueError, match="inner:outer"):
        _pair_deltas_mm(SimpleNamespace(turnback_excursion_bin_pairs_mm="2-4"))

    with pytest.raises(ValueError, match="greater"):
        _pair_deltas_mm(SimpleNamespace(turnback_excursion_bin_pairs_mm="4:4"))


def test_turnback_pair_min_walking_fraction_defaults_to_disabled():
    walking = np.zeros(30, dtype=bool)
    walking[10:20] = True
    exp = _Trajectory(
        [
            {**_episode(10, 0.0, True), "start": 0},
            {**_episode(20, 0.0, False), "start": 10},
        ],
        walking=walking,
    )
    va = _va(trx=[exp], noyc=True, sync_bucket_ranges=[[(0, 30)]])

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _pair_curves(
        [va],
        pairs=((2.0, 3.0),),
    )

    np.testing.assert_allclose(ratio_exp, [[0.5]])
    np.testing.assert_array_equal(turn_exp, [[1.0]])
    np.testing.assert_array_equal(total_exp, [[2]])

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _pair_curves(
        [va],
        pairs=((2.0, 3.0),),
        min_walking_fraction=0.75,
    )

    np.testing.assert_allclose(ratio_exp, [[0.0]])
    np.testing.assert_array_equal(turn_exp, [[0.0]])
    np.testing.assert_array_equal(total_exp, [[1]])


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


def test_turnback_binned_curves_can_exclude_wall_contact_episodes():
    exp = _Trajectory(
        [
            {
                "start": 0,
                "stop": 10,
                "max_outer_delta_mm": 4.0,
                "effective_inner_delta_mm": 2.0,
                "turns_back": True,
            },
            {
                "start": 20,
                "stop": 30,
                "max_outer_delta_mm": 5.0,
                "effective_inner_delta_mm": 2.0,
                "turns_back": True,
            },
            {
                "start": 40,
                "stop": 50,
                "max_outer_delta_mm": 6.0,
                "effective_inner_delta_mm": 2.0,
                "turns_back": True,
            },
        ]
    )
    exp.boundary_event_stats = _wall_regions(slice(22, 24))

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _curves(
        [_va(trx=[exp], noyc=True)],
        edges=(2.0, 8.0),
        exclude_wall_contact=True,
    )

    np.testing.assert_allclose(turn_exp, [[(8.0 - 4.0) / 6.0 + (8.0 - 6.0) / 6.0]])
    np.testing.assert_array_equal(total_exp, [[2]])
    np.testing.assert_allclose(ratio_exp, turn_exp / total_exp)


def test_turnback_pair_curves_can_exclude_wall_contact_episodes():
    exp = _Trajectory(
        [
            {"start": 0, "stop": 10, "turns_back": True},
            {"start": 20, "stop": 30, "turns_back": True},
            {"start": 40, "stop": 50, "turns_back": False},
        ]
    )
    exp.boundary_event_stats = _wall_regions((22, 24))

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _pair_curves(
        [_va(trx=[exp], noyc=True)],
        pairs=((2.0, 4.0),),
        exclude_wall_contact=True,
    )

    np.testing.assert_allclose(turn_exp, [[1.0]])
    np.testing.assert_array_equal(total_exp, [[2]])
    np.testing.assert_allclose(ratio_exp, [[0.5]])


def test_turnback_pair_curves_can_require_min_walking_fraction():
    walking = np.zeros(40, dtype=bool)
    walking[0:10] = True
    walking[20:25] = [True, True, False, False, False]
    walking[30:35] = True
    exp = _Trajectory(
        [
            {"start": 0, "stop": 10, "turns_back": True},
            {"start": 20, "stop": 25, "turns_back": True},
            {"start": 30, "stop": 35, "turns_back": False},
        ],
        walking=walking,
    )

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _pair_curves(
        [_va(trx=[exp], noyc=True)],
        pairs=((2.0, 4.0),),
        min_walking_fraction=0.75,
    )

    np.testing.assert_allclose(turn_exp, [[1.0]])
    np.testing.assert_array_equal(total_exp, [[2]])
    np.testing.assert_allclose(ratio_exp, [[0.5]])


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


def test_turnback_binned_masks_below_min_episode_count():
    exp = _Trajectory(
        [
            _episode(10, 1.0, True),
            _episode(20, 5.0, True),
            _episode(30, 8.0, True),
            _episode(40, 20.0, True),
        ]
    )

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _curves(
        [_va(trx=[exp], noyc=True)],
        min_episodes=5,
    )

    assert np.isnan(ratio_exp).all()
    np.testing.assert_allclose(turn_exp, [[1.5, 3.0]])
    np.testing.assert_array_equal(total_exp, [[4, 4]])


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


def test_independent_pair_curves_call_dual_circle_metric_per_pair():
    exp = _Trajectory([_episode(10, 0.0, True), _episode(20, 0.0, False)])
    ctrl = _Trajectory([_episode(10, 0.0, False)])
    va = _va(trx=[exp, ctrl], sync_bucket_ranges=[[(0, 50)]])

    ratio_exp, ratio_ctrl, turn_exp, turn_ctrl, total_exp, total_ctrl, _ = _pair_curves(
        [va],
        pairs=((2.0, 4.0), (6.0, 8.0), (14.0, 16.0)),
    )

    np.testing.assert_allclose(ratio_exp, [[0.5, 0.5, 0.5]])
    np.testing.assert_allclose(ratio_ctrl, [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(turn_exp, [[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(turn_ctrl, [[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(total_exp, [[2, 2, 2]])
    np.testing.assert_array_equal(total_ctrl, [[1, 1, 1]])

    assert [
        (call["inner_delta_mm"], call["outer_delta_mm"])
        for call in exp.calls
    ] == [(2.0, 4.0), (6.0, 8.0), (14.0, 16.0)]


def test_independent_pair_curves_mask_below_min_episode_count():
    exp = _Trajectory([_episode(10, 0.0, True), _episode(20, 0.0, False)])

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _pair_curves(
        [_va(trx=[exp], noyc=True)],
        pairs=((2.0, 4.0),),
        min_episodes=3,
    )

    assert np.isnan(ratio_exp).all()
    np.testing.assert_allclose(turn_exp, [[1.0]])
    np.testing.assert_array_equal(total_exp, [[2]])


def test_turnback_excursion_bin_export_records_min_episode_filter(tmp_path, monkeypatch):
    exp = _Trajectory(
        [
            _episode(10, 1.0, True),
            _episode(20, 5.0, True),
            _episode(30, 8.0, True),
            _episode(40, 20.0, True),
        ]
    )
    va = _va(trx=[exp], noyc=True, sync_bucket_ranges=[[(0, 50)]])
    opts = SimpleNamespace(
        export_group_label="group",
        turnback_excursion_bin_edges_mm="2,8,16",
        turnback_excursion_bin_radii_mm=None,
        turnback_excursion_bin_radius_pairs_mm=None,
        turnback_excursion_bin_pairs_mm=None,
        turnback_excursion_bin_trainings=None,
        turnback_excursion_bin_skip_first_sync_buckets=0,
        turnback_excursion_bin_keep_first_sync_buckets=0,
        turnback_excursion_bin_last_sync_buckets=0,
        turnback_inner_delta_mm=2.0,
        turnback_inner_radius_mm=None,
        turnback_border_width_mm=0.1,
        turnback_inner_radius_offset_px=0.0,
        turnback_excursion_bin_debug=False,
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=0,
        sli_select_keep_first_sync_buckets=0,
        min_turnback_episodes=5,
        turnback_excursion_bin_debug_episodes_csv=None,
    )
    monkeypatch.setattr(
        "src.exporting.turnback_excursion_bin_sli_bundle._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.1], dtype=float),
            np.asarray([[[0.1]]], dtype=float),
        ),
    )
    out = tmp_path / "turnback_excursion_bin.npz"

    export_turnback_excursion_bin_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        assert int(bundle["min_turnback_episodes"]) == 5
        assert np.isnan(bundle["turnback_excursion_bin_ratio_exp"]).all()
        np.testing.assert_array_equal(
            bundle["turnback_excursion_bin_total_exp"], [[4, 4]]
        )
        assert int(bundle["episode_filter_turnback_excursion_bin_exp_min_episodes"]) == 5
        assert int(bundle["episode_filter_turnback_excursion_bin_exp_unit_count"]) == 2
        assert int(bundle["episode_filter_turnback_excursion_bin_exp_included_count"]) == 0
        assert int(bundle["episode_filter_turnback_excursion_bin_exp_excluded_count"]) == 2
        np.testing.assert_array_equal(
            bundle["episode_filter_turnback_excursion_bin_exp_excluded_episode_counts"],
            [4, 4],
        )

    loaded = load_sli_bundle(str(out))
    assert int(loaded["min_turnback_episodes"]) == 5
    assert np.isnan(loaded["turnback_excursion_bin_ratio_exp"]).all()


def test_turnback_pair_export_applies_exp_target_sync_bucket_filter(tmp_path, monkeypatch):
    exp = _Trajectory([_episode(10, 0.0, True), _episode(20, 0.0, False)])
    va = _va(trx=[exp], noyc=True, sync_bucket_ranges=[[]])
    va.reward_exclusion_mask = [[[True]]]
    opts = SimpleNamespace(
        export_group_label="group",
        turnback_excursion_bin_edges_mm=None,
        turnback_excursion_bin_radii_mm=None,
        turnback_excursion_bin_radius_pairs_mm="2:4",
        turnback_excursion_bin_pairs_mm=None,
        turnback_excursion_bin_trainings="1",
        turnback_excursion_bin_skip_first_sync_buckets=0,
        turnback_excursion_bin_keep_first_sync_buckets=0,
        turnback_excursion_bin_last_sync_buckets=0,
        turnback_inner_delta_mm=2.0,
        turnback_inner_radius_mm=None,
        turnback_border_width_mm=0.1,
        turnback_inner_radius_offset_px=0.0,
        turnback_excursion_bin_debug=False,
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=0,
        sli_select_keep_first_sync_buckets=0,
        min_turnback_episodes=1,
        require_exp_target_sync_bucket=True,
        exp_target_sync_bucket_filter_training=1,
        exp_target_sync_bucket_filter_sync_bucket=1,
        piTh=10,
        turnback_excursion_bin_debug_episodes_csv=None,
    )
    monkeypatch.setattr(
        "src.exporting.turnback_excursion_bin_sli_bundle._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.1], dtype=float),
            np.asarray([[[0.1]]], dtype=float),
        ),
    )
    out = tmp_path / "turnback_pairs.npz"

    export_turnback_excursion_bin_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        assert bool(bundle["exp_target_sync_bucket_filter_enabled"])
        np.testing.assert_array_equal(
            bundle["exp_target_sync_bucket_filter_eligible"], [False]
        )
        np.testing.assert_array_equal(
            bundle["exp_target_sync_bucket_filter_reason"], ["target_sync_bucket_missing"]
        )
        assert np.isnan(bundle["sli"]).all()
        assert np.isnan(bundle["sli_ts"]).all()
        assert np.isnan(bundle["turnback_excursion_bin_ratio_exp"]).all()
        np.testing.assert_array_equal(bundle["turnback_excursion_bin_total_exp"], [[2]])
        np.testing.assert_allclose(bundle["turnback_excursion_bin_turn_exp"], [[1.0]])

    loaded = load_sli_bundle(str(out))
    assert np.isnan(loaded["turnback_excursion_bin_ratio_exp"]).all()
    np.testing.assert_array_equal(loaded["exp_target_sync_bucket_filter_eligible"], [False])


def test_turnback_pair_home_vector_alignment_filter_uses_theta_orientation():
    x = np.zeros(40, dtype=float)
    y = np.zeros(40, dtype=float)
    theta = np.zeros(40, dtype=float)
    x[10] = 1.0
    y[10] = 0.0
    theta[10] = 270.0
    x[20] = 1.0
    y[20] = 0.0
    theta[20] = 90.0
    exp = _Trajectory(
        [
            {
                **_episode(11, 0.0, True),
                "start": 10,
                "event_frame": 10,
                "reward_cx_px": 0.0,
                "reward_cy_px": 0.0,
            },
            {
                **_episode(21, 0.0, True),
                "start": 20,
                "event_frame": 20,
                "reward_cx_px": 0.0,
                "reward_cy_px": 0.0,
            },
            {
                **_episode(30, 0.0, False),
                "event_frame": 29,
                "reward_cx_px": 0.0,
                "reward_cy_px": 0.0,
            },
        ],
        x=x,
        y=y,
        theta=theta,
    )
    va = _va(trx=[exp], noyc=True, sync_bucket_ranges=[[(0, 50)]])

    ratio_exp, _, turn_exp, _, total_exp, _, _ = _pair_curves(
        [va],
        pairs=((2.0, 3.0),),
        require_home_vector_alignment=True,
        home_vector_alignment_threshold=0.0,
        home_vector_alignment_window_radius_frames=2,
        home_vector_alignment_heading_estimator="mean",
        home_vector_alignment_home_vector_anchor="intersection",
        home_vector_alignment_max_interpolated_heading_frames=1,
    )

    np.testing.assert_allclose(ratio_exp, [[1.0 / 3.0]])
    np.testing.assert_array_equal(turn_exp, [[1.0]])
    np.testing.assert_array_equal(total_exp, [[3]])


def test_turnback_pair_export_applies_home_vector_alignment_filter(
    tmp_path, monkeypatch
):
    x = np.zeros(30, dtype=float)
    y = np.zeros(30, dtype=float)
    theta = np.zeros(30, dtype=float)
    x[10] = 1.0
    theta[10] = 270.0
    x[20] = 1.0
    theta[20] = 90.0
    exp = _Trajectory(
        [
            {
                **_episode(11, 0.0, True),
                "start": 10,
                "event_frame": 10,
                "reward_cx_px": 0.0,
                "reward_cy_px": 0.0,
            },
            {
                **_episode(21, 0.0, True),
                "start": 20,
                "event_frame": 20,
                "reward_cx_px": 0.0,
                "reward_cy_px": 0.0,
            },
            {
                **_episode(25, 0.0, False),
                "event_frame": 24,
                "reward_cx_px": 0.0,
                "reward_cy_px": 0.0,
            },
        ],
        x=x,
        y=y,
        theta=theta,
    )
    va = _va(trx=[exp], noyc=True, sync_bucket_ranges=[[(0, 30)]])
    opts = SimpleNamespace(
        export_group_label="group",
        turnback_excursion_bin_edges_mm=None,
        turnback_excursion_bin_radii_mm=None,
        turnback_excursion_bin_radius_pairs_mm="2:3",
        turnback_excursion_bin_pairs_mm=None,
        turnback_excursion_bin_trainings="1",
        turnback_excursion_bin_skip_first_sync_buckets=0,
        turnback_excursion_bin_keep_first_sync_buckets=0,
        turnback_excursion_bin_last_sync_buckets=0,
        turnback_inner_delta_mm=2.0,
        turnback_inner_radius_mm=None,
        turnback_border_width_mm=0.1,
        turnback_inner_radius_offset_px=0.0,
        turnback_excursion_bin_debug=False,
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=0,
        sli_select_keep_first_sync_buckets=0,
        min_turnback_episodes=1,
        require_exp_target_sync_bucket=False,
        exp_target_sync_bucket_filter_training=1,
        exp_target_sync_bucket_filter_sync_bucket=1,
        piTh=10,
        turnback_excursion_bin_debug_episodes_csv=None,
        turnback_excursion_bin_require_home_vector_alignment=True,
        turnback_excursion_bin_home_vector_alignment_threshold=0.0,
        turnback_excursion_bin_home_vector_alignment_window_radius_frames=2,
        turnback_excursion_bin_home_vector_alignment_heading_estimator="mean",
        turnback_excursion_bin_home_vector_alignment_home_vector_anchor="intersection",
        turnback_excursion_bin_home_vector_alignment_max_interpolated_heading_frames=1,
    )
    monkeypatch.setattr(
        "src.exporting.turnback_excursion_bin_sli_bundle._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.1], dtype=float),
            np.asarray([[[0.1]]], dtype=float),
        ),
    )
    out = tmp_path / "turnback_pairs_theta_filter.npz"

    export_turnback_excursion_bin_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        assert bool(bundle["turnback_excursion_bin_require_home_vector_alignment"])
        np.testing.assert_allclose(bundle["turnback_excursion_bin_turn_exp"], [[1.0]])
        np.testing.assert_array_equal(bundle["turnback_excursion_bin_total_exp"], [[3]])
        np.testing.assert_allclose(
            bundle["turnback_excursion_bin_ratio_exp"], [[1.0 / 3.0]]
        )


def test_turnback_pair_debug_csv_uses_video_analysis_fly_index(tmp_path):
    exp = _Trajectory([_episode(10, 0.0, True)])
    ctrl = _Trajectory([_episode(20, 0.0, False)])
    va = _va(
        trx=[exp, ctrl],
        sync_bucket_ranges=[[(0, 50)]],
        noyc=False,
        f=7,
    )
    out_csv = tmp_path / "turnback_pair_debug.csv"

    n_rows = _write_turnback_pair_debug_episodes_csv(
        [va],
        out_csv=str(out_csv),
        inner_deltas_mm=np.asarray([2.0], dtype=float),
        outer_deltas_mm=np.asarray([3.0], dtype=float),
        legacy_pair_deltas=False,
        border_width_mm=0.1,
        radius_offset_px=0.0,
        selected_trainings=[0],
        skip_first=0,
        keep_first=0,
        last_sync_buckets=0,
        exclude_wall_contact=False,
    )

    assert n_rows == 2
    with open(out_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["fly_role"] for row in rows] == ["exp", "ctrl"]
    assert [row["fly_idx"] for row in rows] == ["7", "7"]


def test_turnback_pair_debug_csv_includes_theta_alignment_diagnostics(tmp_path):
    x = np.zeros(25, dtype=float)
    y = np.zeros(25, dtype=float)
    theta = np.zeros(25, dtype=float)
    walking = np.ones(25, dtype=bool)
    walking[20:21] = False
    x[10] = 1.0
    theta[10] = 270.0
    x[20] = 1.0
    theta[20] = 90.0
    exp = _Trajectory(
        [
            {
                **_episode(11, 0.0, True),
                "start": 10,
                "event_frame": 10,
                "reward_cx_px": 0.0,
                "reward_cy_px": 0.0,
            },
            {
                **_episode(21, 0.0, True),
                "start": 20,
                "event_frame": 20,
                "reward_cx_px": 0.0,
                "reward_cy_px": 0.0,
            },
        ],
        x=x,
        y=y,
        theta=theta,
        walking=walking,
    )
    va = _va(trx=[exp], sync_bucket_ranges=[[(0, 25)]], noyc=True)
    out_csv = tmp_path / "turnback_pair_debug_theta.csv"

    n_rows = _write_turnback_pair_debug_episodes_csv(
        [va],
        out_csv=str(out_csv),
        inner_deltas_mm=np.asarray([2.0], dtype=float),
        outer_deltas_mm=np.asarray([3.0], dtype=float),
        legacy_pair_deltas=False,
        border_width_mm=0.1,
        radius_offset_px=0.0,
        selected_trainings=[0],
        skip_first=0,
        keep_first=0,
        last_sync_buckets=0,
        exclude_wall_contact=False,
        require_home_vector_alignment=True,
        home_vector_alignment_threshold=0.0,
        min_walking_fraction=0.75,
    )

    assert n_rows == 2
    with open(out_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert rows[0]["home_vector_alignment_pass"] == "True"
    assert rows[1]["home_vector_alignment_pass"] == "False"
    assert rows[0]["walking_fraction_pass"] == "True"
    assert rows[1]["walking_fraction_pass"] == "False"
    assert float(rows[0]["walking_fraction"]) == pytest.approx(1.0)
    assert float(rows[1]["walking_fraction"]) == pytest.approx(0.0)
    assert float(rows[0]["min_walking_fraction"]) == pytest.approx(0.75)
    assert float(rows[0]["theta_deg"]) == pytest.approx(270.0)
    assert float(rows[1]["theta_deg"]) == pytest.approx(90.0)
    assert float(rows[0]["home_vector_alignment"]) == pytest.approx(1.0)
    assert float(rows[1]["home_vector_alignment"]) == pytest.approx(-1.0)
    assert float(rows[0]["home_vector_alignment_angle_deg"]) == pytest.approx(0.0)
    assert float(rows[1]["home_vector_alignment_angle_deg"]) == pytest.approx(180.0)
    assert float(rows[0]["orientation_x"]) == pytest.approx(-1.0)
    assert float(rows[0]["home_vector_unit_x"]) == pytest.approx(-1.0)
