import csv
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.sli_bundle_utils import load_sli_bundle
from src.exporting.return_prob_excursion_bin_sli_bundle import (
    _bin_edges_mm,
    _compute_return_prob_curves,
    _effective_windowing,
    _integrated_bin_contribution,
    _selected_training_indices,
    _selected_windows_for_va,
    export_return_prob_excursion_bin_sli_bundle,
)
from src.exporting.return_prob_outer_radius_sli_bundle import (
    _compute_return_prob_curves as _compute_return_prob_outer_radius_curves,
    _write_return_prob_outer_radius_debug_episodes_csv,
    export_return_prob_outer_radius_sli_bundle,
)
from src.plotting.return_prob_outer_radius_sli_bundle_plotter import (
    _bundle_to_exported as _return_prob_outer_radius_bundle_to_exported,
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
        self.boundary_event_stats = {}

    def reward_return_excursion_episodes_for_training(self, **kwargs):
        self.calls.append(kwargs)
        return list(self._episodes)


class _OuterRadiusTrajectory(_Trajectory):
    def reward_return_probability_episodes_for_training(self, **kwargs):
        self.calls.append(kwargs)
        return list(self._episodes)


def _episode(stop, max_excursion_mm, returns):
    return {"stop": stop, "max_excursion_mm": max_excursion_mm, "returns": returns}


def _wall_regions(*regions):
    return {"wall": {"all": {"edge": {"boundary_contact_regions": list(regions)}}}}


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
        "reward_delta_mm": 0.0,
        "border_width_mm": 0.1,
        "selected_trainings": [0],
        "skip_first": 0,
        "keep_first": 0,
        "last_sync_buckets": 0,
        "debug": False,
    }
    params.update(overrides)
    return _compute_return_prob_curves(vas, **params)


def test_integrated_bin_contribution_is_exact_threshold_average():
    assert _integrated_bin_contribution(1.0, 2.0, 8.0) == 1.0
    assert _integrated_bin_contribution(2.0, 2.0, 8.0) == 1.0
    assert _integrated_bin_contribution(5.0, 2.0, 8.0) == pytest.approx(0.5)
    assert _integrated_bin_contribution(8.0, 2.0, 8.0) == 0.0
    assert np.isnan(_integrated_bin_contribution(np.nan, 2.0, 8.0))
    assert np.isnan(_integrated_bin_contribution(5.0, 8.0, 8.0))


def test_bin_edges_accept_panel_one_edges_and_reject_prickly_inputs():
    opts = SimpleNamespace(return_prob_excursion_bin_edges_mm="2,8,16,inf")
    np.testing.assert_allclose(_bin_edges_mm(opts), [2.0, 8.0, 16.0, np.inf])

    with pytest.raises(ValueError, match="required"):
        _bin_edges_mm(SimpleNamespace(return_prob_excursion_bin_edges_mm=""))

    with pytest.raises(ValueError, match="at least two"):
        _bin_edges_mm(SimpleNamespace(return_prob_excursion_bin_edges_mm="2"))

    with pytest.raises(ValueError, match="monotonically increasing|inf.*last edge"):
        _bin_edges_mm(SimpleNamespace(return_prob_excursion_bin_edges_mm="2,inf,16"))

    with pytest.raises(
        ValueError, match="monotonically increasing|strictly increasing"
    ):
        _bin_edges_mm(SimpleNamespace(return_prob_excursion_bin_edges_mm="2,8,8"))


def test_training_selection_is_one_based_deduplicated_and_falls_back_to_all():
    vas = [_va(trns=[_Training(name="T1"), _Training(name="T2"), _Training(name="T3")])]

    opts = SimpleNamespace(return_prob_excursion_bin_trainings="3,1-2,2")
    assert _selected_training_indices(vas, opts) == [0, 1, 2]

    opts = SimpleNamespace(return_prob_excursion_bin_trainings="9")
    assert _selected_training_indices(vas, opts) == [0, 1, 2]

    assert _selected_training_indices([_va(skipped=True)], SimpleNamespace()) == []


def test_window_selection_honors_skip_keep_and_last_sync_buckets():
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
        return_prob_excursion_bin_skip_first_sync_buckets=-2,
        return_prob_excursion_bin_keep_first_sync_buckets=None,
        return_prob_excursion_bin_last_sync_buckets=-3,
    )

    assert _effective_windowing(opts) == (0, 6, 0)


def test_return_probability_treats_nonreturning_terminal_excursions_as_censored():
    exp = _Trajectory(
        [
            _episode(10, 4.0, True),
            _episode(20, 10.0, False),
            _episode(30, np.nan, True),
            _episode(80, 4.0, True),
        ]
    )
    ctrl = _Trajectory([_episode(10, 10.0, True), _episode(20, 4.0, False)])
    va = _va(
        trx=[exp, ctrl],
        sync_bucket_ranges=[[(0, 50)]],
    )

    ratio_exp, ratio_ctrl, ret_exp, ret_ctrl, total_exp, total_ctrl, _ = _curves([va])

    np.testing.assert_allclose(ret_exp, [[2.0 / 3.0, 1.0]])
    np.testing.assert_array_equal(total_exp, [[1, 1]])
    np.testing.assert_allclose(ratio_exp, [[2.0 / 3.0, 1.0]])

    np.testing.assert_allclose(ret_ctrl, [[0.0, 0.75]])
    np.testing.assert_array_equal(total_ctrl, [[1, 1]])
    np.testing.assert_allclose(ratio_ctrl, [[0.0, 0.75]])


def test_return_probability_bin_average_is_not_bin_membership():
    exp = _Trajectory(
        [
            _episode(10, 1.0, True),
            _episode(20, 5.0, True),
            _episode(30, 8.0, True),
            _episode(40, 20.0, True),
        ]
    )

    ratio_exp, _, ret_exp, _, total_exp, _, _ = _curves([_va(trx=[exp], noyc=True)])

    np.testing.assert_allclose(ret_exp, [[1.5, 3.0]])
    np.testing.assert_array_equal(total_exp, [[4, 4]])
    np.testing.assert_allclose(ratio_exp, [[0.375, 0.75]])


def test_return_probability_can_exclude_wall_contact_episodes():
    exp = _Trajectory(
        [
            {"start": 0, "stop": 10, "max_excursion_mm": 4.0, "returns": True},
            {"start": 20, "stop": 30, "max_excursion_mm": 5.0, "returns": True},
            {"start": 40, "stop": 50, "max_excursion_mm": 6.0, "returns": True},
        ]
    )
    exp.boundary_event_stats = _wall_regions(slice(22, 24))

    _, _, ret_exp, _, total_exp, _, _ = _curves(
        [_va(trx=[exp], noyc=True)],
        edges=(2.0, 8.0),
        exclude_wall_contact=True,
    )

    np.testing.assert_allclose(ret_exp, [[(8.0 - 4.0) / 6.0 + (8.0 - 6.0) / 6.0]])
    np.testing.assert_array_equal(total_exp, [[2]])


def test_return_probability_leaves_bins_nan_when_no_valid_denominator_exists():
    ratio_exp, ratio_ctrl, ret_exp, ret_ctrl, total_exp, total_ctrl, _ = _curves(
        [_va(trx=[_Trajectory([], bad=True), _Trajectory([])], noyc=True)]
    )

    assert np.isnan(ratio_exp).all()
    assert np.isnan(ratio_ctrl).all()
    np.testing.assert_array_equal(ret_exp, [[0.0, 0.0]])
    np.testing.assert_array_equal(ret_ctrl, [[0.0, 0.0]])
    np.testing.assert_array_equal(total_exp, [[0, 0]])
    np.testing.assert_array_equal(total_ctrl, [[0, 0]])


def test_return_probability_masks_bins_below_min_trajectory_count():
    exp = _Trajectory(
        [
            _episode(10, 1.0, True),
            _episode(20, 5.0, True),
            _episode(30, 8.0, True),
            _episode(40, 20.0, True),
        ]
    )

    ratio_exp, _, ret_exp, _, total_exp, _, _ = _curves(
        [_va(trx=[exp], noyc=True)], min_trajectories=5
    )

    assert np.isnan(ratio_exp).all()
    np.testing.assert_allclose(ret_exp, [[1.5, 3.0]])
    np.testing.assert_array_equal(total_exp, [[4, 4]])


def test_return_probability_outer_radius_masks_below_min_trajectory_count():
    exp = _OuterRadiusTrajectory(
        [
            {"stop": 10, "returns": True},
            {"stop": 20, "returns": False},
            {"stop": 30, "returns": True},
        ]
    )

    ratio_exp, _, ret_exp, _, total_exp, _, _ = (
        _compute_return_prob_outer_radius_curves(
            [_va(trx=[exp], noyc=True)],
            outer_radii_mm=np.asarray([16.0], dtype=float),
            legacy_outer_radii=False,
            reward_radius_mm=None,
            reward_delta_mm=0.0,
            border_width_mm=0.1,
            selected_trainings=[0],
            skip_first=0,
            keep_first=0,
            last_sync_buckets=0,
            debug=False,
            min_trajectories=4,
        )
    )

    np.testing.assert_array_equal(ret_exp, [[2]])
    np.testing.assert_array_equal(total_exp, [[3]])
    assert np.isnan(ratio_exp).all()


def test_return_probability_outer_radius_can_exclude_wall_contact_episodes():
    exp = _OuterRadiusTrajectory(
        [
            {"start": 0, "stop": 10, "returns": True},
            {"start": 20, "stop": 30, "returns": True},
            {"start": 40, "stop": 50, "returns": False},
        ]
    )
    exp.boundary_event_stats = _wall_regions(slice(22, 24))

    ratio_exp, _, ret_exp, _, total_exp, _, _ = (
        _compute_return_prob_outer_radius_curves(
            [_va(trx=[exp], noyc=True)],
            outer_radii_mm=np.asarray([16.0], dtype=float),
            legacy_outer_radii=False,
            reward_radius_mm=None,
            reward_delta_mm=0.0,
            border_width_mm=0.1,
            selected_trainings=[0],
            skip_first=0,
            keep_first=0,
            last_sync_buckets=0,
            debug=False,
            exclude_wall_contact=True,
        )
    )

    np.testing.assert_array_equal(ret_exp, [[1]])
    np.testing.assert_array_equal(total_exp, [[2]])
    np.testing.assert_allclose(ratio_exp, [[0.5]])


def test_open_ended_bin_resolves_from_nonreturning_censored_excursions_but_does_not_count_them():
    exp = _Trajectory([_episode(10, 20.0, False)])

    ratio_exp, _, ret_exp, _, total_exp, _, _ = _curves(
        [_va(trx=[exp], noyc=True)],
        edges=(2.0, 8.0, np.inf),
    )

    assert np.isnan(ratio_exp).all()
    np.testing.assert_allclose(ret_exp, [[0.0, 0.0]])
    np.testing.assert_array_equal(total_exp, [[0, 0]])


def test_open_ended_bin_rejects_when_no_observed_excursion_exceeds_lower_edge():
    exp = _Trajectory([_episode(10, 8.0, True)])

    with pytest.raises(ValueError, match="no observed excursion exceeded it"):
        _curves([_va(trx=[exp], noyc=True)], edges=(2.0, 16.0, np.inf))


def test_return_prob_excursion_bin_export_applies_exp_target_sync_bucket_filter(
    tmp_path, monkeypatch
):
    exp = _Trajectory([_episode(10, 1.0, True), _episode(20, 5.0, True)])
    va = _va(trx=[exp], noyc=True, sync_bucket_ranges=[[]])
    va.reward_exclusion_mask = [[[True]]]
    opts = SimpleNamespace(
        export_group_label="group",
        return_prob_excursion_bin_edges_mm="2,8",
        return_prob_excursion_bin_radii_mm=None,
        return_prob_excursion_bin_trainings="1",
        return_prob_excursion_bin_skip_first_sync_buckets=0,
        return_prob_excursion_bin_keep_first_sync_buckets=0,
        return_prob_excursion_bin_last_sync_buckets=0,
        return_prob_excursion_bin_reward_radius_mm=None,
        return_prob_excursion_bin_reward_delta_mm=0.0,
        return_prob_excursion_bin_border_width_mm=0.1,
        return_prob_excursion_bin_debug=False,
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=0,
        sli_select_keep_first_sync_buckets=0,
        min_between_reward_trajectories=1,
        require_exp_target_sync_bucket=True,
        exp_target_sync_bucket_filter_training=1,
        exp_target_sync_bucket_filter_sync_bucket=1,
        piTh=10,
    )
    monkeypatch.setattr(
        "src.exporting.return_prob_excursion_bin_sli_bundle._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.1], dtype=float),
            np.asarray([[[0.1]]], dtype=float),
        ),
    )
    out = tmp_path / "return_prob_excursion_bin.npz"

    export_return_prob_excursion_bin_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        np.testing.assert_array_equal(
            bundle["exp_target_sync_bucket_filter_eligible"], [False]
        )
        np.testing.assert_array_equal(
            bundle["exp_target_sync_bucket_filter_reason"], ["target_sync_bucket_missing"]
        )
        assert np.isnan(bundle["sli"]).all()
        assert np.isnan(bundle["sli_ts"]).all()
        assert "return_prob_excursion_bin_ratio_exp" not in bundle.files
        assert np.isnan(
            bundle["fraction_within_radius_excursion_bin_ratio_exp"]
        ).all()
        np.testing.assert_array_equal(
            bundle["fraction_within_radius_excursion_bin_total_exp"], [[2]]
        )
        np.testing.assert_allclose(
            bundle["fraction_within_radius_excursion_bin_return_exp"], [[1.5]]
        )

    loaded = load_sli_bundle(str(out))
    assert np.isnan(
        loaded["fraction_within_radius_excursion_bin_ratio_exp"]
    ).all()
    np.testing.assert_array_equal(loaded["exp_target_sync_bucket_filter_eligible"], [False])


def test_return_prob_outer_radius_export_applies_exp_target_sync_bucket_filter(
    tmp_path, monkeypatch
):
    exp = _OuterRadiusTrajectory(
        [{"stop": 10, "returns": True}, {"stop": 20, "returns": False}]
    )
    va = _va(trx=[exp], noyc=True, sync_bucket_ranges=[[]])
    va.reward_exclusion_mask = [[[True]]]
    opts = SimpleNamespace(
        export_group_label="group",
        return_prob_outer_radius_outer_radii_mm="16",
        return_prob_outer_radius_outer_deltas_mm=None,
        return_prob_outer_radius_trainings="1",
        return_prob_outer_radius_skip_first_sync_buckets=0,
        return_prob_outer_radius_keep_first_sync_buckets=0,
        return_prob_outer_radius_last_sync_buckets=0,
        return_prob_reward_radius_mm=None,
        return_prob_reward_delta_mm=0.0,
        return_prob_border_width_mm=0.1,
        return_prob_outer_radius_debug=False,
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=0,
        sli_select_keep_first_sync_buckets=0,
        min_between_reward_trajectories=1,
        require_exp_target_sync_bucket=True,
        exp_target_sync_bucket_filter_training=1,
        exp_target_sync_bucket_filter_sync_bucket=1,
        piTh=10,
    )
    monkeypatch.setattr(
        "src.exporting.return_prob_outer_radius_sli_bundle._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.1], dtype=float),
            np.asarray([[[0.1]]], dtype=float),
        ),
    )
    out = tmp_path / "return_prob_outer_radius.npz"

    export_return_prob_outer_radius_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        np.testing.assert_array_equal(
            bundle["exp_target_sync_bucket_filter_eligible"], [False]
        )
        np.testing.assert_array_equal(
            bundle["exp_target_sync_bucket_filter_reason"], ["target_sync_bucket_missing"]
        )
        assert np.isnan(bundle["sli"]).all()
        assert np.isnan(bundle["sli_ts"]).all()
        assert "return_prob_outer_radius_ratio_exp" not in bundle.files
        assert np.isnan(bundle["fraction_within_radius_outer_radius_ratio_exp"]).all()
        np.testing.assert_array_equal(
            bundle["fraction_within_radius_outer_radius_total_exp"], [[2]]
        )
        np.testing.assert_array_equal(
            bundle["fraction_within_radius_outer_radius_return_exp"], [[1]]
        )


def test_return_prob_outer_radius_export_records_wall_contact_exclusion(
    tmp_path, monkeypatch
):
    exp = _OuterRadiusTrajectory(
        [
            {"start": 0, "stop": 10, "returns": True},
            {"start": 20, "stop": 30, "returns": True},
            {"start": 40, "stop": 50, "returns": False},
        ]
    )
    exp.boundary_event_stats = _wall_regions((22, 24))
    va = _va(trx=[exp], noyc=True)
    opts = SimpleNamespace(
        export_group_label="group",
        return_prob_outer_radius_outer_radii_mm="16",
        return_prob_outer_radius_outer_deltas_mm=None,
        return_prob_outer_radius_trainings="1",
        return_prob_outer_radius_skip_first_sync_buckets=0,
        return_prob_outer_radius_keep_first_sync_buckets=0,
        return_prob_outer_radius_last_sync_buckets=0,
        return_prob_reward_radius_mm=None,
        return_prob_reward_delta_mm=0.0,
        return_prob_border_width_mm=0.1,
        return_prob_outer_radius_debug=False,
        return_prob_outer_radius_exclude_wall_contact=True,
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=0,
        sli_select_keep_first_sync_buckets=0,
        min_between_reward_trajectories=0,
        require_exp_target_sync_bucket=False,
        piTh=10,
    )
    monkeypatch.setattr(
        "src.exporting.return_prob_outer_radius_sli_bundle._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.1], dtype=float),
            np.asarray([[[0.1]]], dtype=float),
        ),
    )
    out = tmp_path / "return_prob_outer_radius_wall_filtered.npz"

    export_return_prob_outer_radius_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        assert bool(
            bundle[
                "fraction_within_radius_outer_radius_exclude_wall_contact"
            ].reshape(()).item()
        )
        np.testing.assert_array_equal(
            bundle["fraction_within_radius_outer_radius_total_exp"], [[2]]
        )
        np.testing.assert_array_equal(
            bundle["fraction_within_radius_outer_radius_return_exp"], [[1]]
        )
        np.testing.assert_allclose(
            bundle["fraction_within_radius_outer_radius_ratio_exp"], [[0.5]]
        )


def test_return_prob_outer_radius_debug_csv_records_episode_audit_fields(tmp_path):
    exp = _OuterRadiusTrajectory(
        [
            {
                "start": 0,
                "stop": 10,
                "anchor_reward": 0,
                "reward_entry": 10,
                "returns": True,
                "end_reason": "reenter_reward",
                "max_radius_mm": 5.0,
                "max_excursion_mm": 2.0,
                "max_dist_frame": 5,
            },
            {
                "start": 20,
                "stop": 30,
                "anchor_reward": 20,
                "reward_entry": None,
                "returns": False,
                "end_reason": "exit_outer",
                "max_radius_mm": 18.0,
                "max_excursion_mm": 15.0,
                "max_dist_frame": 28,
            },
        ]
    )
    exp.boundary_event_stats = _wall_regions((22, 24))
    va = _va(trx=[exp], noyc=True, sync_bucket_ranges=[[(0, 50)]], fn="debug-video")
    out_csv = tmp_path / "return_prob_outer_radius_debug.csv"

    n_rows = _write_return_prob_outer_radius_debug_episodes_csv(
        [va],
        out_csv=str(out_csv),
        outer_radii_mm=np.asarray([16.0], dtype=float),
        legacy_outer_radii=False,
        reward_radius_mm=None,
        reward_delta_mm=0.0,
        border_width_mm=0.1,
        selected_trainings=[0],
        skip_first=0,
        keep_first=0,
        last_sync_buckets=0,
        exclude_wall_contact=True,
    )

    assert n_rows == 2
    with open(out_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert [row["video_id"] for row in rows] == ["debug-video::f-1"] * 2
    assert [row["fly_role"] for row in rows] == ["exp", "exp"]
    assert [row["requested_outer_mm"] for row in rows] == ["16.0", "16.0"]
    assert [row["radius_mode"] for row in rows] == ["radius", "radius"]
    assert [row["returns"] for row in rows] == ["True", "False"]
    assert [row["end_reason"] for row in rows] == ["reenter_reward", "exit_outer"]
    assert [row["wall_overlap"] for row in rows] == ["False", "True"]
    assert [row["included_in_metric"] for row in rows] == ["True", "False"]
    assert [row["event_frame"] for row in rows] == ["9", "29"]
    assert [row["full_trajectory_start"] for row in rows] == ["0", "20"]
    assert [row["full_trajectory_stop"] for row in rows] == ["10", "30"]
    assert [row["full_trajectory_max_radius_mm"] for row in rows] == ["5.0", "18.0"]
    assert [row["full_trajectory_max_excursion_mm"] for row in rows] == ["2.0", "15.0"]
    assert [row["full_trajectory_max_dist_frame"] for row in rows] == ["5", "28"]


def test_return_prob_outer_radius_plotter_preserves_duplicate_filename_fly_ids():
    bundle = {
        "sli": np.asarray([0.1, 0.2], dtype=float),
        "video_ids": np.asarray(["same.mp4::f0", "same.mp4::f1"], dtype=object),
        "fraction_within_radius_outer_radius_outer_radii_mm": np.asarray(
            [16.0], dtype=float
        ),
        "fraction_within_radius_outer_radius_ratio_exp": np.asarray(
            [[0.25], [0.75]], dtype=float
        ),
        "fraction_within_radius_outer_radius_ratio_ctrl": np.asarray(
            [[np.nan], [np.nan]], dtype=float
        ),
    }

    exported = _return_prob_outer_radius_bundle_to_exported(
        bundle,
        label="group",
        mode="exp",
        metric="ratio",
        sub_idx=np.asarray([0, 1], dtype=int),
    )

    np.testing.assert_allclose(
        np.asarray(exported.per_unit_values_panel[0], dtype=float), [0.25, 0.75]
    )
    np.testing.assert_array_equal(
        np.asarray(exported.per_unit_ids_panel[0], dtype=object),
        ["same.mp4::f0", "same.mp4::f1"],
    )
