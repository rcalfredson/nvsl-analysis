import sys
from types import SimpleNamespace

import numpy as np

from src.plotting.between_reward_conditioned_disttrav import (
    BetweenRewardConditionedDistTravConfig,
    BetweenRewardConditionedDistTravPlotter,
)
from src.plotting.between_reward_segment_metrics import (
    dist_traveled_mm_masked,
    max_radial_distance_mm_masked,
    path_length_and_max_radius_mm_masked,
)
from src.plotting.btw_rwd_return_leg_dist_collectors import (
    ReturnLegDistPerFlyCollector,
)
from src.plotting.btw_rwd_return_leg_dist_totals import (
    ReturnLegDistTotalsConfig,
    ReturnLegDistTotalsPlotter,
)
from src.exporting.btw_rwd_return_leg_dist_sli_bundle import (
    build_btw_rwd_return_leg_dist_sli_bundle,
)
from src.exporting.between_reward_maxdist_sli_bundle import (
    _extract_between_reward_maxdist_arrays,
    build_between_reward_maxdist_sli_bundle,
)


class _Trajectory:
    def __init__(self, x, y=None, *, d=None, walking=None, bad=False, f=0):
        self.x = np.asarray(x, dtype=float)
        self.y = np.zeros_like(self.x) if y is None else np.asarray(y, dtype=float)
        if d is None:
            self.d = np.hypot(np.diff(self.x), np.diff(self.y))
        else:
            self.d = np.asarray(d, dtype=float)
        self.walking = walking
        self._bad = bad
        self.f = f

    def bad(self):
        return self._bad


class _Training:
    start = 0
    stop = 10
    n = 1

    def name(self):
        return "T1"

    def circles(self, _fly_idx):
        return [(0.0, 0.0, 0.0)]


class _Video:
    def __init__(self, *, trx, segments_by_fly, noyc=True, sync_bucket_ranges=None):
        self.fn = "fake-video"
        self.f = 7
        self.trns = [_Training()]
        self.trx = trx
        self.flies = list(range(len(trx)))
        self.noyc = noyc
        self._skipped = False
        self.sync_bucket_ranges = sync_bucket_ranges
        self._segments_by_fly = segments_by_fly
        self.xf = SimpleNamespace(fctr=1.0)
        self.ct = SimpleNamespace(pxPerMmFloor=lambda: 2.0)

    def _numRewardsMsg(self, *_args, **_kwargs):
        return 5

    def _syncBucket(self, _trn, _df):
        return 0, 2, None

    def _iter_between_reward_segment_com(self, _trn, fly_idx, **_kwargs):
        yield from self._segments_by_fly.get(fly_idx, [])


class _MaxDistVideo:
    def __init__(self):
        self.fn = "fake-video"
        self.trns = [_Training()]
        self.reward_exclusion_mask = [[[True, True]]]
        self.syncMeanBetweenRewardMaxDist = [
            {"exp": [1.0, 2.0], "ctrl": [3.0, 4.0]}
        ]
        self.syncMeanBetweenRewardMaxDistN = [
            {"exp": [2, 1], "ctrl": [1, 3]}
        ]

    def _numRewardsMsg(self, *_args, **_kwargs):
        return 5

    def _syncBucket(self, _trn, _df):
        return 0, 2, None

    def bySyncBucketMeanBetweenRewardMaxDist(self, **_kwargs):
        return None


def _segment(*, s, e, b_idx=0, max_i=None, max_d_mm=4.0):
    return SimpleNamespace(
        s=int(s), e=int(e), b_idx=int(b_idx), max_d_i=max_i, max_d_mm=float(max_d_mm)
    )


def test_between_reward_distance_traveled_sums_only_consecutive_kept_steps():
    traj = _Trajectory(x=[0, 3, 6, 10, 10], d=[3, 3, 4, 0])

    total = dist_traveled_mm_masked(
        traj=traj,
        s=0,
        e=4,
        fi=0,
        nonwalk_mask=np.asarray([False, False, True, False]),
        exclude_nonwalk=True,
        px_per_mm=2.0,
    )
    distance_from_override = dist_traveled_mm_masked(
        traj=traj,
        s=0,
        e=4,
        fi=0,
        nonwalk_mask=None,
        exclude_nonwalk=False,
        px_per_mm=2.0,
        start_override=2,
    )

    assert total == 1.5
    assert distance_from_override == 2.0


def test_between_reward_distance_traveled_rejects_degenerate_windows_and_scale():
    traj = _Trajectory(x=[0, 3], d=[3])

    assert np.isnan(
        dist_traveled_mm_masked(
            traj=traj,
            s=0,
            e=1,
            fi=0,
            nonwalk_mask=None,
            exclude_nonwalk=False,
            px_per_mm=2.0,
        )
    )
    assert np.isnan(
        dist_traveled_mm_masked(
            traj=traj,
            s=0,
            e=2,
            fi=0,
            nonwalk_mask=None,
            exclude_nonwalk=False,
            px_per_mm=0.0,
        )
    )


def test_between_reward_max_distance_uses_kept_finite_frames_from_reward_center():
    traj = _Trajectory(x=[0, 3, 4, 6], y=[0, 4, 0, 8], d=[5, 5, 10])

    assert (
        max_radial_distance_mm_masked(
            traj=traj,
            s=0,
            e=4,
            fi=0,
            nonwalk_mask=None,
            exclude_nonwalk=False,
            px_per_mm=2.0,
            center_xy=(0.0, 0.0),
        )
        == 5.0
    )

    assert (
        max_radial_distance_mm_masked(
            traj=traj,
            s=0,
            e=4,
            fi=0,
            nonwalk_mask=np.asarray([False, False, False, True]),
            exclude_nonwalk=True,
            px_per_mm=2.0,
            center_xy=(0.0, 0.0),
        )
        == 2.5
    )


def test_path_length_and_max_radius_share_validity_gate():
    traj = _Trajectory(x=[1, 1, 1], y=[0, 0, 0], d=[0, 0])

    path_mm, radius_mm = path_length_and_max_radius_mm_masked(
        traj=traj,
        s=0,
        e=3,
        fi=0,
        nonwalk_mask=None,
        exclude_nonwalk=False,
        px_per_mm=2.0,
        center_xy=(0.0, 0.0),
    )

    assert np.isnan(path_mm)
    assert np.isnan(radius_mm)


def test_conditioned_distance_traveled_bins_total_and_return_leg_per_fly():
    traj = _Trajectory(x=np.arange(12, dtype=float), d=np.ones(11))
    # Segment max_i/max_d_mm are supplied by the fake iterator here; this test
    # exercises downstream binning/aggregation, not max-distance discovery from x/y.
    va = _Video(
        trx=[traj],
        segments_by_fly={
            0: [
                _segment(s=0, e=4, max_i=2, max_d_mm=4.0),
                _segment(s=4, e=8, max_i=5, max_d_mm=7.0),
                _segment(s=8, e=11, max_i=9, max_d_mm=12.0),
            ]
        },
        sync_bucket_ranges=[[(0, 10)]],
    )
    cfg = BetweenRewardConditionedDistTravConfig(
        out_file="unused.png",
        training_index=0,
        x_bin_edges_mm=[0.0, 6.0, 10.0],
        ci_conf=0.95,
    )
    plotter = BetweenRewardConditionedDistTravPlotter(
        [va],
        SimpleNamespace(
            com_exclude_wall_contact=False,
            btw_rwd_conditioned_exclude_nonwalking_frames=False,
        ),
        gls=None,
        customizer=None,
        cfg=cfg,
    )

    y_total, y_tail, unit_ids, meta = plotter._collect_per_fly_binned_means()

    np.testing.assert_allclose(y_total, [[1.5, 1.5]])
    np.testing.assert_allclose(y_tail, [[0.5, 1.0]])
    assert unit_ids.tolist() == ["fake-video|fly=7|trx=0"]
    assert meta["units"] == "mm"


def test_return_leg_collector_averages_by_sync_bucket_and_honors_nonwalk_option():
    exp = _Trajectory(
        x=np.arange(12, dtype=float),
        d=np.ones(11),
        walking=np.asarray([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]),
    )
    ctrl = _Trajectory(x=np.arange(12, dtype=float), d=np.full(11, 2.0), f=1)
    va = _Video(
        trx=[exp, ctrl],
        segments_by_fly={
            0: [
                _segment(s=0, e=4, b_idx=0, max_i=2),
                _segment(s=4, e=8, b_idx=0, max_i=5),
                _segment(s=6, e=10, b_idx=1, max_i=8),
            ],
            1: [_segment(s=0, e=4, b_idx=0, max_i=1)],
        },
    )

    collector = ReturnLegDistPerFlyCollector()
    collector.vas = [va]
    collector.opts = SimpleNamespace(
        com_exclude_wall_contact=False,
        btw_rwd_sync_bucket_min_trajectories=0,
        btw_rwd_return_leg_dist_exclude_nonwalking_frames=True,
        btw_rwd_return_leg_dist_min_walk_frames=2,
    )
    collector.cfg = SimpleNamespace(
        skip_first_sync_buckets=0, keep_first_sync_buckets=0
    )

    mean_exp, mean_ctrl, n_exp, n_ctrl = (
        collector.collect_return_leg_sync_bucket_arrays()
    )

    np.testing.assert_allclose(mean_exp, [[[0.5, 0.5]]])
    np.testing.assert_allclose(mean_ctrl, [[[2.0, np.nan]]])
    np.testing.assert_array_equal(n_exp, [[[2, 1]]])
    np.testing.assert_array_equal(n_ctrl, [[[1, 0]]])


def test_between_reward_maxdist_export_masks_by_min_trajectory_count():
    va = _MaxDistVideo()

    mean_exp, mean_ctrl, n_exp, n_ctrl = _extract_between_reward_maxdist_arrays(
        [va], SimpleNamespace(min_between_reward_trajectories=2)
    )

    np.testing.assert_allclose(mean_exp, [[[1.0, np.nan]]])
    np.testing.assert_allclose(mean_ctrl, [[[np.nan, 4.0]]])
    np.testing.assert_array_equal(n_exp, [[[2, 1]]])
    np.testing.assert_array_equal(n_ctrl, [[[1, 3]]])


def test_between_reward_maxdist_bundle_applies_exp_pi_threshold_filter(monkeypatch):
    va = _MaxDistVideo()

    monkeypatch.setitem(
        sys.modules,
        "analyze",
        SimpleNamespace(bucketLenForType=lambda _bucket_type: (1.0, None)),
    )
    monkeypatch.setattr(
        "src.exporting.bundle_utils._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.1], dtype=float),
            np.asarray([[[0.1, 0.2]]], dtype=float),
        ),
    )

    bundle = build_between_reward_maxdist_sli_bundle(
        [va],
        SimpleNamespace(
            min_between_reward_trajectories=1,
            best_worst_trn=1,
            sli_use_training_mean=True,
            sli_select_skip_first_sync_buckets=0,
            sli_select_keep_first_sync_buckets=0,
            require_exp_pi_threshold_bucket=True,
            exp_pi_threshold_filter_training=1,
            exp_pi_threshold_filter_sync_bucket=1,
            piTh=10,
        ),
        gls=None,
    )

    assert np.isnan(bundle["between_reward_maxdist_exp"]).all()
    np.testing.assert_allclose(bundle["between_reward_maxdist_ctrl"], [[[3.0, 4.0]]])
    np.testing.assert_array_equal(bundle["between_reward_maxdistN_exp"], [[[2, 1]]])
    assert np.isnan(bundle["sli"]).all()
    assert np.isnan(bundle["sli_ts"]).all()
    np.testing.assert_array_equal(bundle["exp_pi_threshold_filter_eligible"], [False])
    np.testing.assert_array_equal(
        bundle["exp_pi_threshold_filter_reason"], ["pi_threshold_failed"]
    )


def test_return_leg_sync_bucket_filter_masks_low_count_buckets():
    exp = _Trajectory(x=np.arange(12, dtype=float), d=np.ones(11))
    va = _Video(
        trx=[exp],
        segments_by_fly={
            0: [
                _segment(s=0, e=4, b_idx=0, max_i=2),
                _segment(s=4, e=7, b_idx=0, max_i=5),
                _segment(s=6, e=10, b_idx=1, max_i=8),
            ],
        },
    )

    collector = ReturnLegDistPerFlyCollector()
    collector.vas = [va]
    collector.opts = SimpleNamespace(
        com_exclude_wall_contact=False,
        min_between_reward_trajectories=2,
        btw_rwd_return_leg_dist_exclude_nonwalking_frames=False,
        btw_rwd_return_leg_dist_min_walk_frames=2,
    )
    collector.cfg = SimpleNamespace(
        skip_first_sync_buckets=0, keep_first_sync_buckets=0
    )

    mean_exp, _mean_ctrl, n_exp, _n_ctrl = (
        collector.collect_return_leg_sync_bucket_arrays()
    )

    np.testing.assert_allclose(mean_exp, [[[0.5, np.nan]]])
    np.testing.assert_array_equal(n_exp, [[[2, 1]]])


def test_return_leg_bundle_applies_exp_pi_threshold_filter(monkeypatch):
    exp = _Trajectory(x=np.arange(12, dtype=float), d=np.ones(11))
    ctrl = _Trajectory(x=np.arange(12, dtype=float), d=np.full(11, 2.0), f=1)
    va = _Video(
        trx=[exp, ctrl],
        segments_by_fly={
            0: [
                _segment(s=0, e=4, b_idx=0, max_i=2),
                _segment(s=4, e=8, b_idx=0, max_i=5),
                _segment(s=6, e=10, b_idx=1, max_i=8),
            ],
            1: [_segment(s=0, e=4, b_idx=0, max_i=1)],
        },
    )
    va.reward_exclusion_mask = [[[True, True]]]

    monkeypatch.setitem(
        sys.modules,
        "analyze",
        SimpleNamespace(bucketLenForType=lambda _bucket_type: (1.0, None)),
    )
    monkeypatch.setattr(
        "src.exporting.bundle_utils._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.1], dtype=float),
            np.asarray([[[0.1, 0.2]]], dtype=float),
        ),
    )

    bundle = build_btw_rwd_return_leg_dist_sli_bundle(
        [va],
        SimpleNamespace(
            com_exclude_wall_contact=False,
            min_between_reward_trajectories=1,
            btw_rwd_return_leg_dist_exclude_nonwalking_frames=False,
            btw_rwd_return_leg_dist_min_walk_frames=2,
            best_worst_trn=1,
            sli_use_training_mean=True,
            sli_select_skip_first_sync_buckets=0,
            sli_select_keep_first_sync_buckets=0,
            require_exp_pi_threshold_bucket=True,
            exp_pi_threshold_filter_training=1,
            exp_pi_threshold_filter_sync_bucket=1,
            piTh=10,
        ),
        gls=None,
    )

    assert np.isnan(bundle["between_reward_return_leg_dist_exp"]).all()
    np.testing.assert_allclose(
        bundle["between_reward_return_leg_dist_ctrl"], [[[2.0, np.nan]]]
    )
    np.testing.assert_array_equal(
        bundle["between_reward_return_leg_distN_exp"], [[[2, 1]]]
    )
    assert np.isnan(bundle["sli"]).all()
    assert np.isnan(bundle["sli_ts"]).all()
    np.testing.assert_array_equal(bundle["exp_pi_threshold_filter_eligible"], [False])
    np.testing.assert_array_equal(
        bundle["exp_pi_threshold_filter_reason"], ["pi_threshold_failed"]
    )


def test_return_leg_scalar_bars_filter_after_pooling_selected_episodes():
    exp = _Trajectory(x=np.arange(12, dtype=float), d=np.ones(11))
    va = _Video(
        trx=[exp],
        segments_by_fly={
            0: [
                _segment(s=0, e=4, b_idx=0, max_i=2),
                _segment(s=4, e=7, b_idx=0, max_i=5),
                _segment(s=6, e=10, b_idx=1, max_i=8),
            ],
        },
    )
    va.trns = [_Training(), _Training()]

    opts = SimpleNamespace(
        com_exclude_wall_contact=False,
        min_between_reward_trajectories=4,
        btw_rwd_return_leg_dist_exclude_nonwalking_frames=False,
        btw_rwd_return_leg_dist_min_walk_frames=2,
    )
    cfg = ReturnLegDistTotalsConfig(
        out_file="unused.png",
        pool_trainings=True,
        trainings=[1, 2],
        skip_first_sync_buckets=0,
        keep_first_sync_buckets=0,
    )
    plotter = ReturnLegDistTotalsPlotter(
        vas=[va], opts=opts, gls=None, customizer=None, cfg=cfg
    )

    data = plotter.compute_scalar_panels()

    assert data["panel_labels"] == ["Selected trainings combined"]
    assert int(data["n_units_panel"][0]) == 1
    np.testing.assert_allclose(
        np.asarray(data["per_unit_values_panel"][0], dtype=float), [0.5]
    )
    assert data["meta"]["training_selection"]["trainings_effective"] == [1, 2]


def test_return_leg_scalar_bars_single_training_filter_uses_selected_episode_count():
    exp = _Trajectory(x=np.arange(12, dtype=float), d=np.ones(11))
    va = _Video(
        trx=[exp],
        segments_by_fly={
            0: [
                _segment(s=0, e=4, b_idx=0, max_i=2),
                _segment(s=4, e=7, b_idx=0, max_i=5),
                _segment(s=6, e=10, b_idx=1, max_i=8),
            ],
        },
    )

    opts = SimpleNamespace(
        com_exclude_wall_contact=False,
        min_between_reward_trajectories=4,
        btw_rwd_return_leg_dist_exclude_nonwalking_frames=False,
        btw_rwd_return_leg_dist_min_walk_frames=2,
    )
    cfg = ReturnLegDistTotalsConfig(
        out_file="unused.png",
        pool_trainings=False,
        trainings=[1],
        skip_first_sync_buckets=0,
        keep_first_sync_buckets=0,
    )
    plotter = ReturnLegDistTotalsPlotter(
        vas=[va], opts=opts, gls=None, customizer=None, cfg=cfg
    )

    data = plotter.compute_scalar_panels()

    assert data["panel_labels"] == ["T1"]
    assert int(data["n_units_panel"][0]) == 0
    assert data["per_unit_values_panel"][0].size == 0
