import sys
from types import SimpleNamespace

import numpy as np

from src.analysis.trajectory import Trajectory
from src.analysis.sli_bundle_utils import load_sli_bundle
from src.analysis.video_analysis import VideoAnalysis
from src.exporting.turnback_sli_bundle import (
    _extract_turnback_arrays,
    export_turnback_sli_bundle,
)


class _CircleTraining:
    def __init__(self, *, start=0, stop=10, circle=True, radius_px=10.0):
        self.start = int(start)
        self.stop = int(stop)
        self._circle = bool(circle)
        self._radius_px = float(radius_px)

    def isCircle(self):
        return self._circle

    def circles(self, _fly_idx):
        return [(0.0, 0.0, self._radius_px)]

    def name(self):
        return "T1"

    def sname(self):
        return self.name()


class _TrajectoryEpisodes:
    def __init__(self, episodes, *, bad=False):
        self._episodes = list(episodes)
        self._bad = bool(bad)
        self.calls = []

    def reward_turnback_dual_circle_episodes_for_training(self, **kwargs):
        self.calls.append(kwargs)
        return list(self._episodes)


def _trajectory_at_distances(distances_px, *, training_stop=None):
    trj = object.__new__(Trajectory)
    trj.x = np.asarray(distances_px, dtype=float)
    trj.y = np.zeros_like(trj.x)
    trj.f = 0
    trj.va = SimpleNamespace(
        xf=SimpleNamespace(fctr=1.0), ct=SimpleNamespace(pxPerMmFloor=lambda: 1.0)
    )
    trn = _CircleTraining(
        start=0,
        stop=len(trj.x) if training_stop is None else training_stop,
        radius_px=10.0,
    )
    return trj, trn


def test_turnback_dual_circle_detects_success_failure_and_excludes_censored_episode():
    trj, trn = _trajectory_at_distances(
        [
            13.0,  # inside inner
            15.0,  # exit inner, still inside outer
            16.0,
            13.0,  # re-enter inner: success
            13.0,
            15.0,  # exit inner
            19.0,  # exit outer before re-entry: failure
            13.0,
            13.0,
            15.0,  # exit inner
            16.0,  # training end before outcome: censored and excluded
        ]
    )

    episodes = trj.reward_turnback_dual_circle_episodes_for_training(
        trn=trn, inner_delta_mm=4.0, outer_delta_mm=8.0, border_width_mm=0.0
    )

    assert [
        {
            "start": ep["start"],
            "stop": ep["stop"],
            "turns_back": ep["turns_back"],
            "end_reason": ep["end_reason"],
        }
        for ep in episodes
    ] == [
        {"start": 1, "stop": 3, "turns_back": True, "end_reason": "reenter_inner"},
        {
            "start": 5,
            "stop": 7,
            "turns_back": False,
            "end_reason": "exit_outer",
        },
    ]


def test_turnback_dual_circle_rejects_invalid_geometry_and_non_circle_training():
    trj, trn = _trajectory_at_distances([13.0, 15.0, 13.0])

    assert (
        trj.reward_turnback_dual_circle_episodes_for_training(
            trn=trn, inner_delta_mm=8.0, outer_delta_mm=4.0, border_width_mm=0.0
        )
        == []
    )
    assert (
        trj.reward_turnback_dual_circle_episodes_for_training(
            trn=_CircleTraining(circle=False),
            inner_delta_mm=4.0,
            outer_delta_mm=8.0,
            border_width_mm=0.0,
        )
        == []
    )


def test_turnback_ratio_bins_by_outcome_frame_and_leaves_empty_buckets_nan():
    exp = _TrajectoryEpisodes(
        [
            {"start": 1, "stop": 4, "turns_back": True},
            {"start": 1, "stop": 6, "turns_back": False},
            {"start": 8, "stop": 10, "turns_back": True},
        ]
    )
    ctrl = _TrajectoryEpisodes([])
    va = SimpleNamespace(
        circle=True,
        opts=SimpleNamespace(
            turnback_inner_delta_mm=4.0,
            turnback_outer_delta_mm=8.0,
            turnback_border_width_mm=0.0,
            turnback_inner_radius_offset_px=0.0,
            turnback_dual_circle_debug=False,
            min_turnback_episodes=1,
        ),
        sync_bucket_ranges=[[(0, 5), (5, 10), (10, 15)]],
        trns=[_CircleTraining(start=0, stop=15)],
        trx=[exp, ctrl],
    )

    VideoAnalysis.analyzeRewardTurnbackDualCircle(va)

    counts = va.reward_turnback_dual_circle_counts
    np.testing.assert_array_equal(counts["turnback"], [[[1, 1, 0], [0, 0, 0]]])
    np.testing.assert_array_equal(counts["total"], [[[1, 2, 0], [0, 0, 0]]])
    np.testing.assert_allclose(
        counts["ratio"], [[[1.0, 0.5, np.nan], [np.nan, np.nan, np.nan]]]
    )

    assert exp.calls[0]["inner_delta_mm"] == 4.0
    assert exp.calls[0]["outer_delta_mm"] == 8.0


def test_turnback_ratio_min_episode_filter_masks_low_total_buckets():
    exp = _TrajectoryEpisodes(
        [
            {"start": 1, "stop": 4, "turns_back": True},
            {"start": 1, "stop": 6, "turns_back": False},
            {"start": 8, "stop": 10, "turns_back": True},
        ]
    )
    va = SimpleNamespace(
        circle=True,
        opts=SimpleNamespace(
            turnback_inner_delta_mm=4.0,
            turnback_outer_delta_mm=8.0,
            turnback_border_width_mm=0.0,
            turnback_inner_radius_offset_px=0.0,
            turnback_dual_circle_debug=False,
            min_turnback_episodes=2,
        ),
        sync_bucket_ranges=[[(0, 5), (5, 10), (10, 15)]],
        trns=[_CircleTraining(start=0, stop=15)],
        trx=[exp],
    )

    VideoAnalysis.analyzeRewardTurnbackDualCircle(va)

    counts = va.reward_turnback_dual_circle_counts
    np.testing.assert_array_equal(counts["total"], [[[1, 2, 0]]])
    np.testing.assert_allclose(counts["ratio"], [[[np.nan, 0.5, np.nan]]])


def test_turnback_bundle_extraction_keeps_exp_ctrl_axes_and_pads_missing_buckets():
    ratio = np.asarray([[[1.0, 0.5], [0.0, np.nan]]], dtype=float)
    total = np.asarray([[[3, 2], [1, 0]]], dtype=int)
    va = SimpleNamespace(
        fn="fake-video",
        trns=[_CircleTraining(start=0, stop=15)],
        reward_turnback_dual_circle_counts={"ratio": ratio, "total": total},
        _numRewardsMsg=lambda *args, **_kwargs: 5,
        _syncBucket=lambda _trn, _df: (0, 3, None),
    )

    ratio_exp, ratio_ctrl, total_exp, total_ctrl = _extract_turnback_arrays([va])

    np.testing.assert_allclose(ratio_exp, [[[1.0, 0.5, np.nan]]])
    np.testing.assert_allclose(ratio_ctrl, [[[0.0, np.nan, np.nan]]])
    np.testing.assert_array_equal(total_exp, [[[3, 2, 0]]])
    np.testing.assert_array_equal(total_ctrl, [[[1, 0, 0]]])


def test_turnback_bundle_export_records_inner_and_outer_radius_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "analyze",
        SimpleNamespace(bucketLenForType=lambda _metric: (10.0, None)),
    )
    ratio = np.asarray([[[1.0, 0.5], [0.0, np.nan]]], dtype=float)
    total = np.asarray([[[3, 2], [1, 0]]], dtype=int)
    va = SimpleNamespace(
        fn="fake-video",
        f=7,
        _skipped=False,
        noyc=False,
        trns=[_CircleTraining(start=0, stop=10)],
        reward_turnback_dual_circle_counts={"ratio": ratio, "total": total},
        _numRewardsMsg=lambda *_args, **_kwargs: 5,
        _syncBucket=lambda _trn, _df: (0, 2, None),
    )
    opts = SimpleNamespace(
        export_group_label="Intact Control>Kir",
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=1,
        sli_select_keep_first_sync_buckets=0,
        turnback_inner_delta_mm=4.0,
        turnback_outer_delta_mm=8.0,
        turnback_inner_radius_offset_px=0.25,
        min_turnback_episodes=1,
    )
    out = tmp_path / "turnback_bundle.npz"

    export_turnback_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        assert float(bundle["turnback_inner_delta_mm"]) == 4.0
        assert float(bundle["turnback_outer_delta_mm"]) == 8.0
        assert float(bundle["turnback_inner_radius_offset_px"]) == 0.25
        assert int(bundle["episode_filter_turnback_sync_exp_min_episodes"]) == 1
        assert int(bundle["episode_filter_turnback_sync_exp_unit_count"]) == 2
        assert int(bundle["episode_filter_turnback_sync_exp_included_count"]) == 2
        assert int(bundle["episode_filter_turnback_sync_exp_excluded_count"]) == 0
        np.testing.assert_array_equal(
            bundle["episode_filter_turnback_sync_exp_episode_counts"], [3, 2]
        )
        assert int(bundle["episode_filter_turnback_sync_ctrl_included_count"]) == 1
        assert int(bundle["episode_filter_turnback_sync_ctrl_excluded_count"]) == 1
        np.testing.assert_array_equal(
            bundle["episode_filter_turnback_sync_ctrl_excluded_episode_counts"], [0]
        )


def test_turnback_bundle_export_applies_exp_pi_threshold_filter(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "analyze",
        SimpleNamespace(bucketLenForType=lambda _metric: (10.0, None)),
    )
    monkeypatch.setattr(
        "src.exporting.turnback_sli_bundle._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.1], dtype=float),
            np.asarray([[[0.1]]], dtype=float),
        ),
    )
    ratio = np.asarray([[[1.0], [np.nan]]], dtype=float)
    total = np.asarray([[[2], [0]]], dtype=int)
    va = SimpleNamespace(
        fn="fake-video",
        f=7,
        _skipped=False,
        noyc=True,
        trns=[_CircleTraining(start=0, stop=10)],
        reward_turnback_dual_circle_counts={"ratio": ratio, "total": total},
        reward_exclusion_mask=[[[True]]],
        _numRewardsMsg=lambda *_args, **_kwargs: 5,
        _syncBucket=lambda _trn, _df: (0, 1, None),
    )
    opts = SimpleNamespace(
        export_group_label="Intact Control>Kir",
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=0,
        sli_select_keep_first_sync_buckets=0,
        turnback_inner_delta_mm=4.0,
        turnback_outer_delta_mm=8.0,
        turnback_inner_radius_offset_px=0.0,
        min_turnback_episodes=1,
        require_exp_pi_threshold_bucket=True,
        exp_pi_threshold_filter_training=1,
        exp_pi_threshold_filter_sync_bucket=1,
        piTh=10,
    )
    out = tmp_path / "turnback_bundle.npz"

    export_turnback_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        np.testing.assert_array_equal(
            bundle["exp_pi_threshold_filter_eligible"], [False]
        )
        np.testing.assert_array_equal(
            bundle["exp_pi_threshold_filter_reason"], ["pi_threshold_failed"]
        )
        assert np.isnan(bundle["sli"]).all()
        assert np.isnan(bundle["sli_ts"]).all()
        assert np.isnan(bundle["turnback_ratio_exp"]).all()
        np.testing.assert_array_equal(bundle["turnback_total_exp"], [[[2]]])

    loaded = load_sli_bundle(str(out))
    assert np.isnan(loaded["turnback_ratio_exp"]).all()
    np.testing.assert_array_equal(loaded["exp_pi_threshold_filter_eligible"], [False])
