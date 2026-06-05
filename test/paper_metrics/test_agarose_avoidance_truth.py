import sys
from types import SimpleNamespace

import numpy as np

from src.analysis.trajectory import Trajectory
from src.analysis.video_analysis import VideoAnalysis
from src.exporting.agarose_sli_bundle import (
    _extract_agarose_arrays,
    export_agarose_sli_bundle,
)
from src.utils.common import CT


class _IdentityXformer:
    fctr = 1.0
    frameSize = (244, 244)

    def t2f(self, x, y, **_kwargs):
        return x, y


class _Training:
    def __init__(self, *, start, stop, name):
        self.start = int(start)
        self.stop = int(stop)
        self._name = name

    def name(self):
        return self._name


class _TrajectoryEpisodes:
    def __init__(self, episodes, *, bad=False):
        self._episodes = list(episodes)
        self._bad = bool(bad)
        self.calls = []

    def calc_agarose_dual_circle_episodes(self, **kwargs):
        self.calls.append(kwargs)
        self.agarose_dual_circle_episodes = list(self._episodes)


def _trajectory_at_well_distances(distances_px):
    xf = _IdentityXformer()
    _radius, centers = CT.large.arenaWells(xf, 0)
    cx, cy = centers[0]
    trj = object.__new__(Trajectory)
    trj.x = cx + np.asarray(distances_px, dtype=float)
    trj.y = np.full_like(trj.x, cy)
    trj.f = 0
    trj.va = SimpleNamespace(ct=CT.large, xf=xf, trxf={0: None})
    return trj


def _video_analysis_stub(**attrs):
    va = object.__new__(VideoAnalysis)
    for name, value in attrs.items():
        setattr(va, name, value)
    return va


def test_agarose_dual_circle_classifies_outer_only_and_inner_contact_episodes():
    trj = _trajectory_at_well_distances(
        [40.0, 32.0, 33.0, 40.0, 32.0, 20.0, 32.0, 40.0]
    )

    trj.calc_agarose_dual_circle_episodes(delta_mm=1.0)

    assert [
        (ep["start"], ep["stop"], ep["avoids_inner"], ep["entered_inner_frame"])
        for ep in trj.agarose_dual_circle_episodes
    ] == [(1, 3, True, None), (4, 7, False, 5)]


def test_agarose_ratio_uses_avoid_over_total_and_min_total_masks_ratio():
    exp = _TrajectoryEpisodes(
        [
            {"start": 1, "stop": 4, "avoids_inner": True},
            {"start": 6, "stop": 8, "avoids_inner": False},
            {"start": 7, "stop": 9, "avoids_inner": True},
        ]
    )
    va = _video_analysis_stub(
        ct=CT.large,
        opts=SimpleNamespace(
            agarose_outer_delta_mm=1.0,
            min_agarose_episodes=2,
            agarose_dual_circle_min_total=1,
            agarose_dual_circle_debug_csv=None,
        ),
        sync_bucket_ranges=[[(0, 5), (5, 10)]],
        trx=[exp],
        trns=[_Training(start=10, stop=20, name="T1")],
        startPre=0,
        _min2f=lambda minutes: int(minutes),
        _f2min=lambda frames: float(frames),
    )

    VideoAnalysis.analyzeAgaroseDualCircleAvoidance(va)

    counts = va.agarose_dual_circle_counts
    np.testing.assert_array_equal(counts["avoid"], [[[1, 1]]])
    np.testing.assert_array_equal(counts["total"], [[[1, 2]]])
    np.testing.assert_allclose(counts["ratio"], [[[np.nan, 0.5]]])
    assert exp.calls[0]["delta_mm"] == 1.0


def test_agarose_episode_assignment_uses_episode_start_frame_for_sync_bucket():
    exp = _TrajectoryEpisodes(
        [
            {"start": 4, "stop": 8, "avoids_inner": True},
            {"start": 5, "stop": 6, "avoids_inner": False},
            {"start": 10, "stop": 12, "avoids_inner": True},
        ]
    )

    va = _video_analysis_stub(
        ct=CT.large,
        opts=SimpleNamespace(
            agarose_outer_delta_mm=1.0,
            min_agarose_episodes=1,
            agarose_dual_circle_debug_csv=None,
        ),
        sync_bucket_ranges=[[(0, 5), (5, 10)], [(10, 15)]],
        trx=[exp],
        trns=[
            _Training(start=10, stop=20, name="T1"),
            _Training(start=30, stop=40, name="T2"),
        ],
        startPre=0,
        _min2f=lambda minutes: int(minutes),
        _f2min=lambda frames: float(frames),
    )

    VideoAnalysis.analyzeAgaroseDualCircleAvoidance(va)

    counts = va.agarose_dual_circle_counts
    np.testing.assert_array_equal(counts["avoid"], [[[1, 0]], [[1, 0]]])
    np.testing.assert_array_equal(counts["total"], [[[1, 1]], [[1, 0]]])
    np.testing.assert_allclose(counts["ratio"], [[[1.0, 0.0]], [[1.0, np.nan]]])


def test_agarose_pre_windows_use_episode_start_frame():
    episodes = [
        {"start": 9, "stop": 12, "avoids_inner": True},
        {"start": 10, "stop": 12, "avoids_inner": False},
        {"start": 29, "stop": 31, "avoids_inner": False},
    ]
    va = _video_analysis_stub(
        opts=SimpleNamespace(min_agarose_episodes=1),
        trx=[SimpleNamespace(_bad=False)],
        trns=[
            _Training(start=10, stop=20, name="T1"),
            _Training(start=30, stop=40, name="T2"),
        ],
        startPre=0,
        _min2f=lambda minutes: int(minutes),
        _f2min=lambda frames: float(frames),
    )

    pre = VideoAnalysis._calcAgaroseDualCirclePreCounts(va, [episodes])
    training_pre = VideoAnalysis._calcAgaroseDualCircleTrainingPreCounts(va, [episodes])

    np.testing.assert_array_equal(pre["avoid"], [1])
    np.testing.assert_array_equal(pre["total"], [1])
    np.testing.assert_allclose(pre["ratio"], [1.0])
    np.testing.assert_array_equal(training_pre["avoid"], [[1], [0]])
    np.testing.assert_array_equal(training_pre["total"], [[1], [1]])
    np.testing.assert_allclose(training_pre["ratio"], [[1.0], [0.0]])


def test_agarose_bundle_extraction_keeps_exp_ctrl_axes_and_counts():
    ratio = np.asarray([[[1.0, 0.5], [0.0, np.nan]]], dtype=float)
    total = np.asarray([[[3, 2], [1, 0]]], dtype=int)
    avoid = np.asarray([[[3, 1], [0, 0]]], dtype=int)
    va = _video_analysis_stub(
        fn="fake-video",
        trns=[_Training(start=0, stop=10, name="T1")],
        agarose_dual_circle_counts={"ratio": ratio, "total": total, "avoid": avoid},
        _numRewardsMsg=lambda *args, **_kwargs: 5,
        _syncBucket=lambda _trn, _df: (0, 3, None),
    )

    ratio_exp, ratio_ctrl, total_exp, total_ctrl, avoid_exp, avoid_ctrl = (
        _extract_agarose_arrays([va])
    )

    np.testing.assert_allclose(ratio_exp, [[[1.0, 0.5, np.nan]]])
    np.testing.assert_allclose(ratio_ctrl, [[[0.0, np.nan, np.nan]]])
    np.testing.assert_array_equal(total_exp, [[[3, 2, 0]]])
    np.testing.assert_array_equal(total_ctrl, [[[1, 0, 0]]])
    np.testing.assert_array_equal(avoid_exp, [[[3, 1, 0]]])
    np.testing.assert_array_equal(avoid_ctrl, [[[0, 0, 0]]])


def test_agarose_bundle_export_records_min_total_metadata(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "analyze",
        SimpleNamespace(bucketLenForType=lambda _metric: (10.0, None)),
    )
    ratio = np.asarray([[[1.0, np.nan], [0.0, np.nan]]], dtype=float)
    total = np.asarray([[[2, 1], [2, 0]]], dtype=int)
    avoid = np.asarray([[[2, 1], [0, 0]]], dtype=int)
    va = _video_analysis_stub(
        fn="fake-video",
        f=7,
        _skipped=False,
        noyc=False,
        trns=[_Training(start=0, stop=10, name="T1")],
        agarose_dual_circle_counts={"ratio": ratio, "total": total, "avoid": avoid},
        _numRewardsMsg=lambda *_args, **_kwargs: 5,
        _syncBucket=lambda _trn, _df: (0, 2, None),
    )
    opts = SimpleNamespace(
        export_group_label="Intact Control>Kir",
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=0,
        sli_select_keep_first_sync_buckets=0,
        min_agarose_episodes=2,
        agarose_dual_circle_min_total=1,
        agarose_sli_include_pre=False,
    )
    out = tmp_path / "agarose_bundle.npz"

    export_agarose_sli_bundle([va], opts, gls=None, out_fn=str(out))

    with np.load(out, allow_pickle=True) as bundle:
        assert int(bundle["min_agarose_episodes"]) == 2
        assert int(bundle["agarose_dual_circle_min_total"]) == 2
        assert int(bundle["episode_filter_agarose_sync_exp_min_episodes"]) == 2
        assert int(bundle["episode_filter_agarose_sync_exp_unit_count"]) == 2
        assert int(bundle["episode_filter_agarose_sync_exp_included_count"]) == 1
        assert int(bundle["episode_filter_agarose_sync_exp_excluded_count"]) == 1
        np.testing.assert_array_equal(
            bundle["episode_filter_agarose_sync_exp_episode_counts"], [2, 1]
        )
        np.testing.assert_array_equal(
            bundle["episode_filter_agarose_sync_exp_excluded_episode_counts"], [1]
        )
