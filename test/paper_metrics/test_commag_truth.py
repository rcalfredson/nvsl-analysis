import sys
from types import SimpleNamespace

import numpy as np

from src.analysis.video_analysis import VideoAnalysis
from src.exporting.com_sli_bundle import _extract_commag_arrays, build_com_sli_bundle


class _Trajectory:
    def __init__(self, x, y, *, bad=False, f=0):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self._bad = bool(bad)
        self.f = int(f)


class _Training:
    start = 0

    def __init__(self, *, stop=20, center=(10.0, 20.0)):
        self.stop = int(stop)
        self._center = tuple(center)

    def name(self):
        return "T1"

    def circles(self, _fly_idx):
        cx, cy = self._center
        return [(float(cx), float(cy), 5.0)]


class _COMVideo:
    def __init__(self, *, trx, on_by_fly, df=5, n_buckets=1, opts=None):
        self.trx = list(trx)
        self.trns = [_Training()]
        self.flies = list(range(len(self.trx)))
        self.opts = opts or SimpleNamespace(
            com_exclude_wall_contact=False,
            com_sli_debug=False,
            btw_rwd_com_exclude_reward_endpoints=False,
            btw_rwd_sync_bucket_min_trajectories=0,
        )
        self.xf = SimpleNamespace(fctr=1.0)
        self.ct = SimpleNamespace(pxPerMmFloor=lambda: 2.0)
        self._on_by_fly = {
            int(k): np.asarray(v, dtype=int) for k, v in on_by_fly.items()
        }
        self._df = int(df)
        self._n_buckets = int(n_buckets)

    def _getOn(self, _trn, _include_ctrl, f=0):
        return self._on_by_fly.get(int(f))

    def _numRewardsMsg(self, *_args, **_kwargs):
        return self._df

    def _syncBucket(self, _trn, _df):
        return 0, self._n_buckets, None

    def _iter_between_reward_segment_com(self, trn, fly_idx, **kwargs):
        yield from VideoAnalysis._iter_between_reward_segment_com(
            self, trn, fly_idx, **kwargs
        )


def _iter_segments(va, **kwargs):
    defaults = dict(
        fi=0,
        df=va._df,
        n_buckets=va._n_buckets,
        complete=[True] * va._n_buckets,
        relative_to_reward=True,
        per_segment_min_meddist_mm=0.0,
        exclude_wall=False,
    )
    defaults.update(kwargs)
    return list(
        VideoAnalysis._iter_between_reward_segment_com(va, va.trns[0], 0, **defaults)
    )


def test_commag_segment_is_distance_of_mean_segment_position_from_reward_center():
    va = _COMVideo(
        trx=[_Trajectory(x=[12, 14, 16], y=[20, 22, 24])], on_by_fly={0: [0, 3]}
    )

    [seg] = _iter_segments(va)

    assert seg.b_idx == 0
    assert seg.s == 0
    assert seg.e == 3
    assert seg.mx_mm == 2.0
    assert seg.my_mm == 1.0
    np.testing.assert_allclose(seg.mag_mm, np.sqrt(5.0))


def test_commag_segment_assignment_uses_between_reward_start_frame():
    va = _COMVideo(
        trx=[_Trajectory(x=np.arange(9), y=np.full(9, 20.0))],
        on_by_fly={0: [0, 5, 8]},
        df=5,
        n_buckets=2,
    )

    segments = _iter_segments(va, complete=[True, True])

    assert [(seg.s, seg.e, seg.b_idx) for seg in segments] == [(0, 5, 0), (5, 8, 1)]


def test_commag_endpoint_exclusion_removes_both_reward_endpoint_frames():
    va = _COMVideo(
        trx=[_Trajectory(x=[100, 14, 18], y=[100, 20, 20])], on_by_fly={0: [0, 3]}
    )

    [seg] = _iter_segments(va, exclude_reward_endpoints=True)

    assert seg.mx_mm == 2.0
    assert seg.my_mm == 0.0
    assert seg.mag_mm == 2.0


def test_commag_segment_filters_nonfinite_coordinates_meddist_and_wall_contact():
    finite_va = _COMVideo(
        trx=[_Trajectory(x=[12, np.nan, 16], y=[20, 20, 24])], on_by_fly={0: [0, 3]}
    )
    [seg] = _iter_segments(finite_va)
    assert seg.mx_mm == 2.0
    assert seg.my_mm == 1.0

    meddist_va = _COMVideo(
        trx=[_Trajectory(x=[11, 11, 11], y=[20, 20, 20])], on_by_fly={0: [0, 3]}
    )
    [skip] = _iter_segments(
        meddist_va, per_segment_min_meddist_mm=1.0, yield_skips=True
    )
    assert skip.why == "meddist_filtered"

    wall_va = _COMVideo(
        trx=[_Trajectory(x=[12, 14, 16], y=[20, 22, 24])], on_by_fly={0: [0, 3]}
    )
    [skip] = _iter_segments(
        wall_va,
        exclude_wall=True,
        wc=np.asarray([False, True, False]),
        yield_skips=True,
    )
    assert skip.why == "wall_contact"


def test_by_sync_bucket_com_averages_segment_magnitudes_not_component_mean_vector():
    va = _COMVideo(
        trx=[
            _Trajectory(x=[12, 12, 8, 8, 16, 16], y=[20, 20, 20, 20, 20, 20]),
            _Trajectory(x=[10, 10, 10, 10, 10, 10], y=[22, 22, 22, 22, 22, 22]),
        ],
        on_by_fly={0: [0, 2, 4], 1: [0, 2, 4]},
        df=5,
        n_buckets=1,
    )

    VideoAnalysis.bySyncBucketCOM(va, relative_to_reward=True, store_mag=True)

    assert va.syncCOMMagN[0]["exp"] == [2]
    assert va.syncCOMMag[0]["exp"] == [1.0]
    assert va.syncCOMMag[0]["ctrl"] == [1.0]


def test_extract_commag_arrays_applies_min_segment_count_masking():
    va = SimpleNamespace(
        syncCOMMag=[{"exp": [1.0, 2.0], "ctrl": [3.0, 4.0]}],
        syncCOMMagN=[{"exp": [2, 1], "ctrl": [0, 3]}],
    )

    exp, ctrl, n_exp, n_ctrl = _extract_commag_arrays(
        [va], SimpleNamespace(btw_rwd_sync_bucket_min_trajectories=2)
    )

    np.testing.assert_allclose(exp, [[[1.0, np.nan]]])
    np.testing.assert_allclose(ctrl, [[[np.nan, 4.0]]])
    np.testing.assert_array_equal(n_exp, [[[2, 1]]])
    np.testing.assert_array_equal(n_ctrl, [[[0, 3]]])


def test_extract_commag_arrays_prefers_episode_threshold_option():
    va = SimpleNamespace(
        syncCOMMag=[{"exp": [1.0, 2.0], "ctrl": [3.0, 4.0]}],
        syncCOMMagN=[{"exp": [2, 1], "ctrl": [1, 3]}],
    )

    exp, ctrl, n_exp, n_ctrl = _extract_commag_arrays(
        [va],
        SimpleNamespace(
            min_between_reward_trajectories=2,
            btw_rwd_sync_bucket_min_trajectories=99,
        ),
    )

    np.testing.assert_allclose(exp, [[[1.0, np.nan]]])
    np.testing.assert_allclose(ctrl, [[[np.nan, 4.0]]])
    np.testing.assert_array_equal(n_exp, [[[2, 1]]])
    np.testing.assert_array_equal(n_ctrl, [[[1, 3]]])


def test_com_bundle_applies_exp_pi_threshold_filter(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "analyze",
        SimpleNamespace(bucketLenForType=lambda _metric: (10.0, None)),
    )
    monkeypatch.setattr(
        "src.exporting.com_sli_bundle._compute_sli_scalar_and_timeseries_from_rpid",
        lambda _vas, _opts: (
            np.asarray([0.1], dtype=float),
            np.asarray([[[0.1, 0.2]]], dtype=float),
        ),
    )
    va = SimpleNamespace(
        _skipped=False,
        syncCOMMag=[{"exp": [1.0, 2.0], "ctrl": [3.0, 4.0]}],
        syncCOMMagN=[{"exp": [2, 2], "ctrl": [2, 2]}],
        trns=[_Training()],
        fn="fake-video",
        reward_exclusion_mask=[[[True]]],
    )
    opts = SimpleNamespace(
        export_group_label="group",
        best_worst_trn=1,
        sli_use_training_mean=True,
        sli_select_skip_first_sync_buckets=0,
        sli_select_keep_first_sync_buckets=0,
        min_between_reward_trajectories=1,
        require_exp_target_sync_bucket=True,
        exp_target_sync_bucket_filter_training=1,
        exp_target_sync_bucket_filter_sync_bucket=1,
        piTh=10,
        com_sli_debug=False,
    )

    bundle = build_com_sli_bundle([va], opts, gls=None)

    assert np.isnan(bundle["sli"]).all()
    assert np.isnan(bundle["sli_ts"]).all()
    assert np.isnan(bundle["commag_exp"]).all()
    np.testing.assert_allclose(bundle["commag_ctrl"], [[[3.0, 4.0]]])
    np.testing.assert_array_equal(bundle["commagN_exp"], [[[2, 2]]])
    np.testing.assert_array_equal(bundle["exp_pi_threshold_filter_eligible"], [False])
    np.testing.assert_array_equal(
        bundle["exp_pi_threshold_filter_reason"], ["pi_threshold_failed"]
    )
