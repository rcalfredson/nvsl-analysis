from types import SimpleNamespace

import numpy as np
import pytest

import src.plotting.cross_fly_correlations as corr
from src.plotting.cross_fly_correlations import SLIContext
from src.plotting.rewards_per_distance_totals import (
    RewardsPerDistanceTotalsConfig,
    RewardsPerDistanceTotalsPlotter,
)


class _Trajectory:
    def __init__(self, distances, *, x=None, y=None):
        self._distances = np.asarray(distances, dtype=float)
        self.x = np.asarray(x if x is not None else np.zeros(len(distances)), dtype=float)
        self.y = np.asarray(y if y is not None else np.zeros(len(distances)), dtype=float)

    def bad(self):
        return False

    def distTrav(self, start, stop):
        return float(np.sum(self._distances[int(start) : int(stop)]))


class _Training:
    start = 0
    stop = 20

    def name(self):
        return "training 1"

    def circles(self, _f=0):
        return [(0.0, 0.0, 1.0)]


class _VideoAnalysis:
    def __init__(self, trajectory, reward_frames=(), excluded_buckets=()):
        self.trns = [_Training()]
        self.trx = [trajectory]
        self.sync_bucket_ranges = [[(0, 10), (10, 20)]]
        self.xf = SimpleNamespace(fctr=1.0)
        self.ct = SimpleNamespace(pxPerMmFloor=lambda: 1.0)
        self._reward_frames = np.asarray(reward_frames, dtype=int)
        self._excluded_buckets = set(excluded_buckets)

    def is_excluded_pair(self, _f, _training_idx, bucket_idx):
        return int(bucket_idx) in self._excluded_buckets

    def _idxSync(self, _sync_type, _trn, start, _stop):
        return int(start)

    def _countOn(self, start, stop, *, calc, ctrl, f):
        assert calc and not ctrl and f == 0
        return int(
            np.count_nonzero(
                (self._reward_frames >= int(start))
                & (self._reward_frames < int(stop))
            )
        )


def test_pooled_rpd_uses_total_rewards_over_total_distance():
    trajectory = _Trajectory([9.0] * 10 + [1.0] * 10)
    va = _VideoAnalysis(
        trajectory,
        reward_frames=[5, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    )
    ctx = SLIContext(
        training_idx=0,
        average_over_buckets=True,
        keep_first_sync_buckets=2,
    )

    pooled = corr._pooled_rewards_per_distance_for_context(va, ctx=ctx, f=0)
    bucketwise_mean = np.mean([1.0 / 0.09, 9.0 / 0.01])

    assert pooled == pytest.approx(10.0 / 0.1)
    assert pooled != pytest.approx(bucketwise_mean)


def test_window_rpd_uses_pooled_reward_minimum_not_per_bucket_masks():
    trajectory = _Trajectory(np.ones(20))
    va = _VideoAnalysis(
        trajectory,
        reward_frames=[1, 2, 11, 12, 13],
        excluded_buckets=[0],
    )
    ctx = SLIContext(
        training_idx=0,
        average_over_buckets=True,
        keep_first_sync_buckets=2,
    )

    window_value = corr._pooled_rewards_per_distance_for_context(
        va,
        ctx=ctx,
        f=0,
        validity_policy="window",
        min_rewards=5,
    )
    all_buckets_value = corr._pooled_rewards_per_distance_for_context(
        va,
        ctx=ctx,
        f=0,
        validity_policy="all-buckets",
        min_rewards=5,
    )

    assert window_value == pytest.approx(5.0 / 0.02)
    assert np.isnan(all_buckets_value)


def test_window_rpd_rejects_fewer_than_five_pooled_rewards():
    trajectory = _Trajectory(np.ones(20))
    va = _VideoAnalysis(trajectory, reward_frames=[1, 2, 11, 12])
    ctx = SLIContext(
        training_idx=0,
        average_over_buckets=True,
        keep_first_sync_buckets=2,
    )

    value = corr._pooled_rewards_per_distance_for_context(
        va,
        ctx=ctx,
        f=0,
    )

    assert np.isnan(value)


def test_rpd_total_collector_uses_the_same_window_validity_policy():
    trajectory = _Trajectory(np.ones(20))
    va = _VideoAnalysis(
        trajectory,
        reward_frames=[1, 2, 11, 12, 13],
        excluded_buckets=[0],
    )

    def _data(policy):
        cfg = RewardsPerDistanceTotalsConfig(
            out_file="",
            trainings=(1,),
            keep_first_sync_buckets=2,
            validity_policy=policy,
            min_rewards=5,
        )
        return RewardsPerDistanceTotalsPlotter(
            vas=[va],
            opts=SimpleNamespace(),
            gls=None,
            customizer=None,
            cfg=cfg,
        ).compute_scalar_panels()

    window_data = _data("window")
    all_buckets_data = _data("all-buckets")

    assert window_data["mean"] == pytest.approx([5.0 / 0.02])
    assert all_buckets_data["panel_labels"] == []


def test_pooled_speed_weights_bucket_means_by_valid_frame_count(monkeypatch):
    arrays = {
        "speed_exp": np.asarray([[[1.0, 9.0]]]),
        "speedN_exp": np.asarray([[[9, 1]]]),
    }
    monkeypatch.setattr(corr, "_extract_speed_arrays", lambda _vas, _opts: arrays)
    ctx = SLIContext(
        training_idx=0,
        average_over_buckets=True,
        keep_first_sync_buckets=2,
    )

    pooled = corr._extract_exp_speed_for_context(
        [object()],
        SimpleNamespace(),
        ctx,
        aggregation="pooled",
    )
    bucketwise = corr._extract_exp_speed_for_context(
        [object()],
        SimpleNamespace(),
        ctx,
        aggregation="bucketwise",
    )

    assert pooled == pytest.approx([1.8])
    assert bucketwise == pytest.approx([5.0])


def test_pooled_median_distance_uses_all_selected_frames():
    xs = [0.0] * 9 + [100.0] + [50.0] + [np.nan] * 9
    trajectory = _Trajectory(np.ones(20), x=xs, y=np.zeros(20))
    va = _VideoAnalysis(trajectory)
    ctx = SLIContext(
        training_idx=0,
        average_over_buckets=True,
        keep_first_sync_buckets=2,
    )

    pooled = corr._pooled_median_distance_for_context(va, ctx)
    legacy_median_of_bucket_medians = np.median([0.0, 50.0])

    assert pooled == pytest.approx(0.0)
    assert pooled != pytest.approx(legacy_median_of_bucket_medians)


def test_window_aggregation_defaults_to_pooled_and_supports_legacy_mode():
    assert corr._window_aggregation_mode(SimpleNamespace()) == "pooled"
    assert (
        corr._window_aggregation_mode(
            SimpleNamespace(corr_window_metric_aggregation="bucketwise")
        )
        == "bucketwise"
    )
