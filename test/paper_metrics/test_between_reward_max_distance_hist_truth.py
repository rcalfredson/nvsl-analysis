from types import SimpleNamespace

import numpy as np

from src.plotting.between_reward_max_distance_hist import (
    DEFAULT_MAX_DISTANCE_BIN_EDGES_MM,
    BetweenRewardMaxDistanceHistogramConfig,
    BetweenRewardMaxDistanceHistogramPlotter,
)


class _Training:
    def __init__(self, idx):
        self.idx = idx
        self.start = idx * 100
        self.stop = self.start + 100

    def name(self):
        return f"T{self.idx + 1}"


class _Trajectory:
    def bad(self):
        return False


class _Video:
    def __init__(self, t2_bucket_count=5):
        self.fn = "video.avi"
        self.f = 0
        self.flies = [0]
        self.noyc = True
        self.trx = [_Trajectory()]
        self.trns = [_Training(0), _Training(1)]
        self.sync_bucket_ranges = [
            [(i * 20, (i + 1) * 20) for i in range(5)],
            [(100 + i * 20, 100 + (i + 1) * 20) for i in range(t2_bucket_count)],
        ]

    def _iter_between_reward_segment_com(self, trn, _fly_idx, **_kwargs):
        values = [1.0, 6.0] if trn.idx == 0 else [1.0, 7.0, 12.0, 18.0, 35.0]
        for value in values:
            yield SimpleNamespace(max_d_mm=value)


def _plotter(va, *, require_bucket=True):
    opts = SimpleNamespace(
        min_between_reward_trajectories=5,
        require_exp_target_sync_bucket=require_bucket,
        skip_first_sync_buckets=0,
    )
    cfg = BetweenRewardMaxDistanceHistogramConfig(
        out_file="unused.png",
        bin_edges=DEFAULT_MAX_DISTANCE_BIN_EDGES_MM,
        normalize=True,
        per_fly=True,
        min_segs_per_fly=0,
        ci=True,
        trainings=[2],
    )
    return BetweenRewardMaxDistanceHistogramPlotter(
        vas=[va],
        opts=opts,
        gls=[],
        customizer=SimpleNamespace(),
        cfg=cfg,
    )


def test_max_distance_histogram_defaults_to_five_mm_edges():
    np.testing.assert_allclose(
        DEFAULT_MAX_DISTANCE_BIN_EDGES_MM,
        [0, 5, 10, 15, 20, 25, 30],
    )


def test_five_trajectory_gate_is_applied_before_histogram_range_clipping():
    data = _plotter(_Video()).compute_histograms()

    assert data["panel_labels"] == ["T2"]
    assert data["n_units_panel"].tolist() == [1]
    assert data["n_raw"].tolist() == [5]
    assert data["n_used"].tolist() == [4]
    np.testing.assert_allclose(data["mean"][0], [0.25, 0.25, 0.25, 0.25, 0, 0])
    assert data["meta"]["min_trajectories_per_fly_window"] == 5
    assert data["meta"]["min_trajectories_applied_before_range_clipping"] is True


def test_t2_sync_bucket_five_filter_excludes_ineligible_fly():
    data = _plotter(_Video(t2_bucket_count=4)).compute_histograms()

    assert data["panel_labels"] == []
    assert data["meta"]["exp_target_sync_bucket_filter_eligible_count"] == 0


def test_filter_metadata_excludes_skipped_videos():
    skipped = _Video()
    skipped._skipped = True
    plotter = _plotter(_Video())
    plotter.vas.append(skipped)

    data = plotter.compute_histograms()

    assert data["meta"]["exp_target_sync_bucket_filter_total_count"] == 1
