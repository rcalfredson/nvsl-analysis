from types import SimpleNamespace

import numpy as np

from src.plotting.between_reward_tortuosity_mean_swarm import (
    BetweenRewardTortuosityMeanSwarmConfig,
    BetweenRewardTortuosityMeanSwarmPlotter,
)


class _Trajectory:
    def __init__(self):
        self.x = np.arange(12, dtype=float)
        self.y = np.zeros_like(self.x)
        self.d = np.ones(11, dtype=float)
        self.pxPerMmFloor = 1.0

    def bad(self):
        return False


class _Training:
    def __init__(self, n, label):
        self.n = int(n)
        self.start = 0
        self.stop = 12
        self._label = str(label)

    def name(self):
        return self._label

    def circles(self, _fly_idx):
        return [(0.0, 0.0, 0.0)]


class _Video:
    def __init__(self):
        self.fn = "fake-tortuosity-video"
        self.f = 7
        self.trns = [_Training(1, "T1"), _Training(2, "T2")]
        self.trx = [_Trajectory()]
        self.flies = [0]
        self.noyc = True
        self._skipped = False
        self.xf = SimpleNamespace(fctr=1.0)
        self.reward_exclusion_mask = [[[False, False, False, False]]]
        self.sync_bucket_ranges = [
            [(0, 3), (3, 6), (6, 9), (9, 12)],
            [(0, 3), (3, 6), (6, 9), (9, 12)],
        ]

    def _iter_between_reward_segment_com(self, _trn, _fly_idx, **_kwargs):
        yield SimpleNamespace(s=0, e=3, b_idx=0)
        yield SimpleNamespace(s=3, e=6, b_idx=1)
        yield SimpleNamespace(s=6, e=9, b_idx=2)


def _plotter(*, pool_trainings):
    opts = SimpleNamespace(min_between_reward_trajectories=4)
    cfg = BetweenRewardTortuosityMeanSwarmConfig(
        out_file="unused.png",
        pool_trainings=bool(pool_trainings),
        trainings=[1, 2] if pool_trainings else [1],
        skip_first_sync_buckets=0,
        keep_first_sync_buckets=0,
        metric_mode="path_over_displacement",
        segment_scope="full",
        min_segments_per_fly=1,
    )
    return BetweenRewardTortuosityMeanSwarmPlotter(
        vas=[_Video()], opts=opts, gls=None, customizer=None, cfg=cfg
    )


def test_tortuosity_mean_swarm_single_training_filter_uses_segment_count():
    data = _plotter(pool_trainings=False).compute_scalar_panels()

    assert data["panel_labels"] == ["T1"]
    assert int(data["n_units_panel"][0]) == 0
    assert data["per_unit_values_panel"][0].size == 0


def test_tortuosity_mean_swarm_filter_after_pooling_selected_segments():
    data = _plotter(pool_trainings=True).compute_scalar_panels()

    assert data["panel_labels"] == ["Selected trainings combined"]
    assert int(data["n_units_panel"][0]) == 1
    np.testing.assert_allclose(
        np.asarray(data["per_unit_values_panel"][0], dtype=float), [1.0]
    )
    assert data["meta"]["training_selection"]["trainings_effective"] == [1, 2]
    assert data["meta"]["min_segments_per_fly"] == 4


def test_tortuosity_mean_swarm_applies_exp_pi_threshold_filter():
    va = _Video()
    va.reward_exclusion_mask = [[[True, False, False, False]]]
    opts = SimpleNamespace(
        min_between_reward_trajectories=1,
        require_exp_pi_threshold_bucket=True,
        exp_pi_threshold_filter_training=1,
        exp_pi_threshold_filter_sync_bucket=1,
        piTh=10,
    )
    cfg = BetweenRewardTortuosityMeanSwarmConfig(
        out_file="unused.png",
        pool_trainings=False,
        trainings=[1],
        skip_first_sync_buckets=0,
        keep_first_sync_buckets=0,
        metric_mode="path_over_displacement",
        segment_scope="full",
        min_segments_per_fly=1,
    )
    plotter = BetweenRewardTortuosityMeanSwarmPlotter(
        vas=[va], opts=opts, gls=None, customizer=None, cfg=cfg
    )

    data = plotter.compute_scalar_panels()

    assert data["panel_labels"] == []
    assert data["per_unit_values_panel"].size == 0
