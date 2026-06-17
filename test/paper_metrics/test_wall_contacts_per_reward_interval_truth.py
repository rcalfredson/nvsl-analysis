import csv
from types import SimpleNamespace

import numpy as np

from src.plotting.wall_contacts_per_reward_interval_totals import (
    WallContactsPerRewardIntervalTotalsConfig,
    WallContactsPerRewardIntervalTotalsPlotter,
)


class _Region:
    def __init__(self, start):
        self.start = int(start)


class _Trajectory:
    def __init__(self, starts, *, f=0, bad=False):
        self.f = f
        self._bad = bad
        self.x = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
        self.y = np.zeros(8, dtype=float)
        self.d = np.ones(8, dtype=float)
        self.boundary_event_stats = {
            "wall": {
                "all": {
                    "edge": {
                        "boundary_contact_regions": [_Region(s) for s in starts]
                    }
                }
            }
        }

    def bad(self):
        return self._bad

    def distTrav(self, i1, i2):
        return float(np.sum(self.d[int(i1) : int(i2)]))


class _Training:
    def __init__(self, n, label):
        self.n = int(n)
        self.start = 0
        self.stop = 8
        self._label = str(label)

    def name(self):
        return self._label

    def circles(self, _f):
        return [(0.0, 0.0, 1.0)]


class _Transform:
    fctr = 1.0


class _Chamber:
    @staticmethod
    def pxPerMmFloor():
        return 1.0


class _Video:
    def __init__(self):
        self.fn = "fake-wall-contact-video"
        self.f = 7
        self.trns = [_Training(1, "T1"), _Training(2, "T2")]
        self.trx = [_Trajectory([1, 3, 5])]
        self.flies = [0]
        self.noyc = True
        self._skipped = False
        self.fps = 2.0
        self.xf = _Transform()
        self.ct = _Chamber()
        self.reward_exclusion_mask = [[[False, False, False, False]]]
        self.sync_bucket_ranges = [
            [(0, 2), (2, 4), (4, 6), (6, 8)],
            [(0, 2), (2, 4), (4, 6), (6, 8)],
        ]

    def _getOn(self, _trn, *, calc=False, f=0):
        return np.asarray([0, 2, 4, 6], dtype=int)


def test_wall_contact_scalar_bars_single_training_filter_uses_interval_count():
    opts = SimpleNamespace(min_between_reward_trajectories=4)
    cfg = WallContactsPerRewardIntervalTotalsConfig(
        out_file="unused.png",
        pool_trainings=False,
        trainings=[1],
        skip_first_sync_buckets=0,
        keep_first_sync_buckets=0,
    )
    plotter = WallContactsPerRewardIntervalTotalsPlotter(
        vas=[_Video()], opts=opts, gls=None, customizer=None, cfg=cfg
    )

    data = plotter.compute_scalar_panels()

    assert data["panel_labels"] == ["T1"]
    assert int(data["n_units_panel"][0]) == 0
    assert data["per_unit_values_panel"][0].size == 0


def test_wall_contact_scalar_bars_filter_after_pooling_selected_intervals():
    opts = SimpleNamespace(min_between_reward_trajectories=4)
    cfg = WallContactsPerRewardIntervalTotalsConfig(
        out_file="unused.png",
        pool_trainings=True,
        trainings=[1, 2],
        skip_first_sync_buckets=0,
        keep_first_sync_buckets=0,
    )
    plotter = WallContactsPerRewardIntervalTotalsPlotter(
        vas=[_Video()], opts=opts, gls=None, customizer=None, cfg=cfg
    )

    data = plotter.compute_scalar_panels()

    assert data["panel_labels"] == ["Selected trainings combined"]
    assert int(data["n_units_panel"][0]) == 1
    np.testing.assert_allclose(
        np.asarray(data["per_unit_values_panel"][0], dtype=float), [1.0]
    )
    assert data["meta"]["training_selection"]["trainings_effective"] == [1, 2]


def test_wall_contact_scalar_bars_apply_exp_target_sync_bucket_filter():
    va = _Video()
    va.reward_exclusion_mask = [[[True, False, False, False]]]
    va.sync_bucket_ranges[0] = []
    opts = SimpleNamespace(
        min_between_reward_trajectories=1,
        require_exp_target_sync_bucket=True,
        exp_target_sync_bucket_filter_training=1,
        exp_target_sync_bucket_filter_sync_bucket=1,
        piTh=10,
    )
    cfg = WallContactsPerRewardIntervalTotalsConfig(
        out_file="unused.png",
        pool_trainings=False,
        trainings=[1],
        skip_first_sync_buckets=0,
        keep_first_sync_buckets=0,
    )
    plotter = WallContactsPerRewardIntervalTotalsPlotter(
        vas=[va], opts=opts, gls=None, customizer=None, cfg=cfg
    )

    data = plotter.compute_scalar_panels()

    assert data["panel_labels"] == []
    assert data["per_unit_values_panel"].size == 0


def test_contactless_fraction_pools_intervals_before_dividing():
    va = _Video()
    va.trx = [_Trajectory([3])]
    opts = SimpleNamespace(min_between_reward_trajectories=5)
    cfg = WallContactsPerRewardIntervalTotalsConfig(
        out_file="unused.png",
        pool_trainings=False,
        trainings=[2],
        skip_first_sync_buckets=1,
        keep_first_sync_buckets=3,
        metric="contactless_fraction",
    )
    plotter = WallContactsPerRewardIntervalTotalsPlotter(
        vas=[va], opts=opts, gls=None, customizer=None, cfg=cfg
    )

    data = plotter.compute_scalar_panels()

    assert data["panel_labels"] == ["T2"]
    assert int(data["n_units_panel"][0]) == 0

    plotter.opts.min_between_reward_trajectories = 2
    data = plotter.compute_scalar_panels()
    np.testing.assert_allclose(
        np.asarray(data["per_unit_values_panel"][0], dtype=float), [0.5]
    )
    assert data["meta"]["y_label"] == (
        "Fraction of trajectories without wall contact"
    )


def test_wall_contact_trajectory_length_uses_contact_intervals_only():
    va = _Video()
    va.trx = [_Trajectory([3])]
    va.trx[0].d = np.asarray([1, 1, 10, 10, 100, 100, 1000, 1000], dtype=float)
    opts = SimpleNamespace(min_between_reward_trajectories=1)
    cfg = WallContactsPerRewardIntervalTotalsConfig(
        out_file="unused.png",
        pool_trainings=False,
        trainings=[2],
        skip_first_sync_buckets=1,
        keep_first_sync_buckets=3,
        metric="wall_contact_trajectory_length",
    )
    plotter = WallContactsPerRewardIntervalTotalsPlotter(
        vas=[va], opts=opts, gls=None, customizer=None, cfg=cfg
    )

    data = plotter.compute_scalar_panels()

    assert data["panel_labels"] == ["T2"]
    np.testing.assert_allclose(
        np.asarray(data["per_unit_values_panel"][0], dtype=float), [20.0]
    )
    assert data["meta"]["y_label"] == "Length of wall-contact trajectories (mm)"


def test_contactless_episode_csv_reconstructs_per_fly_fraction(tmp_path):
    va = _Video()
    va.trx = [_Trajectory([3])]
    opts = SimpleNamespace(
        min_between_reward_trajectories=2,
        export_group_label="Test group",
    )
    cfg = WallContactsPerRewardIntervalTotalsConfig(
        out_file="unused.png",
        pool_trainings=False,
        trainings=[2],
        skip_first_sync_buckets=1,
        keep_first_sync_buckets=3,
        metric="contactless_fraction",
    )
    plotter = WallContactsPerRewardIntervalTotalsPlotter(
        vas=[va], opts=opts, gls=None, customizer=None, cfg=cfg
    )
    out_csv = tmp_path / "episodes.csv"
    scalar_data = plotter.compute_scalar_panels()

    plotter.export_episode_csv(str(out_csv))

    with out_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {row["group"] for row in rows} == {"Test group"}
    assert [int(row["wall_contact_event_count"]) for row in rows] == [1, 0]
    assert [row["contactless"] for row in rows] == ["False", "True"]
    assert {int(row["pooled_episode_count"]) for row in rows} == {2}
    assert {int(row["pooled_contactless_episode_count"]) for row in rows} == {1}
    assert {float(row["pooled_contactless_fraction"]) for row in rows} == {0.5}
    assert {int(row["pooled_metric_contributing_episode_count"]) for row in rows} == {
        2
    }
    np.testing.assert_allclose(
        np.asarray(scalar_data["per_unit_values_panel"][0], dtype=float),
        [float(rows[0]["pooled_contactless_fraction"])],
    )
    assert {row["passes_minimum_episode_filter"] for row in rows} == {"True"}
    assert {row["included_in_metric"] for row in rows} == {"True"}
    assert [float(row["max_distance_from_reward_center_mm"]) for row in rows] == [
        3.0,
        5.0,
    ]
    assert [float(row["trajectory_length_mm"]) for row in rows] == [2.0, 2.0]
