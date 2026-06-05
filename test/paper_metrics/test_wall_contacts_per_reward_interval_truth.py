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


class _Training:
    def __init__(self, n, label):
        self.n = int(n)
        self.start = 0
        self.stop = 8
        self._label = str(label)

    def name(self):
        return self._label


class _Video:
    def __init__(self):
        self.fn = "fake-wall-contact-video"
        self.f = 7
        self.trns = [_Training(1, "T1"), _Training(2, "T2")]
        self.trx = [_Trajectory([1, 3, 5])]
        self.flies = [0]
        self.noyc = True
        self._skipped = False
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
