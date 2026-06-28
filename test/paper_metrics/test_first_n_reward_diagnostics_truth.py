from types import SimpleNamespace

import numpy as np

from src.plotting.first_n_reward_diagnostics import (
    FirstNRewardDiagnosticsConfig,
    FirstNRewardDiagnosticsPlotter,
    validate_first_n_reward_diagnostic_rows,
)


class _Trajectory:
    def __init__(self, px_per_frame=10.0):
        self.px_per_frame = float(px_per_frame)
        self.pxPerMmFloor = 10.0

    def bad(self):
        return False

    def distTrav(self, start, stop):
        return abs(int(stop) - int(start)) * self.px_per_frame


class _FloorCalibration:
    @staticmethod
    def pxPerMmFloor():
        return 10.0


class _Transform:
    fctr = 1.0


class _Training:
    start = 0
    stop = 1000


class _FirstNRewardVideo:
    def __init__(self, *, rewards, fps=10.0, sync_bucket_ranges=None):
        self.fn = "first_n_reward_test.mp4"
        self.f = 0
        self.fps = float(fps)
        self.trns = [_Training()]
        self.trx = [_Trajectory()]
        self.ct = _FloorCalibration()
        self.xf = _Transform()
        self.sync_bucket_ranges = sync_bucket_ranges or [[(0, 1000)]]
        self._rewards = np.asarray(rewards, dtype=int)

    def _getOn(self, _trn, *, calc=False, ctrl=False, f=0):
        if ctrl:
            return np.asarray([], dtype=int)
        return self._rewards


def _rows_for_rewards(
    rewards,
    *,
    first_n_rewards=10,
    skip_first_sync_buckets=0,
    keep_first_sync_buckets=0,
    sync_bucket_ranges=None,
):
    plotter = FirstNRewardDiagnosticsPlotter(
        vas=[
            _FirstNRewardVideo(
                rewards=rewards,
                sync_bucket_ranges=sync_bucket_ranges,
            )
        ],
        opts=SimpleNamespace(),
        gls=None,
        cfg=FirstNRewardDiagnosticsConfig(
            csv_out="",
            trainings=(1,),
            skip_first_sync_buckets=skip_first_sync_buckets,
            keep_first_sync_buckets=keep_first_sync_buckets,
            first_n_rewards=first_n_rewards,
            sli_values=[0.25],
            reward_event_type="calc",
        ),
    )
    return plotter.compute_all_rows()


def test_selected_reward_rate_uses_between_reward_intervals_not_reward_count():
    rewards = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]

    [row] = _rows_for_rewards(rewards)

    assert row.eligible_for_nth_reward_cutoff
    assert row.time_to_first_selected_reward_s == 5.0
    assert row.time_to_nth_selected_reward_s == 95.0
    assert row.first_n_selected_reward_span_s == 90.0
    assert row.selected_reward_rate_to_nth_per_min == 6.0
    assert row.selected_reward_rate_to_nth_per_min != 10 * 60.0 / 95.0


def test_selected_rewards_per_distance_uses_first_to_nth_path_length():
    rewards = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]

    [row] = _rows_for_rewards(rewards)

    assert row.eligible_for_nth_reward_cutoff
    assert row.first_n_selected_reward_distance_traveled_mm == 900.0
    assert row.selected_reward_rate_to_nth_per_m == 10.0


def test_selected_reward_rate_is_nan_without_a_positive_first_to_nth_interval():
    [single] = _rows_for_rewards([50], first_n_rewards=1)
    assert single.eligible_for_nth_reward_cutoff
    assert np.isnan(single.selected_reward_rate_to_nth_per_min)
    assert np.isnan(single.selected_reward_rate_to_nth_per_m)

    [duplicate_frame] = _rows_for_rewards([50, 50], first_n_rewards=2)
    assert duplicate_frame.eligible_for_nth_reward_cutoff
    assert duplicate_frame.first_n_selected_reward_span_s == 0.0
    assert np.isnan(duplicate_frame.selected_reward_rate_to_nth_per_min)
    assert np.isnan(duplicate_frame.selected_reward_rate_to_nth_per_m)

    [missing] = _rows_for_rewards([50], first_n_rewards=2)
    assert not missing.eligible_for_nth_reward_cutoff
    assert np.isnan(missing.selected_reward_rate_to_nth_per_min)
    assert np.isnan(missing.selected_reward_rate_to_nth_per_m)


def test_first_n_reward_timing_uses_selected_sync_bucket_window():
    [row] = _rows_for_rewards(
        [50, 110, 130],
        first_n_rewards=2,
        skip_first_sync_buckets=1,
        keep_first_sync_buckets=1,
        sync_bucket_ranges=[[(0, 100), (100, 200)]],
    )

    assert row.eligible_for_nth_reward_cutoff
    assert row.selected_reward_count_in_selected_window == 2
    assert row.time_to_first_selected_reward_s == 1.0
    assert row.time_to_nth_selected_reward_s == 3.0
    assert row.first_n_selected_reward_span_s == 2.0
    assert row.selected_reward_rate_to_nth_per_min == 30.0
    assert row.first_n_selected_reward_distance_traveled_mm == 20.0
    assert row.selected_reward_rate_to_nth_per_m == 50.0


def test_first_n_reward_validator_rejects_old_reward_count_rate_formula():
    [row] = _rows_for_rewards([50, 150, 250, 350, 450, 550, 650, 750, 850, 950])
    bad = row.__dict__.copy()
    bad["selected_reward_rate_to_nth_per_min"] = 10 * 60.0 / 95.0

    try:
        validate_first_n_reward_diagnostic_rows([bad])
    except ValueError as exc:
        assert "inconsistent selected reward rate" in str(exc)
    else:
        raise AssertionError("validator accepted the old off-by-one rate formula")


def test_first_n_reward_validator_rejects_bad_distance_rate_formula():
    [row] = _rows_for_rewards([50, 150, 250, 350, 450, 550, 650, 750, 850, 950])
    bad = row.__dict__.copy()
    bad["selected_reward_rate_to_nth_per_m"] = 10.0 / (
        row.first_n_selected_reward_distance_traveled_mm / 1000.0
    )

    try:
        validate_first_n_reward_diagnostic_rows([bad])
    except ValueError as exc:
        assert "inconsistent selected reward distance rate" in str(exc)
    else:
        raise AssertionError("validator accepted the old distance-rate formula")
