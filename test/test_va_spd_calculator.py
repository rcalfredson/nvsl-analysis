from types import SimpleNamespace

import numpy as np

from src.analysis.va_spd_calculator import VASpeedCalculator


def _calculator(va, opts=None):
    calculator = VASpeedCalculator.__new__(VASpeedCalculator)
    calculator.va = va
    calculator.opts = opts or SimpleNamespace(excl_wall_for_spd=False)
    return calculator


def test_speed_ranges_preserve_windows_when_reward_pi_is_undefined():
    va = SimpleNamespace(
        fps=10,
        trns=[
            SimpleNamespace(n=1, start=6000),
            SimpleNamespace(n=2, start=12000),
        ],
        buckets=[
            np.asarray([6100, 6200, 6300]),
            np.asarray([12100, 12200, 12300, 12400]),
        ],
        reward_ranges=[slice(np.nan, np.nan)] * 3,
        pair_exclude=[True, True, True],
    )

    ranges = _calculator(va)._unfiltered_speed_ranges()

    assert ranges == [slice(0, 6000), slice(6100, 6200), slice(12200, 12300)]


def test_missing_designated_later_bucket_does_not_fall_back_to_earlier_bucket():
    va = SimpleNamespace(
        fps=10,
        trns=[SimpleNamespace(n=1, start=6000), SimpleNamespace(n=2, start=12000)],
        buckets=[
            np.asarray([6100, 6200, 6300, 6400, 6500, 6600, 6700]),
            np.asarray([12100, 12200, 12300, 12400, np.nan, np.nan, np.nan]),
        ],
    )

    later_range = _calculator(va)._unfiltered_speed_ranges()[2]

    assert np.isnan(later_range.start)
    assert np.isnan(later_range.stop)


def test_average_speed_does_not_depend_on_pair_exclusion():
    trajectory = SimpleNamespace(
        sp=np.asarray([2.0, 4.0, 6.0]),
        nan=np.zeros(3, dtype=bool),
        pxPerMmFloor=2.0,
        bad=lambda: False,
    )
    va = SimpleNamespace(trx=[trajectory], pair_exclude=[True])

    values = _calculator(va)._calc_average_speeds(0, 3)

    np.testing.assert_allclose(values, [2.5])


def _trajectory(speed, lost):
    return SimpleNamespace(
        sp=np.asarray(speed, dtype=float),
        nan=np.asarray(lost, dtype=bool),
        pxPerMmFloor=2.0,
        bad=lambda: False,
    )


def test_average_speed_excludes_both_steps_touching_interpolated_position():
    traj = _trajectory([0, 4, 100, 100, 8], [False, False, True, False, False])
    values = _calculator(SimpleNamespace(trx=[traj]))._calc_average_speeds(1, 5)
    np.testing.assert_allclose(values, [3.0])


def test_average_speed_checks_endpoint_before_window_and_keeps_stationary_frames():
    traj = _trajectory([0, 100, 100, 0, 2], [False, True, False, False, False])
    values = _calculator(SimpleNamespace(trx=[traj]))._calc_average_speeds(2, 5)
    np.testing.assert_allclose(values, [0.5])


def test_average_speed_combines_tracking_and_wall_filters():
    traj = _trajectory([0, 4, 100, 100, 8], [False, False, True, False, False])
    traj.boundary_event_stats = {
        "wall": {
            "all": {
                "opp_edge": {
                    "boundary_contact": np.asarray(
                        [False, True, False, False, False]
                    )
                }
            }
        }
    }
    calculator = _calculator(
        SimpleNamespace(trx=[traj]), SimpleNamespace(excl_wall_for_spd=True)
    )
    np.testing.assert_allclose(calculator._calc_average_speeds(1, 5), [4.0])


def test_average_speed_is_missing_when_no_originally_tracked_steps_remain():
    traj = _trajectory([0, 100, 100], [False, True, False])
    values = _calculator(SimpleNamespace(trx=[traj]))._calc_average_speeds(0, 3)
    assert np.isnan(values[0])
