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


def test_average_speed_does_not_depend_on_pair_exclusion():
    trajectory = SimpleNamespace(
        sp=np.asarray([2.0, 4.0, 6.0]),
        pxPerMmFloor=2.0,
        bad=lambda: False,
    )
    va = SimpleNamespace(trx=[trajectory], pair_exclude=[True])

    values = _calculator(va)._calc_average_speeds(0, 3)

    np.testing.assert_allclose(values, [2.0])
