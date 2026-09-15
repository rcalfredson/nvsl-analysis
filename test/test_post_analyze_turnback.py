from types import SimpleNamespace

import numpy as np

from src.utils.post_analyze import turnback_counts_exp_by_bucket


class _Trajectory:
    _bad = False

    def reward_turnback_dual_circle_episodes_for_training(self, **_kwargs):
        return [
            {"start": 1, "stop": 4, "turns_back": True},
            {"start": 1, "stop": 6, "turns_back": False},
            {"start": 8, "stop": 10, "turns_back": True},
        ]


class _Training:
    def isCircle(self):
        return True


def test_turnback_sensitivity_counts_require_full_bucket_containment():
    va = SimpleNamespace(
        sync_bucket_ranges=[[(0, 5), (5, 10)]],
        trx=[_Trajectory()],
        trns=[_Training()],
    )

    turns, totals = turnback_counts_exp_by_bucket(
        va,
        inner_delta_mm=0.0,
        outer_delta_mm=2.0,
        border_width_mm=0.1,
    )

    np.testing.assert_array_equal(turns, [[1, 1]])
    np.testing.assert_array_equal(totals, [[1, 1]])
