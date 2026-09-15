from types import SimpleNamespace

import numpy as np

from src.analysis.reward_range_calculator import unfiltered_sync_measurement_ranges


def test_unfiltered_ranges_ignore_pi_filtered_ranges_and_pair_exclusions():
    va = SimpleNamespace(
        fps=10,
        trns=[SimpleNamespace(n=1, start=6000), SimpleNamespace(n=2, start=12000)],
        buckets=[
            np.asarray([6100, 6200, 6300]),
            np.asarray([12100, 12200, 12300, 12400]),
        ],
        reward_ranges=[slice(np.nan, np.nan)] * 3,
        pair_exclude=[True, True, True],
    )

    assert unfiltered_sync_measurement_ranges(va) == [
        slice(0, 6000),
        slice(6100, 6200),
        slice(12200, 12300),
    ]
