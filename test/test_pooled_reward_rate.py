from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.reward_rate import pooled_rewards_per_minute


def video(rewards=40):
    calls = []
    def count(start, stop, *, calc, ctrl, f):
        calls.append((start, stop, calc, ctrl, f))
        return rewards
    return SimpleNamespace(
        trx=[SimpleNamespace(bad=lambda: False)], fps=1,
        trns=[SimpleNamespace(start=0, stop=3000)],
        sync_bucket_ranges=[[(i * 600, (i + 1) * 600) for i in range(5)]],
        reward_exclusion_mask=[[[False, False, False, True, True]]],
        rwdsPerMinBySyncBucket=[[9, 2, 2, np.nan, np.nan]],
        _countOn=count,
    ), calls


def pooled_rate(va):
    return pooled_rewards_per_minute(
        va, training_idx=0, skip_first_sync_buckets=1, keep_first_sync_buckets=4,
    )


@pytest.mark.parametrize("rewards, expected", [(40, 1), (0, 0)])
def test_full_duration_includes_pi_excluded_buckets(rewards, expected):
    va, calls = video(rewards)
    assert pooled_rate(va) == expected
    assert calls == [(600, 3000, True, False, 0)]


@pytest.mark.parametrize("invalid", ["missing", "gap", "bad", "fps", "rewards"])
def test_invalid_observations_do_not_become_zero_reward_time(invalid):
    va, _ = video(np.nan if invalid == "rewards" else 40)
    if invalid == "missing":
        va.sync_bucket_ranges[0].pop()
    elif invalid == "gap":
        va.sync_bucket_ranges[0][3] = (1900, 2400)
    elif invalid == "bad":
        va.trx[0].bad = lambda: True
    elif invalid == "fps":
        va.fps = 0
    assert np.isnan(pooled_rate(va))


def test_equal_complete_bucket_rates_agree_with_pooled_rate():
    va, _ = video(80)
    assert pooled_rate(va) == np.mean([2]*4)
