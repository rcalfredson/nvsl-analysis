import numpy as np
import pytest

from src.analysis.rewards_per_distance import rewards_per_distance


@pytest.mark.parametrize("rewards", [0, 1, 4])
def test_rewards_per_distance_retains_finite_reward_counts(rewards):
    assert rewards_per_distance(rewards, 0.02) == pytest.approx(rewards / 0.02)


@pytest.mark.parametrize("rewards", [np.nan, np.inf, -np.inf])
def test_rewards_per_distance_rejects_nonfinite_reward_counts(rewards):
    assert np.isnan(rewards_per_distance(rewards, 0.02))


@pytest.mark.parametrize("distance_m", [0, -0.02, np.nan, np.inf, -np.inf])
def test_rewards_per_distance_requires_finite_positive_distance(distance_m):
    assert np.isnan(rewards_per_distance(2, distance_m))


def test_rewards_per_distance_eligibility_is_independent_between_paired_flies():
    experimental = rewards_per_distance(2, 0.02)
    yoked = rewards_per_distance(2, 0)

    assert experimental == pytest.approx(100)
    assert np.isnan(yoked)
