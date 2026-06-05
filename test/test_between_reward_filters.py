from types import SimpleNamespace

import numpy as np

from src.analysis.between_reward_filters import (
    DEFAULT_MIN_BETWEEN_REWARD_SYNC_BUCKET_TRAJECTORIES,
    mask_metric_by_min_between_reward_trajectories,
    min_between_reward_sync_bucket_trajectories,
)


def test_between_reward_min_trajectory_threshold_defaults_to_pi_like_value():
    assert (
        min_between_reward_sync_bucket_trajectories(SimpleNamespace())
        == DEFAULT_MIN_BETWEEN_REWARD_SYNC_BUCKET_TRAJECTORIES
    )


def test_between_reward_min_trajectory_threshold_can_be_disabled():
    opts = SimpleNamespace(btw_rwd_sync_bucket_min_trajectories=0)
    assert min_between_reward_sync_bucket_trajectories(opts) == 0


def test_between_reward_new_episode_threshold_option_takes_precedence():
    opts = SimpleNamespace(
        min_between_reward_trajectories=3,
        btw_rwd_sync_bucket_min_trajectories=9,
    )

    assert min_between_reward_sync_bucket_trajectories(opts) == 3


def test_between_reward_metric_mask_is_per_bucket_and_keeps_counts_external():
    values = np.array([[1.0, 2.0, 3.0]])
    counts = np.array([[9, 10, 11]])

    masked = mask_metric_by_min_between_reward_trajectories(values, counts, 10)

    assert np.isnan(masked[0, 0])
    np.testing.assert_allclose(masked[0, 1:], [2.0, 3.0])
