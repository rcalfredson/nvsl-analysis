from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.episode_filters import (
    DEFAULT_MIN_EPISODES,
    EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY,
    EPISODE_TYPE_INNER_EXIT_REENTRY,
    EPISODE_TYPE_OUTER_ENTRY_REEXIT,
    eligible_by_min_episode_count,
    mask_metric_by_min_episode_count,
    min_episode_count_for_type,
)


def test_episode_filter_defaults_are_episode_type_level():
    opts = SimpleNamespace()

    assert (
        min_episode_count_for_type(opts, EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY)
        == DEFAULT_MIN_EPISODES
    )
    assert min_episode_count_for_type(opts, EPISODE_TYPE_INNER_EXIT_REENTRY) == 5
    assert min_episode_count_for_type(opts, EPISODE_TYPE_OUTER_ENTRY_REEXIT) == 5


def test_between_reward_episode_filter_accepts_legacy_option():
    opts = SimpleNamespace(btw_rwd_sync_bucket_min_trajectories=7)

    assert (
        min_episode_count_for_type(opts, EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY)
        == 7
    )


def test_primary_episode_filter_option_wins_over_legacy_option():
    opts = SimpleNamespace(
        min_between_reward_trajectories=3,
        btw_rwd_sync_bucket_min_trajectories=7,
    )

    assert (
        min_episode_count_for_type(opts, EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY)
        == 3
    )


def test_episode_filter_thresholds_can_be_disabled():
    opts = SimpleNamespace(min_turnback_episodes=0)

    assert min_episode_count_for_type(opts, EPISODE_TYPE_INNER_EXIT_REENTRY) == 0


def test_invalid_episode_filter_values_fall_back_to_default():
    opts = SimpleNamespace(min_agarose_episodes="not-an-int")

    assert min_episode_count_for_type(opts, EPISODE_TYPE_OUTER_ENTRY_REEXIT) == 5


def test_agarose_episode_filter_accepts_legacy_option():
    opts = SimpleNamespace(agarose_dual_circle_min_total=8)

    assert min_episode_count_for_type(opts, EPISODE_TYPE_OUTER_ENTRY_REEXIT) == 8


def test_unknown_episode_filter_type_raises():
    with pytest.raises(ValueError, match="unknown episode filter type"):
        min_episode_count_for_type(SimpleNamespace(), "mystery")


def test_eligible_by_min_episode_count_uses_inclusive_threshold():
    counts = np.array([4, 5, 6])

    np.testing.assert_array_equal(
        eligible_by_min_episode_count(counts, 5),
        np.array([False, True, True]),
    )


def test_mask_metric_by_min_episode_count_keeps_counts_external():
    values = np.array([[1.0, 2.0, 3.0]])
    counts = np.array([[4, 5, 6]])

    masked = mask_metric_by_min_episode_count(values, counts, 5)

    np.testing.assert_allclose(masked, [[np.nan, 2.0, 3.0]])
    np.testing.assert_array_equal(counts, [[4, 5, 6]])
