from __future__ import annotations

from src.analysis.episode_filters import (
    DEFAULT_MIN_EPISODES,
    EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY,
    mask_metric_by_min_episode_count,
    min_episode_count_for_type,
)


DEFAULT_MIN_BETWEEN_REWARD_SYNC_BUCKET_TRAJECTORIES = DEFAULT_MIN_EPISODES
BTW_RWD_MIN_TRAJECTORIES_OPT = "btw_rwd_sync_bucket_min_trajectories"


def min_between_reward_sync_bucket_trajectories(opts) -> int:
    """Return the per-fly, per-sync-bucket between-reward trajectory threshold."""
    return min_episode_count_for_type(opts, EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY)


def mask_metric_by_min_between_reward_trajectories(
    values,
    counts,
    min_trajectories: int,
):
    """
    Mask metric values whose contributing between-reward trajectory count is too low.

    The filter is per fly/condition and per sync bucket. Counts are returned separately
    by callers so they remain the raw number of contributing trajectories.
    """
    return mask_metric_by_min_episode_count(values, counts, min_trajectories)
