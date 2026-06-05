from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY = "between_reward_trajectory"
EPISODE_TYPE_INNER_EXIT_REENTRY = "inner_exit_reentry"
EPISODE_TYPE_OUTER_ENTRY_REEXIT = "outer_entry_reexit"

DEFAULT_MIN_EPISODES = 5


@dataclass(frozen=True)
class EpisodeFilterSpec:
    episode_type: str
    primary_opt: str
    default_min_episodes: int = DEFAULT_MIN_EPISODES
    legacy_opts: tuple[str, ...] = ()


EPISODE_FILTER_SPECS: Mapping[str, EpisodeFilterSpec] = {
    EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY: EpisodeFilterSpec(
        episode_type=EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY,
        primary_opt="min_between_reward_trajectories",
        legacy_opts=("btw_rwd_sync_bucket_min_trajectories",),
    ),
    EPISODE_TYPE_INNER_EXIT_REENTRY: EpisodeFilterSpec(
        episode_type=EPISODE_TYPE_INNER_EXIT_REENTRY,
        primary_opt="min_turnback_episodes",
    ),
    EPISODE_TYPE_OUTER_ENTRY_REEXIT: EpisodeFilterSpec(
        episode_type=EPISODE_TYPE_OUTER_ENTRY_REEXIT,
        primary_opt="min_agarose_episodes",
        legacy_opts=("agarose_dual_circle_min_total",),
    ),
}


def _normalize_min_episode_count(raw, *, default: int) -> int:
    try:
        return max(0, int(raw if raw is not None else default))
    except (TypeError, ValueError):
        return max(0, int(default))


def min_episode_count_for_type(opts, episode_type: str) -> int:
    """
    Resolve the minimum episode count for a metric episode type.

    Primary option names win when present and non-None. Legacy option names are
    checked next so older CLI flags remain functional while metric code can move
    toward episode-type-level configuration.
    """
    spec = EPISODE_FILTER_SPECS.get(str(episode_type))
    if spec is None:
        raise ValueError(f"unknown episode filter type: {episode_type!r}")

    raw = getattr(opts, spec.primary_opt, None)
    if raw is None:
        for opt_name in spec.legacy_opts:
            raw = getattr(opts, opt_name, None)
            if raw is not None:
                break

    return _normalize_min_episode_count(raw, default=spec.default_min_episodes)


def eligible_by_min_episode_count(counts, min_episodes: int) -> np.ndarray:
    """Return a boolean mask of entries whose episode count passes the threshold."""
    n = np.asarray(counts, dtype=int)
    min_n = max(0, int(min_episodes or 0))
    if min_n <= 0:
        return np.ones(n.shape, dtype=bool)
    return n >= min_n


def mask_metric_by_min_episode_count(values, counts, min_episodes: int):
    """
    Mask metric values whose contributing episode count is too low.

    The function intentionally returns masked metric values only; callers should
    keep raw counts beside those values for diagnostics and pooled reductions.
    """
    arr = np.asarray(values, dtype=float)
    keep = eligible_by_min_episode_count(counts, min_episodes)
    return np.where(keep, arr, np.nan)
