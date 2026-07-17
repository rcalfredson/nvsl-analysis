"""Shared default y-axis limits for per-sync-bucket metric plots."""

from __future__ import annotations


_DEFAULT_YLIMS = {
    "commag": (0.0, 10.0),
    "rrd_mean_dist": (0.0, 220.0),
    "between_reward_return_leg_dist": (0.0, 220.0),
}


def default_sync_bucket_ylim(metric: str) -> tuple[float, float]:
    """Return the shared absolute-value y range for a sync-bucket metric."""
    return _DEFAULT_YLIMS[metric]
