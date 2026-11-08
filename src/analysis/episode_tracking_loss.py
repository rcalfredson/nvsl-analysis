"""Tracking exposure for fixed, already detected episodes.

This module never fills missing coordinates or redefines an episode. Half-open
frame intervals are explicit so decision frames can be audited separately from
the interval used by the metric.
"""
from __future__ import annotations

import numpy as np


class GapIndex:
    def __init__(self, missing, fps):
        self.missing = np.asarray(missing, dtype=bool)
        if self.missing.ndim != 1 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Expected a one-dimensional mask and positive fps")
        self.fps = float(fps)
        changes = np.diff(np.r_[False, self.missing, False].astype(int))
        self.starts = np.flatnonzero(changes == 1)
        self.stops = np.flatnonzero(changes == -1)
        self.prefix = np.r_[0, np.cumsum(self.missing)]

    def exposure(self, start, stop):
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= len(self.missing):
            raise ValueError(f"Invalid frame interval [{start}, {stop})")
        first = np.searchsorted(self.stops, start, side="right")
        last = np.searchsorted(self.starts, stop, side="left")
        lengths = np.minimum(self.stops[first:last], stop) - np.maximum(
            self.starts[first:last], start
        )
        lost = int(self.prefix[stop] - self.prefix[start])
        return dict(
            frames=stop - start,
            lost_frames=lost,
            loss_fraction=lost / (stop - start),
            gap_count=int(last - first),
            max_gap_overlap_s=float(np.max(lengths) / self.fps) if lengths.size else 0.,
            max_intersecting_gap_s=(
                float(np.max(self.stops[first:last] - self.starts[first:last]) / self.fps)
                if last > first else 0.
            ),
        )


def reduce_episodes(episodes, *, policy, minimum=5):
    """Mean episode value; binary values give a ratio with its proper denominator."""
    if policy not in ("baseline", "complete_episode", "complete_support"):
        raise ValueError(f"Unknown episode policy: {policy}")
    selected = [
        ep for ep in episodes
        if np.isfinite(ep["value"])
        and (policy == "baseline" or ep[
            "episode_lost_frames" if policy == "complete_episode" else "support_lost_frames"
        ] == 0)
    ]
    n = len(selected)
    total = float(sum(ep["value"] for ep in selected))
    return dict(n_episodes=n, value=total / n if n >= minimum and n else float("nan"),
                episode_value_sum=total)
