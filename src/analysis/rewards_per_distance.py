"""Metric-local rewards-per-distance calculations."""

from __future__ import annotations

import numpy as np


def rewards_per_distance(rewards, distance_m) -> float:
    """Return rewards per meter when numerator and denominator are measurable.

    Zero rewards are a valid outcome. Missing/nonfinite reward counts and
    missing/nonpositive traveled distance make the measurement undefined.
    """
    if not np.isfinite(rewards) or not np.isfinite(distance_m) or distance_m <= 0:
        return np.nan
    return float(rewards / distance_m)
