"""Shared default y-axis limits for per-sync-bucket metric plots."""

from __future__ import annotations

import math

from matplotlib.ticker import MultipleLocator, ScalarFormatter


MAX_EXPLICIT_Y_TICKS = 100


_DEFAULT_YLIMS = {
    "commag": (0.0, 10.0),
    "rrd_mean_dist": (0.0, 220.0),
    "between_reward_return_leg_dist": (0.0, 220.0),
}


def default_sync_bucket_ylim(metric: str) -> tuple[float, float]:
    """Return the shared absolute-value y range for a sync-bucket metric."""
    return _DEFAULT_YLIMS[metric]


def apply_sync_bucket_ytick_spacing(axes, spacing: float | None) -> None:
    """Apply a safe major-y-tick interval without changing axis limits.

    Axes for which the interval would generate more than
    ``MAX_EXPLICIT_Y_TICKS`` retain their existing automatic locator. This
    prevents one interval intended for a compact metric such as SLI from
    producing thousands of ticks on another plot generated in the same run.
    """
    if spacing is None:
        return

    spacing = float(spacing)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("y-tick spacing must be a positive finite number")

    for ax in axes:
        limits = ax.get_ylim()
        estimated_tick_count = math.ceil(abs(limits[1] - limits[0]) / spacing) + 3
        if estimated_tick_count > MAX_EXPLICIT_Y_TICKS:
            continue
        ax.yaxis.set_major_locator(MultipleLocator(spacing))
        # A prior PlotCustomizer pass may have installed an integer-only
        # formatter. Restore Matplotlib's adaptive numeric formatting so, for
        # example, a requested 0.3 interval is displayed faithfully.
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(*limits)
