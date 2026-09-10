"""Shared default y-axis limits for per-sync-bucket metric plots."""

from __future__ import annotations

import math

import numpy as np

from matplotlib.ticker import MultipleLocator, ScalarFormatter

MAX_EXPLICIT_Y_TICKS = 100
DEFAULT_TARGET_Y_TICK_INTERVALS = 6
MIN_AUTO_Y_TICK_SPACING = 0.1


_DEFAULT_YLIMS = {
    "commag": (0.0, 10.0),
    "rrd_mean_dist": (0.0, 220.0),
    "between_reward_return_leg_dist": (0.0, 220.0),
}


def default_sync_bucket_ylim(metric: str) -> tuple[float, float]:
    """Return the shared absolute-value y range for a sync-bucket metric."""
    return _DEFAULT_YLIMS[metric]


def apply_sync_bucket_xticks(ax, bucket_positions) -> None:
    """Anchor major x ticks to every plotted sync-bucket position."""
    positions = np.asarray(bucket_positions, dtype=float)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("bucket positions must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(positions)):
        raise ValueError("bucket positions must be finite")
    ax.set_xticks(positions)


def _auto_sync_bucket_ytick_spacing(limits) -> float:
    """Choose a readable major-y-tick interval no finer than one tenth."""
    span = abs(float(limits[1]) - float(limits[0]))
    raw_spacing = max(
        span / DEFAULT_TARGET_Y_TICK_INTERVALS,
        MIN_AUTO_Y_TICK_SPACING,
    )

    exponent = math.floor(math.log10(raw_spacing))
    scale = 10.0**exponent
    normalized = raw_spacing / scale

    if normalized <= 1.0:
        nice = 1.0
    elif normalized <= 2.0:
        nice = 2.0
    elif normalized <= 5.0:
        nice = 5.0
    else:
        nice = 10.0

    return max(nice * scale, MIN_AUTO_Y_TICK_SPACING)


def apply_sync_bucket_ytick_spacing(axes, spacing: float | None) -> None:
    """Apply readable major-y-tick intervals without changing axis limits.

    When ``spacing`` is None, choose a conventional 1/2/5-based interval
    automatically, with 0.1 as the finest permitted interval. Explicit
    spacing values are honored as requested.

    Axes for which a requested explicit interval would generate more than
    ``MAX_EXPLICIT_Y_TICKS`` retain their existing automatic locator.
    """
    if spacing is not None:
        spacing = float(spacing)
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("y-tick spacing must be a positive finite number")

    for ax in axes:
        limits = ax.get_ylim()
        axis_spacing = (
            _auto_sync_bucket_ytick_spacing(limits) if spacing is None else spacing
        )
        estimated_tick_count = math.ceil(abs(limits[1] - limits[0]) / axis_spacing) + 3
        if estimated_tick_count > MAX_EXPLICIT_Y_TICKS:
            continue

        ax.yaxis.set_major_locator(MultipleLocator(axis_spacing))
        # A prior PlotCustomizer pass may have installed an integer-only
        # formatter. Restore Matplotlib's adaptive numeric formatting so
        # fractional tick intervals are displayed faithfully.
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(*limits)
