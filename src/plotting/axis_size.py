"""Helpers for giving saved plots a consistent physical data-axis size."""

from __future__ import annotations

from typing import Sequence

import numpy as np


# Width and height of the data rectangle, in inches. Figure dimensions may grow
# to accommodate labels, titles, legends, and colorbars.
DEFAULT_PLOT_AXIS_SIZE_INCHES = (4.5, 3.375)


def normalize_axis_size_inches(size: Sequence[float]) -> tuple[float, float]:
    """Return a validated ``(width, height)`` axis size in inches."""
    if len(size) != 2:
        raise ValueError("axis size must contain exactly width and height")
    width, height = (float(value) for value in size)
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("axis width and height must be finite and greater than zero")
    return width, height


def set_axis_size_inches(ax, size: Sequence[float]) -> None:
    """Resize the canvas so *ax* has the requested physical width and height.

    Call this after layout. Layout-computed margins are preserved in inches,
    because text remains a fixed physical size when the canvas changes. Other
    axes (for example, a colorbar) retain their physical size and offset from
    the main axis.
    """
    target_width, target_height = normalize_axis_size_inches(size)
    fig = ax.figure
    fig.canvas.draw()
    fig_width, fig_height = fig.get_size_inches()
    main_position = ax.get_position()
    main_bounds = (
        main_position.x0 * fig_width,
        main_position.y0 * fig_height,
        main_position.x1 * fig_width,
        main_position.y1 * fig_height,
    )
    left, bottom, right, top = main_bounds
    old_width = right - left
    old_height = top - bottom
    if old_width <= 0 or old_height <= 0:
        raise ValueError("cannot size an axis with an empty bounding box")

    new_fig_width = left + target_width + (fig_width - right)
    new_fig_height = bottom + target_height + (fig_height - top)

    # Capture all positions before resizing. Auxiliary axes after the main axis
    # move with its far edge, while retaining their own physical dimensions.
    positions_inches = []
    for other_ax in fig.axes:
        position = other_ax.get_position()
        positions_inches.append(
            (
                other_ax,
                position.x0 * fig_width,
                position.y0 * fig_height,
                position.x1 * fig_width,
                position.y1 * fig_height,
            )
        )

    delta_width = target_width - old_width
    delta_height = target_height - old_height

    def transform(value, start, end, target_span, delta):
        if value <= start:
            return value
        if value >= end:
            return value + delta
        return start + (value - start) * target_span / (end - start)

    fig.set_size_inches(new_fig_width, new_fig_height, forward=True)
    for other_ax, x0, y0, x1, y1 in positions_inches:
        new_bounds = (
            transform(x0, left, right, target_width, delta_width) / new_fig_width,
            transform(y0, bottom, top, target_height, delta_height) / new_fig_height,
            transform(x1, left, right, target_width, delta_width) / new_fig_width,
            transform(y1, bottom, top, target_height, delta_height) / new_fig_height,
        )
        other_ax.set_axes_locator(None)
        other_ax.set_position(
            (
                new_bounds[0],
                new_bounds[1],
                new_bounds[2] - new_bounds[0],
                new_bounds[3] - new_bounds[1],
            )
        )

    # An explicit physical width and height supersede an earlier box-aspect
    # request (the correlation customizer normally requests the same 4:3 ratio).
    ax.set_box_aspect(None)
