"""Shared styling helpers for occupancy heatmaps."""

from __future__ import annotations

import math


def apply_heatmap_text_layout(
    heatmap_axes,
    title_pairs,
    colorbar_ax,
    font_size: float,
    *,
    title_pad_points: float = 6.0,
    title_colorbar_pad_points: float = 3.0,
) -> None:
    """Size heatmap text uniformly and resolve title/colorbar crowding."""
    font_size = float(font_size)
    if not math.isfinite(font_size) or font_size <= 0:
        raise ValueError("heatmap font size must be a positive finite number")

    heatmap_axes = list(heatmap_axes)
    title_pairs = list(title_pairs)
    fig = colorbar_ax.figure

    def displayed_colorbar_ticklabels():
        """Return visible tick labels whose locations fall inside the colorbar."""
        lower, upper = sorted(float(value) for value in colorbar_ax.get_ylim())
        tolerance = max(abs(lower), abs(upper), abs(upper - lower), 1e-12) * 1e-9
        return [
            tick
            for tick in colorbar_ax.get_yticklabels()
            if tick.get_visible()
            and lower - tolerance <= float(tick.get_position()[1]) <= upper + tolerance
        ]

    for header, sample_size in title_pairs:
        # Supplying an explicit y disables Matplotlib's automatic title
        # positioning. That positioning can treat adjacent plot decorations as
        # obstacles and move a title far above a short heatmap axes.
        if header is not None:
            header.axes.set_title(
                header.get_text(), loc="left", y=1.0, fontsize=font_size
            )
        if sample_size is not None:
            sample_size.axes.set_title(
                sample_size.get_text(), loc="right", y=1.0, fontsize=font_size
            )
    colorbar_ax.tick_params(labelsize=font_size)

    if not heatmap_axes:
        return

    fig.canvas.draw()

    # A colorbar occupies its outer GridSpec cell, while equal-aspect heatmaps
    # can be shorter inside their nested cells. Keep the bar out of the title
    # band above the highest heatmap.
    heatmap_top = max(float(ax.get_position().y1) for ax in heatmap_axes)
    colorbar_pos = colorbar_ax.get_position()
    if float(colorbar_pos.y1) > heatmap_top:
        colorbar_ax.set_position(
            [
                float(colorbar_pos.x0),
                float(colorbar_pos.y0),
                float(colorbar_pos.width),
                max(0.0, heatmap_top - float(colorbar_pos.y0)),
            ]
        )
        fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    title_pad_px = float(title_pad_points) * fig.dpi / 72.0
    rightmost_ax = max(heatmap_axes, key=lambda ax: float(ax.get_position().x1))
    colorbar_right_px = max(
        [float(colorbar_ax.get_window_extent(renderer=renderer).x1)]
        + [
            float(tick.get_window_extent(renderer=renderer).x1)
            for tick in displayed_colorbar_ticklabels()
        ]
    )
    for header, sample_size in title_pairs:
        if sample_size is None:
            continue
        sample_bbox = sample_size.get_window_extent(renderer=renderer)
        shift_px = 0.0

        if sample_size.axes is rightmost_ax:
            sample_size.set_ha("right")
            shift_px = max(
                shift_px,
                colorbar_right_px - float(sample_bbox.x1),
            )

        if header is not None:
            header_bbox = header.get_window_extent(renderer=renderer)
            vertically_aligned = min(header_bbox.y1, sample_bbox.y1) > max(
                header_bbox.y0, sample_bbox.y0
            )
            if vertically_aligned:
                shift_px = max(
                    shift_px,
                    float(header_bbox.x1) + title_pad_px - float(sample_bbox.x0),
                )

        if shift_px <= 0.0:
            continue

        axes_width_px = float(
            sample_size.axes.get_window_extent(renderer=renderer).width
        )
        if axes_width_px <= 0.0:
            continue
        sample_size.set_x(
            float(sample_size.get_position()[0]) + shift_px / axes_width_px
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    # The top colorbar tick is centered on the end of the bar and can therefore
    # protrude into the title band even after the bar itself has been capped.
    # Lift the final panel's complete title line just enough to clear that tick.
    rightmost_title_pairs = [
        (header, sample_size)
        for header, sample_size in title_pairs
        if sample_size is not None and sample_size.axes is rightmost_ax
    ]
    vertical_pad_px = float(title_colorbar_pad_points) * fig.dpi / 72.0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_ticks = displayed_colorbar_ticklabels()
    if not visible_ticks or not rightmost_title_pairs:
        return

    top_tick = max(
        visible_ticks,
        key=lambda tick: float(tick.get_window_extent(renderer=renderer).y1),
    )
    top_tick_bbox = top_tick.get_window_extent(renderer=renderer)
    for header, sample_size in rightmost_title_pairs:
        sample_bbox = sample_size.get_window_extent(renderer=renderer)
        horizontally_overlaps = (
            sample_bbox.x1 > top_tick_bbox.x0
            and sample_bbox.x0 < top_tick_bbox.x1
        )
        if not horizontally_overlaps:
            continue

        excess_px = (
            float(top_tick_bbox.y1) + vertical_pad_px - float(sample_bbox.y0)
        )
        if excess_px <= 0.0:
            continue

        # Axes title y coordinates are relative to the title's own axes. Use
        # that axes' display height and apply the correction only once, so a
        # stacked panel with the same x position cannot amplify the movement.
        title_ax = sample_size.axes
        axes_height_px = float(
            title_ax.get_window_extent(renderer=renderer).height
        )
        if axes_height_px <= 0.0:
            continue
        old_y = float(sample_size.get_position()[1])
        new_y = old_y + excess_px / axes_height_px
        if header is not None:
            header.axes.set_title(
                header.get_text(),
                loc="left",
                y=new_y,
                fontsize=font_size,
            )
        title_ax.set_title(
            sample_size.get_text(),
            loc="right",
            y=new_y,
            fontsize=font_size,
        )
