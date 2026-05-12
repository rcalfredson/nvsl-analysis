from __future__ import annotations

import numpy as np


def _expanded_text_bbox(bbox, pad_px: float):
    return bbox.expanded(
        (float(bbox.width) + 2.0 * pad_px) / max(float(bbox.width), 1.0),
        (float(bbox.height) + 2.0 * pad_px) / max(float(bbox.height), 1.0),
    )


def _move_text_by_display_dy(ax, text, dy_px: float) -> float:
    x, y = text.get_position()
    x_disp, y_disp = ax.transData.transform((float(x), float(y)))
    _x_new, y_new = ax.transData.inverted().transform(
        (x_disp, y_disp + float(dy_px))
    )
    text.set_y(float(y_new))
    text._final_y_ = float(y_new)
    return float(y_new)


def resolve_annotation_text_overlaps(ax, texts, ylim, *, pad_px=None, max_iter=16):
    """
    Resolve annotation collisions using rendered text extents.

    First-pass annotation placement often works in data coordinates, which can
    under-estimate text height when plot font size is increased. This helper
    measures actual display-space text boxes and shifts later annotations upward
    only when their padded boxes collide. ``ylim`` is mutated if extra headroom
    is needed.
    """
    texts = [t for t in texts if t is not None and t.get_visible()]
    if len(texts) < 2:
        return

    fig = ax.figure
    fig.canvas.draw()
    if pad_px is None:
        max_font_px = max(float(t.get_fontsize()) * fig.dpi / 72.0 for t in texts)
        pad_px = max(3.0, 0.22 * max_font_px)

    for _ in range(int(max_iter)):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ordered = sorted(
            (
                (
                    _expanded_text_bbox(t.get_window_extent(renderer=renderer), pad_px),
                    t,
                )
                for t in texts
            ),
            key=lambda item: (item[0].y0, item[0].x0),
        )

        moved = False
        placed = []
        for bbox, text in ordered:
            needed_shift_px = 0.0
            for prev_bbox in placed:
                if not bbox.overlaps(prev_bbox):
                    continue
                overlap_px = prev_bbox.y1 - bbox.y0
                if overlap_px >= 0.0:
                    needed_shift_px = max(needed_shift_px, overlap_px + pad_px)

            if needed_shift_px > 0.0:
                y_new = _move_text_by_display_dy(ax, text, needed_shift_px)
                ylim[1] = max(
                    float(ylim[1]),
                    y_new + 0.05 * (float(ylim[1]) - float(ylim[0])),
                )
                moved = True
                fig.canvas.draw()
                bbox = _expanded_text_bbox(
                    text.get_window_extent(renderer=fig.canvas.get_renderer()),
                    pad_px,
                )

            placed.append(bbox)

        if not moved:
            break
