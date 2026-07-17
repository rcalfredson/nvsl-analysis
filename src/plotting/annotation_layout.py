from __future__ import annotations

import numpy as np


_OVERLAY_CANDIDATES = (
    (0.03, 0.97, "left", "top"),
    (0.97, 0.97, "right", "top"),
    (0.03, 0.03, "left", "bottom"),
    (0.97, 0.03, "right", "bottom"),
    (0.50, 0.97, "center", "top"),
    (0.50, 0.03, "center", "bottom"),
)


def _bbox_overlap_area(a, b) -> float:
    width = max(
        0.0, min(float(a.x1), float(b.x1)) - max(float(a.x0), float(b.x0))
    )
    height = max(
        0.0, min(float(a.y1), float(b.y1)) - max(float(a.y0), float(b.y0))
    )
    return width * height


def _artist_display_bboxes(ax, renderer):
    bboxes = []
    for line in ax.lines:
        if not line.get_visible() or not np.asarray(line.get_xdata()).size:
            continue
        try:
            bbox = line.get_path().transformed(line.get_transform()).get_extents()
        except (AttributeError, ValueError):
            continue
        if np.all(np.isfinite(bbox.extents)):
            bboxes.append(bbox)
    for collection in ax.collections:
        if not collection.get_visible():
            continue
        try:
            bbox = collection.get_datalim(ax.transData).transformed(ax.transData)
        except (AttributeError, ValueError):
            continue
        if np.all(np.isfinite(bbox.extents)):
            bboxes.append(bbox)
    return bboxes


def place_flexible_overlay_texts(ax, texts, *, pad_px: float = 6.0):
    """Place axes-anchored text using rendered extents and several clear candidates.

    This is intended for labels such as AUC summaries that may move independently
    of the data.  Unlike data annotations, their placement never changes axis
    limits and their complete rendered bounding boxes remain inside the axes.
    """
    texts = [text for text in texts if text is not None and text.get_visible()]
    if not texts:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    safe_bbox = axes_bbox.padded(-float(pad_px))
    flexible_ids = {id(text) for text in texts}
    text_obstacles = [
        text.get_window_extent(renderer=renderer)
        for text in ax.texts
        if text.get_visible() and id(text) not in flexible_ids
    ]
    legend = ax.get_legend()
    legend_bbox = (
        legend.get_window_extent(renderer=renderer)
        if legend is not None and legend.get_visible()
        else None
    )
    geometry_bboxes = _artist_display_bboxes(ax, renderer)
    placed_bboxes = []

    for text in texts:
        best = None
        for order, (x, y, ha, va) in enumerate(_OVERLAY_CANDIDATES):
            text.set_transform(ax.transAxes)
            text.set_position((x, y))
            text.set_ha(ha)
            text.set_va(va)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = text.get_window_extent(renderer=renderer)

            outside = (
                max(0.0, safe_bbox.x0 - bbox.x0)
                + max(0.0, bbox.x1 - safe_bbox.x1)
                + max(0.0, safe_bbox.y0 - bbox.y0)
                + max(0.0, bbox.y1 - safe_bbox.y1)
            )
            score = 1e6 * outside
            score += 4.0 * sum(_bbox_overlap_area(bbox, b) for b in text_obstacles)
            score += 6.0 * sum(_bbox_overlap_area(bbox, b) for b in placed_bboxes)
            score += 0.15 * sum(_bbox_overlap_area(bbox, b) for b in geometry_bboxes)
            if legend_bbox is not None:
                score += 8.0 * _bbox_overlap_area(bbox, legend_bbox)
            candidate = (score, order, x, y, ha, va, bbox)
            if best is None or candidate[:2] < best[:2]:
                best = candidate

        _score, _order, x, y, ha, va, _bbox = best
        text.set_transform(ax.transAxes)
        text.set_position((x, y))
        text.set_ha(ha)
        text.set_va(va)
        fig.canvas.draw()
        placed_bboxes.append(
            text.get_window_extent(renderer=fig.canvas.get_renderer())
        )


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


def _expand_ylim_if_text_exceeds_top(ax, text, bbox, ylim, *, pad_px: float) -> None:
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    excess_px = float(bbox.y1) + float(pad_px) - float(axes_bbox.y1)
    if excess_px <= 0.0:
        return

    x, _y = text.get_position()
    _x_disp, top_disp = ax.transData.transform((float(x), float(ylim[1])))
    _x_new, y_needed = ax.transData.inverted().transform(
        (_x_disp, top_disp + excess_px)
    )
    if np.isfinite(y_needed):
        ylim[1] = max(float(ylim[1]), float(y_needed))


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
                _move_text_by_display_dy(ax, text, needed_shift_px)
                moved = True
                fig.canvas.draw()
                bbox = _expanded_text_bbox(
                    text.get_window_extent(renderer=fig.canvas.get_renderer()),
                    pad_px,
                )
                _expand_ylim_if_text_exceeds_top(
                    ax,
                    text,
                    bbox,
                    ylim,
                    pad_px=pad_px,
                )

            placed.append(bbox)

        if not moved:
            break
