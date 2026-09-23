from __future__ import annotations

import numpy as np


ANNOTATION_STACK_GAP_POINTS = 4.0
SIGNIFICANCE_GAP_RATIO = 0.5


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


def keep_text_box_inside_axes(
    ax,
    text,
    *,
    pad_px: float = 1.0,
    left_pad_px: float | None = None,
    right_pad_px: float | None = None,
) -> bool:
    """Nudge a rendered text patch inside the visible inner axes boundary.

    Matplotlib's axes extent follows each spine's centerline, so the safe
    inset includes the inward half of every visible spine plus ``pad_px``.
    The text is translated only as far as necessary and its alignment,
    transform, font, and wording are left unchanged. ``False`` is returned
    when the rendered box is physically too large to fit.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    patch = text.get_bbox_patch()
    text_bbox = (
        patch.get_window_extent(renderer=renderer)
        if patch is not None
        else text.get_window_extent(renderer=renderer)
    )

    def _edge_inset_px(spine_name, edge_pad_px):
        spine = ax.spines.get(spine_name)
        half_spine_width_px = 0.0
        if spine is not None and spine.get_visible():
            half_spine_width_px = (
                0.5 * float(spine.get_linewidth()) * fig.dpi / 72.0
            )
        return float(edge_pad_px) + half_spine_width_px

    left_pad_px = pad_px if left_pad_px is None else left_pad_px
    right_pad_px = pad_px if right_pad_px is None else right_pad_px
    left_inset = _edge_inset_px("left", left_pad_px)
    right_inset = _edge_inset_px("right", right_pad_px)
    bottom_inset = _edge_inset_px("bottom", pad_px)
    top_inset = _edge_inset_px("top", pad_px)
    available_width = float(axes_bbox.width) - left_inset - right_inset
    available_height = float(axes_bbox.height) - bottom_inset - top_inset
    if text_bbox.width > available_width or text_bbox.height > available_height:
        return False

    dx_px = max(0.0, axes_bbox.x0 + left_inset - text_bbox.x0)
    dx_px -= max(0.0, text_bbox.x1 - (axes_bbox.x1 - right_inset))
    dy_px = max(0.0, axes_bbox.y0 + bottom_inset - text_bbox.y0)
    dy_px -= max(0.0, text_bbox.y1 - (axes_bbox.y1 - top_inset))
    if dx_px or dy_px:
        transform = text.get_transform()
        position_display = transform.transform(text.get_position())
        text.set_position(
            transform.inverted().transform(
                position_display + np.array([dx_px, dy_px], dtype=float)
            )
        )
        fig.canvas.draw()
    text._keep_inside_axes_after_layout = True
    return True



def _fit_auc_p_value_on_one_line(ax, text, *, pad_px: float) -> bool:
    """Keep the AUC prefix at full size and size only its p-value to fit."""
    original = text.get_text()
    separator = " (p ="
    if separator not in original or "\n" in original:
        return False

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    original_bbox = text.get_window_extent(renderer)
    original_position = text.get_position()
    original_ha, original_va = text.get_ha(), text.get_va()
    prefix, _separator, tail = original.partition(separator)
    p_value = "(p =" + tail
    font_size = float(text.get_fontsize())
    minimum_p_size = max(12.0, 0.55 * font_size)

    text.set_text(prefix)
    text.set_ha("left")
    text.set_va("baseline")
    p_font = text.get_fontproperties().copy()
    p_font.set_size(minimum_p_size)
    p_text = ax.text(
        *original_position,
        p_value,
        transform=text.get_transform(),
        ha="left",
        va="baseline",
        fontproperties=p_font,
        color=text.get_color(),
        alpha=text.get_alpha(),
        zorder=text.get_zorder(),
    )

    def restore():
        p_text.remove()
        text.set_text(original)
        text.set_ha(original_ha)
        text.set_va(original_va)
        text.set_position(original_position)
        fig.canvas.draw()
        return False

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)
    prefix_width = text.get_window_extent(renderer).width
    gap_px = max(3.0, 0.16 * font_size * fig.dpi / 72.0)
    def half_spine(side):
        spine = ax.spines[side]
        return (
            0.5 * float(spine.get_linewidth()) * fig.dpi / 72.0
            if spine.get_visible() else 0.0
        )

    safe_x0 = axes_bbox.x0 + pad_px + half_spine("left")
    safe_x1 = (
        axes_bbox.x1 - pad_px - half_spine("right")
        - 0.5 * font_size * fig.dpi / 72.0
    )
    available_p_width = safe_x1 - safe_x0 - prefix_width - gap_px
    if available_p_width <= 0:
        return restore()

    def p_width(size):
        p_text.set_fontsize(size)
        fig.canvas.draw()
        return p_text.get_window_extent(fig.canvas.get_renderer()).width

    if p_width(minimum_p_size) > available_p_width:
        return restore()
    low, high = minimum_p_size, font_size
    for _ in range(9):
        mid = (low + high) / 2.0
        if p_width(mid) <= available_p_width:
            low = mid
        else:
            high = mid
    p_width(low)

    transform = text.get_transform()
    anchor_display = transform.transform(original_position)
    left_px = min(
        max(float(original_bbox.x0), float(safe_x0)),
        float(safe_x1 - prefix_width - gap_px - p_text.get_window_extent(fig.canvas.get_renderer()).width),
    )
    text.set_position(transform.inverted().transform((left_px, anchor_display[1])))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    prefix_bbox = text.get_window_extent(renderer)
    anchor_display = transform.transform(text.get_position())
    top_shift = float(original_bbox.y1 - prefix_bbox.y1)
    if top_shift:
        text.set_position(
            transform.inverted().transform(anchor_display + np.array([0.0, top_shift]))
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        prefix_bbox = text.get_window_extent(renderer)
        anchor_display = transform.transform(text.get_position())
    p_text.set_position(
        transform.inverted().transform(
            (prefix_bbox.x1 + gap_px, anchor_display[1])
        )
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    prefix_bbox = text.get_window_extent(renderer)
    p_bbox = p_text.get_window_extent(renderer)
    safe_y0 = axes_bbox.y0 + pad_px + half_spine("bottom")
    safe_y1 = axes_bbox.y1 - pad_px - half_spine("top")
    shift_y = max(0.0, safe_y0 - min(prefix_bbox.y0, p_bbox.y0))
    shift_y -= max(0.0, max(prefix_bbox.y1, p_bbox.y1) - safe_y1)
    if shift_y:
        for label in (text, p_text):
            display_position = transform.transform(label.get_position())
            label.set_position(
                transform.inverted().transform(
                    display_position + np.array([0.0, shift_y])
                )
            )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        prefix_bbox = text.get_window_extent(renderer)
        p_bbox = p_text.get_window_extent(renderer)
    if (
        prefix_bbox.x0 < safe_x0 - 0.5
        or p_bbox.x1 > safe_x1 + 0.5
        or min(prefix_bbox.y0, p_bbox.y0) < safe_y0 - 0.5
        or max(prefix_bbox.y1, p_bbox.y1) > safe_y1 + 0.5
    ):
        return restore()
    text._auc_p_value_text = p_text
    return True


def fit_auc_annotation_inside_axes(ax, text, *, pad_px: float = 2.0) -> bool:
    """Keep an AUC/ABC annotation inside its axes without shrinking its font.

    First retain the one-line label and nudge its rendered box inside the axes.
    If the label is intrinsically too wide, reduce only the p-value font size.
    Wrap the p-value as a fallback. Return ``False`` only if neither fits.
    """
    if text is None or not text.get_visible():
        return True

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    original_bbox = text.get_window_extent(renderer=renderer)
    original = text.get_text()

    # Leave a small physical reserve for backend-specific text-metric
    # differences, particularly mathtext/superscripts in PDF output.
    backend_slack_px = (
        0.5 * float(text.get_fontsize()) * float(fig.dpi) / 72.0
    )
    if keep_text_box_inside_axes(
        ax,
        text,
        pad_px=pad_px,
        right_pad_px=pad_px + backend_slack_px,
    ):
        return True

    p_value_separator = " (p ="
    if "\n" in original or p_value_separator not in original:
        return False
    if _fit_auc_p_value_on_one_line(ax, text, pad_px=pad_px):
        return True

    text.set_text(original.replace(p_value_separator, "\n(p =", 1))
    fig.canvas.draw()
    wrapped_bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())

    # With baseline-aligned Matplotlib text, adding a line expands the block in
    # both vertical directions.  Keep the first line's top edge where the
    # original one-line label was and let the new p-value line extend downward.
    # The final constraint below can still move the block vertically if this
    # preserved placement genuinely crosses an axes edge.
    dy_px = float(original_bbox.y1) - float(wrapped_bbox.y1)
    if dy_px:
        transform = text.get_transform()
        position_display = transform.transform(text.get_position())
        text.set_position(
            transform.inverted().transform(
                position_display + np.array([0.0, dy_px], dtype=float)
            )
        )

    if keep_text_box_inside_axes(ax, text, pad_px=pad_px):
        return True

    text.set_text(original)
    return False


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


def move_two_group_legend_below_data_if_annotation_overlap(
    ax, legend, annotation_texts
):
    """Move a colliding two-group legend into clear space below the data.

    Keep the original legend when it misses annotations or when the proposed
    one-row placement would touch plot data, text, or the axes boundary.
    """
    if legend is None or not legend.get_visible():
        return legend

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    original_bbox = legend.get_window_extent(renderer=renderer)
    annotations = [text for text in annotation_texts if text.get_visible()]
    if not any(
        original_bbox.overlaps(text.get_window_extent(renderer=renderer))
        for text in annotations
    ):
        return legend

    handles, labels = ax.get_legend_handles_labels()
    if len(labels) != 2:
        return legend

    candidate = ax.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        prop=legend.prop,
        frameon=legend.get_frame_on(),
        handlelength=legend.handlelength,
        handletextpad=legend.handletextpad,
        borderpad=legend.borderpad,
        borderaxespad=legend.borderaxespad,
        columnspacing=legend.columnspacing,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = candidate.get_window_extent(renderer=renderer)
    axes_bbox = ax.get_window_extent(renderer=renderer).padded(-2.0)
    obstacles = [
        text.get_window_extent(renderer=renderer)
        for text in ax.texts
        if text.get_visible()
    ] + _artist_display_bboxes(ax, renderer)
    clearance_bbox = bbox.padded(3.0)
    if (
        bbox.x0 < axes_bbox.x0
        or bbox.x1 > axes_bbox.x1
        or bbox.y0 < axes_bbox.y0
        or bbox.y1 > axes_bbox.y1
        or any(clearance_bbox.overlaps(obstacle) for obstacle in obstacles)
    ):
        ax.legend_ = legend
        return legend
    return candidate


def place_flexible_overlay_texts(ax, texts, *, pad_px: float = 6.0, upper_only=False):
    """Place axes-anchored text using rendered extents and several clear candidates.

    This is intended for labels such as AUC summaries that may move independently
    of the data.  Unlike data annotations, their placement never changes axis
    limits and their complete rendered bounding boxes remain inside the axes.
    ``upper_only`` restricts candidates to the upper half, for post-training
    summaries whose negative curves should remain unobscured.
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
        candidates = (
            [
                (x, y, ha, "top")
                for y in (0.97, 0.75, 0.60)
                for x, ha in ((0.03, "left"), (0.97, "right"), (0.50, "center"))
            ]
            if upper_only else _OVERLAY_CANDIDATES
        )
        for order, (x, y, ha, va) in enumerate(candidates):
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


def _linked_sample_size_texts(stars):
    sample_sizes = getattr(stars, "_sample_size_texts_", None)
    if sample_sizes is None:
        sample_size = getattr(stars, "_sample_size_text_", None)
        sample_sizes = () if sample_size is None else (sample_size,)
    return tuple(
        sample_size
        for sample_size in sample_sizes
        if sample_size is not None and sample_size.get_visible()
    )


def _choose_sample_size_sides(ax, texts, gap_px: float) -> None:
    """Place a lower count below its trace when an above label is too tight.

    Counts remain above their data markers by default.  At a shared x position,
    an above-label candidate for a lower trace is moved below that trace when
    its rendered box would approach the next-higher marker.  The switch is made
    only when the full below-label box fits inside the axes.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    trace_clearance_px = 2.0 * fig.dpi / 72.0

    counts = [
        text
        for text in texts
        if getattr(text, "_data_point_y_", None) is not None
        and np.isfinite(getattr(text, "_data_point_y_", np.nan))
    ]
    by_x: dict[float, list] = {}
    for text in counts:
        x, _y = text.get_position()
        x_px = float(ax.transData.transform((float(x), 0.0))[0])
        # Texts belonging to one bucket share an exact data x in plotRewards.
        # Pixel rounding also makes this tolerant of negligible float noise.
        by_x.setdefault(round(x_px, 3), []).append(text)

    for bucket_counts in by_x.values():
        ordered = sorted(
            bucket_counts,
            key=lambda text: (
                float(text._data_point_y_),
                int(getattr(text, "_group_idx_", 0)),
            ),
        )
        for text in ordered:
            text._sample_size_side_ = "above"

        for lower, upper in zip(ordered, ordered[1:]):
            lower_bbox = lower.get_window_extent(renderer=renderer)
            lower_x, _lower_y = lower.get_position()
            _lower_x_px, lower_marker_y_px = ax.transData.transform(
                (float(lower_x), float(lower._data_point_y_))
            )
            upper_x, _upper_y = upper.get_position()
            _upper_x_px, upper_marker_y_px = ax.transData.transform(
                (float(upper_x), float(upper._data_point_y_))
            )
            lower_marker_half_px = (
                0.5
                * float(getattr(lower, "_data_marker_size_points_", 0.0))
                * fig.dpi
                / 72.0
            )
            upper_marker_half_px = (
                0.5
                * float(getattr(upper, "_data_marker_size_points_", 0.0))
                * fig.dpi
                / 72.0
            )

            above_top_px = (
                lower_marker_y_px
                + lower_marker_half_px
                + gap_px
                + float(lower_bbox.height)
            )
            upper_marker_bottom_px = upper_marker_y_px - upper_marker_half_px
            below_bottom_px = (
                lower_marker_y_px
                - lower_marker_half_px
                - gap_px
                - float(lower_bbox.height)
            )
            approaches_upper_trace = (
                above_top_px + trace_clearance_px >= upper_marker_bottom_px
            )
            fits_below = below_bottom_px >= float(axes_bbox.y0) + trace_clearance_px
            if approaches_upper_trace and fits_below:
                lower._sample_size_side_ = "below"


def _position_linked_significance_texts(ax, texts, gap_px: float) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for stars in texts:
        sample_sizes = _linked_sample_size_texts(stars)
        if not sample_sizes:
            continue

        top_sample = max(
            sample_sizes,
            key=lambda sample_size: float(
                sample_size.get_window_extent(renderer=renderer).y1
            ),
        )
        sample_bbox = top_sample.get_window_extent(renderer=renderer)
        stars_bbox = stars.get_window_extent(renderer=renderer)
        target_stars_bottom_px = (
            float(sample_bbox.y1) + SIGNIFICANCE_GAP_RATIO * gap_px
        )
        _move_text_by_display_dy(
            ax,
            stars,
            target_stars_bottom_px - float(stars_bbox.y0),
        )

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def _position_linked_annotation_stacks(ax, texts) -> None:
    """Use compact physical gaps for linked marker/count/star stacks."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_px = ANNOTATION_STACK_GAP_POINTS * fig.dpi / 72.0
    _choose_sample_size_sides(ax, texts, gap_px)

    for sample_size in texts:
        data_y = getattr(sample_size, "_data_point_y_", None)
        if data_y is None or not np.isfinite(data_y):
            continue

        sample_bbox = sample_size.get_window_extent(renderer=renderer)
        sample_x, _sample_y = sample_size.get_position()
        _data_x_px, data_y_px = ax.transData.transform(
            (float(sample_x), float(data_y))
        )
        marker_size_points = float(
            getattr(sample_size, "_data_marker_size_points_", 0.0)
        )
        marker_half_px = 0.5 * marker_size_points * fig.dpi / 72.0
        if getattr(sample_size, "_sample_size_side_", "above") == "below":
            marker_bottom_px = data_y_px - marker_half_px
            target_sample_top_px = marker_bottom_px - gap_px
            shift_px = target_sample_top_px - float(sample_bbox.y1)
        else:
            marker_top_px = data_y_px + marker_half_px
            target_sample_bottom_px = marker_top_px + gap_px
            shift_px = target_sample_bottom_px - float(sample_bbox.y0)
        _move_text_by_display_dy(ax, sample_size, shift_px)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    _position_linked_significance_texts(ax, texts, gap_px)


def dodge_annotation_reference_line(ax, texts, *, y=0.0, gap_points=2.0):
    """Lift count/star labels intersecting a horizontal reference line.

    Run after final axis layout. Only linked data annotations participate;
    lifting a count also lifts its significance text by the same amount.
    Existing marker clearance and axis limits are preserved.
    """
    texts = [t for t in texts if t is not None and t.get_visible()]
    fig = ax.figure
    fig.canvas.draw()
    line_px = ax.transData.transform((0, y))[1]
    bounds = ax.get_window_extent(fig.canvas.get_renderer())
    if not bounds.y0 < line_px < bounds.y1:
        return
    gap_px = gap_points * fig.dpi / 72.0
    counts = [t for t in texts if hasattr(t, "_data_point_y_")]
    stars = [t for t in texts if _linked_sample_size_texts(t)]
    for text in counts + stars:
        renderer = fig.canvas.get_renderer()
        bbox = text.get_window_extent(renderer)
        if bbox.y0 >= line_px + gap_px or bbox.y1 <= line_px - gap_px:
            continue
        shift = line_px + gap_px - bbox.y0
        linked = [s for s in stars if text in _linked_sample_size_texts(s)]
        group = [text] + linked
        if any(t.get_window_extent(renderer).y1 + shift > bounds.y1 for t in group):
            continue
        for member in group:
            _move_text_by_display_dy(ax, member, shift)
        fig.canvas.draw()


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
    under-estimate text height when plot font size is increased. Linked
    sample-size/significance stacks are first given compact visible gaps, with
    the sample-to-significance gap half the marker-to-sample gap. This helper
    then measures actual display-space text boxes and shifts later annotations
    upward only when their padded boxes collide. ``ylim`` is mutated if extra
    headroom is needed.
    """
    texts = [t for t in texts if t is not None and t.get_visible()]
    if not texts:
        return

    fig = ax.figure
    fig.canvas.draw()
    _position_linked_annotation_stacks(ax, texts)
    if len(texts) < 2:
        return

    if pad_px is None:
        max_font_px = max(float(t.get_fontsize()) * fig.dpi / 72.0 for t in texts)
        pad_px = max(3.0, 0.22 * max_font_px)

    for _ in range(int(max_iter)):
        gap_px = ANNOTATION_STACK_GAP_POINTS * fig.dpi / 72.0
        _position_linked_significance_texts(ax, texts, gap_px)
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
            for prev_bbox, prev_text in placed:
                if any(
                    sample_size is prev_text
                    for sample_size in _linked_sample_size_texts(text)
                ):
                    continue
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

            placed.append((bbox, text))

        if not moved:
            break
