from __future__ import annotations

from types import MethodType

import numpy as np


def _draw_axes_contained_text(text, renderer):
    # PDF/SVG and raster output have different glyph metrics. Recheck the
    # padded patch before painting, using the renderer that will export it.
    options = text._axes_constraint_options.copy()
    scale = text.figure.dpi / text._axes_constraint_dpi
    for key in ("pad_px", "left_pad_px", "right_pad_px"):
        options[key] *= scale
    keep_text_box_inside_axes(
        text.axes, text,
        **options,
        _renderer=renderer,
    )
    text._axes_constraint_original_draw(renderer)


ANNOTATION_STACK_GAP_POINTS = 4.0
SIGNIFICANCE_GAP_RATIO = 0.5
AUC_INSET_FONT_RATIO = 0.4
# Fixed clearance from the inner edge of each vertical spine, independent
# of font size. Vertical padding still scales to clear stars and curves.
AUC_HORIZONTAL_INSET_POINTS = 1.0
AUC_TOP_INSET_FONT_RATIO = 0.9


def tick_label_fontsize(ax) -> float:
    """Use the tick-label size for statistics, independent of axis titles."""
    from matplotlib.font_manager import FontProperties
    import matplotlib as mpl

    ticks = [*ax.get_xticklabels(), *ax.get_yticklabels()]
    if ticks:
        return max(float(tick.get_fontsize()) for tick in ticks)
    return float(
        FontProperties(size=mpl.rcParams["xtick.labelsize"]).get_size_in_points()
    )


def fit_stats_box_inside_axes(ax, text) -> bool:
    """Fit statistics at tick size, wrapping or moving outside as needed."""
    text.set_fontsize(tick_label_fontsize(ax))
    if keep_text_box_inside_axes(ax, text):
        return True
    original = text.get_text()
    text.set_text(original.replace(", ", ",\n"))
    if keep_text_box_inside_axes(ax, text):
        return True
    text.set_text(original)
    text.set_transform(ax.transAxes)
    text.set_position((1.02, 1.0))
    text.set_ha("left")
    text.set_va("top")
    text._keep_inside_axes_after_layout = False
    return False


def pad_sample_size_labels_over_markers(ax, texts) -> None:
    """Add a small translucent box only to counts overlapping visible markers.

    Call after axis limits, annotation positions and figure layout are final.
    Existing boxes (such as agarose's unconditional background) are preserved.
    Only texts tagged with ``_data_point_y_`` are sample-size candidates.
    """
    from matplotlib.lines import _mark_every_path
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Affine2D

    counts = [
        t for t in texts
        if t is not None and t.get_visible() and hasattr(t, "_data_point_y_")
    ]
    if not counts:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)
    markers = []
    for line in ax.lines:
        if not line.get_visible() or line.get_alpha() == 0:
            continue
        marker = MarkerStyle(line.get_marker())
        if not len(marker.get_path().vertices) or line.get_markersize() <= 0:
            continue
        path = line.get_path()
        transform = line.get_transform()
        if line.get_markevery() is not None:
            # Use the same subsampling as Line2D.draw, including distance-based
            # markevery values, so unpainted vertices cannot trigger a box.
            path = _mark_every_path(
                line.get_markevery(), path, transform, ax
            )
        centers = transform.transform(path.vertices)
        marker_path = marker.get_path().transformed(marker.get_transform())
        scale = line.get_markersize() * fig.dpi / 72.0
        marker_path = marker_path.transformed(Affine2D().scale(scale))
        edge_px = 0.5 * line.get_markeredgewidth() * fig.dpi / 72.0
        for x, y in centers:
            if not np.isfinite([x, y]).all():
                continue
            painted_path = marker_path.transformed(Affine2D().translate(x, y))
            bounds = painted_path.get_extents().padded(edge_px)
            if line.get_clip_on() and not bounds.overlaps(axes_bbox):
                continue
            markers.append((painted_path, bounds, edge_px, line.get_zorder()))

    for text in counts:
        managed = hasattr(text, "_marker_overlap_original_zorder_")
        if text.get_bbox_patch() is not None and not managed:
            continue
        bounds = text.get_window_extent(renderer)
        overlaps = [
            zorder for path, marker_bounds, edge_px, zorder in markers
            if bounds.overlaps(marker_bounds)
            and path.intersects_bbox(bounds.padded(edge_px), filled=True)
            and bounds.overlaps(axes_bbox)
        ]
        if overlaps:
            if not managed:
                text._marker_overlap_original_zorder_ = text.get_zorder()
            text.set_bbox(dict(
                boxstyle="round,pad=0.12", facecolor="white",
                edgecolor="none", alpha=0.78,
            ))
            text.set_zorder(max(
                text._marker_overlap_original_zorder_, max(overlaps) + 1, 5,
            ))
        elif managed:
            text.set_bbox(None)
            text.set_zorder(text._marker_overlap_original_zorder_)
            del text._marker_overlap_original_zorder_


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
    min_fontsize: float | None = None,
    _renderer=None,
) -> bool:
    """Nudge a rendered text patch inside the visible inner axes boundary.

    Matplotlib's axes extent follows each spine's centerline, so the safe
    inset includes the inward half of every visible spine plus ``pad_px``.
    The text is translated only as far as necessary and its alignment,
    transform and wording are left unchanged. When ``min_fontsize`` is set,
    shrink the font only if the rendered box exceeds the available width,
    stopping at that minimum. Otherwise the font is unchanged. ``False`` is
    returned when the rendered box is physically too large to fit. Successful
    constraints are rechecked before drawing with the output renderer.
    """
    fig = ax.figure

    def measure_renderer():
        if _renderer is None:
            fig.canvas.draw()
            return fig.canvas.get_renderer()
        text.update_bbox_position_size(_renderer)
        return _renderer

    renderer = measure_renderer()
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
    if min_fontsize is not None:
        if not np.isfinite(min_fontsize) or min_fontsize <= 0:
            raise ValueError("min_fontsize must be finite and positive")
        minimum = min(float(min_fontsize), float(text.get_fontsize()))
        while text_bbox.width > available_width and text.get_fontsize() > minimum:
            # Include the bbox patch's padding in each measurement. A small
            # reserve avoids landing exactly on the edge with mathtext.
            size = float(text.get_fontsize())
            scaled_size = size * max(0.0, available_width) / text_bbox.width * 0.98
            text.set_fontsize(max(minimum, min(size - 0.25, scaled_size)))
            renderer = measure_renderer()
            text_bbox = (
                patch.get_window_extent(renderer=renderer)
                if patch is not None
                else text.get_window_extent(renderer=renderer)
            )
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
        measure_renderer()
    text._keep_inside_axes_after_layout = True
    if _renderer is None:
        text._axes_constraint_dpi = fig.dpi
        text._axes_constraint_options = dict(
            pad_px=pad_px, left_pad_px=left_pad_px,
            right_pad_px=right_pad_px, min_fontsize=min_fontsize,
        )
        if not hasattr(text, "_axes_constraint_original_draw"):
            text._axes_constraint_original_draw = text.draw
            text.draw = MethodType(_draw_axes_contained_text, text)
    return True



def _auc_narrow_font_properties(original, wording):
    """Find an installed narrow face without silently using a wider default."""
    import re
    from matplotlib.font_manager import findfont, get_font

    plain_text = re.sub(r"\$[^$]*\$", "", wording)
    for family in ("Arial Narrow", "Liberation Sans Narrow"):
        properties = original.copy()
        properties.set_file(None)
        properties.set_family(family)
        try:
            path = findfont(properties, fallback_to_default=False)
        except ValueError:
            continue
        charmap = get_font(path).get_charmap()
        if any(ord(char) not in charmap for char in plain_text if not char.isspace()):
            continue
        return properties
    return None


def fit_auc_annotation_inside_axes(
    ax, text, *, pad_px: float = 2.0, horizontal_pad_px: float | None = None,
) -> bool:
    """Keep an AUC/ABC annotation inside its axes without shrinking its font.

    First retain the one-line label and nudge its rendered box inside the axes.
    If the label is intrinsically too wide, try thin spaces around equals signs,
    then after commas and around multiplication symbols.
    Next try Arial Narrow (or installed Liberation Sans Narrow) at the same
    size before wrapping the p-value. Return ``False``
    only if neither fits.
    ``horizontal_pad_px`` can tighten side margins independently of vertical
    clearance; by default each side uses 1 pt from the inner spine edge.
    """
    if text is None or not text.get_visible():
        return True

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    original_bbox = text.get_window_extent(renderer=renderer)
    original = text.get_text()
    if horizontal_pad_px is None:
        horizontal_pad_px = AUC_HORIZONTAL_INSET_POINTS * fig.dpi / 72.0
    if keep_text_box_inside_axes(
        ax,
        text,
        pad_px=pad_px,
        left_pad_px=horizontal_pad_px,
        right_pad_px=horizontal_pad_px,
    ):
        return True

    if "\n" not in original:
        # Match correlation plots: mathtext thin spaces work with Arial
        # without requiring a Unicode thin-space glyph.
        compact = original.replace(" = ", r"$\,$=$\,$")
        # Time plots also tighten sample-size lists and scientific notation.
        # Bracing \times suppresses its automatic binary-operator spacing;
        # explicit thin spaces would otherwise add to the existing gaps.
        tighter = compact.replace(", ", r",$\,$").replace(
            r" \times ", r"\,{\times}\,"
        ).replace(" × ", r"$\,$×$\,$")
        for candidate in dict.fromkeys((compact, tighter)):
            if candidate == original:
                continue
            text.set_text(candidate)
            if keep_text_box_inside_axes(
                ax, text, pad_px=pad_px,
                left_pad_px=horizontal_pad_px,
                right_pad_px=horizontal_pad_px,
            ):
                return True
            text.set_text(original)

        original_font = text.get_fontproperties().copy()
        narrow_font = _auc_narrow_font_properties(original_font, original)
        if narrow_font is not None:
            text.set_fontproperties(narrow_font)
            # Prefer ordinary spacing if the narrow face alone is enough.
            for candidate in dict.fromkeys((original, compact, tighter)):
                text.set_text(candidate)
                if keep_text_box_inside_axes(
                    ax, text, pad_px=pad_px,
                    left_pad_px=horizontal_pad_px,
                    right_pad_px=horizontal_pad_px,
                ):
                    return True
            text.set_fontproperties(original_font)
            text.set_text(original)

    p_value_separator = " (p ="
    if "\n" in original or p_value_separator not in original:
        return False

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

    if keep_text_box_inside_axes(
        ax, text, pad_px=pad_px,
        left_pad_px=horizontal_pad_px, right_pad_px=horizontal_pad_px,
    ):
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


def _text_ink_bbox(text, renderer):
    """Measure visible glyph outlines rather than font ascent/descent reserves."""
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Bbox

    anchor = text.get_transform().transform(text.get_position())
    scale = text.figure.dpi / 72.0
    boxes = []
    # Matplotlib supplies each line's baseline offset, including wrapped text.
    # TextPath uses the same font/mathtext outlines as vector output.
    for line, _size, x, y in text._get_layout(renderer)[1]:
        if not line.strip():
            continue
        ink = TextPath(
            (0, 0), line, prop=text.get_fontproperties(), usetex=text.get_usetex(),
        ).get_extents()
        if np.all(np.isfinite(ink.extents)):
            boxes.append(Bbox.from_extents(
                anchor[0] + x + ink.x0 * scale,
                anchor[1] + y + ink.y0 * scale,
                anchor[0] + x + ink.x1 * scale,
                anchor[1] + y + ink.y1 * scale,
            ))
    return Bbox.union(boxes) if boxes else text.get_window_extent(renderer)


def place_auc_annotation(ax, text, *, upper_only=False) -> bool:
    """Fit an AUC block near its supplied anchor with minimum border insets.

    Prefer the position chosen by the plot relative to its curves. Only move
    it to clear plot geometry, annotations, a legend or the axes boundary.
    The entire annotation moves together. The top inset follows the main AUC
    line's capital letters. Superscripts get space
    above that line instead of pushing decimal-p-value annotations upward.
    """
    from matplotlib.transforms import Bbox

    if text is None or not text.get_visible():
        return True

    # Restore ordinary spacing and font before rechecking a changed layout.
    if hasattr(text, "_auc_original_text"):
        text.set_text(text._auc_original_text)
        original_font = text._auc_original_font.copy()
        original_font.set_size(text.get_fontsize())
        text.set_fontproperties(original_font)
    else:
        text._auc_original_text = text.get_text()
        text._auc_original_font = text.get_fontproperties().copy()
        text._auc_preferred_placement = (
            text.get_transform(), text.get_position(), text.get_ha(), text.get_va(),
        )

    transform, position, ha, va = text._auc_preferred_placement
    text.set_transform(transform)
    text.set_position(position)
    text.set_ha(ha)
    text.set_va(va)

    fig = ax.figure
    pad_px = AUC_INSET_FONT_RATIO * text.get_fontsize() * fig.dpi / 72.0
    horizontal_pad_px = AUC_HORIZONTAL_INSET_POINTS * fig.dpi / 72.0
    if not fit_auc_annotation_inside_axes(
        ax, text, pad_px=pad_px, horizontal_pad_px=horizontal_pad_px,
    ):
        return False

    labels = [text]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)

    def inset(side):
        spine = ax.spines[side]
        half_width = (
            0.5 * spine.get_linewidth() * fig.dpi / 72.0
            if spine.get_visible() else 0.0
        )
        if side == "top":
            padding = AUC_TOP_INSET_FONT_RATIO * text.get_fontsize() * fig.dpi / 72.0
        elif side in ("left", "right"):
            padding = horizontal_pad_px
        else:
            padding = pad_px
        return padding + half_width

    block = Bbox.union([label.get_window_extent(renderer) for label in labels])
    ink = Bbox.union([_text_ink_bbox(label, renderer) for label in labels])
    from matplotlib.textpath import TextPath

    # Measure the main line's cap height independently of the p-value. Its
    # first baseline can be offset from the anchor when the label is wrapped.
    first_line_y = text._get_layout(renderer)[1][0][3]
    cap_height = TextPath(
        (0, 0), "AUC", prop=text.get_fontproperties(),
    ).get_extents().y1 * fig.dpi / 72.0
    main_top = (
        text.get_transform().transform(text.get_position())[1]
        + first_line_y + cap_height
    )
    superscript_headroom = max(0.0, ink.y1 - main_top)
    # Unusually tall expressions still retain clearance from the spine.
    top_inset = max(inset("top"), superscript_headroom + inset("bottom"))
    safe_bbox = Bbox.from_extents(
        axes_bbox.x0 + inset("left"),
        axes_bbox.y0 + inset("bottom"),
        axes_bbox.x1 - inset("right"),
        axes_bbox.y1 - top_inset,
    )
    block = Bbox.from_extents(block.x0, ink.y0, block.x1, main_top)
    if block.width > safe_bbox.width or block.height > safe_bbox.height:
        return False

    label_ids = {id(label) for label in labels}
    obstacles = [
        other.get_window_extent(renderer)
        for other in ax.texts
        if other.get_visible() and id(other) not in label_ids
    ]
    geometry = _artist_display_bboxes(ax, renderer)
    legend = ax.get_legend()
    legend_bbox = (
        legend.get_window_extent(renderer)
        if legend is not None and legend.get_visible() else None
    )
    # Coordinates select the left/right/center and top/bottom of the safe
    # placement region, rather than measuring padding in data coordinates.
    fallback_candidates = (
        [(x, y) for y in (1.0, 0.75, 0.60) for x in (0.0, 1.0, 0.5)]
        if upper_only else
        [(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0),
         (0.5, 1.0), (0.5, 0.0)]
    )
    available_width = safe_bbox.width - block.width
    available_height = safe_bbox.height - block.height
    preferred_x = float(np.clip(block.x0, safe_bbox.x0, safe_bbox.x0 + available_width))
    preferred_y = float(np.clip(block.y0, safe_bbox.y0, safe_bbox.y0 + available_height))
    if upper_only:
        preferred_y = max(preferred_y, safe_bbox.y0 + 0.60 * available_height)
    # Search nearby clear positions before falling back to the edges. Distance
    # breaks overlap-score ties, so empty space near the curves wins over corners.
    step_px = float(text.get_fontsize()) * fig.dpi / 72.0
    candidates = [(preferred_x, preferred_y)]
    for offset in (1, 2, 3):
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            left = float(np.clip(preferred_x + dx * offset * step_px,
                                 safe_bbox.x0, safe_bbox.x0 + available_width))
            bottom = float(np.clip(preferred_y + dy * offset * step_px,
                                   safe_bbox.y0, safe_bbox.y0 + available_height))
            if not upper_only or bottom >= safe_bbox.y0 + 0.60 * available_height:
                candidates.append((left, bottom))
    candidates.extend(
        (safe_bbox.x0 + x * available_width, safe_bbox.y0 + y * available_height)
        for x, y in fallback_candidates
    )
    best = None
    for order, (left, bottom) in enumerate(candidates):
        candidate = Bbox.from_bounds(
            left, bottom, block.width, block.height + superscript_headroom,
        )
        score = 4.0 * sum(_bbox_overlap_area(candidate, b) for b in obstacles)
        score += 0.15 * sum(_bbox_overlap_area(candidate, b) for b in geometry)
        if legend_bbox is not None:
            score += 8.0 * _bbox_overlap_area(candidate, legend_bbox)
        distance = (left - preferred_x) ** 2 + (bottom - preferred_y) ** 2
        choice = (score, distance, order, left - block.x0, bottom - block.y0)
        if best is None or choice[:3] < best[:3]:
            best = choice

    _score, _distance, _order, dx, dy = best
    for label in labels:
        transform = label.get_transform()
        display_position = transform.transform(label.get_position())
        label.set_position(
            transform.inverted().transform(display_position + np.array([dx, dy]))
        )
    fig.canvas.draw()
    return True


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


def _annotation_layout_renderer(ax, renderer=None):
    """Measure moved Text directly when the figure geometry is already settled.

    Text.get_window_extent recalculates its position using the supplied renderer;
    moving a label does not require repainting all the figure's subplots.
    Passing None retains the full-draw path for layouts with draw-time effects.
    """
    if renderer is None:
        ax.figure.canvas.draw()
        return ax.figure.canvas.get_renderer()
    return renderer


def _can_reuse_annotation_renderer(fig):
    from matplotlib.text import Text

    # Automatic layout and callbacks may change transforms on every draw.
    # Custom text draws include the axes-containment hook in this module.
    texts = (text for ax in fig.axes for text in ax.texts)
    return not (
        getattr(fig, "get_layout_engine", lambda: None)() is not None
        or fig.get_constrained_layout()
        or fig.get_tight_layout()
        or fig.canvas.callbacks.callbacks.get("draw_event")
        or any(
            "draw" in text.__dict__ or type(text).draw is not Text.draw
            for text in texts
        )
        or any(
            "draw" in text.__dict__ or type(text).draw is not Text.draw
            for text in fig.texts
        )
    )


def _choose_sample_size_sides(ax, texts, gap_px: float, *, _renderer=None) -> None:
    """Place a lower count below its trace when an above label is too tight.

    Counts remain above their data markers by default.  At a shared x position,
    an above-label candidate for a lower trace is moved below that trace when
    its rendered box would approach the next-higher marker.  The switch is made
    only when the full below-label box fits inside the axes.
    """
    fig = ax.figure
    renderer = _annotation_layout_renderer(ax, _renderer)
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


def _position_linked_significance_texts(ax, texts, gap_px: float, *, _renderer=None) -> None:
    renderer = _annotation_layout_renderer(ax, _renderer)

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

        renderer = _annotation_layout_renderer(ax, _renderer)


def _position_linked_annotation_stacks(ax, texts, *, _renderer=None) -> None:
    """Use compact physical gaps for linked marker/count/star stacks."""
    fig = ax.figure
    renderer = _annotation_layout_renderer(ax, _renderer)
    gap_px = ANNOTATION_STACK_GAP_POINTS * fig.dpi / 72.0
    _choose_sample_size_sides(ax, texts, gap_px, _renderer=_renderer)

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

        renderer = _annotation_layout_renderer(ax, _renderer)

    _position_linked_significance_texts(ax, texts, gap_px, _renderer=_renderer)


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
    renderer_hint = (
        fig.canvas.get_renderer() if _can_reuse_annotation_renderer(fig) else None
    )
    _position_linked_annotation_stacks(ax, texts, _renderer=renderer_hint)
    if len(texts) < 2:
        return

    if pad_px is None:
        max_font_px = max(float(t.get_fontsize()) * fig.dpi / 72.0 for t in texts)
        pad_px = max(3.0, 0.22 * max_font_px)

    for _ in range(int(max_iter)):
        gap_px = ANNOTATION_STACK_GAP_POINTS * fig.dpi / 72.0
        _position_linked_significance_texts(ax, texts, gap_px, _renderer=renderer_hint)
        renderer = _annotation_layout_renderer(ax, renderer_hint)
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
                renderer = _annotation_layout_renderer(ax, renderer_hint)
                bbox = _expanded_text_bbox(
                    text.get_window_extent(renderer=renderer),
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
