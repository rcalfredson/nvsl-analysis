import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.plotting.plot_customizer import PlotCustomizer

from src.plotting.annotation_layout import (
    AUC_INSET_FONT_RATIO,
    AUC_TOP_INSET_FONT_RATIO,
    dodge_annotation_reference_line,
    fit_auc_annotation_inside_axes,
    keep_text_box_inside_axes,
    place_auc_annotation,
    place_flexible_overlay_texts,
)


def _visible_top_inset_px(ax, text):
    """Find the painted AUC capitals, excluding the p-value's superscript."""
    ax.figure.canvas.draw()
    bounds = ax.get_window_extent(ax.figure.canvas.get_renderer())
    pixels = np.asarray(ax.figure.canvas.buffer_rgba())
    top = pixels.shape[0] - bounds.y1
    start = int(np.ceil(top)) + 3
    left = text.get_transform().transform(text.get_position())[0]
    prefix_width = 1.8 * text.get_fontsize() * ax.figure.dpi / 72
    crop = pixels[
        start:int(top + bounds.height * 0.5),
        int(np.ceil(left)) + 3:int(np.floor(left + prefix_width)),
        :3,
    ]
    dark_rows = np.flatnonzero((crop < 100).all(axis=2).any(axis=1))
    assert len(dark_rows)
    return start + dark_rows[0] - top


@pytest.mark.parametrize("dpi,font_size", [(100, 15), (100, 27), (200, 27)])
@pytest.mark.parametrize("wording", [
    "AUC (n = 74, 70): *** (p = 0.000611)",
    r"AUC (n = 41, 77): **** (p = $2.69 \times 10^{-6}$)",
    r"AUC (n = 119, 118): **** (p = $1.19 \times 10^{-5}$)",
    r"AUC (n = 39, 41): **** (p = $1.38 \times 10^{-13}$)",
])
def test_auc_blocks_keep_minimum_rendered_insets(dpi, font_size, wording):
    from matplotlib.transforms import Bbox

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=dpi)
    ax = axes[1]
    ax.set_ylim(0, 1.2)
    text = ax.text(0.5, 0.99, wording, transform=ax.transAxes,
                   ha="center", va="top", fontsize=font_size)
    original_size = fig.get_size_inches().copy()
    original_position = ax.get_position().bounds
    try:
        for _ in range(2):
            assert place_auc_annotation(ax, text)
            renderer = fig.canvas.get_renderer()
            labels = [text]
            p_value = getattr(text, "_auc_p_value_text", None)
            if p_value is not None:
                labels.append(p_value)
            block = Bbox.union([label.get_window_extent(renderer) for label in labels])
            bounds = ax.get_window_extent(renderer)
            pad = AUC_INSET_FONT_RATIO * font_size * dpi / 72
            left_half_spine = 0.5 * ax.spines["left"].get_linewidth() * dpi / 72
            top_half_spine = 0.5 * ax.spines["top"].get_linewidth() * dpi / 72
            assert block.x0 - bounds.x0 >= pad + left_half_spine - 1e-6
            expected_top = AUC_TOP_INSET_FONT_RATIO * font_size * dpi / 72 + top_half_spine
            assert _visible_top_inset_px(ax, text) == pytest.approx(expected_top, abs=2)
            assert block.x1 < bounds.x1 - pad
            assert text.get_fontsize() == font_size
            assert ax.get_ylim() == (0, 1.2)
            assert ax.get_position().bounds == original_position
            np.testing.assert_array_equal(fig.get_size_inches(), original_size)
    finally:
        plt.close(fig)


def test_auc_fitted_block_avoids_legend_and_keeps_minimum_insets():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([0, 1], [0, 0], label="Control group")
    legend = ax.legend(loc="upper left", fontsize=16)
    text = ax.text(0.03, 0.97, "AUC (n = 74, 70): ***",
                   transform=ax.transAxes, fontsize=16)
    try:
        assert place_auc_annotation(ax, text, upper_only=True)
        renderer = fig.canvas.get_renderer()
        bounds = ax.get_window_extent(renderer)
        bbox = text.get_window_extent(renderer)
        assert not bbox.overlaps(legend.get_window_extent(renderer))
        assert bbox.y0 > bounds.y0 + 0.5 * bounds.height
        expected_top_inset = (
            AUC_TOP_INSET_FONT_RATIO * 16 + 0.5 * ax.spines["top"].get_linewidth()
        ) * fig.dpi / 72
        assert _visible_top_inset_px(ax, text) >= expected_top_inset - 2
    finally:
        plt.close(fig)


def test_wrapped_auc_block_keeps_minimum_border_insets():
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    text = ax.text(.03, .97, "AUC (n = 106, 90): **** (p = 1.41 × 10⁻¹³)",
                   transform=ax.transAxes, fontsize=27)
    try:
        assert place_auc_annotation(ax, text)
        assert "\n(p =" in text.get_text()
        renderer = fig.canvas.get_renderer()
        bounds = ax.get_window_extent(renderer)
        bbox = text.get_window_extent(renderer)
        inset = (AUC_INSET_FONT_RATIO * 27 + 0.4) * fig.dpi / 72
        assert bbox.x0 - bounds.x0 >= inset - 1e-6
        top_inset = (AUC_TOP_INSET_FONT_RATIO * 27 + 0.4) * fig.dpi / 72
        assert _visible_top_inset_px(ax, text) == pytest.approx(top_inset, abs=2)
        assert text.get_fontsize() == 27
    finally:
        plt.close(fig)


def test_auc_collision_check_includes_separately_sized_p_value():
    from matplotlib.transforms import Bbox

    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    ax = axes[1]
    text = ax.text(.03, .97,
                   r"AUC (n = 119, 118): **** (p = $1.19 \times 10^{-5}$)",
                   transform=ax.transAxes, fontsize=24, fontfamily="Arial")
    try:
        assert place_auc_annotation(ax, text)
        renderer = fig.canvas.get_renderer()
        p_bbox = text._auc_p_value_text.get_window_extent(renderer)
        obstacle = ax.text(*ax.transAxes.inverted().transform(p_bbox.p0),
                           "Other summary", transform=ax.transAxes,
                           va="bottom", fontsize=16)
        assert place_auc_annotation(ax, text)
        renderer = fig.canvas.get_renderer()
        block = Bbox.union([
            label.get_window_extent(renderer)
            for label in (text, text._auc_p_value_text)
        ])
        assert not block.overlaps(obstacle.get_window_extent(renderer))
        assert len(ax.texts) == 3  # Rechecking does not leave duplicate p-values.
    finally:
        plt.close(fig)


@pytest.mark.parametrize("curve_top", [0.1, 0.45])
def test_auc_retains_clear_data_anchor_and_follows_curve_height(curve_top):
    from src.plotting.time_series_auc import AUCTestResult, add_auc_label

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 15)
    ax.set_ylim(-1, 1)
    ax.plot([3, 6, 9, 12, 15], np.full(5, curve_top))
    text = add_auc_label(
        ax, x=0.8, y_anchor=curve_top, ylim=[-1, 1], span=2,
        result=AUCTestResult(0.003, (58, 47), "Welch t-test"), font_size=15,
    )
    position = text.get_position()
    try:
        for _ in range(2):
            assert place_auc_annotation(ax, text)
            assert text.get_transform() == ax.transData
            np.testing.assert_allclose(text.get_position(), position)
        assert ax.get_ylim() == (-1, 1)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("rpi,r_diff", [(False, True), (True, False)])
def test_direct_sli_and_ri_auc_use_panel_curve_anchor(rpi, r_diff):
    import ast
    from collections import defaultdict
    from pathlib import Path
    from src.utils import util

    source = ast.parse((Path(__file__).resolve().parents[1] / "analyze.py").read_text())
    plot_function = next(n for n in source.body
                         if isinstance(n, ast.FunctionDef) and n.name == "plotRewards")
    helpers = ast.Module(body=[n for n in plot_function.body
                              if isinstance(n, ast.FunctionDef)
                              and n.name in ("_auc_base_y", "_add_auc_text")],
                         type_ignores=[])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 15)
    ax.set_ylim(-1, 1)
    texts = []
    namespace = dict(
        np=np, util=util, rpi=rpi, r_diff=r_diff,
        global_geom_top=0.9, ylim=[-1, 1],
        auc_texts_by_ax=defaultdict(list), _track_annotation_text=lambda ax, t: texts.append(t),
    )
    try:
        exec(compile(helpers, "analyze.py", "exec"), namespace)
        base_y = namespace["_auc_base_y"](0.1, 2)
        assert base_y == pytest.approx(0.42)  # This panel, rather than another panel's peak.
        text = namespace["_add_auc_text"](
            ax, 0.8, base_y, "AUC (n = 58, 47): **", size=15, base_y=base_y,
        )
        assert text in texts
        assert place_auc_annotation(ax, text)
        assert text.get_transform() == ax.transData
        np.testing.assert_allclose(text.get_position(), [0.8, 0.42])
        assert ax.get_ylim() == (-1, 1)
    finally:
        plt.close(fig)


def test_auc_uses_nearby_clear_space_before_a_corner():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 15)
    ax.set_ylim(-1, 1)
    text = ax.text(0.8, 0.1, "AUC (n = 58, 47): **", fontsize=15)
    obstacle = ax.text(0.8, 0.1, "Overlapping summary", fontsize=15)
    try:
        assert place_auc_annotation(ax, text)
        renderer = fig.canvas.get_renderer()
        assert not text.get_window_extent(renderer).overlaps(
            obstacle.get_window_extent(renderer)
        )
        assert abs(text.get_position()[1] - 0.1) < 0.3
        assert text.get_position()[0] == pytest.approx(0.8)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("rpi", [False, True])
def test_fixed_axis_final_pass_restores_physical_count_star_gap(rpi):
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from src.plotting.annotation_layout import resolve_annotation_text_overlaps

    source = ast.parse((Path(__file__).resolve().parents[1] / "analyze.py").read_text())
    plot_function = next(n for n in source.body
                         if isinstance(n, ast.FunctionDef) and n.name == "plotRewards")
    final_pass = next(n for n in plot_function.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == 'rpi or (sli_axis is not None and sli_axis.fixed)')
    fig, ax = plt.subplots(figsize=(8, 4.68))
    ax.set(xlim=(0, 60), ylim=(-0.5, 2.5))
    data_y = 0.5 if rpi else 0.9
    count = ax.text(20, 0.9, "99", ha="center", fontsize=27)
    count._data_point_y_ = data_y
    count._data_marker_size_points_ = 3.0
    stars = ax.text(20, 1.1, "****", ha="center", fontsize=27, weight="bold")
    stars._sample_size_texts_ = (count,)
    try:
        resolve_annotation_text_overlaps(ax, [count, stars], [-0.5, 2.5])
        fig.set_size_inches(9, 5.1)
        limits = (-1, 1) if rpi else (-0.5, 1.5)
        ax.set_ylim(*limits)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert stars.get_window_extent(renderer).y0 - count.get_window_extent(renderer).y1 > 10
        # AUC belongs to its later placement pass even if its old anchor
        # overlaps the compact stars. It must not push this stack upward.
        auc = ax.text(20, stars.get_position()[1], "AUC summary", fontsize=27)
        auc_position = auc.get_position()
        namespace = dict(
            rpi=rpi, sli_axis=SimpleNamespace(fixed=True),
            annotation_texts_by_ax={ax: [count, stars, auc]}, auc_texts_by_ax={ax: [auc]},
            resolve_annotation_text_overlaps=resolve_annotation_text_overlaps,
        )
        exec(compile(ast.Module(body=[final_pass], type_ignores=[]), 'analyze.py', 'exec'), namespace)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        gap = stars.get_window_extent(renderer).y0 - count.get_window_extent(renderer).y1
        assert gap * 72 / fig.dpi == pytest.approx(2, abs=0.5)
        marker_top = ax.transData.transform((20, data_y))[1] + 1.5 * fig.dpi / 72
        gap = count.get_window_extent(renderer).y0 - marker_top
        assert gap * 72 / fig.dpi == pytest.approx(4, abs=0.5)
        assert ax.get_ylim() == limits
        assert auc.get_position() == auc_position
    finally:
        plt.close(fig)


@pytest.mark.parametrize("axes_width, expect_shrink", [(0.45, True), (0.85, False)])
def test_stats_box_shrinks_only_when_wider_than_axes(axes_width, expect_shrink):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([0.1, 0.15, axes_width, 0.75])
    ax.set_xlabel("Reward rate", fontsize=23)
    wording = r"n = 89, r = 0.424, p = $3.54 \times 10^{-5}$"
    text = ax.text(
        0.02, 0.98, wording, transform=ax.transAxes,
        ha="left", va="top", fontsize=20.7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white"),
    )
    try:
        assert keep_text_box_inside_axes(ax, text, min_fontsize=12)
        renderer = fig.canvas.get_renderer()
        axes_bbox = ax.get_window_extent(renderer)
        box = text.get_bbox_patch().get_window_extent(renderer)
        assert box.x0 > axes_bbox.x0
        assert box.x1 < axes_bbox.x1
        assert box.y0 > axes_bbox.y0
        assert box.y1 < axes_bbox.y1
        assert (text.get_fontsize() < 20.7) is expect_shrink
        assert text.get_fontsize() >= 12
        assert text.get_text() == wording
        assert ax.xaxis.label.get_fontsize() == 23
        size = text.get_fontsize()
        assert keep_text_box_inside_axes(ax, text, min_fontsize=12)
        assert text.get_fontsize() == size
    finally:
        plt.close(fig)


def test_stats_box_respects_minimum_fontsize_when_it_cannot_fit():
    fig, ax = plt.subplots(figsize=(2, 3))
    text = ax.text(
        0.02, 0.98, "Long statistics annotation " * 10,
        transform=ax.transAxes, fontsize=23,
    )
    try:
        assert not keep_text_box_inside_axes(ax, text, min_fontsize=12)
        assert text.get_fontsize() == 12
    finally:
        plt.close(fig)


@pytest.mark.parametrize("font_size", [12, 20])
def test_zero_line_dodge_lifts_only_intersecting_stack(font_size):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="black")
    count = ax.text(0.3, 0, "87", va="center", fontsize=font_size)
    count._data_point_y_ = -0.08
    stars = ax.text(0.3, 0.2, "****", fontsize=font_size)
    stars._sample_size_texts_ = (count,)
    unaffected = ax.text(0.7, -0.5, "89", fontsize=font_size)
    unaffected._data_point_y_ = -0.6
    before = [t.get_position()[1] for t in (count, stars, unaffected)]
    dodge_annotation_reference_line(ax, [count, stars, unaffected])
    renderer = fig.canvas.get_renderer()
    zero_px = ax.transData.transform((0, 0))[1]
    assert count.get_window_extent(renderer).y0 >= zero_px + 2 * fig.dpi / 72 - 1e-6
    assert stars.get_position()[1] - before[1] == pytest.approx(count.get_position()[1] - before[0])
    assert unaffected.get_position()[1] == before[2]
    assert ax.get_ylim() == (-1, 1)
    positions = [t.get_position() for t in (count, stars, unaffected)]
    dodge_annotation_reference_line(ax, [count, stars, unaffected])
    np.testing.assert_allclose([t.get_position() for t in (count, stars, unaffected)], positions)
    plt.close(fig)


def test_flexible_overlay_bbox_stays_inside_axes():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylim(-0.5, 2.0)
    text = ax.text(
        0.03,
        0.97,
        "AUC (n = 50, 21): **** (p = 1.81e-10)",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    place_flexible_overlay_texts(ax, [text], pad_px=4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    text_bbox = text.get_window_extent(renderer=renderer)

    assert text.get_transform() == ax.transAxes
    assert text_bbox.x0 >= axes_bbox.x0
    assert text_bbox.x1 <= axes_bbox.x1
    assert text_bbox.y0 >= axes_bbox.y0
    assert text_bbox.y1 <= axes_bbox.y1
    assert ax.get_ylim() == (-0.5, 2.0)
    plt.close(fig)


def test_flexible_overlay_avoids_legend_corner():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], label="learners")
    ax.legend(loc="upper left")
    text = ax.text(
        0.03,
        0.97,
        "AUC summary",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    place_flexible_overlay_texts(ax, [text])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert not text.get_window_extent(renderer=renderer).overlaps(
        ax.get_legend().get_window_extent(renderer=renderer)
    )
    plt.close(fig)


@pytest.mark.parametrize("font_size", [12, 20])
def test_post_ri_auc_stays_above_negative_curves_on_matching_fixed_axes(font_size):
    with plt.rc_context():
        customizer = PlotCustomizer()
        customizer.update_font_size(font_size)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax in axes:
            ax.plot([3, 6, 9, 12], [-0.2, -0.4, -0.5, -0.6], label="Experimental")
            ax.plot([3, 6, 9, 12], [-0.1, -0.2, -0.3, -0.4], label="Yoked")
            ax.set_ylim(-1.7, 1.2)  # Mimic legacy layout expansion.
        axes[1].legend(loc="upper right")
        text = axes[1].text(
            0.03, 0.97, "AUC (n = 88): ****", transform=axes[1].transAxes,
            ha="left", va="top", fontsize=font_size,
        )
        customizer.set_fixed_y_axes(axes, (-1, 1))
        place_flexible_overlay_texts(axes[1], [text], upper_only=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = text.get_window_extent(renderer)
        bounds = axes[1].get_window_extent(renderer)
        assert bbox.y0 > axes[1].transData.transform((0, 0))[1]
        assert bounds.contains(bbox.x0, bbox.y0)
        assert bounds.contains(bbox.x1, bbox.y1)
        assert not bbox.overlaps(axes[1].get_legend().get_window_extent(renderer))
        assert all(ax.get_ylim() == (-1, 1) for ax in axes)
        np.testing.assert_array_equal(axes[0].get_yticks(), axes[1].get_yticks())
        plt.close(fig)


def test_text_box_constraint_clears_visible_spine_inner_edge():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.spines["top"].set_linewidth(8.0)
    text = ax.text(
        0.02,
        0.99,
        "n = 87, r = 0.514, p = 3.63e-7",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "boxstyle": "round,pad=0.25"},
    )

    assert keep_text_box_inside_axes(ax, text)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    patch_bbox = text.get_bbox_patch().get_window_extent(renderer=renderer)
    half_spine_width_px = 0.5 * 8.0 * fig.dpi / 72.0

    assert patch_bbox.y1 <= axes_bbox.y1 - half_spine_width_px - 0.5
    plt.close(fig)


def test_auc_annotation_is_nudged_before_wrapping():
    fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
    text = ax.text(
        0.98,
        0.8,
        "AUC (n = 106, 90): ****",
        transform=ax.transAxes,
        ha="left",
        fontsize=27,
    )
    original_position = text.get_position()

    assert fit_auc_annotation_inside_axes(ax, text)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    text_bbox = text.get_window_extent(renderer=renderer)

    assert text.get_text() == "AUC (n = 106, 90): ****"
    assert text.get_fontsize() == 27
    assert text.get_position()[0] < original_position[0]
    assert text_bbox.x0 >= axes_bbox.x0
    assert text_bbox.x1 <= axes_bbox.x1
    plt.close(fig)

def test_auc_annotation_leaves_backend_safety_margin_at_right_edge():
    # This width lets the full-size mathtext label fit with the PDF reserve.
    fig, ax = plt.subplots(figsize=(9, 3), dpi=100)
    text = ax.text(
        0.05,
        0.8,
        r"AUC (n = 59, 30): **** (p = $4.17 \times 10^{-5}$)",
        transform=ax.transAxes,
        ha="left",
        fontsize=24,
    )

    assert fit_auc_annotation_inside_axes(ax, text)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    text_bbox = text.get_window_extent(renderer=renderer)

    backend_slack_px = (
        0.5 * text.get_fontsize() * fig.dpi / 72.0
    )

    assert "\n" not in text.get_text()
    assert text_bbox.x1 <= axes_bbox.x1 - backend_slack_px

    plt.close(fig)

def test_auc_annotation_wraps_p_value_when_one_line_cannot_fit():
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    text = ax.text(
        0.05,
        0.8,
        "AUC (n = 106, 90): **** (p = 1.41 × 10⁻¹³)",
        transform=ax.transAxes,
        ha="left",
        fontsize=27,
    )
    fig.canvas.draw()
    original_bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())

    assert fit_auc_annotation_inside_axes(ax, text)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    text_bbox = text.get_window_extent(renderer=renderer)

    assert text.get_text() == "AUC (n = 106, 90): ****\n(p = 1.41 × 10⁻¹³)"
    assert text.get_fontsize() == 27
    assert text_bbox.y1 == pytest.approx(original_bbox.y1, abs=0.5)
    assert text_bbox.x0 >= axes_bbox.x0
    assert text_bbox.x1 <= axes_bbox.x1
    assert text_bbox.y0 >= axes_bbox.y0
    assert text_bbox.y1 <= axes_bbox.y1
    plt.close(fig)



def test_auc_shrinks_only_p_value_to_preserve_panel_size():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    ax = axes[1]
    text = ax.text(
        0.5,
        0.97,
        r"AUC (n = 39, 41): **** (p = $\mathregular{1.38 \times 10^{-13}}$)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=24,
        fontfamily="Arial",
    )
    old_size = fig.get_size_inches().copy()
    old_wspace = fig.subplotpars.wspace

    assert fit_auc_annotation_inside_axes(ax, text)
    p_value = text._auc_p_value_text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)
    main_bbox = text.get_window_extent(renderer)
    p_bbox = p_value.get_window_extent(renderer)

    assert text.get_text() == "AUC (n = 39, 41): ****"
    assert p_value.get_text().startswith("(p =")
    assert text.get_fontsize() == 24
    assert 12 <= p_value.get_fontsize() < text.get_fontsize()
    assert main_bbox.x0 >= axes_bbox.x0
    assert p_bbox.x0 > main_bbox.x1
    assert p_bbox.x1 <= axes_bbox.x1
    assert np.array_equal(fig.get_size_inches(), old_size)
    assert fig.subplotpars.wspace == old_wspace
    plt.close(fig)
