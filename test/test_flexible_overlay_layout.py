import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.plotting.plot_customizer import PlotCustomizer

from src.plotting.annotation_layout import (
    dodge_annotation_reference_line,
    fit_auc_annotation_inside_axes,
    keep_text_box_inside_axes,
    place_flexible_overlay_texts,
)


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
