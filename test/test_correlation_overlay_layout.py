import matplotlib.pyplot as plt
import numpy as np
import pytest
import src.plotting.cross_fly_correlations as correlations

from src.plotting.axis_size import DEFAULT_PLOT_AXIS_SIZE_INCHES
from src.plotting.cross_fly_correlations import (
    _add_smart_stats_box,
    _correlation_axis_size_for_font,
    _finalize_correlation_layout,
    _place_correlation_overlays,
)
from src.plotting.plot_customizer import PlotCustomizer


def _legend_handles(count=4):
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=f"C{idx}",
            label=f"Group {idx}",
        )
        for idx in range(count)
    ]


def test_oversized_stats_box_skips_internal_layout_search(monkeypatch):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = np.linspace(-1.0, 1.0, 24)
    y = np.linspace(-1.0, 1.0, 24)
    scatter = ax.scatter(x, y)
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    draw_count = 0
    original_draw = fig.canvas.draw

    def counted_draw(*args, **kwargs):
        nonlocal draw_count
        draw_count += 1
        return original_draw(*args, **kwargs)

    monkeypatch.setattr(fig.canvas, "draw", counted_draw)
    legend, stats = _place_correlation_overlays(
        ax,
        _legend_handles(),
        "A statistics annotation that is deliberately far too wide " * 8,
        x,
        y,
        scatter_artist=scatter,
        configured_font_size=10.0,
    )

    assert stats.get_position()[0] == 1.02
    assert ax.get_xlim() == original_xlim
    assert ax.get_ylim() == original_ylim
    assert draw_count < 20
    assert legend.axes is ax
    plt.close(fig)


def test_layout_trial_budget_forces_bounded_fallback(monkeypatch):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    grid = np.linspace(-1.0, 1.0, 20)
    x, y = np.meshgrid(grid, grid)
    x = x.ravel()
    y = y.ravel()
    scatter = ax.scatter(x, y)

    draw_count = 0
    original_draw = fig.canvas.draw

    def counted_draw(*args, **kwargs):
        nonlocal draw_count
        draw_count += 1
        return original_draw(*args, **kwargs)

    monkeypatch.setattr(fig.canvas, "draw", counted_draw)
    _legend, stats = _place_correlation_overlays(
        ax,
        _legend_handles(),
        "n=400, r=0.1, p=0.2",
        x,
        y,
        scatter_artist=scatter,
        configured_font_size=10.0,
        max_layout_trials=1,
    )

    assert stats.get_position()[0] == 1.02
    assert draw_count < 15
    plt.close(fig)


def test_preferred_corners_use_headroom_for_legend_and_keep_stats_inside():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = np.linspace(-1.0, 1.0, 30)
    y = 0.9 + 0.2 * x
    scatter = ax.scatter(x, y)
    ax.plot(x, y, linestyle="--")
    ax.set_ylim(-1.0, 1.2)
    original_top = ax.get_ylim()[1]

    legend, stats = _place_correlation_overlays(
        ax,
        _legend_handles(),
        "n = 30, r = 0.999, p = 1e-12",
        x,
        y,
        scatter_artist=scatter,
        configured_font_size=20.0,
    )

    assert legend._loc == 2  # upper left
    assert stats.get_position() == (0.97, 0.03)  # lower right
    assert ax.get_ylim()[1] > original_top
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer=renderer)
    assert axes_box.contains(*stats.get_bbox_patch().get_window_extent(renderer).p0)
    plt.close(fig)


def test_joint_overlay_stats_match_standard_correlation_size():
    with plt.rc_context():
        customizer = PlotCustomizer()
        customizer.update_font_size(20.0)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        try:
            ax.set_xlabel("Strong reward preference")
            ax.set_ylabel("Fast reward preference")
            x = np.linspace(-1.0, 1.0, 30)
            y = 0.9 + 0.2 * x
            scatter = ax.scatter(x, y)
            ax.plot(x, y, linestyle="--")

            _legend, stats = _place_correlation_overlays(
                ax,
                _legend_handles(),
                "n = 30, r = 0.999, p = 1e-12",
                x,
                y,
                scatter_artist=scatter,
                configured_font_size=20.0,
            )

            reference_size = max(
                ax.xaxis.label.get_size(),
                ax.yaxis.label.get_size(),
                *(tick.get_size() for tick in ax.get_xticklabels()),
                *(tick.get_size() for tick in ax.get_yticklabels()),
            )
            expected_size = max(
                correlations.STATS_BOX_MIN_FONTSIZE,
                0.90 * reference_size,
            )

            assert stats.get_fontsize() == pytest.approx(expected_size)
        finally:
            plt.close(fig)


def test_trial_budget_reaches_later_layout_styles(monkeypatch):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    grid = np.linspace(-1.0, 1.0, 20)
    x, y = np.meshgrid(grid, grid)
    x, y = x.ravel(), y.ravel()
    scatter = ax.scatter(x, y)
    messages = []
    monkeypatch.setattr(correlations, "_log_correlation_layout", messages.append)

    _legend, stats = _place_correlation_overlays(
        ax,
        _legend_handles(),
        "n = 400, r = 0.1, p = 0.2",
        x,
        y,
        scatter_artist=scatter,
        configured_font_size=20.0,
        max_layout_trials=48,
    )

    assert "mode=joint_internal layout=annotation_band" in messages[-1]
    assert stats.get_position()[0] == 0.5
    plt.close(fig)


def test_smart_stats_box_patch_is_fully_inside_axes():
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.array([0.45, 0.55])
    y = np.array([0.05, 0.10])
    ax.scatter(x, y)

    stats = _add_smart_stats_box(
        ax,
        "r = 0.50",
        x,
        y,
        fontsize=48.0,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    patch_bbox = stats.get_bbox_patch().get_window_extent(renderer=renderer)

    assert patch_bbox.x0 >= axes_bbox.x0 + 0.5
    assert patch_bbox.x1 <= axes_bbox.x1 - 0.5
    assert patch_bbox.y0 >= axes_bbox.y0 + 0.5
    assert patch_bbox.y1 <= axes_bbox.y1 - 0.5
    # The upper-corner choice is retained; only its anchor is nudged inward.
    assert stats.get_verticalalignment() == "top"
    assert stats.get_position()[1] < 0.95
    plt.close(fig)


def test_smart_stats_box_clears_inner_half_of_top_spine():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.spines["top"].set_linewidth(8.0)
    x = np.array([0.45, 0.55])
    y = np.array([0.05, 0.10])
    ax.scatter(x, y)

    stats = _add_smart_stats_box(
        ax,
        "r = 0.50",
        x,
        y,
        fontsize=48.0,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    patch_bbox = stats.get_bbox_patch().get_window_extent(renderer=renderer)
    half_spine_width_px = 0.5 * 8.0 * fig.dpi / 72.0

    assert patch_bbox.y1 <= axes_bbox.y1 - half_spine_width_px - 0.5
    plt.close(fig)


def test_oversized_smart_stats_box_uses_fitting_fallback():
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.array([0.45, 0.55])
    y = np.array([0.05, 0.10])
    ax.scatter(x, y)

    requested_fontsize = 32.0
    stats = _add_smart_stats_box(
        ax,
        "All flies: r = 0.741, p = 2e-16",
        x,
        y,
        fontsize=requested_fontsize,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    patch_bbox = stats.get_bbox_patch().get_window_extent(renderer=renderer)

    assert stats.get_fontsize() < requested_fontsize
    assert patch_bbox.x0 >= axes_bbox.x0 + 0.5
    assert patch_bbox.x1 <= axes_bbox.x1 - 0.5
    assert patch_bbox.y0 >= axes_bbox.y0 + 0.5
    assert patch_bbox.y1 <= axes_bbox.y1 - 0.5
    plt.close(fig)


def test_long_smart_stats_label_wraps_when_font_tiers_cannot_fit():
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.array([0.45, 0.55])
    y = np.array([0.05, 0.10])
    ax.scatter(x, y)

    stats = _add_smart_stats_box(
        ax,
        (
            "Top SLI-selected (top 20%) (n = 17): r = 0.741, p = 2.33e-16\n"
            "Bottom SLI-selected (bottom 50%) (n = 43): r = 0.214, p = 0.17"
        ),
        x,
        y,
        fontsize=32.0,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    patch_bbox = stats.get_bbox_patch().get_window_extent(renderer=renderer)

    assert stats.get_text().count("\n") > 2
    assert ":\n" in stats.get_text()
    assert patch_bbox.x0 >= axes_bbox.x0 + 0.5
    assert patch_bbox.x1 <= axes_bbox.x1 - 0.5
    assert patch_bbox.y0 >= axes_bbox.y0 + 0.5
    assert patch_bbox.y1 <= axes_bbox.y1 - 0.5
    plt.close(fig)


def test_opposed_band_places_stats_close_to_but_clear_of_markers():
    with plt.rc_context():
        customizer = PlotCustomizer()
        customizer.update_font_size(20.0)
        x = np.array(
            [
                1.0000, 0.9290, 0.3215, 0.5959, 0.5427, 0.3606,
                0.3285, 0.6006, 0.7309, 0.3931, 0.2542, 0.6120,
                0.7552, 0.7064, 0.3781, 0.5633, 0.3398, 0.1769,
                0.4909, 0.6428, 0.7842, 0.0000, 0.3848, 0.5523,
                0.1537, 0.3801, 0.3419, 0.5689, 0.2482, 0.3121,
            ]
        )
        y = np.array(
            [
                0.9558, 0.8871, 0.3249, 0.4260, 0.5041, 0.6264,
                0.4545, 0.7738, 0.6575, 0.2996, 0.5956, 0.6594,
                0.8782, 0.5854, 0.2910, 0.4638, 0.5099, 0.2914,
                0.6149, 0.5376, 0.8670, 0.0000, 0.2942, 1.0000,
                0.3469, 0.3068, 0.2313, 0.5939, 0.2507, 0.4325,
            ]
        )

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        try:
            scatter = ax.scatter(x, y)
            slope, intercept = np.polyfit(x, y, 1)
            order = np.argsort(x)
            ax.plot(x[order], slope * x[order] + intercept, linestyle="--")
            ax.set_xlabel("SLI for T1 SB1")
            ax.set_ylabel("Mean SLI over T2 SB2–5")
            ax.set_title("Initial SLI and later SLI", pad=10)
            handles = _legend_handles()
            for handle, label in zip(
                handles,
                (
                    "Fast learners only",
                    "Strong learners only",
                    "Fast + strong",
                    "Other flies",
                ),
            ):
                handle.set_label(label)

            scaled_axis_size = _correlation_axis_size_for_font(
                customizer,
                base_size=DEFAULT_PLOT_AXIS_SIZE_INCHES,
            )
            _finalize_correlation_layout(
                fig,
                customizer,
                rect=(0, 0, 1, 0.96),
                axis_size_inches=scaled_axis_size,
            )
            base_y0, base_y1 = ax.get_ylim()
            base_y_span = base_y1 - base_y0
            stats_text = correlations._format_corr_annotation(
                0.754, 8.3e-17, 85
            )

            legend, stats = _place_correlation_overlays(
                ax,
                handles,
                stats_text,
                x,
                y,
                scatter_artist=scatter,
                configured_font_size=20.0,
                max_headroom_frac=0.50,
                split_corner_max_lower_frac=0.20,
                annotation_band_max_headroom_frac=0.90,
            )

            final_y0, final_y1 = ax.get_ylim()
            assert stats.get_position() == (0.97, 0.03)
            assert legend._loc == 9  # upper center
            assert legend._ncols == 2
            assert stats.get_text() == stats_text.replace(", p =", ",\np =", 1)
            assert (final_y1 - base_y1) / base_y_span <= 0.25
            assert final_y0 == pytest.approx(base_y0)

            reference_size = max(
                ax.xaxis.label.get_size(), ax.yaxis.label.get_size(),
                *(tick.get_size() for tick in ax.get_xticklabels()),
                *(tick.get_size() for tick in ax.get_yticklabels()),
            )
            assert stats.get_fontsize() == pytest.approx(0.90 * reference_size)

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            stats_bbox = stats.get_bbox_patch().get_window_extent(renderer)
            marker_radius_px = (
                0.5 * np.sqrt(float(np.max(scatter.get_sizes()))) * fig.dpi / 72.0
            )
            pad_px = marker_radius_px + 0.5
            padded_stats_bbox = stats_bbox.expanded(
                (stats_bbox.width + 2.0 * pad_px) / stats_bbox.width,
                (stats_bbox.height + 2.0 * pad_px) / stats_bbox.height,
            )
            points_display = ax.transData.transform(np.column_stack([x, y]))
            obscured = (
                (points_display[:, 0] >= padded_stats_bbox.x0)
                & (points_display[:, 0] <= padded_stats_bbox.x1)
                & (points_display[:, 1] >= padded_stats_bbox.y0)
                & (points_display[:, 1] <= padded_stats_bbox.y1)
            )
            assert not np.any(obscured)
        finally:
            plt.close(fig)
