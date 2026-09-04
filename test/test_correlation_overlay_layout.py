import matplotlib.pyplot as plt
import numpy as np

from src.plotting.cross_fly_correlations import (
    _add_smart_stats_box,
    _place_correlation_overlays,
)


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
