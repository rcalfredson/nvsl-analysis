import matplotlib.pyplot as plt
import numpy as np

from src.plotting.cross_fly_correlations import _place_correlation_overlays


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
