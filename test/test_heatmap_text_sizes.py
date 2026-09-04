import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.plotting.heatmap_style import apply_heatmap_text_layout


def test_heatmap_text_layout_separates_titles_from_header_and_colorbar():
    font_size = 20.0
    fig = plt.figure(figsize=(5.0, 4.0), dpi=100)
    ax = fig.add_axes([0.1, 0.15, 0.55, 0.6])
    colorbar_ax = fig.add_axes([0.82, 0.15, 0.05, 0.75])
    image = ax.imshow(
        np.geomspace(1e-6, 1e-3, 4).reshape(2, 2),
        norm=mpl.colors.LogNorm(vmin=1e-6, vmax=1e-3),
    )
    fig.colorbar(
        image,
        cax=colorbar_ax,
        ticks=mpl.ticker.LogLocator(subs=(1.0, 3.0)),
        format=mpl.ticker.LogFormatter(minor_thresholds=(10, 10)),
    )
    header = ax.set_title("A long heatmap header", loc="left")
    sample_size = ax.set_title("n=12", loc="right")

    apply_heatmap_text_layout(
        [ax],
        [(header, sample_size)],
        colorbar_ax,
        font_size,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    header_bbox = header.get_window_extent(renderer=renderer)
    sample_bbox = sample_size.get_window_extent(renderer=renderer)
    title_gap_px = sample_bbox.x0 - header_bbox.x1
    expected_gap_px = 6.0 * fig.dpi / 72.0

    assert header.get_fontsize() == font_size
    assert sample_size.get_fontsize() == font_size
    assert sample_size.get_position()[0] > 1.0
    colorbar_right_px = max(
        tick.get_window_extent(renderer=renderer).x1
        for tick in colorbar_ax.get_yticklabels()
    )
    assert sample_bbox.x1 >= colorbar_right_px - 0.5
    assert title_gap_px >= expected_gap_px - 0.5
    displayed_ticks = [
        tick
        for tick in colorbar_ax.get_yticklabels()
        if 1e-6 <= tick.get_position()[1] <= 1e-3
    ]
    top_tick_bbox = max(
        (tick.get_window_extent(renderer=renderer) for tick in displayed_ticks),
        key=lambda bbox: bbox.y1,
    )
    expected_vertical_gap_px = 3.0 * fig.dpi / 72.0
    assert sample_bbox.y0 - top_tick_bbox.y1 >= expected_vertical_gap_px - 0.5
    assert header.get_position()[1] == pytest.approx(sample_size.get_position()[1])
    assert header.get_position()[1] > 1.0
    assert header.get_position()[1] < 1.5
    assert colorbar_ax.get_position().y1 == pytest.approx(ax.get_position().y1)
    assert colorbar_ax.get_yticklabels()
    assert all(
        tick.get_fontsize() == font_size for tick in colorbar_ax.get_yticklabels()
    )
    plt.close(fig)


def test_heatmap_text_layout_right_aligns_final_sample_size_when_titles_fit():
    fig = plt.figure(figsize=(6.0, 4.0), dpi=100)
    ax = fig.add_axes([0.1, 0.15, 0.65, 0.6])
    colorbar_ax = fig.add_axes([0.85, 0.15, 0.04, 0.6])
    image = ax.imshow(np.arange(4, dtype=float).reshape(2, 2))
    fig.colorbar(image, cax=colorbar_ax)
    header = ax.set_title("T1", loc="left")
    sample_size = ax.set_title("n=12", loc="right")

    apply_heatmap_text_layout(
        [ax],
        [(header, sample_size)],
        colorbar_ax,
        20.0,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    colorbar_right_px = max(
        tick.get_window_extent(renderer=renderer).x1
        for tick in colorbar_ax.get_yticklabels()
    )

    assert sample_size.get_window_extent(renderer=renderer).x1 == pytest.approx(
        colorbar_right_px,
        abs=0.5,
    )
    plt.close(fig)


def test_heatmap_text_layout_resets_excessive_title_displacement():
    fig = plt.figure(figsize=(3.1, 6.0), dpi=100)
    ax = fig.add_axes([0.05, 0.5, 0.7, 0.35])
    colorbar_ax = fig.add_axes([0.82, 0.05, 0.05, 0.8])
    image = ax.imshow(np.arange(4, dtype=float).reshape(2, 2))
    fig.colorbar(image, cax=colorbar_ax)
    header = ax.set_title("T2 SB5, last 5 min", loc="left", y=2.5)
    sample_size = ax.set_title("n=18", loc="right", y=2.5)

    apply_heatmap_text_layout(
        [ax],
        [(header, sample_size)],
        colorbar_ax,
        20.0,
    )

    assert header.get_position()[1] < 1.5
    assert sample_size.get_position()[1] == pytest.approx(
        header.get_position()[1]
    )
    plt.close(fig)


@pytest.mark.parametrize("font_size", [0, -1, float("nan"), float("inf")])
def test_heatmap_text_size_rejects_invalid_values(font_size):
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="positive finite"):
        apply_heatmap_text_layout([], [], ax, font_size)

    plt.close(fig)
