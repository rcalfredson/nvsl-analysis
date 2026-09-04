import matplotlib.pyplot as plt
import pytest

from src.plotting.annotation_layout import (
    ANNOTATION_STACK_GAP_POINTS,
    resolve_annotation_text_overlaps,
)


def _resolved_vertical_gaps(font_size):
    fig, ax = plt.subplots(figsize=(6.0, 4.68), dpi=100)
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(-0.5, 2.5)
    sample_size = ax.text(
        10.0,
        0.37,
        "82",
        fontsize=font_size,
        ha="center",
        va="baseline",
    )
    stars = ax.text(
        10.0,
        0.6,
        "****",
        fontsize=font_size,
        ha="center",
        va="baseline",
        weight="bold",
    )
    sample_size._data_point_y_ = 0.2
    sample_size._data_marker_size_points_ = 3.0
    stars._sample_size_text_ = sample_size

    resolve_annotation_text_overlaps(ax, [sample_size, stars], [-0.5, 2.5])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    marker_top_px = ax.transData.transform((10.0, sample_size._data_point_y_))[1]
    marker_top_px += 0.5 * sample_size._data_marker_size_points_ * fig.dpi / 72.0
    lower_gap = sample_size.get_window_extent(renderer=renderer).y0 - marker_top_px
    upper_gap = (
        stars.get_window_extent(renderer=renderer).y0
        - sample_size.get_window_extent(renderer=renderer).y1
    )
    plt.close(fig)
    return lower_gap, upper_gap


def test_annotation_stack_uses_equal_gaps_at_small_and_large_fonts():
    for font_size in (11, 23):
        lower_gap, upper_gap = _resolved_vertical_gaps(font_size)
        expected_gap = ANNOTATION_STACK_GAP_POINTS * 100.0 / 72.0

        assert lower_gap == pytest.approx(expected_gap, abs=0.5)
        assert upper_gap == pytest.approx(lower_gap, abs=0.5)


def test_sample_size_without_stars_uses_compact_physical_gap():
    fig, ax = plt.subplots(figsize=(6.0, 4.68), dpi=100)
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(-0.5, 2.5)
    sample_size = ax.text(10.0, 0.5, "82", fontsize=23, ha="center")
    sample_size._data_point_y_ = 0.2
    sample_size._data_marker_size_points_ = 3.0

    resolve_annotation_text_overlaps(ax, [sample_size], [-0.5, 2.5])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    marker_top_px = ax.transData.transform((10.0, sample_size._data_point_y_))[1]
    marker_top_px += 0.5 * sample_size._data_marker_size_points_ * fig.dpi / 72.0
    lower_gap = sample_size.get_window_extent(renderer=renderer).y0 - marker_top_px
    expected_gap = ANNOTATION_STACK_GAP_POINTS * fig.dpi / 72.0
    plt.close(fig)

    assert lower_gap == pytest.approx(expected_gap, abs=0.5)
