import matplotlib.pyplot as plt
import pytest

from src.plotting.annotation_layout import (
    ANNOTATION_STACK_GAP_POINTS,
    SIGNIFICANCE_GAP_RATIO,
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


def test_annotation_stack_uses_half_gap_above_sample_size_at_all_fonts():
    for font_size in (11, 23):
        lower_gap, upper_gap = _resolved_vertical_gaps(font_size)
        expected_gap = ANNOTATION_STACK_GAP_POINTS * 100.0 / 72.0

        assert lower_gap == pytest.approx(expected_gap, abs=0.5)
        assert upper_gap == pytest.approx(
            SIGNIFICANCE_GAP_RATIO * lower_gap,
            abs=0.5,
        )


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


def test_two_group_stars_use_half_gap_above_resolved_sample_stack():
    fig, ax = plt.subplots(figsize=(6.0, 4.68), dpi=100)
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(-0.5, 2.5)
    sample_sizes = [
        ax.text(10.0, 0.4, label, fontsize=23, ha="center", va="baseline")
        for label in ("34", "38")
    ]
    for sample_size in sample_sizes:
        sample_size._data_point_y_ = 0.2
        sample_size._data_marker_size_points_ = 3.0
    stars = ax.text(
        10.0,
        0.8,
        "****",
        fontsize=23,
        ha="center",
        va="baseline",
        weight="bold",
    )
    stars._sample_size_texts_ = tuple(sample_sizes)

    resolve_annotation_text_overlaps(
        ax,
        [*sample_sizes, stars],
        [-0.5, 2.5],
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sample_bboxes = [
        sample_size.get_window_extent(renderer=renderer)
        for sample_size in sample_sizes
    ]
    top_sample_bbox = max(sample_bboxes, key=lambda bbox: bbox.y1)
    star_gap = stars.get_window_extent(renderer=renderer).y0 - top_sample_bbox.y1
    expected_gap = (
        SIGNIFICANCE_GAP_RATIO
        * ANNOTATION_STACK_GAP_POINTS
        * fig.dpi
        / 72.0
    )
    plt.close(fig)

    assert not sample_bboxes[0].overlaps(sample_bboxes[1])
    assert star_gap == pytest.approx(expected_gap, abs=0.5)
