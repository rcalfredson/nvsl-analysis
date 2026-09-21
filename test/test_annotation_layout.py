import matplotlib.pyplot as plt
import pytest

from src.plotting.annotation_layout import (
    ANNOTATION_STACK_GAP_POINTS,
    SIGNIFICANCE_GAP_RATIO,
    move_two_group_legend_below_data_if_annotation_overlap,
    resolve_annotation_text_overlaps,
)
from src.plotting.plot_customizer import compact_legend_spacing


@pytest.mark.parametrize(
    ('star_x', 'extra_low_line', 'expect_move'),
    ((30.0, False, True), (50.0, False, False), (30.0, True, False)),
)
def test_two_group_sli_legend_moves_only_to_clear_lower_space(
    star_x, extra_low_line, expect_move
):
    fig, ax = plt.subplots(figsize=(8.0, 4.68), dpi=100)
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(-0.5, 2.5)
    ax.plot([10, 20, 30, 40, 50], [1.0, 1.1, 1.3, 1.4, 1.5], label='Ctrl')
    ax.fill_between(
        [10, 20, 30, 40, 50],
        [0.9, 1.0, 1.2, 1.3, 1.4],
        [1.1, 1.2, 1.4, 1.5, 1.6],
        alpha=0.2,
    )
    ax.plot(
        [10, 20, 30, 40, 50],
        [0.7, 0.8, 0.75, 0.85, 0.8],
        label='Antennae removed',
    )
    ax.fill_between(
        [10, 20, 30, 40, 50],
        [0.6, 0.7, 0.65, 0.75, 0.7],
        [0.8, 0.9, 0.85, 0.95, 0.9],
        alpha=0.2,
    )
    ax.axhline(0.0, color='k')
    if extra_low_line:
        ax.axhline(-0.3, color='k')
    stars = ax.text(star_x, 1.9 if star_x == 30 else 2.1, '****', fontsize=27)
    original = ax.legend(
        loc='upper left',
        prop={'size': 27, 'style': 'italic'},
        **compact_legend_spacing(27),
    )

    legend = move_two_group_legend_below_data_if_annotation_overlap(
        ax, original, [stars]
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert (legend is not original) is expect_move
    assert ax.get_legend() is legend
    if expect_move:
        legend_bbox = legend.get_window_extent(renderer)
        assert not legend_bbox.overlaps(stars.get_window_extent(renderer))
        assert legend_bbox.y1 < ax.transData.transform((0.0, 0.0))[1]
    plt.close(fig)


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


def test_lower_sample_size_moves_below_when_above_would_approach_upper_trace():
    fig, ax = plt.subplots(figsize=(6.0, 4.68), dpi=100)
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(0.0, 2.0)
    lower = ax.text(10.0, 0.9, "100", fontsize=23, ha="center")
    upper = ax.text(10.0, 1.0, "110", fontsize=23, ha="center")
    for group_idx, (sample_size, data_y) in enumerate(
        ((lower, 0.78), (upper, 0.90))
    ):
        sample_size._data_point_y_ = data_y
        sample_size._data_marker_size_points_ = 3.0
        sample_size._group_idx_ = group_idx

    resolve_annotation_text_overlaps(ax, [lower, upper], [0.0, 2.0])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_px = ANNOTATION_STACK_GAP_POINTS * fig.dpi / 72.0
    lower_marker_bottom = ax.transData.transform((10.0, 0.78))[1]
    lower_marker_bottom -= 0.5 * 3.0 * fig.dpi / 72.0
    upper_marker_top = ax.transData.transform((10.0, 0.90))[1]
    upper_marker_top += 0.5 * 3.0 * fig.dpi / 72.0

    assert lower._sample_size_side_ == "below"
    assert upper._sample_size_side_ == "above"
    assert lower_marker_bottom - lower.get_window_extent(renderer).y1 == pytest.approx(
        gap_px, abs=0.5
    )
    assert upper.get_window_extent(renderer).y0 - upper_marker_top == pytest.approx(
        gap_px, abs=0.5
    )
    plt.close(fig)


def test_spacious_sample_sizes_remain_above_their_traces():
    fig, ax = plt.subplots(figsize=(6.0, 4.68), dpi=100)
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(0.0, 2.0)
    sample_sizes = [
        ax.text(10.0, 0.5, "80", fontsize=23, ha="center"),
        ax.text(10.0, 1.5, "90", fontsize=23, ha="center"),
    ]
    for group_idx, (sample_size, data_y) in enumerate(
        zip(sample_sizes, (0.3, 1.3))
    ):
        sample_size._data_point_y_ = data_y
        sample_size._data_marker_size_points_ = 3.0
        sample_size._group_idx_ = group_idx

    resolve_annotation_text_overlaps(ax, sample_sizes, [0.0, 2.0])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for sample_size, data_y in zip(sample_sizes, (0.3, 1.3)):
        marker_top = ax.transData.transform((10.0, data_y))[1]
        marker_top += 0.5 * 3.0 * fig.dpi / 72.0
        assert sample_size._sample_size_side_ == "above"
        assert sample_size.get_window_extent(renderer).y0 > marker_top
    plt.close(fig)
