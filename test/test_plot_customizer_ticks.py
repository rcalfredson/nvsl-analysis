import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.plot_customizer import (
    PlotCustomizer,
    _fixed_endpoint_ticks,
    _tick_decimal_precision,
)


def test_tick_decimal_precision_preserves_point_fifteen_spacing():
    ticks = np.arange(-0.15, 0.91, 0.15)

    precision = _tick_decimal_precision(ticks)
    labels = [f"{tick:.{precision}f}" for tick in ticks]

    assert precision == 2
    np.testing.assert_allclose([float(label) for label in labels], ticks)


def test_adjusted_y_tick_labels_match_their_coordinates():
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_ylim(-0.2, 1.0)
    customizer = PlotCustomizer()
    customizer.update_font_size(20)

    customizer.adjust_padding_proportionally()
    fig.canvas.draw()

    visible = [
        (position, label.get_text())
        for position, label in zip(ax.get_yticks(), ax.get_yticklabels())
        if -0.2 <= position <= 1.0 and label.get_text()
    ]
    assert visible
    np.testing.assert_allclose(
        [float(label) for _position, label in visible],
        [position for position, _label in visible],
    )
    plt.close(fig)


def test_explicit_legend_line_break_is_preserved():
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.plot([0, 1], [0, 1], label="Control genotype\nwith treatment")
    legend = ax.legend()
    customizer = PlotCustomizer()

    customizer.adjust_padding_proportionally()
    fig.canvas.draw()

    assert legend.get_texts()[0].get_text() == "Control genotype\nwith treatment"
    plt.close(fig)


def test_fixed_endpoint_ticks_include_limits_with_nice_uniform_spacing():
    ticks = _fixed_endpoint_ticks((-0.2, 1.0), max_intervals=8)

    np.testing.assert_allclose(ticks, np.arange(-0.2, 1.01, 0.2), atol=1e-12)
    np.testing.assert_allclose(np.diff(ticks), 0.2)


def test_fixed_y_axis_anchors_ticks_to_both_limits():
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    customizer = PlotCustomizer()
    customizer.update_font_size(20)
    customizer.adjust_padding_proportionally()

    customizer.set_fixed_y_axis(ax, (-0.2, 1.0))
    fig.canvas.draw()

    np.testing.assert_allclose(ax.get_ylim(), (-0.2, 1.0))
    np.testing.assert_allclose(
        ax.get_yticks(), np.arange(-0.2, 1.01, 0.2), atol=1e-12
    )
    assert [label.get_text() for label in ax.get_yticklabels()] == [
        "-0.2",
        "0.0",
        "0.2",
        "0.4",
        "0.6",
        "0.8",
        "1.0",
    ]
    plt.close(fig)


def test_fixed_ticks_reduce_density_to_stay_within_precision_budget():
    ticks = _fixed_endpoint_ticks((0.01, 0.98))
    precision = _tick_decimal_precision(ticks, max_precision=3)

    assert len(ticks) == 6
    assert precision <= 3
    np.testing.assert_allclose(
        [float(f"{tick:.{precision}f}") for tick in ticks],
        ticks,
        atol=1e-12,
    )
    np.testing.assert_allclose(np.diff(ticks), np.diff(ticks)[0])


def test_fixed_tick_grid_does_not_depend_on_axis_height():
    customizer = PlotCustomizer()
    figures_and_axes = [plt.subplots(figsize=(5.0, height)) for height in (3.0, 8.0)]

    for _fig, ax in figures_and_axes:
        customizer.set_fixed_y_axis(ax, (0.01, 0.98))

    np.testing.assert_allclose(
        figures_and_axes[0][1].get_yticks(),
        figures_and_axes[1][1].get_yticks(),
    )
    for fig, _ax in figures_and_axes:
        plt.close(fig)


def test_fixed_y_ticks_are_visible_only_on_leftmost_column():
    fig, axes = plt.subplots(2, 2)
    customizer = PlotCustomizer()

    customizer.set_fixed_y_axes(fig.get_axes(), (-0.2, 1.0))
    fig.canvas.draw()

    for ax in axes[:, 0]:
        assert any(label.get_visible() for label in ax.get_yticklabels())
    for ax in axes[:, 1]:
        assert not any(label.get_visible() for label in ax.get_yticklabels())
    plt.close(fig)
