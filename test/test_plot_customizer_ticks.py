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
