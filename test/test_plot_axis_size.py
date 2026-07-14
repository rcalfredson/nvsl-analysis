import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.plotting.axis_size import (
    DEFAULT_PLOT_AXIS_SIZE_INCHES,
    normalize_axis_size_inches,
    set_axis_size_inches,
)
from src.plotting.cross_fly_correlations import _finalize_correlation_layout
from src.plotting.plot_customizer import PlotCustomizer


def _axis_size_inches(ax):
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent().transformed(
        ax.figure.dpi_scale_trans.inverted()
    )
    return bbox.width, bbox.height


def test_set_axis_size_inches_after_tight_layout():
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.set_xlabel("A deliberately long label that changes the margins")
    fig.tight_layout()

    set_axis_size_inches(ax, (3.2, 2.1))

    assert _axis_size_inches(ax) == pytest.approx((3.2, 2.1))
    plt.close(fig)


def test_set_axis_size_preserves_physical_margins_for_labels():
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlabel("reward rate during first 10 rewards", fontsize=18)
    ax.set_ylabel("SLI for T1 SB1\n(mean)", fontsize=18)
    fig.tight_layout()
    fig.canvas.draw()
    old_axis = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    old_left_margin = old_axis.x0
    old_bottom_margin = old_axis.y0

    set_axis_size_inches(ax, DEFAULT_PLOT_AXIS_SIZE_INCHES)

    new_axis = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    assert new_axis.x0 == pytest.approx(old_left_margin)
    assert new_axis.y0 == pytest.approx(old_bottom_margin)
    assert _axis_size_inches(ax) == pytest.approx(DEFAULT_PLOT_AXIS_SIZE_INCHES)
    plt.close(fig)


def test_correlation_layout_uses_configured_physical_axis_size():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    customizer = PlotCustomizer()
    customizer.standard_plot_axis_size = (4.0, 3.0)

    _finalize_correlation_layout(fig, customizer)

    assert _axis_size_inches(ax) == pytest.approx((4.0, 3.0))
    plt.close(fig)


def test_default_axis_size_is_four_by_three():
    width, height = DEFAULT_PLOT_AXIS_SIZE_INCHES
    assert width / height == pytest.approx(4.0 / 3.0)


@pytest.mark.parametrize("size", [(0, 2), (2, -1), (float("nan"), 2), (1,)])
def test_axis_size_validation_rejects_invalid_values(size):
    with pytest.raises(ValueError):
        normalize_axis_size_inches(size)
