import argparse

import pytest
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
import numpy as np

from src.plotting.sync_bucket_axis_limits import (
    apply_sync_bucket_ytick_spacing,
    default_sync_bucket_ylim,
)
from src.utils.parsers import positive_finite_float


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("commag", (0.0, 10.0)),
        ("rrd_mean_dist", (0.0, 220.0)),
        ("between_reward_return_leg_dist", (0.0, 220.0)),
    ],
)
def test_shared_sync_bucket_axis_limits(metric, expected):
    assert default_sync_bucket_ylim(metric) == expected


def test_explicit_ytick_spacing_preserves_limits_and_sets_interval():
    fig, ax = plt.subplots()
    ax.set_ylim(0.0, 1.0)

    apply_sync_bucket_ytick_spacing([ax], 0.3)

    assert ax.get_ylim() == (0.0, 1.0)
    visible_ticks = ax.get_yticks()
    visible_ticks = visible_ticks[(visible_ticks >= 0.0) & (visible_ticks <= 1.0)]
    np.testing.assert_allclose(np.diff(visible_ticks), 0.3)
    fig.canvas.draw()
    assert "0.3" in [label.get_text() for label in ax.get_yticklabels()]
    plt.close(fig)


def test_explicit_ytick_spacing_keeps_automatic_locator_for_large_range():
    fig, ax = plt.subplots()
    ax.set_ylim(0.0, 1600.0)
    original_locator = ax.yaxis.get_major_locator()
    assert isinstance(original_locator, AutoLocator)

    apply_sync_bucket_ytick_spacing([ax], 0.3)

    assert ax.yaxis.get_major_locator() is original_locator
    assert len(ax.get_yticks()) < 100
    plt.close(fig)


@pytest.mark.parametrize("spacing", [0, -0.1, float("nan"), float("inf")])
def test_explicit_ytick_spacing_rejects_invalid_values(spacing):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="positive finite"):
        apply_sync_bucket_ytick_spacing([ax], spacing)
    plt.close(fig)


def test_positive_finite_float_accepts_tick_spacing():
    assert positive_finite_float("0.3") == 0.3


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf", "not-a-number"])
def test_positive_finite_float_rejects_invalid_tick_spacing(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_finite_float(value)
