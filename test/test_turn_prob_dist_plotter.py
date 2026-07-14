from types import SimpleNamespace

import numpy as np
import pytest

from src.plotting.turn_prob_dist_plotter import TurnProbabilityByDistancePlotter
from scripts.plot_turn_prob_dist_sli_bundle import (
    _resolve_image_format,
    _selected_source,
)


def _va(gidx, values):
    return SimpleNamespace(
        gidx=gidx,
        turn_prob_by_distance={
            distance: [[exp], [ctrl]]
            for distance, exp, ctrl in values
        },
    )


@pytest.mark.parametrize("grouped", [False, True])
def test_union_filter_supports_unhashable_va_records(grouped):
    vas = [
        _va(0, [(1.0, (0.2, 0.3), (0.4, 0.5)), (2.0, (np.nan, 0.3), (0.4, 0.5))]),
        _va(0, [(1.0, (0.6, 0.7), (0.8, 0.9)), (2.0, (0.6, 0.7), (0.8, 0.9))]),
    ]
    opts = SimpleNamespace(use_union_filter=True)
    plotter = TurnProbabilityByDistancePlotter(
        vas, gls=["group"] if grouped else None, opts=opts
    )
    plotter.timeframes = ["pre_trn"]

    excluded = plotter.filter_flies()
    expected = {(0, 0, 0)} if grouped else {(0, 0)}
    assert excluded == ({"group": expected} if grouped else expected)

    plotter.average_turn_probabilities()
    results = plotter.results_toward[1.0]
    if grouped:
        results = results["group"]
    assert results["exp"]["means"] == pytest.approx([0.6])
    assert results["ctrl"]["means"] == pytest.approx([0.8])


def test_bundle_plot_source_uses_requested_format():
    assert _selected_source("t2_end", "all", "exp_across_groups", "pdf") == (
        "imgs/turn_probability_t2_end_all_exp_across_groups.pdf"
    )


def test_bundle_plot_format_defaults_to_output_extension():
    assert _resolve_image_format("plot.svg", None) == "svg"


def test_bundle_plot_rejects_extension_format_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        _resolve_image_format("plot.pdf", "png")
