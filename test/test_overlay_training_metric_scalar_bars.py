import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from src.plotting.overlay_training_metric_scalar_bars import (
    ExportedTrainingScalarBars,
    plot_overlays,
)


def test_fraction_overlay_uses_even_ticks_and_wraps_ylabel():
    export = ExportedTrainingScalarBars(
        group="Control",
        panel_labels=["T2 SB2-5"],
        per_unit_values_panel=np.asarray(
            [np.asarray([0.5, 0.75, 1.0], dtype=float)],
            dtype=object,
        ),
        per_unit_ids_panel=np.asarray(
            [np.asarray(["a", "b", "c"], dtype=object)],
            dtype=object,
        ),
        mean=np.asarray([0.75], dtype=float),
        ci_lo=np.asarray([0.5], dtype=float),
        ci_hi=np.asarray([1.0], dtype=float),
        n_units_panel=np.asarray([3], dtype=int),
        meta={"ci": True, "ci_conf": 0.95},
    )

    fig = plot_overlays(
        [export],
        ylabel="Fraction of trajectories without wall contact",
        ymax=1.0,
        ytick_step=0.2,
        opts=SimpleNamespace(
            imageFormat="png",
            fontSize=16,
            fontFamily=None,
        ),
    )
    ax = fig.axes[0]
    ticks = ax.get_yticks()
    ymin, ymax = ax.get_ylim()
    visible_tick_indices = [
        i
        for i, value in enumerate(ticks)
        if ymin - 1e-10 <= value <= ymax + 1e-10
    ]

    np.testing.assert_allclose(np.diff(ticks), 0.2)
    assert ax.get_ylabel() == "Fraction of trajectories\nwithout wall contact"
    fig.canvas.draw()
    tick_labels = ax.get_yticklabels()
    assert [tick_labels[i].get_text() for i in visible_tick_indices] == [
        "0.0",
        "0.2",
        "0.4",
        "0.6",
        "0.8",
        "1.0",
    ]
    label_bbox = ax.yaxis.get_label().get_window_extent(
        renderer=fig.canvas.get_renderer()
    )
    assert label_bbox.x0 >= fig.bbox.x0
    assert label_bbox.y0 >= fig.bbox.y0
    assert label_bbox.y1 <= fig.bbox.y1
    plt.close(fig)
