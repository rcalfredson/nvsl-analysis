from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection

from src.plotting.overlay_training_metric_scalar_bars import (
    ExportedTrainingScalarBars,
    OmnibusLearnerEntry,
    plot_omnibus_learner_overlays,
    plot_overlays,
)


def _export(group, values):
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    return ExportedTrainingScalarBars(
        group=group,
        panel_labels=["T2 SB2-5"],
        per_unit_values_panel=np.asarray([values], dtype=object),
        per_unit_ids_panel=np.asarray(
            [np.asarray([f"{group}_{i}" for i in range(values.size)], dtype=object)],
            dtype=object,
        ),
        mean=np.asarray([mean], dtype=float),
        ci_lo=np.asarray([lo], dtype=float),
        ci_hi=np.asarray([hi], dtype=float),
        n_units_panel=np.asarray([values.size], dtype=int),
        meta={"ci": True, "ci_conf": 0.95},
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


def test_confidence_intervals_are_drawn_above_point_overlay():
    fig = plot_overlays([_export("Control", [0.25, 0.5, 0.75])], show_points=True)
    ax = fig.axes[0]

    point_zorders = [
        artist.get_zorder()
        for artist in ax.collections
        if isinstance(artist, PathCollection)
    ]
    ci_zorders = [
        artist.get_zorder()
        for artist in ax.collections
        if isinstance(artist, LineCollection)
    ]

    assert point_zorders
    assert ci_zorders
    assert min(ci_zorders) > max(point_zorders)
    plt.close(fig)


def test_omnibus_learner_overlay_clusters_top_then_bottom():
    entries = [
        OmnibusLearnerEntry("Top 20% learners", "Ctrl>Kir FLC", _export("a", [0.9, 0.8, 0.85])),
        OmnibusLearnerEntry("Top 20% learners", "PFNd>Kir FLC", _export("b", [0.5, 0.45, 0.55])),
        OmnibusLearnerEntry("Bottom 50% learners", "Ctrl>Kir FLC", _export("c", [0.35, 0.30, 0.40])),
        OmnibusLearnerEntry("Bottom 50% learners", "PFNd>Kir FLC", _export("d", [0.2, 0.25, 0.15])),
    ]

    fig = plot_omnibus_learner_overlays(
        entries,
        ylabel="Fraction of trajectories without wall contact",
        ymax=1.0,
        ytick_step=0.2,
        stats=True,
        opts=SimpleNamespace(
            imageFormat="png",
            fontSize=16,
            fontFamily=None,
        ),
    )

    ax = fig.axes[0]
    tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert tick_labels == [
        "Ctrl>Kir FLC\n(n=3)",
        "PFNd>Kir FLC\n(n=3)",
        "Ctrl>Kir FLC\n(n=3)",
        "PFNd>Kir FLC\n(n=3)",
    ]
    cluster_labels = [text.get_text() for text in ax.texts]
    assert "Top 20% learners" in cluster_labels
    assert "Bottom 50% learners" in cluster_labels
    np.testing.assert_allclose(np.diff(ax.get_yticks()), 0.2)
    plt.close(fig)
