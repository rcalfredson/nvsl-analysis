from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.analysis.sli_bundle_utils import load_sli_bundle
from src.plotting.overlay_training_metric_scalar_bars import (
    ExportedTrainingScalarBars,
    plot_overlays,
)
from src.plotting.return_prob_outer_radius_sli_bundle_plotter import (
    _ci_triplet,
    _selected_groups,
)
from src.utils.common import writeImage


def _mode_values(bundle, mode: str) -> np.ndarray:
    exp = np.asarray(bundle["post_wall_departure_tortuosity_exp"], dtype=float)
    ctrl = np.asarray(bundle["post_wall_departure_tortuosity_ctrl"], dtype=float)
    if mode == "exp":
        return exp
    if mode == "ctrl":
        return ctrl
    if mode == "exp_minus_ctrl":
        return exp - ctrl
    raise ValueError(f"Unknown mode={mode!r}")


def _bundle_to_exported(
    bundle,
    *,
    label: str,
    mode: str,
    sub_idx: np.ndarray,
) -> ExportedTrainingScalarBars:
    values = np.asarray(_mode_values(bundle, mode)[sub_idx, 0], dtype=float)
    ids = np.asarray(bundle["video_ids"], dtype=object).reshape(-1)[sub_idx]
    finite = np.isfinite(values)
    samples = np.asarray(values[finite], dtype=float)
    sample_ids = np.asarray(ids[finite], dtype=object)
    mean, lo, hi, n = _ci_triplet(samples)
    ylabel = (
        "Post-wall departure tortuosity (exp - yok)"
        if mode == "exp_minus_ctrl"
        else "Post-wall departure tortuosity (yoked)"
        if mode == "ctrl"
        else "Post-wall departure tortuosity"
    )
    return ExportedTrainingScalarBars(
        group=label,
        panel_labels=["Wall-contact trajectories"],
        per_unit_values_panel=np.asarray([samples], dtype=object),
        per_unit_ids_panel=np.asarray([sample_ids], dtype=object),
        mean=np.asarray([mean], dtype=float),
        ci_lo=np.asarray([lo], dtype=float),
        ci_hi=np.asarray([hi], dtype=float),
        n_units_panel=np.asarray([n], dtype=int),
        meta={
            "y_label": ylabel,
            "base_title": "Tortuosity after final wall departure",
            "pool_trainings": False,
            "ci_conf": 0.95,
            "mode": mode,
            "metric": "post_wall_departure_tortuosity",
            "metric_palette_family": "between_reward_tortuosity",
        },
    )


def plot_post_wall_departure_tortuosity_sli_bundles(
    bundles,
    out,
    *,
    labels: list[str] | None = None,
    mode: str = "exp",
    sli_extremes=None,
    sli_fraction=None,
    sli_top_fraction=None,
    sli_bottom_fraction=None,
    standalone_extreme_labels: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymax: float | None = None,
    stats: bool = False,
    stats_alpha: float = 0.05,
    stats_paired: bool = False,
    debug: bool = False,
    bar_alpha: float = 0.90,
    opts=None,
):
    if opts is None:
        opts = SimpleNamespace(imageFormat="png", fontSize=None, fontFamily=None)
    loaded = [load_sli_bundle(path) for path in bundles]
    if sli_top_fraction is None:
        sli_top_fraction = sli_fraction
    if sli_bottom_fraction is None:
        sli_bottom_fraction = sli_fraction

    exported = []
    for idx, bundle in enumerate(loaded):
        group_label = (
            labels[idx]
            if labels is not None and idx < len(labels)
            else str(bundle["group_label"])
        )
        for subset_label, subset_idx in _selected_groups(
            bundle,
            group_label=group_label,
            sli_extremes=sli_extremes,
            sli_top_fraction=sli_top_fraction,
            sli_bottom_fraction=sli_bottom_fraction,
            standalone_extreme_labels=standalone_extreme_labels,
        ):
            exported.append(
                _bundle_to_exported(
                    bundle,
                    label=subset_label,
                    mode=mode,
                    sub_idx=subset_idx,
                )
            )
    if not exported:
        raise ValueError("No non-empty plotted groups after SLI filtering.")

    fig = plot_overlays(
        exported,
        title=title or "Tortuosity after final wall departure",
        xlabel=xlabel or "Fly group",
        ylabel=ylabel or exported[0].meta["y_label"],
        ymax=ymax,
        stats=bool(stats),
        stats_alpha=float(stats_alpha),
        stats_paired=bool(stats_paired),
        debug=bool(debug),
        bar_alpha=bar_alpha,
        opts=opts,
    )
    writeImage(out, format=getattr(opts, "imageFormat", "png"))
    return fig
