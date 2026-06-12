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
    exp = np.asarray(bundle["return_leg_tortuosity_excursion_bin_exp"], dtype=float)
    ctrl = np.asarray(bundle["return_leg_tortuosity_excursion_bin_ctrl"], dtype=float)
    if mode == "exp":
        return exp
    if mode == "ctrl":
        return ctrl
    if mode == "exp_minus_ctrl":
        return exp - ctrl
    raise ValueError(f"Unknown mode={mode!r}")


def _panel_labels(bundle) -> list[str]:
    pair_mode = bool(
        np.asarray(
            bundle.get("return_leg_tortuosity_excursion_bin_pair_mode", False)
        )
        .reshape(())
        .item()
    )
    if pair_mode:
        lower = np.asarray(
            bundle["return_leg_tortuosity_excursion_bin_pair_lower_mm"], dtype=float
        ).reshape(-1)
        upper = np.asarray(
            bundle["return_leg_tortuosity_excursion_bin_pair_upper_mm"], dtype=float
        ).reshape(-1)
        return [f"[{lo:g}, {hi:g}) mm" for lo, hi in zip(lower, upper)]

    edges = np.asarray(
        bundle["return_leg_tortuosity_excursion_bin_edges_mm"], dtype=float
    ).reshape(-1)
    open_ended = bool(
        np.asarray(
            bundle.get(
                "return_leg_tortuosity_excursion_bin_open_ended_upper_bin", False
            )
        )
        .reshape(())
        .item()
    )
    return [
        f"{lo:g}+ mm"
        if open_ended and idx == len(edges) - 2
        else f"[{lo:g}, {hi:g}) mm"
        for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))
    ]


def _bundle_to_exported(
    bundle,
    *,
    label: str,
    mode: str,
    sub_idx: np.ndarray,
) -> ExportedTrainingScalarBars:
    values = np.asarray(_mode_values(bundle, mode)[sub_idx, :], dtype=float)
    ids = np.asarray(bundle["video_ids"], dtype=object).reshape(-1)[sub_idx]

    samples_by_panel = []
    ids_by_panel = []
    means = []
    ci_lo = []
    ci_hi = []
    counts = []
    for panel_idx in range(values.shape[1]):
        column = values[:, panel_idx]
        finite = np.isfinite(column)
        samples = np.asarray(column[finite], dtype=float)
        sample_ids = np.asarray(ids[finite], dtype=object)
        mean, lo, hi, n = _ci_triplet(samples)
        samples_by_panel.append(samples)
        ids_by_panel.append(sample_ids)
        means.append(mean)
        ci_lo.append(lo)
        ci_hi.append(hi)
        counts.append(n)

    top_fraction = float(
        np.asarray(
            bundle.get("return_leg_tortuosity_excursion_bin_top_fraction", 1.0)
        )
        .reshape(())
        .item()
    )
    return_start_mode = str(
        np.asarray(
            bundle.get(
                "return_leg_tortuosity_excursion_bin_return_start_mode",
                "global_max",
            )
        )
        .reshape(())
        .item()
    )
    tail_prefix = (
        ""
        if top_fraction == 1.0
        else f"Top {100.0 * top_fraction:g}% mean "
    )
    start_prefix = (
        ""
        if return_start_mode == "global_max"
        else "Post-wall "
    )
    metric_label = f"{tail_prefix}{start_prefix}return-leg tortuosity"
    ylabel = (
        f"{metric_label} (exp - yok)"
        if mode == "exp_minus_ctrl"
        else f"{metric_label} (yoked)"
        if mode == "ctrl"
        else metric_label
    )
    return ExportedTrainingScalarBars(
        group=label,
        panel_labels=_panel_labels(bundle),
        per_unit_values_panel=np.asarray(samples_by_panel, dtype=object),
        per_unit_ids_panel=np.asarray(ids_by_panel, dtype=object),
        mean=np.asarray(means, dtype=float),
        ci_lo=np.asarray(ci_lo, dtype=float),
        ci_hi=np.asarray(ci_hi, dtype=float),
        n_units_panel=np.asarray(counts, dtype=int),
        meta={
            "y_label": ylabel,
            "base_title": (
                "Return-leg tortuosity by maximum distance bin"
                if top_fraction == 1.0 and return_start_mode == "global_max"
                else f"{metric_label} by maximum distance bin"
            ),
            "pool_trainings": False,
            "ci_conf": 0.95,
            "mode": mode,
            "metric": "return_leg_tortuosity_excursion_bin",
            "metric_palette_family": "between_reward_tortuosity",
            "top_fraction": top_fraction,
            "return_start_mode": return_start_mode,
        },
    )


def plot_return_leg_tortuosity_excursion_bin_sli_bundles(
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
        title=title,
        xlabel=xlabel or "Maximum distance bin from reward circle (mm)",
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
