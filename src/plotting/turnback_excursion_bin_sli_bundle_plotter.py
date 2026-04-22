from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.plotting.return_prob_outer_radius_sli_bundle_plotter import (
    _ci_triplet,
    _selected_groups,
)
from src.plotting.overlay_training_metric_scalar_bars import (
    ExportedTrainingScalarBars,
    plot_overlays,
)
from src.utils.common import writeImage


def _metric_arrays(bundle, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "ratio":
        exp = np.asarray(bundle["turnback_excursion_bin_ratio_exp"], dtype=float)
        ctrl = np.asarray(bundle["turnback_excursion_bin_ratio_ctrl"], dtype=float)
        return exp, ctrl

    if metric == "success":
        exp = np.asarray(bundle["turnback_excursion_bin_turn_exp"], dtype=float)
        ctrl = np.asarray(bundle["turnback_excursion_bin_turn_ctrl"], dtype=float)
        return exp, ctrl

    if metric == "total":
        exp = np.asarray(bundle["turnback_excursion_bin_total_exp"], dtype=float)
        ctrl = np.asarray(bundle["turnback_excursion_bin_total_ctrl"], dtype=float)
        return exp, ctrl

    if metric == "failure":
        exp_total = np.asarray(bundle["turnback_excursion_bin_total_exp"], dtype=float)
        ctrl_total = np.asarray(
            bundle["turnback_excursion_bin_total_ctrl"], dtype=float
        )
        exp_succ = np.asarray(bundle["turnback_excursion_bin_turn_exp"], dtype=float)
        ctrl_succ = np.asarray(bundle["turnback_excursion_bin_turn_ctrl"], dtype=float)
        return exp_total - exp_succ, ctrl_total - ctrl_succ

    raise ValueError(f"Unknown metric={metric!r}")


def _mode_values(bundle, mode: str, metric: str) -> np.ndarray:
    exp, ctrl = _metric_arrays(bundle, metric)
    if mode == "exp":
        return exp
    if mode == "ctrl":
        return ctrl
    if mode == "exp_minus_ctrl":
        return exp - ctrl
    raise ValueError(f"Unknown mode={mode!r}")


def _panel_labels(bundle) -> list[str]:
    edges = np.asarray(bundle["turnback_excursion_bin_edges_mm"], dtype=float).reshape(
        -1
    )
    return [f"[{float(a):g}, {float(b):g}) mm" for a, b in zip(edges[:-1], edges[1:])]


def _bundle_to_exported(
    bundle,
    *,
    label: str,
    mode: str,
    metric: str,
    sub_idx: np.ndarray,
) -> ExportedTrainingScalarBars:
    vals = _mode_values(bundle, mode, metric)
    video_ids_full = np.asarray(bundle["video_ids"], dtype=object).reshape(-1)
    vals = np.asarray(vals[sub_idx, :], dtype=float)
    video_ids = np.asarray(video_ids_full[sub_idx], dtype=object)

    panel_labels = _panel_labels(bundle)
    per_unit_values_panel = []
    per_unit_ids_panel = []
    means = []
    ci_lo = []
    ci_hi = []
    n_units_panel = []

    for p_idx in range(vals.shape[1]):
        col = np.asarray(vals[:, p_idx], dtype=float)
        mask = np.isfinite(col)
        used_vals = np.asarray(col[mask], dtype=float)
        used_ids = np.asarray(video_ids[mask], dtype=object)
        m, lo, hi, n = _ci_triplet(used_vals)
        per_unit_values_panel.append(used_vals)
        per_unit_ids_panel.append(used_ids)
        means.append(m)
        ci_lo.append(lo)
        ci_hi.append(hi)
        n_units_panel.append(n)

    if metric == "ratio":
        ylabel = (
            "Turnback ratio (exp - yok)"
            if mode == "exp_minus_ctrl"
            else (
                "Turnback ratio (yoked)"
                if mode == "ctrl"
                else "Turnback ratio"
            )
        )
        base_title = "Bin-averaged turnback ratio vs outer-radius bin"
    elif metric == "success":
        ylabel = (
            "Turnback success mass per fly (exp - yok)"
            if mode == "exp_minus_ctrl"
            else (
                "Turnback success mass per fly (yoked)"
                if mode == "ctrl"
                else "Turnback success mass per fly"
            )
        )
        base_title = "Integrated turnback success mass vs outer-radius bin"
    elif metric == "failure":
        ylabel = (
            "Non-turnback mass per fly (exp - yok)"
            if mode == "exp_minus_ctrl"
            else (
                "Non-turnback mass per fly (yoked)"
                if mode == "ctrl"
                else "Non-turnback mass per fly"
            )
        )
        base_title = "Integrated non-turnback mass vs outer-radius bin"
    elif metric == "total":
        ylabel = (
            "Resolved turnback excursions per fly (exp - yok)"
            if mode == "exp_minus_ctrl"
            else (
                "Resolved turnback excursions per fly (yoked)"
                if mode == "ctrl"
                else "Resolved turnback excursions per fly"
            )
        )
        base_title = "Resolved turnback excursion count vs outer-radius bin"
    else:
        raise ValueError(f"Unknown metric={metric!r}")

    meta = {
        "y_label": ylabel,
        "base_title": base_title,
        "pool_trainings": False,
        "ci_conf": 0.95,
        "mode": mode,
        "metric": metric,
        "metric_palette_family": "turnback_ratio",
    }
    return ExportedTrainingScalarBars(
        group=label,
        panel_labels=panel_labels,
        per_unit_values_panel=np.asarray(per_unit_values_panel, dtype=object),
        per_unit_ids_panel=np.asarray(per_unit_ids_panel, dtype=object),
        mean=np.asarray(means, dtype=float),
        ci_lo=np.asarray(ci_lo, dtype=float),
        ci_hi=np.asarray(ci_hi, dtype=float),
        n_units_panel=np.asarray(n_units_panel, dtype=int),
        meta=meta,
    )


def plot_turnback_excursion_bin_sli_bundles(
    bundles,
    out,
    *,
    labels: list[str] | None = None,
    mode: str = "exp",
    metric: str = "ratio",
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

    loaded = [np.load(path, allow_pickle=True) for path in bundles]

    if sli_top_fraction is None:
        sli_top_fraction = sli_fraction
    if sli_bottom_fraction is None:
        sli_bottom_fraction = sli_fraction

    exported = []
    for i, bundle in enumerate(loaded):
        bundle_label = (
            labels[i]
            if labels is not None and i < len(labels)
            else str(bundle["group_label"].reshape(()).item())
        )
        for subset_label, idx in _selected_groups(
            bundle,
            group_label=bundle_label,
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
                    metric=metric,
                    sub_idx=idx,
                )
            )

    if not exported:
        raise ValueError("No non-empty plotted groups after SLI filtering.")

    xlabel = xlabel or "Outer-radius bin from reward circle (mm)"
    ylabel = ylabel or exported[0].meta.get("y_label", "Turnback ratio")
    fig = plot_overlays(
        exported,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
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
