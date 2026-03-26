from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.analysis.sli_tools import select_fractional_groups
from src.plotting.overlay_training_metric_scalar_bars import (
    ExportedTrainingScalarBars,
    plot_overlays,
)
from src.utils.util import meanConfInt


def _pct_label(which, frac):
    if frac is None:
        return which
    return f"{which} {int(round(100 * frac))}%"


def _selected_indices(
    bundle, *, sli_extremes=None, top_fraction=None, bottom_fraction=None
):
    n = len(bundle["sli"])
    if sli_extremes is None:
        return np.arange(n, dtype=int)

    sli = np.asarray(bundle["sli"], dtype=float)
    bottom, top = select_fractional_groups(
        pd.Series(sli),
        top_fraction=top_fraction,
        bottom_fraction=bottom_fraction,
    )
    bottom = np.asarray([] if bottom is None else bottom, dtype=int)
    top = np.asarray([] if top is None else top, dtype=int)

    if sli_extremes == "top":
        return top
    if sli_extremes == "bottom":
        return bottom
    if sli_extremes == "both":
        return {"bottom": bottom, "top": top}
    raise ValueError(f"Unsupported sli_extremes={sli_extremes!r}")


def _subset_plot_label(
    group_label: str,
    *,
    which: str | None,
    sli_extremes,
    sli_top_fraction,
    sli_bottom_fraction,
    standalone_extreme_labels: bool,
) -> str:
    label = str(group_label)
    if which == "top":
        extreme_label = _pct_label("Top", sli_top_fraction)
        return (
            extreme_label
            if standalone_extreme_labels
            else f"{label} ({extreme_label.lower()})"
        )
    if which == "bottom":
        extreme_label = _pct_label("Bottom", sli_bottom_fraction)
        return (
            extreme_label
            if standalone_extreme_labels
            else f"{label} ({extreme_label.lower()})"
        )
    if sli_extremes == "top":
        return f"{label} ({_pct_label('top', sli_top_fraction)})"
    if sli_extremes == "bottom":
        return f"{label} ({_pct_label('bottom', sli_bottom_fraction)})"
    return label


def _mode_values(bundle, mode: str) -> np.ndarray:
    exp = np.asarray(bundle["return_prob_outer_radius_ratio_exp"], dtype=float)
    ctrl = np.asarray(bundle["return_prob_outer_radius_ratio_ctrl"], dtype=float)
    if mode == "exp":
        return exp
    if mode == "ctrl":
        return ctrl
    if mode == "exp_minus_ctrl":
        return exp - ctrl
    raise ValueError(f"Unknown mode={mode!r}")


def _panel_labels(bundle) -> list[str]:
    outer_deltas = np.asarray(
        bundle["return_prob_outer_radius_outer_deltas_mm"], dtype=float
    ).reshape(-1)
    return [f"+{float(x):g} mm" for x in outer_deltas]


def _ci_triplet(x: np.ndarray) -> tuple[float, float, float, int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan, 0
    m, lo, hi, n = meanConfInt(x, conf=0.95)
    return float(m), float(lo), float(hi), int(n)


def _bundle_to_exported(
    bundle,
    *,
    label: str,
    mode: str,
    sub_idx: np.ndarray,
) -> ExportedTrainingScalarBars:
    vals = _mode_values(bundle, mode)
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

    ylabel = (
        "Return probability (exp - yok)"
        if mode == "exp_minus_ctrl"
        else ("Return probability (yoked)" if mode == "ctrl" else "Return probability")
    )
    meta = {
        "y_label": ylabel,
        "base_title": "Return probability vs outer-circle radius",
        "pool_trainings": False,
        "ci_conf": 0.95,
        "mode": mode,
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


def _iter_plot_groups(
    bundles: Iterable[dict],
    labels: list[str] | None,
    *,
    mode: str,
    sli_extremes,
    sli_top_fraction,
    sli_bottom_fraction,
    standalone_extreme_labels: bool,
):
    plotted = []
    for i, bundle in enumerate(bundles):
        bundle_label = (
            labels[i]
            if labels is not None and i < len(labels)
            else str(bundle["group_label"].reshape(()).item())
        )
        selected = _selected_indices(
            bundle,
            sli_extremes=sli_extremes,
            top_fraction=sli_top_fraction,
            bottom_fraction=sli_bottom_fraction,
        )
        if isinstance(selected, dict):
            for which in ("bottom", "top"):
                idx = np.asarray(selected[which], dtype=int)
                if idx.size == 0:
                    continue
                plotted.append(
                    _bundle_to_exported(
                        bundle,
                        label=_subset_plot_label(
                            bundle_label,
                            which=which,
                            sli_extremes=sli_extremes,
                            sli_top_fraction=sli_top_fraction,
                            sli_bottom_fraction=sli_bottom_fraction,
                            standalone_extreme_labels=standalone_extreme_labels,
                        ),
                        mode=mode,
                        sub_idx=idx,
                    )
                )
        else:
            idx = np.asarray(selected, dtype=int)
            if idx.size == 0:
                continue
            plotted.append(
                _bundle_to_exported(
                    bundle,
                    label=_subset_plot_label(
                        bundle_label,
                        which=None,
                        sli_extremes=sli_extremes,
                        sli_top_fraction=sli_top_fraction,
                        sli_bottom_fraction=sli_bottom_fraction,
                        standalone_extreme_labels=standalone_extreme_labels,
                    ),
                    mode=mode,
                    sub_idx=idx,
                )
            )
    return plotted


def plot_return_prob_outer_radius_sli_bundles(
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
    ymax: float | None = None,
    stats: bool = False,
    stats_alpha: float = 0.05,
    stats_paired: bool = False,
    debug: bool = False,
):
    loaded = [np.load(path, allow_pickle=True) for path in bundles]

    if sli_top_fraction is None:
        sli_top_fraction = sli_fraction
    if sli_bottom_fraction is None:
        sli_bottom_fraction = sli_fraction

    exported = _iter_plot_groups(
        loaded,
        labels,
        mode=mode,
        sli_extremes=sli_extremes,
        sli_top_fraction=sli_top_fraction,
        sli_bottom_fraction=sli_bottom_fraction,
        standalone_extreme_labels=standalone_extreme_labels,
    )
    if not exported:
        raise ValueError("No non-empty plotted groups after SLI filtering.")

    ylabel = exported[0].meta.get("y_label", "Return probability")
    fig = plot_overlays(
        exported,
        title=title,
        xlabel="Outer-circle radius delta from reward circle (mm)",
        ylabel=ylabel,
        ymax=ymax,
        stats=bool(stats),
        stats_alpha=float(stats_alpha),
        stats_paired=bool(stats_paired),
        debug=bool(debug),
    )
    fig.savefig(out, bbox_inches="tight")
    return fig
