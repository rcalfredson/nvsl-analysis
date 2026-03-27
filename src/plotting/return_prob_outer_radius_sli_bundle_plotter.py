from __future__ import annotations

import csv
from typing import Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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


def _selected_groups(
    bundle,
    *,
    group_label: str,
    sli_extremes,
    sli_top_fraction,
    sli_bottom_fraction,
    standalone_extreme_labels: bool,
):
    selected = _selected_indices(
        bundle,
        sli_extremes=sli_extremes,
        top_fraction=sli_top_fraction,
        bottom_fraction=sli_bottom_fraction,
    )
    out = []
    if isinstance(selected, dict):
        for which in ("bottom", "top"):
            idx = np.asarray(selected[which], dtype=int)
            if idx.size == 0:
                continue
            out.append(
                (
                    _subset_plot_label(
                        group_label,
                        which=which,
                        sli_extremes=sli_extremes,
                        sli_top_fraction=sli_top_fraction,
                        sli_bottom_fraction=sli_bottom_fraction,
                        standalone_extreme_labels=standalone_extreme_labels,
                    ),
                    idx,
                )
            )
        return out

    idx = np.asarray(selected, dtype=int)
    if idx.size == 0:
        return out
    out.append(
        (
            _subset_plot_label(
                group_label,
                which=None,
                sli_extremes=sli_extremes,
                sli_top_fraction=sli_top_fraction,
                sli_bottom_fraction=sli_bottom_fraction,
                standalone_extreme_labels=standalone_extreme_labels,
            ),
            idx,
        )
    )
    return out


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


def _metric_arrays(bundle, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "ratio":
        exp = np.asarray(bundle["return_prob_outer_radius_ratio_exp"], dtype=float)
        ctrl = np.asarray(bundle["return_prob_outer_radius_ratio_ctrl"], dtype=float)
        return exp, ctrl

    if metric == "success":
        exp = np.asarray(bundle["return_prob_outer_radius_return_exp"], dtype=float)
        ctrl = np.asarray(bundle["return_prob_outer_radius_return_ctrl"], dtype=float)
        return exp, ctrl

    if metric == "total":
        exp = np.asarray(bundle["return_prob_outer_radius_total_exp"], dtype=float)
        ctrl = np.asarray(bundle["return_prob_outer_radius_total_ctrl"], dtype=float)
        return exp, ctrl

    if metric == "failure":
        exp_total = np.asarray(bundle["return_prob_outer_radius_total_exp"], dtype=float)
        ctrl_total = np.asarray(
            bundle["return_prob_outer_radius_total_ctrl"], dtype=float
        )
        exp_succ = np.asarray(bundle["return_prob_outer_radius_return_exp"], dtype=float)
        ctrl_succ = np.asarray(
            bundle["return_prob_outer_radius_return_ctrl"], dtype=float
        )
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


def _constant_positive_n(n_units_panel: np.ndarray) -> int | None:
    arr = np.asarray(n_units_panel, dtype=int).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return None
    uniq = np.unique(arr)
    if uniq.size != 1:
        return None
    return int(uniq[0])


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
            "Return probability (exp - yok)"
            if mode == "exp_minus_ctrl"
            else (
                "Return probability (yoked)"
                if mode == "ctrl"
                else "Return probability"
            )
        )
        base_title = "Return probability vs outer-circle radius"
    elif metric == "success":
        ylabel = (
            "Successful returns per fly (exp - yok)"
            if mode == "exp_minus_ctrl"
            else (
                "Successful returns per fly (yoked)"
                if mode == "ctrl"
                else "Successful returns per fly"
            )
        )
        base_title = "Successful return count vs outer-circle radius"
    elif metric == "failure":
        ylabel = (
            "Failed returns per fly (exp - yok)"
            if mode == "exp_minus_ctrl"
            else (
                "Failed returns per fly (yoked)"
                if mode == "ctrl"
                else "Failed returns per fly"
            )
        )
        base_title = "Failed return count vs outer-circle radius"
    elif metric == "total":
        ylabel = (
            "Resolved excursions per fly (exp - yok)"
            if mode == "exp_minus_ctrl"
            else (
                "Resolved excursions per fly (yoked)"
                if mode == "ctrl"
                else "Resolved excursions per fly"
            )
        )
        base_title = "Resolved excursion count vs outer-circle radius"
    else:
        raise ValueError(f"Unknown metric={metric!r}")

    meta = {
        "y_label": ylabel,
        "base_title": base_title,
        "pool_trainings": False,
        "ci_conf": 0.95,
        "mode": mode,
        "metric": metric,
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
    metric: str,
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
        for subset_label, idx in _selected_groups(
            bundle,
            group_label=bundle_label,
            sli_extremes=sli_extremes,
            sli_top_fraction=sli_top_fraction,
            sli_bottom_fraction=sli_bottom_fraction,
            standalone_extreme_labels=standalone_extreme_labels,
        ):
            plotted.append(
                _bundle_to_exported(
                    bundle,
                    label=subset_label,
                    mode=mode,
                    metric=metric,
                    sub_idx=idx,
                )
            )
    return plotted


def export_return_prob_outer_radius_csv(
    bundles,
    csv_out,
    *,
    labels: list[str] | None = None,
    mode: str = "exp",
    outer_delta_mm: float,
    sli_extremes=None,
    sli_fraction=None,
    sli_top_fraction=None,
    sli_bottom_fraction=None,
    standalone_extreme_labels: bool = False,
):
    if mode == "exp_minus_ctrl":
        raise ValueError(
            "CSV export does not support mode='exp_minus_ctrl'; use exp or ctrl."
        )

    loaded = [np.load(path, allow_pickle=True) for path in bundles]
    if sli_top_fraction is None:
        sli_top_fraction = sli_fraction
    if sli_bottom_fraction is None:
        sli_bottom_fraction = sli_fraction

    rows = []
    for i, bundle in enumerate(loaded):
        bundle_label = (
            labels[i]
            if labels is not None and i < len(labels)
            else str(bundle["group_label"].reshape(()).item())
        )
        deltas = np.asarray(
            bundle["return_prob_outer_radius_outer_deltas_mm"], dtype=float
        ).reshape(-1)
        matches = np.where(np.isclose(deltas, float(outer_delta_mm), atol=1e-9))[0]
        if matches.size == 0:
            raise ValueError(
                f"Requested outer_delta_mm={outer_delta_mm:g} not found in bundle "
                f"{bundle_label!r}; available deltas={deltas.tolist()}"
            )
        p_idx = int(matches[0])

        succ_exp = np.asarray(bundle["return_prob_outer_radius_return_exp"], dtype=float)
        succ_ctrl = np.asarray(
            bundle["return_prob_outer_radius_return_ctrl"], dtype=float
        )
        total_exp = np.asarray(bundle["return_prob_outer_radius_total_exp"], dtype=float)
        total_ctrl = np.asarray(
            bundle["return_prob_outer_radius_total_ctrl"], dtype=float
        )
        ratio_exp = np.asarray(bundle["return_prob_outer_radius_ratio_exp"], dtype=float)
        ratio_ctrl = np.asarray(
            bundle["return_prob_outer_radius_ratio_ctrl"], dtype=float
        )

        if mode == "exp":
            succ = succ_exp[:, p_idx]
            total = total_exp[:, p_idx]
            ratio = ratio_exp[:, p_idx]
        else:
            succ = succ_ctrl[:, p_idx]
            total = total_ctrl[:, p_idx]
            ratio = ratio_ctrl[:, p_idx]
        fail = total - succ

        video_ids = np.asarray(bundle["video_ids"], dtype=object).reshape(-1)
        video_fns = np.asarray(
            bundle["video_fns"] if "video_fns" in bundle else video_ids, dtype=object
        ).reshape(-1)
        if "fly_ids" in bundle:
            fly_ids = np.asarray(bundle["fly_ids"], dtype=int).reshape(-1)
        else:
            fly_ids = np.full(video_ids.shape, -1, dtype=int)

        for subset_label, idx in _selected_groups(
            bundle,
            group_label=bundle_label,
            sli_extremes=sli_extremes,
            sli_top_fraction=sli_top_fraction,
            sli_bottom_fraction=sli_bottom_fraction,
            standalone_extreme_labels=standalone_extreme_labels,
        ):
            for vi in np.asarray(idx, dtype=int):
                rows.append(
                    {
                        "group_label": str(bundle_label),
                        "subset_label": str(subset_label),
                        "mode": str(mode),
                        "outer_delta_mm": float(deltas[p_idx]),
                        "video_id": str(video_ids[vi]),
                        "video_fn": str(video_fns[vi]),
                        "fly_id": int(fly_ids[vi]),
                        "success_count": int(succ[vi]),
                        "failure_count": int(fail[vi]),
                        "total_count": int(total[vi]),
                        "ratio": (
                            float(ratio[vi]) if np.isfinite(ratio[vi]) else np.nan
                        ),
                    }
                )

    if not rows:
        raise ValueError("No CSV rows to write after SLI filtering.")

    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _stacked_group_summary(
    bundle,
    *,
    mode: str,
    sub_idx: np.ndarray,
):
    if mode == "exp_minus_ctrl":
        raise ValueError(
            "metric='stacked' does not support mode='exp_minus_ctrl'; use exp or ctrl."
        )

    succ_exp = np.asarray(bundle["return_prob_outer_radius_return_exp"], dtype=float)
    succ_ctrl = np.asarray(bundle["return_prob_outer_radius_return_ctrl"], dtype=float)
    total_exp = np.asarray(bundle["return_prob_outer_radius_total_exp"], dtype=float)
    total_ctrl = np.asarray(bundle["return_prob_outer_radius_total_ctrl"], dtype=float)

    if mode == "exp":
        succ = succ_exp
        total = total_exp
    else:
        succ = succ_ctrl
        total = total_ctrl

    fail = total - succ
    succ = np.asarray(succ[sub_idx, :], dtype=float)
    fail = np.asarray(fail[sub_idx, :], dtype=float)

    succ_mean = []
    fail_mean = []
    n_units_panel = []
    for p_idx in range(succ.shape[1]):
        succ_col = np.asarray(succ[:, p_idx], dtype=float)
        fail_col = np.asarray(fail[:, p_idx], dtype=float)
        mask = np.isfinite(succ_col) & np.isfinite(fail_col)
        if not np.any(mask):
            succ_mean.append(np.nan)
            fail_mean.append(np.nan)
            n_units_panel.append(0)
            continue
        succ_mean.append(float(np.mean(succ_col[mask])))
        fail_mean.append(float(np.mean(fail_col[mask])))
        n_units_panel.append(int(mask.sum()))

    return (
        np.asarray(succ_mean, dtype=float),
        np.asarray(fail_mean, dtype=float),
        np.asarray(n_units_panel, dtype=int),
    )


def _plot_return_prob_outer_radius_stacked(
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
):
    loaded = [np.load(path, allow_pickle=True) for path in bundles]

    if sli_top_fraction is None:
        sli_top_fraction = sli_fraction
    if sli_bottom_fraction is None:
        sli_bottom_fraction = sli_fraction

    plotted = []
    for i, bundle in enumerate(loaded):
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
                succ_mean, fail_mean, n_units_panel = _stacked_group_summary(
                    bundle, mode=mode, sub_idx=idx
                )
                plotted.append(
                    {
                        "label": _subset_plot_label(
                            bundle_label,
                            which=which,
                            sli_extremes=sli_extremes,
                            sli_top_fraction=sli_top_fraction,
                            sli_bottom_fraction=sli_bottom_fraction,
                            standalone_extreme_labels=standalone_extreme_labels,
                        ),
                        "succ_mean": succ_mean,
                        "fail_mean": fail_mean,
                        "n_units_panel": n_units_panel,
                    }
                )
        else:
            idx = np.asarray(selected, dtype=int)
            if idx.size == 0:
                continue
            succ_mean, fail_mean, n_units_panel = _stacked_group_summary(
                bundle, mode=mode, sub_idx=idx
            )
            plotted.append(
                {
                    "label": _subset_plot_label(
                        bundle_label,
                        which=None,
                        sli_extremes=sli_extremes,
                        sli_top_fraction=sli_top_fraction,
                        sli_bottom_fraction=sli_bottom_fraction,
                        standalone_extreme_labels=standalone_extreme_labels,
                    ),
                    "succ_mean": succ_mean,
                    "fail_mean": fail_mean,
                    "n_units_panel": n_units_panel,
                }
            )

    if not plotted:
        raise ValueError("No non-empty plotted groups after SLI filtering.")

    panel_labels = _panel_labels(loaded[0])
    P = len(panel_labels)
    G = len(plotted)
    x_centers = np.arange(P, dtype=float)

    fig_w = max(7.0, 1.15 * P + 1.8 * G)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 4.8))

    group_band = 0.80
    bar_w = group_band / max(1, G)
    gpos = np.arange(G) - (G - 1) / 2.0
    offsets = gpos * bar_w

    colors = list(plt.cm.tab10.colors)
    all_n = [d["n_units_panel"] for d in plotted]
    const_ns = [_constant_positive_n(n) for n in all_n]
    global_legend_n = None
    if const_ns and all(n is not None for n in const_ns):
        uniq = np.unique(np.asarray(const_ns, dtype=int))
        if uniq.size == 1:
            global_legend_n = int(uniq[0])

    group_handles = []
    max_height = 0.0
    for gi, d in enumerate(plotted):
        color = colors[gi % len(colors)]
        xg = x_centers + offsets[gi]
        succ = np.asarray(d["succ_mean"], dtype=float)
        fail = np.asarray(d["fail_mean"], dtype=float)
        succ_plot = np.where(np.isfinite(succ), succ, 0.0)
        fail_plot = np.where(np.isfinite(fail), fail, 0.0)
        total_plot = succ_plot + fail_plot
        if total_plot.size:
            max_height = max(max_height, float(np.nanmax(total_plot)))

        ax.bar(
            xg,
            succ_plot,
            width=bar_w,
            align="center",
            color=color,
            edgecolor=color,
            linewidth=0.8,
        )
        ax.bar(
            xg,
            fail_plot,
            width=bar_w,
            align="center",
            bottom=succ_plot,
            color=color,
            alpha=0.30,
            edgecolor=color,
            linewidth=0.8,
            hatch="//",
        )
        label = (
            f"{d['label']} (n={global_legend_n})"
            if global_legend_n is not None
            else d["label"]
        )
        group_handles.append(mpatches.Patch(facecolor=color, edgecolor=color, label=label))

    need_per_panel_n = global_legend_n is None
    if need_per_panel_n and max_height > 0:
        y_pad = 0.03 * max_height
        for p in range(P):
            ns = []
            y_top = 0.0
            for d in plotted:
                npp = int(d["n_units_panel"][p]) if p < len(d["n_units_panel"]) else 0
                if npp > 0:
                    ns.append(npp)
                succ = float(d["succ_mean"][p]) if np.isfinite(d["succ_mean"][p]) else 0.0
                fail = float(d["fail_mean"][p]) if np.isfinite(d["fail_mean"][p]) else 0.0
                y_top = max(y_top, succ + fail)
            if not ns:
                continue
            uniq = sorted(set(ns))
            n_text = (
                f"n={uniq[0]}" if len(uniq) == 1 else "n=" + "/".join(map(str, ns))
            )
            ax.text(
                float(x_centers[p]),
                float(y_top + y_pad),
                n_text,
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.2",
                clip_on=False,
                zorder=9,
            )

    component_handles = [
        mpatches.Patch(facecolor="0.35", edgecolor="0.35", label="Success"),
        mpatches.Patch(
            facecolor="0.75", edgecolor="0.35", hatch="//", label="Failure"
        ),
    ]

    ax.set_xticks(x_centers)
    ax.set_xticklabels(panel_labels, rotation=30, ha="right")
    ax.set_xlabel("Outer-circle radius delta from reward circle (mm)")
    ax.set_ylabel(
        "Resolved excursions per fly"
        if mode == "exp"
        else "Resolved excursions per fly (yoked)"
    )
    ax.set_ylim(bottom=0.0)
    if ymax is not None:
        ax.set_ylim(top=float(ymax))

    title_use = title
    if title_use is None:
        title_use = (
            "Return-probability success/failure composition vs outer-circle radius"
        )
    fig.suptitle(title_use)

    group_legend = ax.legend(handles=group_handles, loc="upper right", fontsize=8)
    ax.add_artist(group_legend)
    ax.legend(handles=component_handles, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    return fig


def plot_return_prob_outer_radius_sli_bundles(
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
    ymax: float | None = None,
    stats: bool = False,
    stats_alpha: float = 0.05,
    stats_paired: bool = False,
    debug: bool = False,
):
    if metric == "stacked":
        if stats:
            raise ValueError("metric='stacked' does not currently support --stats.")
        return _plot_return_prob_outer_radius_stacked(
            bundles,
            out,
            labels=labels,
            mode=mode,
            sli_extremes=sli_extremes,
            sli_fraction=sli_fraction,
            sli_top_fraction=sli_top_fraction,
            sli_bottom_fraction=sli_bottom_fraction,
            standalone_extreme_labels=standalone_extreme_labels,
            title=title,
            ymax=ymax,
        )

    loaded = [np.load(path, allow_pickle=True) for path in bundles]

    if sli_top_fraction is None:
        sli_top_fraction = sli_fraction
    if sli_bottom_fraction is None:
        sli_bottom_fraction = sli_fraction

    exported = _iter_plot_groups(
        loaded,
        labels,
        mode=mode,
        metric=metric,
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
