from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.between_reward_tortuosity_distance_box import (
    BetweenRewardTortuosityDistanceBoxResult,
)
from src.plotting.palettes import group_metric_edge_color, group_metric_fill_color
from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import writeImage


def _values(result: BetweenRewardTortuosityDistanceBoxResult, bin_idx: int) -> np.ndarray:
    if bin_idx < 0 or bin_idx >= len(result.values_by_bin):
        return np.asarray([], dtype=float)
    vals = np.asarray(result.values_by_bin[bin_idx], dtype=float)
    return vals[np.isfinite(vals)]


def _validate_results(results: list[BetweenRewardTortuosityDistanceBoxResult]) -> None:
    if not results:
        raise ValueError("at least one tortuosity box result is required")
    e0 = np.asarray(results[0].x_edges_mm, dtype=float)
    t0 = int(results[0].meta.get("training_index", -1))
    segment_level0 = bool(getattr(results[0], "segment_level", True))
    unit_stat0 = str(getattr(results[0], "unit_stat", "raw_segment"))
    for r in results[1:]:
        e = np.asarray(r.x_edges_mm, dtype=float)
        if e.shape != e0.shape or not np.allclose(e, e0, rtol=0, atol=1e-12):
            raise ValueError("x_edges_mm differ across inputs; re-export with shared bins")
        if int(r.meta.get("training_index", -1)) != t0:
            raise ValueError("training_index differs across inputs")
        if bool(getattr(r, "segment_level", True)) != segment_level0:
            raise ValueError("segment_level differs across inputs")
        if str(getattr(r, "unit_stat", "raw_segment")) != unit_stat0:
            raise ValueError("unit_stat differs across inputs")


def plot_box_results(
    results: list[BetweenRewardTortuosityDistanceBoxResult],
    *,
    group_labels: list[str],
    out_file: str,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    showfliers: bool = False,
    ymax: float | None = None,
    opts=None,
) -> plt.Figure:
    _validate_results(results)
    if opts is None:
        opts = SimpleNamespace(imageFormat="png", fontSize=None, fontFamily=None)
    if len(group_labels) != len(results):
        raise ValueError("group_labels length must match results length")

    customizer = PlotCustomizer()
    font_size = getattr(opts, "fontSize", None)
    if font_size is not None:
        customizer.update_font_size(font_size)
    customizer.update_font_family(getattr(opts, "fontFamily", None))

    edges = np.asarray(results[0].x_edges_mm, dtype=float)
    labels = [str(x) for x in results[0].bin_labels]
    B = int(len(labels))
    G = int(len(results))

    width = max(7.5, 0.72 * B + 1.1 * G)
    fig, ax = plt.subplots(1, 1, figsize=(width, 4.4))

    centers = np.arange(B, dtype=float)
    group_width = 0.78
    box_width = group_width / max(1, G)
    offsets = (np.arange(G, dtype=float) - (G - 1) / 2.0) * box_width

    legend_handles = []
    for g_idx, (res, label) in enumerate(zip(results, group_labels)):
        data = []
        positions = []
        for b_idx in range(B):
            vals = _values(res, b_idx)
            if vals.size == 0:
                continue
            data.append(vals)
            positions.append(float(centers[b_idx] + offsets[g_idx]))
        if not data:
            continue

        fill = group_metric_fill_color(g_idx, "between_reward_distance")
        edge = group_metric_edge_color(g_idx, "between_reward_distance")
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width * 0.82,
            patch_artist=True,
            showfliers=showfliers,
            manage_ticks=False,
            boxprops={"facecolor": fill, "edgecolor": edge, "linewidth": 1.1},
            whiskerprops={"color": edge, "linewidth": 1.0},
            capprops={"color": edge, "linewidth": 1.0},
            medianprops={"color": "black", "linewidth": 1.25},
            flierprops={
                "marker": "o",
                "markersize": 2.5,
                "markerfacecolor": fill,
                "markeredgecolor": edge,
                "alpha": 0.45,
            },
        )
        if bp["boxes"]:
            legend_handles.append(bp["boxes"][0])
            bp["boxes"][0].set_label(label)

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlim(-0.75, B - 0.25)
    ax.set_xlabel(
        xlabel
        or str(
            results[0].meta.get(
                "x_label", "Maximum distance from reward circle center [mm]"
            )
        )
    )
    if ylabel is None:
        ylabel = str(results[0].meta.get("y_label", "Loop tortuosity"))
        if not bool(getattr(results[0], "segment_level", True)):
            stat = str(getattr(results[0], "unit_stat", "median"))
            ylabel = f"Per-fly {stat} {ylabel.lower()}"
    ax.set_ylabel(ylabel)
    if title is None:
        t_idx = int(results[0].meta.get("training_index", 0)) + 1
        title = f"training {t_idx}"
    if title:
        ax.set_title(title)
    if ymax is not None:
        ax.set_ylim(top=float(ymax))
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=max(8, min(float(customizer.in_plot_font_size), 13.0)),
        )

    fig.tight_layout()
    fig.subplots_adjust(right=min(fig.subplotpars.right, 0.80), bottom=max(fig.subplotpars.bottom, 0.24))
    writeImage(out_file, format=getattr(opts, "imageFormat", "png"))
    return fig


def plot_single_box_result(
    result: BetweenRewardTortuosityDistanceBoxResult,
    *,
    out_file: str,
    customizer: PlotCustomizer | None = None,
    showfliers: bool = False,
    ymax: float | None = None,
) -> plt.Figure:
    opts = SimpleNamespace(imageFormat="png", fontSize=None, fontFamily=None)
    if customizer is not None:
        opts.fontSize = getattr(customizer, "font_size", None)
    return plot_box_results(
        [result],
        group_labels=[""],
        out_file=out_file,
        showfliers=showfliers,
        ymax=ymax,
        opts=opts,
    )
