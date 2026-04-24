from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.bin_edges import normalize_panel_edges
from src.plotting.overlay_training_metric_hist import (
    ExportedTrainingHistogram,
    load_export_npz,
    validate_alignment,
)
from src.plotting.palettes import group_metric_fill_color, group_metric_edge_color
from src.plotting.plot_customizer import PlotCustomizer


def _panel_y(h: ExportedTrainingHistogram, p_idx: int) -> np.ndarray:
    if h.per_fly:
        if h.mean is None or h.mean.shape[0] <= p_idx:
            return np.full((h.bins,), np.nan, dtype=float)
        return np.asarray(h.mean[p_idx], dtype=float)

    if h.counts.shape[0] <= p_idx:
        return np.zeros((h.bins,), dtype=float)
    counts = np.asarray(h.counts[p_idx], dtype=float)
    total = float(np.nansum(counts))
    if total <= 0:
        return np.full((h.bins,), np.nan, dtype=float)
    return counts / total


def _panel_n_label(h: ExportedTrainingHistogram, p_idx: int) -> str:
    if h.per_fly:
        if h.n_units_panel is not None and h.n_units_panel.shape[0] > p_idx:
            return str(int(h.n_units_panel[p_idx]))
        return "?"
    if h.n_used.shape[0] > p_idx:
        return str(int(h.n_used[p_idx]))
    return "?"


def _fmt_tick(v: float) -> str:
    if not np.isfinite(v):
        return ""
    if np.isclose(v, round(v), atol=1e-10):
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _flat_edges(edges_item) -> np.ndarray:
    edges = normalize_panel_edges(edges_item)
    if isinstance(edges, list):
        parts: list[np.ndarray] = []
        for g in edges:
            gg = np.asarray(g, dtype=float).ravel()
            if gg.size < 2:
                continue
            if parts and np.isclose(parts[-1][-1], gg[0]):
                parts.append(gg[1:])
            else:
                parts.append(gg)
        if not parts:
            return np.asarray([], dtype=float)
        return np.concatenate(parts)
    return np.asarray(edges, dtype=float).ravel()


def plot_filled_hist_overlays(
    hists: list[ExportedTrainingHistogram],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    alpha: float = 0.38,
    ymax: float | None = None,
    opts=None,
) -> plt.Figure:
    """
    Plot exported training histograms as transparent overlaid distributions.

    This intentionally omits confidence intervals and statistical annotations; it is
    meant for visual comparison of distribution shape.
    """
    if opts is None:
        opts = SimpleNamespace(imageFormat="png", fontSize=None, fontFamily=None)

    validate_alignment(hists)

    customizer = PlotCustomizer()
    font_size = getattr(opts, "fontSize", None)
    if font_size is not None:
        customizer.update_font_size(font_size)
    customizer.update_font_family(getattr(opts, "fontFamily", None))
    font_scale = max(float(customizer.increase_factor), 1.0)

    panel_labels = hists[0].panel_labels
    n_panels = len(panel_labels)
    bins = int(hists[0].bins)
    min_width = 7.5 * n_panels if bins >= 12 else 5.0 * n_panels
    width = max(min_width, min(12.0 * n_panels, 0.50 * bins * n_panels))
    height = 4.1 * min(1.0 + 0.10 * (font_scale - 1.0), 1.18)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(width, height),
        squeeze=False,
        sharey=True,
    )
    axes = axes[0]

    alpha = float(alpha)
    if not np.isfinite(alpha):
        alpha = 0.38
    alpha = min(max(alpha, 0.05), 1.0)

    for p_idx, (ax, panel_label) in enumerate(zip(axes, panel_labels)):
        edges = _flat_edges(hists[0].bin_edges[p_idx])
        if edges.size < 2:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue

        any_data = False
        for g_idx, h in enumerate(hists):
            y = _panel_y(h, p_idx)
            y = np.asarray(y, dtype=float).ravel()
            if y.size != edges.size - 1:
                continue
            if not np.any(np.isfinite(y)):
                continue

            any_data = True
            y_plot = np.where(np.isfinite(y), y, 0.0)
            centers = 0.5 * (edges[:-1] + edges[1:])
            x_poly = np.concatenate([[edges[0]], centers, [edges[-1]]])
            y_poly = np.concatenate([[0.0], y_plot, [0.0]])
            label = f"{h.group} (n={_panel_n_label(h, p_idx)})"
            fill_color = group_metric_fill_color(g_idx, None)
            edge_color = group_metric_edge_color(g_idx, None)

            ax.fill_between(
                x_poly,
                y_poly,
                color=fill_color,
                alpha=alpha,
                linewidth=0,
                label=label,
            )
            ax.plot(
                x_poly,
                y_poly,
                color=edge_color,
                linewidth=1.4,
                alpha=min(0.95, alpha + 0.35),
            )

        if not any_data:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue

        ax.set_title(panel_label)
        ax.set_xlim(float(edges[0]), float(edges[-1]))
        ax.set_ylim(bottom=0)
        if ymax is not None:
            ax.set_ylim(top=float(ymax))
        if xlabel:
            ax.set_xlabel(xlabel)
        if p_idx == 0 and ylabel:
            ax.set_ylabel(ylabel)

        tick_count = min(7, max(3, bins // 3))
        ticks = np.linspace(float(edges[0]), float(edges[-1]), tick_count)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_tick(t) for t in ticks])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            fontsize=max(8, min(float(customizer.in_plot_font_size), 13.0)),
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(right=min(fig.subplotpars.right, 0.76))
    return fig
