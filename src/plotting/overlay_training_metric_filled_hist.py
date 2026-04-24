from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class DistributionOperationSpec:
    label: str
    term1: str
    operation: str
    term2: str


def _panel_y(h: ExportedTrainingHistogram, p_idx: int) -> np.ndarray:
    if h.meta.get("filled_hist_derived_y", False):
        if h.mean is None or h.mean.shape[0] <= p_idx:
            return np.full((h.bins,), np.nan, dtype=float)
        return np.asarray(h.mean[p_idx], dtype=float)

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


def _normalize_operation(operation: str) -> str:
    op = str(operation).strip().lower()
    aliases = {
        "+": "add",
        "plus": "add",
        "sum": "add",
        "add": "add",
        "-": "subtract",
        "minus": "subtract",
        "sub": "subtract",
        "subtract": "subtract",
    }
    if op not in aliases:
        raise ValueError(
            f"Unsupported distribution operation {operation!r}; use add or subtract."
        )
    return aliases[op]


def parse_distribution_operation_spec(spec: str) -> DistributionOperationSpec:
    """
    Parse a derived-distribution spec.

    Expected form:
        "Result label=Term 1 label,subtract,Term 2 label"

    Operation may be add, subtract, +, or -.
    """
    if "=" not in spec:
        raise ValueError(
            "Derived distribution specs must be 'Result=Term1,operation,Term2'."
        )
    label, expr = spec.split("=", 1)
    label = label.strip()
    parts = [p.strip() for p in expr.split(",", 2)]
    if not label or len(parts) != 3 or not all(parts):
        raise ValueError(
            "Derived distribution specs must be 'Result=Term1,operation,Term2'."
        )
    return DistributionOperationSpec(
        label=label,
        term1=parts[0],
        operation=_normalize_operation(parts[1]),
        term2=parts[2],
    )


def _derived_n_used(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.zeros_like(a, dtype=int)
    return np.minimum(a, b).astype(int)


def _renormalize_distribution(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    finite = np.isfinite(y)
    total = float(np.nansum(y[finite]))
    if not finite.any() or np.isclose(total, 0.0, atol=1e-12):
        return y
    y[finite] = y[finite] / total
    return y


def apply_distribution_operations(
    hists: list[ExportedTrainingHistogram],
    specs: list[DistributionOperationSpec],
    *,
    include_inputs: bool = True,
    renormalize: bool = False,
) -> list[ExportedTrainingHistogram]:
    """
    Return input histograms plus optional derived add/subtract distributions.

    Derived distributions are computed from the same panel y-values that this filled
    overlay plot draws, so pooled exports are combined after conversion to
    proportions and per-fly exports are combined from their exported means. If
    renormalize is enabled, each derived panel is divided by its finite bin sum
    after the operation. Panels with zero finite sum are left unchanged.
    """
    if not specs:
        return list(hists) if include_inputs else []

    validate_alignment(hists)
    by_label = {h.group: h for h in hists}
    label_counts: dict[str, int] = {}
    for h in hists:
        label_counts[h.group] = label_counts.get(h.group, 0) + 1
    duplicate_labels = sorted(label for label, n in label_counts.items() if n > 1)
    if duplicate_labels:
        raise ValueError(
            "Distribution operation terms require unique --input labels; duplicates: "
            + ", ".join(duplicate_labels)
        )

    derived: list[ExportedTrainingHistogram] = []
    for spec in specs:
        if spec.term1 not in by_label:
            raise ValueError(f"Unknown term1 label {spec.term1!r} in {spec.label!r}.")
        if spec.term2 not in by_label:
            raise ValueError(f"Unknown term2 label {spec.term2!r} in {spec.label!r}.")
        if spec.label in by_label:
            raise ValueError(f"Derived label {spec.label!r} already exists.")

        left = by_label[spec.term1]
        right = by_label[spec.term2]
        panel_values = []
        for p_idx in range(len(left.panel_labels)):
            y1 = _panel_y(left, p_idx)
            y2 = _panel_y(right, p_idx)
            if y1.shape != y2.shape:
                raise ValueError(
                    f"Cannot combine {spec.term1!r} and {spec.term2!r}; panel "
                    f"{left.panel_labels[p_idx]!r} y-shapes differ."
                )
            if spec.operation == "add":
                y = y1 + y2
            elif spec.operation == "subtract":
                y = y1 - y2
            else:
                raise ValueError(
                    f"Unsupported distribution operation {spec.operation!r}."
                )
            if renormalize:
                y = _renormalize_distribution(y)
            panel_values.append(y)

        meta = dict(left.meta)
        meta["filled_hist_derived_y"] = True
        meta["filled_hist_derived_renormalized"] = bool(renormalize)
        meta["filled_hist_operation"] = {
            "term1": spec.term1,
            "operation": spec.operation,
            "term2": spec.term2,
        }
        mean = np.asarray(panel_values, dtype=float)
        n_used = _derived_n_used(left.n_used, right.n_used)
        n_units_panel = None
        if left.n_units_panel is not None and right.n_units_panel is not None:
            n_units_panel = _derived_n_used(left.n_units_panel, right.n_units_panel)

        derived_hist = ExportedTrainingHistogram(
            group=spec.label,
            panel_labels=list(left.panel_labels),
            counts=np.zeros_like(left.counts, dtype=float),
            mean=mean,
            ci_lo=None,
            ci_hi=None,
            n_units=None,
            n_units_panel=n_units_panel,
            per_unit_panel=None,
            per_unit_ids_panel=None,
            bin_edges=left.bin_edges,
            n_used=n_used,
            meta=meta,
        )
        derived.append(derived_hist)
        by_label[spec.label] = derived_hist

    if include_inputs:
        return list(hists) + derived
    return derived


def _fmt_tick(v: float) -> str:
    if not np.isfinite(v):
        return ""
    if np.isclose(v, round(v), atol=1e-10):
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _wrapped_axis_label_text(text: str) -> str:
    text = str(text)
    if "\n" in text:
        return text
    if " from " in text:
        return text.replace(" from ", "\nfrom ", 1)
    if " (" in text:
        return text.replace(" (", "\n(", 1)
    if "(" in text and text.index("(") > 8:
        return text.replace("(", "\n(", 1)
    return text


def _ensure_axis_labels_visible(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    labels = []
    xlabels = []
    for ax in axes:
        xlabel = ax.xaxis.get_label()
        ylabel = ax.yaxis.get_label()
        if xlabel.get_text():
            labels.append(xlabel)
            xlabels.append(xlabel)
        if ylabel.get_text():
            labels.append(ylabel)
    if not labels:
        return

    pad_px = 6.0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox

    def _overflow_px() -> tuple[float, float, float, float]:
        left = right = bottom = top = 0.0
        for label in labels:
            bbox = label.get_window_extent(renderer=renderer)
            left = max(left, max((fig_bbox.x0 + pad_px) - bbox.x0, 0.0))
            right = max(right, max(bbox.x1 - (fig_bbox.x1 - pad_px), 0.0))
            bottom = max(bottom, max((fig_bbox.y0 + pad_px) - bbox.y0, 0.0))
            top = max(top, max(bbox.y1 - (fig_bbox.y1 - pad_px), 0.0))
        return left, right, bottom, top

    overflow = _overflow_px()
    if not any(v > 0 for v in overflow):
        return

    changed = False
    for label in xlabels:
        wrapped = _wrapped_axis_label_text(label.get_text())
        if wrapped != label.get_text():
            label.set_text(wrapped)
            changed = True

    if changed:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overflow = _overflow_px()
        if not any(v > 0 for v in overflow):
            return

    fig_w_px = max(fig.get_size_inches()[0] * fig.dpi, 1.0)
    fig_h_px = max(fig.get_size_inches()[1] * fig.dpi, 1.0)
    overflow_left, overflow_right, overflow_bottom, overflow_top = overflow

    left = min(fig.subplotpars.left + overflow_left / fig_w_px + 0.01, 0.35)
    right = max(fig.subplotpars.right - overflow_right / fig_w_px - 0.01, 0.55)
    bottom = min(fig.subplotpars.bottom + overflow_bottom / fig_h_px + 0.01, 0.40)
    top = max(fig.subplotpars.top - overflow_top / fig_h_px - 0.01, 0.60)

    if right > left and top > bottom:
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        fig.canvas.draw()


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
        data_min = np.inf
        data_max = -np.inf
        for g_idx, h in enumerate(hists):
            y = _panel_y(h, p_idx)
            y = np.asarray(y, dtype=float).ravel()
            if y.size != edges.size - 1:
                continue
            if not np.any(np.isfinite(y)):
                continue

            any_data = True
            finite_y = y[np.isfinite(y)]
            data_min = min(data_min, float(np.nanmin(finite_y)))
            data_max = max(data_max, float(np.nanmax(finite_y)))
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
        if np.isfinite(data_min) and data_min < 0:
            ax.axhline(0, color="0.25", linewidth=0.8, alpha=0.45)
            ax.set_ylim(bottom=float(data_min) * 1.08)
        else:
            ax.set_ylim(bottom=0)
        if ymax is not None:
            ax.set_ylim(top=float(ymax))
        elif np.isfinite(data_max) and data_min < 0:
            ax.set_ylim(top=max(float(data_max) * 1.08, 0.0))
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
    _ensure_axis_labels_visible(fig, list(axes))
    return fig
