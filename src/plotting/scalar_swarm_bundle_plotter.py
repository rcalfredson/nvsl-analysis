from __future__ import annotations

import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway

from src.plotting.overlay_training_metric_scalar_bars import (
    ExportedTrainingScalarBars,
    _ensure_xlabel_visible,
    _ensure_ylabel_visible,
    validate_alignment,
)
from src.plotting.palettes import (
    NEUTRAL_DARK,
    group_metric_edge_color,
    group_metric_fill_color,
    normalize_metric_palette_family,
)
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.stats_bars import StatAnnotConfig, annotate_grouped_bars_per_bin


def savefig_with_format(
    out_path: str, image_format: str, *, log_tag: str = "scalar_swarm"
) -> str:
    file_extension = "." + str(image_format).lstrip(".")
    base, ext = os.path.splitext(out_path)
    if ext.lower() != file_extension.lower():
        out_path = base + file_extension
        print(
            f"The file extension has been changed to {file_extension} to coincide with the specified format."
        )
    plt.savefig(out_path, bbox_inches="tight", format=image_format)
    print(f"[{log_tag}] wrote {out_path}")
    return out_path


def metric_palette_family(xs: list[ExportedTrainingScalarBars]) -> str | None:
    families = []
    for x in xs:
        meta = dict(x.meta or {})
        family = normalize_metric_palette_family(
            meta.get("metric_palette_family") or meta.get("metric")
        )
        if family is not None:
            families.append(family)
    uniq = []
    for family in families:
        if family not in uniq:
            uniq.append(family)
    return uniq[0] if len(uniq) == 1 else None


def group_value_matrix(x: ExportedTrainingScalarBars) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a (max_panel_n, P) matrix of plotted values for independent stats.

    Row identity is irrelevant for independent group tests, so this preserves
    every plotted value even when unit IDs repeat.
    """
    P = len(x.panel_labels)
    vals_by_panel = []
    max_n = 0
    for p in range(P):
        vals = np.asarray(x.per_unit_values_panel[p], dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        vals_by_panel.append(vals)
        max_n = max(max_n, int(vals.size))

    M = np.full((max_n, P), np.nan, dtype=float)
    I = np.asarray([f"row_{i}" for i in range(max_n)], dtype=object)
    for p, vals in enumerate(vals_by_panel):
        if vals.size:
            M[: vals.size, p] = vals

    return M, I


def wrap_long_ylabel(text: str) -> str:
    text = str(text)
    if "\n" in text or len(text) < 42:
        return text
    if ": " in text:
        text = text.replace(": ", ":\n", 1)
    if " vs " in text:
        text = text.replace(" vs ", "\nvs ", 1)
    return text


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    if p <= 0.0 or p < 1e-300:
        return "p<1e-300"
    if p < 1e-4:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def _anova_text(
    xs: list[ExportedTrainingScalarBars], *, min_n_per_group: int = 3
) -> str | None:
    if len(xs) < 3:
        return None
    P = len(xs[0].panel_labels)
    lines: list[str] = []
    for p in range(P):
        samples = []
        for x in xs:
            vals = np.asarray(x.per_unit_values_panel[p], dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size < int(min_n_per_group):
                samples = []
                break
            samples.append(vals)
        if len(samples) < 3:
            continue
        try:
            f_stat, p_val = f_oneway(*samples)
        except Exception:
            continue
        n_total = int(sum(s.size for s in samples))
        df_between = len(samples) - 1
        df_within = n_total - len(samples)
        if not np.isfinite(f_stat) or df_within <= 0:
            continue
        line = f"ANOVA F({df_between}, {df_within})={float(f_stat):.2f}, {_fmt_p(float(p_val))}"
        if P > 1:
            line = f"{xs[0].panel_labels[p]}: {line}"
        lines.append(line)
    return "\n".join(lines) if lines else None


def _add_anova_box(ax: plt.Axes, text: str, *, loc: str, fontsize: float) -> None:
    loc = str(loc or "upper_left").lower()
    if loc == "lower_right":
        xy = (0.97, 0.04)
        ha = "right"
        va = "bottom"
    elif loc == "lower_left":
        xy = (0.03, 0.04)
        ha = "left"
        va = "bottom"
    elif loc == "upper_right":
        xy = (0.97, 0.96)
        ha = "right"
        va = "top"
    else:
        xy = (0.03, 0.96)
        ha = "left"
        va = "top"
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.86,
            "linewidth": 0.6,
        },
        zorder=12,
    )


def plot_swarm_overlays(
    xs: list[ExportedTrainingScalarBars],
    *,
    out: str,
    image_format: str = "png",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymax: float | None = None,
    ymin: float | None = None,
    zero_line: bool = False,
    stats: bool = False,
    stats_alpha: float = 0.05,
    stats_debug: bool = False,
    show_anova: bool = False,
    anova_loc: str = "upper_left",
    log_tag: str = "scalar_swarm",
    opts=None,
):
    if opts is None:
        opts = SimpleNamespace(fontSize=None, fontFamily=None)
    validate_alignment(xs)

    customizer = PlotCustomizer()
    if getattr(opts, "fontSize", None) is not None:
        customizer.update_font_size(getattr(opts, "fontSize"))
    customizer.update_font_family(getattr(opts, "fontFamily", None))

    labels = xs[0].panel_labels
    P = len(labels)
    G = len(xs)
    single_panel = P == 1
    palette_family = metric_palette_family(xs)

    fig_w = max(5.5, (1.0 * G if single_panel else 1.2 * P))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 4.4))

    rng = np.random.default_rng(0)
    if single_panel:
        centers = np.arange(G, dtype=float)
        panel_offsets = np.zeros((G,), dtype=float)
        tick_pos = centers
        tick_labels = []
    else:
        centers = np.arange(P, dtype=float)
        group_band = 0.76
        step = group_band / max(1, G)
        panel_offsets = (np.arange(G, dtype=float) - (G - 1) / 2.0) * step
        tick_pos = centers
        tick_labels = labels

    legend_handles = []
    xpos_by_group = []
    hi_by_group = []
    per_unit_by_group = []
    per_unit_ids_by_group = []

    for gi, x in enumerate(xs):
        fill = group_metric_fill_color(gi, palette_family)
        edge = group_metric_edge_color(gi, palette_family)
        x_positions = np.full((P,), np.nan, dtype=float)
        y_tops = np.full((P,), np.nan, dtype=float)
        for p in range(P):
            vals = np.asarray(x.per_unit_values_panel[p], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            xpos = centers[gi] if single_panel else centers[p] + panel_offsets[gi]
            x_positions[p] = float(xpos)
            y_tops[p] = float(np.nanmax(vals))
            jitter_width = 0.13 if single_panel else min(0.11, 0.32 / max(1, G))
            jitter = rng.uniform(-jitter_width, jitter_width, size=vals.size)
            ax.scatter(
                np.full(vals.size, xpos) + jitter,
                vals,
                s=22,
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.7,
                alpha=0.78,
                zorder=3,
            )
            mean = float(np.nanmean(vals))
            half = 0.22 if single_panel else min(0.18, 0.42 / max(1, G))
            (mean_line,) = ax.plot(
                [xpos - half, xpos + half],
                [mean, mean],
                color=NEUTRAL_DARK,
                linewidth=1.35,
                zorder=4,
            )
            if not single_panel and p == 0:
                mean_line.set_label(x.group)
                legend_handles.append(mean_line)
        if single_panel:
            n = int(x.n_units_panel[0]) if x.n_units_panel.size else 0
            tick_labels.append(f"{x.group}\n(n={n})" if n > 0 else x.group)
        xpos_by_group.append(x_positions)
        hi_by_group.append(y_tops)
        M, I = group_value_matrix(x)
        per_unit_by_group.append(M)
        per_unit_ids_by_group.append(I)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=25 if single_panel else 30, ha="right")
    if xlabel and not (single_panel and str(xlabel).strip().lower() == "training"):
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(wrap_long_ylabel(ylabel))
    if zero_line:
        ax.axhline(0, color="0.45", linewidth=0.8, linestyle="--", zorder=1)

    all_vals = np.concatenate(
        [
            np.asarray(panel, dtype=float).ravel()
            for x in xs
            for panel in x.per_unit_values_panel
        ]
    )
    finite_vals = all_vals[np.isfinite(all_vals)]
    if ymin is not None or ymax is not None:
        if ymin is not None:
            ax.set_ylim(bottom=float(ymin))
        if ymax is not None:
            ax.set_ylim(top=float(ymax))
    elif finite_vals.size:
        y0 = float(np.nanmin(finite_vals))
        y1 = float(np.nanmax(finite_vals))
        span = y1 - y0
        if not np.isfinite(span) or span <= 0:
            span = max(abs(y1), 1.0)
        bottom = min(0.0, y0)
        top = max(0.0, y1) if zero_line else y1
        ax.set_ylim(bottom=bottom - 0.06 * span, top=top + 0.12 * span)

    if stats and G >= 2:
        cfg_stats = StatAnnotConfig(
            alpha=float(stats_alpha),
            min_n_per_group=3,
            headroom_frac=0.30,
            stack_gap_frac=0.050,
            gap_above_bars_frac=0.045,
            nlabel_off_frac=0.0,
            bracket_fontsize=max(
                7, min(float(customizer.in_plot_font_size) - 5.0, 10.0)
            ),
        )
        stats_x_centers = (
            np.asarray([0.5 * max(G - 1, 0)], dtype=float)
            if single_panel
            else tick_pos
        )
        annotate_grouped_bars_per_bin(
            ax,
            x_centers=stats_x_centers,
            xpos_by_group=xpos_by_group,
            per_unit_by_group=per_unit_by_group,
            per_unit_ids_by_group=per_unit_ids_by_group,
            hi_by_group=hi_by_group,
            group_names=[x.group for x in xs],
            cfg=cfg_stats,
            paired=False,
            panel_label=None,
            debug=bool(stats_debug),
        )
        if show_anova:
            text = _anova_text(xs, min_n_per_group=int(cfg_stats.min_n_per_group))
            if text:
                _add_anova_box(
                    ax,
                    text,
                    loc=anova_loc,
                    fontsize=max(
                        7, min(float(customizer.in_plot_font_size) - 5.0, 9.0)
                    ),
                )
    if title:
        ax.set_title(title)
    if not single_panel and G > 1 and legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            prop={
                "style": "italic",
                "size": max(8, min(float(customizer.in_plot_font_size), 13.0)),
            },
        )
    ax.grid(True, axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if not single_panel and G > 1:
        fig.subplots_adjust(right=min(fig.subplotpars.right, 0.78))
    _ensure_xlabel_visible(fig, ax)
    _ensure_ylabel_visible(fig, ax)
    savefig_with_format(out, image_format, log_tag=log_tag)
    return fig
