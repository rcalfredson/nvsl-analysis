#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.overlay_training_metric_scalar_bars import (
    load_export_npz,
    validate_alignment,
)
from src.plotting.palettes import (
    NEUTRAL_DARK,
    group_metric_edge_color,
    group_metric_fill_color,
    normalize_metric_palette_family,
)
from src.plotting.plot_customizer import PlotCustomizer


def _savefig(out_path: str, image_format: str) -> None:
    file_extension = "." + str(image_format).lstrip(".")
    base, ext = os.path.splitext(out_path)
    if ext.lower() != file_extension.lower():
        out_path = base + file_extension
        print(
            f"The file extension has been changed to {file_extension} to coincide with the specified format."
        )
    plt.savefig(out_path, bbox_inches="tight", format=image_format)
    print(f"[btw_rwd_tortuosity_mean_swarm] wrote {out_path}")


def _metric_palette_family(xs) -> str | None:
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


def plot_swarm_overlays(
    xs,
    *,
    out: str,
    image_format: str = "png",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymax: float | None = None,
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
    metric_palette_family = _metric_palette_family(xs)

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
    for gi, x in enumerate(xs):
        fill = group_metric_fill_color(gi, metric_palette_family)
        edge = group_metric_edge_color(gi, metric_palette_family)
        for p in range(P):
            vals = np.asarray(x.per_unit_values_panel[p], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            xpos = centers[gi] if single_panel else centers[p] + panel_offsets[gi]
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

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=25 if single_panel else 30, ha="right")
    if xlabel and not (single_panel and str(xlabel).strip().lower() == "training"):
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    if ymax is not None:
        ax.set_ylim(top=float(ymax))
    if title:
        ax.set_title(title)
    if not single_panel and G > 1 and legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=max(8, min(float(customizer.in_plot_font_size), 13.0)),
        )
    ax.grid(True, axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if not single_panel and G > 1:
        fig.subplots_adjust(right=min(fig.subplotpars.right, 0.78))
    _savefig(out, image_format)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot per-fly mean between-reward tortuosity exports as side-by-side "
            "swarm plots."
        )
    )
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable: 'GroupLabel=/path/to/export.npz'",
    )
    p.add_argument("--out", required=True, help="Output image path.")
    p.add_argument(
        "--image-format",
        "--imgFormat",
        dest="image_format",
        default="png",
        help="Output image format, for example png, pdf, or svg.",
    )
    p.add_argument("--title", default=None, help="Figure title override.")
    p.add_argument("--xlabel", default="training", help="X-axis label override.")
    p.add_argument("--ylabel", default=None, help="Y-axis label override.")
    p.add_argument("--ymax", type=float, default=None, help="Optional fixed y max.")
    p.add_argument(
        "--fontFamily",
        dest="font_family",
        type=str,
        default=None,
        help="Override the default font family for plots.",
    )
    p.add_argument(
        "--fs",
        dest="font_size",
        type=float,
        default=None,
        help="Font size for plot text.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    xs = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(
                f"--input must be of the form 'Label=path.npz' (got: {spec})"
            )
        label, path = spec.split("=", 1)
        xs.append(load_export_npz(label.strip(), path.strip()))

    ylabel = args.ylabel or xs[0].meta.get("y_label", "Mean tortuosity per fly")
    opts = SimpleNamespace(fontSize=args.font_size, fontFamily=args.font_family)
    fig = plot_swarm_overlays(
        xs,
        out=args.out,
        image_format=args.image_format,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=ylabel,
        ymax=args.ymax,
        opts=opts,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
