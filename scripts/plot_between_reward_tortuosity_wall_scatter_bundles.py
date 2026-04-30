#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.plotting.palettes import group_metric_edge_color, group_metric_fill_color
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
    print(f"[btw_rwd_tortuosity_wall_scatter] wrote {out_path}")


def _load_bundle(label: str, path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    meta_json = z["meta_json"].item() if "meta_json" in z.files else "{}"
    if isinstance(meta_json, (bytes, bytearray)):
        meta_json = meta_json.decode("utf-8")
    meta = json.loads(str(meta_json))
    return {
        "group": label,
        "path": path,
        "wall_pct": np.asarray(z["wall_pct"], dtype=float),
        "tortuosity": np.asarray(z["tortuosity"], dtype=float),
        "meta": meta,
    }


def _clean_xy(bundle: dict, *, y_transform: str, xmax: float | None):
    x = np.asarray(bundle["wall_pct"], dtype=float)
    y = np.asarray(bundle["tortuosity"], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if xmax is not None:
        mask &= x <= float(xmax)
    if y_transform == "log10":
        mask &= y > 0
        y = np.where(y > 0, np.log10(y), np.nan)
    return x[mask], y[mask]


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    if p <= 0.0 or p < 1e-300:
        return "p<1e-300"
    if p < 1e-4:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def _corr_text(x: np.ndarray, y: np.ndarray, *, method: str, min_n: int) -> str | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < int(max(3, min_n)):
        return None
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return None

    method = str(method or "pearson").lower()
    lines = []
    if method in {"pearson", "both"}:
        r, p = stats.pearsonr(x, y)
        lines.append(f"Pearson r={r:.3f}, {_fmt_p(float(p))}")
    if method in {"spearman", "both"}:
        r, p = stats.spearmanr(x, y, nan_policy="omit")
        lines.append(f"Spearman rho={r:.3f}, {_fmt_p(float(p))}")
    if not lines:
        return None
    lines.append(f"n={x.size}")
    return "\n".join(lines)


def plot_wall_scatter_bundles(
    bundles: list[dict],
    *,
    out: str,
    image_format: str = "png",
    kind: str = "scatter",
    y_transform: str = "log10",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xmax: float | None = None,
    ymax: float | None = None,
    alpha: float = 0.18,
    gridsize: int = 45,
    mincnt: int = 1,
    show_corr: bool = False,
    corr_method: str = "pearson",
    corr_min_n: int = 3,
    corr_loc: str = "upper_left",
    opts=None,
):
    if opts is None:
        opts = SimpleNamespace(fontSize=None, fontFamily=None)

    customizer = PlotCustomizer()
    if getattr(opts, "fontSize", None) is not None:
        customizer.update_font_size(getattr(opts, "fontSize"))
    customizer.update_font_family(getattr(opts, "fontFamily", None))

    G = len(bundles)
    if G <= 0:
        raise ValueError("at least one bundle is required")

    fig, axes = plt.subplots(
        1,
        G,
        figsize=(max(4.0 * G, 5.0), 4.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    kind = str(kind or "scatter").lower()
    y_transform = str(y_transform or "log10").lower()
    if xmax is None:
        xmax = 100.0

    for gi, (ax, bundle) in enumerate(zip(axes, bundles)):
        x, y = _clean_xy(bundle, y_transform=y_transform, xmax=xmax)
        fill = group_metric_fill_color(gi, "between_reward_distance")
        edge = group_metric_edge_color(gi, "between_reward_distance")
        if x.size == 0:
            ax.text(0.5, 0.5, "no segments", ha="center", va="center", transform=ax.transAxes)
        elif kind == "hexbin":
            hb = ax.hexbin(
                x,
                y,
                gridsize=int(max(5, gridsize)),
                mincnt=int(max(1, mincnt)),
                cmap="viridis",
                linewidths=0.0,
            )
            cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("segments")
        else:
            ax.scatter(
                x,
                y,
                s=8,
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.25,
                alpha=float(alpha),
            )
        if show_corr:
            text = _corr_text(
                x,
                y,
                method=str(corr_method or "pearson"),
                min_n=int(corr_min_n or 3),
            )
            if text:
                loc = str(corr_loc or "upper_left").lower()
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
                    fontsize=max(
                        7, min(float(customizer.in_plot_font_size) - 5.0, 9.0)
                    ),
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "0.75",
                        "alpha": 0.82,
                        "linewidth": 0.6,
                    },
                )
        ax.set_title(f"{bundle['group']}\n(n={x.size})")
        ax.grid(True, alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        ylabel
        or (
            r"$\log_{10}$(between-reward tortuosity)"
            if y_transform == "log10"
            else "Between-reward tortuosity"
        )
    )
    for ax in axes:
        ax.set_xlabel(xlabel or "Wall-contact frames in segment [%]")
        ax.set_xlim(0, float(xmax))
        if ymax is not None:
            ax.set_ylim(top=float(ymax))
    if y_transform != "log10":
        for ax in axes:
            ax.set_ylim(bottom=0)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if title:
        fig.subplots_adjust(top=0.82)
    _savefig(out, image_format)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot segment-level between-reward tortuosity versus percent "
            "wall-contact frames from exported bundles."
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
    p.add_argument("--kind", choices=("scatter", "hexbin"), default="scatter")
    p.add_argument("--y-transform", choices=("log10", "none"), default="log10")
    p.add_argument("--title", default=None)
    p.add_argument("--xlabel", default=None)
    p.add_argument("--ylabel", default=None)
    p.add_argument("--xmax", type=float, default=100.0)
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--alpha", type=float, default=0.18)
    p.add_argument("--gridsize", type=int, default=45)
    p.add_argument("--mincnt", type=int, default=1)
    p.add_argument(
        "--show-corr",
        action="store_true",
        help=(
            "Annotate each panel with the correlation between wall-contact "
            "percentage and the plotted tortuosity value."
        ),
    )
    p.add_argument(
        "--corr-method",
        choices=("pearson", "spearman", "both"),
        default="pearson",
        help="Correlation type to annotate when --show-corr is set.",
    )
    p.add_argument(
        "--corr-min-n",
        type=int,
        default=3,
        help="Minimum number of points required to annotate a correlation.",
    )
    p.add_argument(
        "--corr-loc",
        choices=("upper_left", "upper_right", "lower_left", "lower_right"),
        default="upper_left",
        help="Panel location for correlation annotations.",
    )
    p.add_argument("--fontFamily", dest="font_family", type=str, default=None)
    p.add_argument("--fs", dest="font_size", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundles = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(
                f"--input must be of the form 'Label=path.npz' (got: {spec})"
            )
        label, path = spec.split("=", 1)
        bundles.append(_load_bundle(label.strip(), path.strip()))
    opts = SimpleNamespace(fontSize=args.font_size, fontFamily=args.font_family)
    fig = plot_wall_scatter_bundles(
        bundles,
        out=args.out,
        image_format=args.image_format,
        kind=args.kind,
        y_transform=args.y_transform,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        xmax=args.xmax,
        ymax=args.ymax,
        alpha=args.alpha,
        gridsize=args.gridsize,
        mincnt=args.mincnt,
        show_corr=bool(args.show_corr),
        corr_method=args.corr_method,
        corr_min_n=int(args.corr_min_n),
        corr_loc=args.corr_loc,
        opts=opts,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
