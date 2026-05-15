#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.correlation_stats import (
    fisher_independent_correlation_test,
    pearson_correlation_summary,
)  # noqa: E402
from src.plotting.palettes import (  # noqa: E402
    group_metric_edge_color,
    group_metric_fill_color,
)
from src.plotting.plot_customizer import PlotCustomizer  # noqa: E402


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    if p <= 0.0 or p < 1e-300:
        return "p<1e-300"
    if p < 1e-4:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def _npz_scalar(z, key: str, default: str | None = None) -> str | None:
    if key not in z.files:
        return default
    val = z[key]
    if getattr(val, "shape", None) == ():
        val = val.item()
    return str(val)


def load_correlation_export(label: str, path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    meta_json = z["meta_json"].item() if "meta_json" in z.files else "{}"
    if isinstance(meta_json, (bytes, bytearray)):
        meta_json = meta_json.decode("utf-8")
    meta = json.loads(str(meta_json))
    x = np.asarray(z["x"], dtype=float)
    y = np.asarray(z["y"], dtype=float)
    summary = pearson_correlation_summary(x, y)
    export_label = _npz_scalar(z, "group", meta.get("group"))
    label = label or export_label or os.path.basename(path)
    return {
        "label": label,
        "path": path,
        "x": x,
        "y": y,
        "summary": summary,
        "meta": meta,
        "x_label": _npz_scalar(z, "x_label", meta.get("x_label")),
        "y_label": _npz_scalar(z, "y_label", meta.get("y_label")),
        "title": _npz_scalar(z, "title", meta.get("title")),
    }


def _savefig(out_path: str, image_format: str) -> None:
    file_extension = "." + str(image_format).lstrip(".")
    base, ext = os.path.splitext(out_path)
    if ext.lower() != file_extension.lower():
        out_path = base + file_extension
        print(
            f"The file extension has been changed to {file_extension} to coincide with the specified format."
        )
    plt.savefig(out_path, bbox_inches="tight", format=image_format)
    print(f"[cross_fly_correlation_compare] wrote {out_path}")


def _add_fit_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, color: str) -> None:
    if x.size < 2 or np.nanstd(x) <= 0:
        return
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.asarray([np.nanmin(x), np.nanmax(x)], dtype=float)
    ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.4, alpha=0.9)


def _parse_axis_limits(value: str | None, option_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 2:
        raise ValueError(f"{option_name} must be formatted as MIN,MAX.")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"{option_name} values must be numeric.") from exc
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"{option_name} must have finite MIN < MAX.")
    return lo, hi


def plot_comparison(
    bundles: list[dict],
    *,
    out: str,
    image_format: str,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    alpha: float,
    point_size: float,
    show_fit: bool,
    stats_loc: str,
    legend_loc: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    opts,
) -> None:
    if len(bundles) != 2:
        raise ValueError("exactly two inputs are required for the Fisher comparison.")

    customizer = PlotCustomizer()
    if getattr(opts, "fontSize", None) is not None:
        customizer.update_font_size(getattr(opts, "fontSize"))
    customizer.update_font_family(getattr(opts, "fontFamily", None))

    try:
        result = fisher_independent_correlation_test(
            bundles[0]["summary"].r,
            bundles[0]["summary"].n,
            bundles[1]["summary"].r,
            bundles[1]["summary"].n,
        )
    except ValueError as exc:
        labels = ", ".join(
            f"{b['label']} n={b['summary'].n}, r={b['summary'].r}"
            for b in bundles
        )
        raise ValueError(f"{exc} Inputs: {labels}") from exc

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    lines = []
    for gi, bundle in enumerate(bundles):
        fill = group_metric_fill_color(gi, "between_reward_distance")
        edge = group_metric_edge_color(gi, "between_reward_distance")
        label = bundle["label"]
        x = bundle["x"]
        y = bundle["y"]
        s = bundle["summary"]
        ax.scatter(
            x,
            y,
            s=float(point_size),
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.45,
            alpha=float(alpha),
            label=f"{label} (r={s.r:.3f}, n={s.n})",
        )
        if show_fit:
            _add_fit_line(ax, x, y, color=edge)
        lines.append(f"{label}: r={s.r:.3f}, {_fmt_p(s.p)}, n={s.n}")

    lines.append(
        f"Fisher r-to-z: Z={result.z_stat:.3f}, {_fmt_p(result.p_two_sided)}"
    )

    loc = str(stats_loc or "upper_left").lower()
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
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=max(7, min(float(customizer.in_plot_font_size) - 5.0, 9.0)),
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.86,
            "linewidth": 0.6,
        },
        zorder=10,
    )

    ax.set_xlabel(xlabel or bundles[0].get("x_label") or "x")
    ax.set_ylabel(ylabel or bundles[0].get("y_label") or "y")
    ax.set_title(title or bundles[0].get("title") or "Cross-fly correlation comparison")
    ax.grid(True, alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(frameon=True, loc=str(legend_loc or "upper_right").replace("_", " "))
    customizer.adjust_padding_proportionally()
    fig.tight_layout()
    _savefig(out, image_format)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Overlay two exported cross-fly correlation scatters and run "
            "Fisher's r-to-z test for independent correlations."
        )
    )
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Repeat exactly twice: 'GroupLabel=/path/to/correlation_export.npz'. "
            "Use '=path.npz' to use the label stored in the export."
        ),
    )
    p.add_argument("--out", required=True, help="Output image path.")
    p.add_argument(
        "--image-format",
        "--imgFormat",
        dest="image_format",
        default="png",
        help="Output image format, for example png, pdf, or svg.",
    )
    p.add_argument("--title", default=None)
    p.add_argument("--xlabel", default=None)
    p.add_argument("--ylabel", default=None)
    p.add_argument("--xlim", default=None, metavar="MIN,MAX")
    p.add_argument("--ylim", default=None, metavar="MIN,MAX")
    p.add_argument("--alpha", type=float, default=0.78)
    p.add_argument("--point-size", type=float, default=28.0)
    p.add_argument("--no-fit", action="store_true", help="Hide fitted linear trend lines.")
    p.add_argument(
        "--stats-loc",
        choices=("upper_left", "upper_right", "lower_left", "lower_right"),
        default="upper_left",
    )
    p.add_argument(
        "--legend-loc",
        choices=("best", "upper_left", "upper_right", "lower_left", "lower_right"),
        default="upper_right",
        help="Legend location. Defaults away from the stats box.",
    )
    p.add_argument("--fontFamily", dest="font_family", type=str, default=None)
    p.add_argument("--fs", dest="font_size", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.input) != 2:
        raise SystemExit("--input must be provided exactly twice.")
    bundles = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(
                f"--input must be of the form 'Label=path.npz' (got: {spec})"
            )
        label, path = spec.split("=", 1)
        path = path.strip()
        if not path:
            raise SystemExit(f"--input is missing a path: {spec}")
        bundles.append(load_correlation_export(label.strip(), path))

    opts = SimpleNamespace(fontSize=args.font_size, fontFamily=args.font_family)
    try:
        xlim = _parse_axis_limits(args.xlim, "--xlim")
        ylim = _parse_axis_limits(args.ylim, "--ylim")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        plot_comparison(
            bundles,
            out=args.out,
            image_format=args.image_format,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            alpha=args.alpha,
            point_size=args.point_size,
            show_fit=not args.no_fit,
            stats_loc=args.stats_loc,
            legend_loc=args.legend_loc,
            xlim=xlim,
            ylim=ylim,
            opts=opts,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
