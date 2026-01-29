#!/usr/bin/env python3
# scripts/plot_overlay_training_metric_hist.py

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt

from src.plotting.overlay_training_metric_hist import load_export_npz, plot_overlays


def _savefig(out_path: str) -> None:
    base, ext = os.path.splitext(out_path)
    if ext == "":
        out_path = base + ".png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[overlay_hist] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay training-panel histograms exported as NPZ, plotting PDF or CDF."
    )
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable: 'GroupLabel=/path/to/export.npz'",
    )
    p.add_argument("--out", required=True, help="Output image path (png/pdf/etc).")
    p.add_argument(
        "--mode",
        choices=("pdf", "cdf"),
        default="pdf",
        help="Overlay as probability distribution (pdf) or cumulative distribution (cdf).",
    )
    p.add_argument("--title", default=None, help="Figure title override.")
    p.add_argument("--xlabel", default=None, help="X-axis label override.")
    p.add_argument("--ylabel", default=None, help="Y-axis label override.")
    p.add_argument("--ymax", type=float, default=None, help="Optional fixed y max.")
    p.add_argument(
        "--xmax-plot",
        type=float,
        default=None,
        help="Truncate the plotted x-range at this value (same units as histogram x). "
        "Bins beyond xmax are omitted (PDF grouped bars).",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Add per-bin one-way ANOVA + Holm/Welch posthoc stars (per-fly PDF only).",
    )
    p.add_argument(
        "--stats-alpha",
        type=float,
        default=0.05,
        help="Alpha for stats (default 0.05).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    hists = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(
                f"--input must be of the form 'Label=path.npz' (got: {spec})"
            )
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        hists.append(load_export_npz(label, path))

    # Reasonable defaults if not provided
    if args.xlabel is None:
        args.xlabel = hists[0].meta.get("x_label", None)
    if args.ylabel is None:
        args.ylabel = "proportion" if args.mode == "pdf" else "cumulative proportion"
    if args.title is None:
        base = hists[0].meta.get("base_title", "Overlay histogram")
        args.title = f"{base}\n({args.mode.upper()} overlay)"

    fig = plot_overlays(
        hists,
        mode=args.mode,
        layout="grouped" if args.mode == "pdf" else "overlay",
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        ymax=args.ymax,
        stats=args.stats,
        stats_alpha=args.stats_alpha,
        xmax_plot=args.xmax_plot, 
    )
    _savefig(args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
