#!/usr/bin/env python3
# scripts/plot_overlay_training_metric_hist.py

from __future__ import annotations

import argparse
from types import SimpleNamespace

import matplotlib.pyplot as plt

from src.plotting.overlay_training_metric_hist import load_export_npz, plot_overlays
from src.utils.common import writeImage


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
        "--image-format",
        "--imgFormat",
        dest="image_format",
        default="png",
        help="Output image format (for example png, pdf, or svg).",
    )
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
    p.add_argument(
        "--mode",
        choices=("pdf", "cdf"),
        default="pdf",
        help="Overlay as probability distribution (pdf) or cumulative distribution (cdf).",
    )
    p.add_argument("--title", default=None, help="Figure title override.")
    p.add_argument(
        "--suptitle",
        action="store_true",
        help=(
            "Show a figure suptitle. If --title is not provided, a default title is generated "
            "from the first input's base_title and the selected mode."
        ),
    )
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
        help="Add per-bin one-way ANOVA + Holm posthoc stars (Welch or paired t-tests depending on --stats-paired).",
    )
    p.add_argument(
        "--stats-paired",
        action="store_true",
        help="When --stats is enabled, use paired t-tests (ttest_rel) when unit IDs overlap between groups.",
    )
    p.add_argument(
        "--stats-alpha",
        type=float,
        default=0.05,
        help="Alpha for stats (default 0.05).",
    )
    p.add_argument(
        "--stats-debug",
        action="store_true",
        help="Additional debug output when stats are enabled.",
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
    # Title behavior:
    # - default: no suptitle
    # - is user provides --title: use it
    # - if --suptitle and no --title: generate the old default
    title = args.title
    if title is None and args.suptitle:
        base = hists[0].meta.get("base_title", "Overlay histogram")
        title = f"{base}\n({args.mode.upper()} overlay)"

    opts = SimpleNamespace(
        imageFormat=args.image_format,
        fontFamily=args.font_family,
        fontSize=args.font_size,
    )

    fig = plot_overlays(
        hists,
        mode=args.mode,
        layout="grouped" if args.mode == "pdf" else "overlay",
        title=title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        ymax=args.ymax,
        stats=args.stats,
        stats_alpha=args.stats_alpha,
        stats_paired=args.stats_paired,
        xmax_plot=args.xmax_plot,
        debug=args.stats_debug,
        opts=opts,
    )
    writeImage(args.out, format=args.image_format)
    plt.close(fig)


if __name__ == "__main__":
    main()
