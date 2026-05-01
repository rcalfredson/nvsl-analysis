#!/usr/bin/env python3
from __future__ import annotations

import argparse
from types import SimpleNamespace

import matplotlib.pyplot as plt

from src.plotting.overlay_training_metric_scalar_bars import (
    load_export_npz,
)
from src.plotting.scalar_swarm_bundle_plotter import plot_swarm_overlays


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
        "--stats",
        action="store_true",
        help=(
            "Add per-training significance brackets. With two groups, this uses "
            "an independent Welch t-test; with more groups, it uses the shared "
            "ANOVA/post-hoc t-test annotation path."
        ),
    )
    p.add_argument(
        "--stats-alpha",
        type=float,
        default=0.05,
        help="Alpha for significance stars (default: %(default)s).",
    )
    p.add_argument(
        "--stats-debug",
        action="store_true",
        help="Print additional debug output for stats annotations.",
    )
    p.add_argument(
        "--show-anova",
        action="store_true",
        help=(
            "When --stats is enabled and there are 3+ groups, display the "
            "omnibus one-way ANOVA result computed from the plotted per-fly values."
        ),
    )
    p.add_argument(
        "--anova-loc",
        choices=("upper_left", "upper_right", "lower_left", "lower_right"),
        default="upper_left",
        help="Location for the optional ANOVA text box.",
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
        stats=bool(args.stats),
        stats_alpha=float(args.stats_alpha),
        stats_debug=bool(args.stats_debug),
        show_anova=bool(args.stats and args.show_anova),
        anova_loc=args.anova_loc,
        log_tag="btw_rwd_tortuosity_mean_swarm",
        opts=opts,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
