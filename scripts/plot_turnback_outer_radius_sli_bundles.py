#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt

from src.plotting.turnback_outer_radius_sli_bundle_plotter import (
    plot_turnback_outer_radius_sli_bundles,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bundles",
        required=True,
        help="Comma-separated list of turnback-outer-radius .npz bundles.",
    )
    p.add_argument(
        "--labels",
        default=None,
        help="Optional comma-separated labels overriding bundle labels.",
    )
    p.add_argument("--out", required=True, help="Output image filename (png/pdf).")
    p.add_argument(
        "--mode",
        default="exp",
        choices=["exp", "ctrl", "exp_minus_ctrl"],
        help="Which turnback curve to plot.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional figure title override.",
    )
    p.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional fixed y-axis max.",
    )
    p.add_argument(
        "--sli-extremes",
        default=None,
        choices=["top", "bottom", "both"],
        help="Optional within-bundle SLI filtering.",
    )
    p.add_argument(
        "--standalone-extreme-labels",
        action="store_true",
        help="With --sli-extremes both on a single bundle, use only percentile labels in the legend.",
    )
    p.add_argument(
        "--best-worst-fraction",
        type=float,
        default=argparse.SUPPRESS,
        help="Legacy shared fraction for SLI extremes.",
    )
    p.add_argument(
        "--top-sli-fraction",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of videos to include in the top-SLI subset.",
    )
    p.add_argument(
        "--bottom-sli-fraction",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of videos to include in the bottom-SLI subset.",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Add per-radius one-way ANOVA + Holm posthoc stars (Welch or paired t-tests depending on --stats-paired).",
    )
    p.add_argument(
        "--stats-paired",
        action="store_true",
        help="When --stats is enabled, use paired t-tests when video IDs overlap between groups.",
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
    args = p.parse_args()

    shared_frac = getattr(args, "best_worst_fraction", None)
    top_frac = getattr(args, "top_sli_fraction", None)
    bottom_frac = getattr(args, "bottom_sli_fraction", None)
    if top_frac is None and shared_frac is not None:
        top_frac = shared_frac
    if bottom_frac is None and shared_frac is not None:
        bottom_frac = shared_frac

    for opt_name, frac in (
        ("--best-worst-fraction", shared_frac),
        ("--top-sli-fraction", top_frac),
        ("--bottom-sli-fraction", bottom_frac),
    ):
        if frac is not None and not (0 < float(frac) <= 1):
            raise SystemExit(f"{opt_name} must be in the interval (0, 1].")

    bundles = [s.strip() for s in args.bundles.split(",") if s.strip()]
    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]

    fig = plot_turnback_outer_radius_sli_bundles(
        bundles,
        args.out,
        labels=labels,
        mode=args.mode,
        sli_extremes=args.sli_extremes,
        sli_fraction=shared_frac,
        sli_top_fraction=top_frac,
        sli_bottom_fraction=bottom_frac,
        standalone_extreme_labels=bool(args.standalone_extreme_labels),
        title=args.title,
        ymax=args.ymax,
        stats=bool(args.stats),
        stats_alpha=float(args.stats_alpha),
        stats_paired=bool(args.stats_paired),
        debug=bool(args.stats_debug),
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
