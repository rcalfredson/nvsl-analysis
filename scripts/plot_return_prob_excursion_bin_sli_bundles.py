#!/usr/bin/env python3
import argparse
from types import SimpleNamespace

from src.plotting.return_prob_excursion_bin_sli_bundle_plotter import (
    plot_return_prob_excursion_bin_sli_bundles,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bundles",
        required=True,
        help="Comma-separated list of return-prob-excursion-bin .npz bundles.",
    )
    p.add_argument("--out", required=True, help="Output image filename (png/pdf).")
    p.add_argument("--labels", default=None, help="Optional comma-separated labels.")
    p.add_argument(
        "--mode",
        default="exp",
        choices=["exp", "ctrl", "exp_minus_ctrl"],
        help="Which fly role to plot.",
    )
    p.add_argument(
        "--metric",
        default="ratio",
        choices=["ratio", "success", "failure", "total"],
        help="Which excursion-bin metric to plot.",
    )
    p.add_argument(
        "--sli-extremes",
        default=None,
        choices=["top", "bottom", "both"],
        help="Optional SLI filtering within each bundle.",
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
    p.add_argument("--title", default=None, help="Optional plot title override.")
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument(
        "--stats",
        action="store_true",
        help="Add per-bin one-way ANOVA + Holm posthoc stars.",
    )
    p.add_argument("--stats-alpha", type=float, default=0.05)
    p.add_argument(
        "--stats-paired",
        action="store_true",
        help="Use paired tests for posthoc comparisons.",
    )
    p.add_argument("--debug", action="store_true")
    p.add_argument("--bar-alpha", type=float, default=0.90)
    p.add_argument(
        "--standalone-extreme-labels",
        action="store_true",
        help="Use just 'Top/Bottom XX%%' as labels instead of appending to group names.",
    )
    p.add_argument(
        "--image-format",
        "--imgFormat",
        dest="image_format",
        default="png",
    )
    p.add_argument("--fs", dest="font_size", type=float, default=None)
    p.add_argument("--fontFamily", dest="font_family", type=str, default=None)
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

    bundles = [s for s in args.bundles.split(",") if s]
    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]

    opts = SimpleNamespace(
        imageFormat=args.image_format,
        fontSize=args.font_size,
        fontFamily=args.font_family,
    )

    plot_return_prob_excursion_bin_sli_bundles(
        bundles,
        args.out,
        labels=labels,
        mode=args.mode,
        metric=args.metric,
        sli_extremes=args.sli_extremes,
        sli_fraction=shared_frac,
        sli_top_fraction=top_frac,
        sli_bottom_fraction=bottom_frac,
        standalone_extreme_labels=args.standalone_extreme_labels,
        title=args.title,
        ymax=args.ymax,
        stats=args.stats,
        stats_alpha=args.stats_alpha,
        stats_paired=args.stats_paired,
        debug=args.debug,
        bar_alpha=args.bar_alpha,
        opts=opts,
    )


if __name__ == "__main__":
    main()
