#!/usr/bin/env python3
import argparse
from types import SimpleNamespace

from src.plotting.post_wall_departure_tortuosity_sli_bundle_plotter import (
    plot_post_wall_departure_tortuosity_sli_bundles,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument(
        "--mode", default="exp", choices=["exp", "ctrl", "exp_minus_ctrl"]
    )
    parser.add_argument(
        "--sli-extremes", default=None, choices=["top", "bottom", "both"]
    )
    parser.add_argument(
        "--best-worst-fraction", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument("--top-sli-fraction", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--bottom-sli-fraction", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument("--standalone-extreme-labels", action="store_true")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xlabel", default=None)
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--stats-alpha", type=float, default=0.05)
    parser.add_argument("--stats-paired", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--bar-alpha", type=float, default=0.90)
    parser.add_argument(
        "--image-format", "--imgFormat", dest="image_format", default="png"
    )
    parser.add_argument("--fs", dest="font_size", type=float, default=None)
    parser.add_argument("--fontFamily", dest="font_family", default=None)
    args = parser.parse_args()

    shared = getattr(args, "best_worst_fraction", None)
    top = getattr(args, "top_sli_fraction", None)
    bottom = getattr(args, "bottom_sli_fraction", None)
    top = shared if top is None else top
    bottom = shared if bottom is None else bottom
    for name, value in (
        ("--best-worst-fraction", shared),
        ("--top-sli-fraction", top),
        ("--bottom-sli-fraction", bottom),
    ):
        if value is not None and not (0 < float(value) <= 1):
            raise SystemExit(f"{name} must be in the interval (0, 1].")

    plot_post_wall_departure_tortuosity_sli_bundles(
        [item for item in args.bundles.split(",") if item],
        args.out,
        labels=(
            [item.strip() for item in args.labels.split(",")]
            if args.labels
            else None
        ),
        mode=args.mode,
        sli_extremes=args.sli_extremes,
        sli_fraction=shared,
        sli_top_fraction=top,
        sli_bottom_fraction=bottom,
        standalone_extreme_labels=args.standalone_extreme_labels,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        ymax=args.ymax,
        stats=args.stats,
        stats_alpha=args.stats_alpha,
        stats_paired=args.stats_paired,
        debug=args.debug,
        bar_alpha=args.bar_alpha,
        opts=SimpleNamespace(
            imageFormat=args.image_format,
            fontSize=args.font_size,
            fontFamily=args.font_family,
        ),
    )


if __name__ == "__main__":
    main()
