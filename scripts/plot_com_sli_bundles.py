#!/usr/bin/env python3
import argparse
from types import SimpleNamespace

from src.plotting.com_sli_bundle_plotter import plot_com_sli_bundles


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bundles", required=True, help="Comma-separated list of .npz bundles."
    )
    p.add_argument(
        "--metric",
        default="commag",
        choices=["commag", "sli", "turnback", "agarose", "wallpct", "lgturn_startdist"],
    )
    p.add_argument(
        "--turnback-mode",
        default="exp",
        choices=["exp", "ctrl", "exp_minus_ctrl"],
        help="For metric in {turnback, agarose}: which curve to plot.",
    )
    p.add_argument(
        "--labels",
        default=None,
        help="Optional comma-separated labels (overrides bundle labels).",
    )
    p.add_argument("--out", required=True, help="Output image filename (png/pdf).")
    p.add_argument(
        "--num-trainings",
        type=int,
        default=None,
        help="Optionally limit trainings displayed.",
    )
    p.add_argument(
        "--include-ctrl",
        action="store_true",
        help="Overlay ctrl curves (same linestyle as group, different color).",
    )
    p.add_argument(
        "--sli-extremes",
        default=None,
        choices=["top", "bottom", "both"],
        help="Optional SLI filtering within each bundle.",
    )
    p.add_argument(
        "--sli-fraction",
        type=float,
        default=0.2,
        help="Fraction for SLI extremes (e.g. 0.2 for top 20%).",
    )
    p.add_argument("--wspace", type=float, default=0.35)
    p.add_argument("--image-format", default="png")
    args = p.parse_args()

    bundles = [s for s in args.bundles.split(",") if s]
    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]

    opts = SimpleNamespace(
        wspace=args.wspace,
        imageFormat=args.image_format,
    )

    plot_com_sli_bundles(
        bundles,
        args.out,
        labels=labels,
        num_trainings=args.num_trainings,
        include_ctrl=args.include_ctrl,
        sli_extremes=args.sli_extremes,
        sli_fraction=args.sli_fraction,
        opts=opts,
        metric=args.metric,
        turnback_mode=args.turnback_mode,
    )


if __name__ == "__main__":
    main()
