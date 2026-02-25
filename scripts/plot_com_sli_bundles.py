#!/usr/bin/env python3
import argparse
import os
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
        choices=[
            "commag",
            "sli",
            "weaving",
            "turnback",
            "agarose",
            "wallpct",
            "lgturn_startdist",
            "reward_lgturn_pathlen",
            "reward_lgturn_prevalence",
            "reward_lv",
        ],
    )
    p.add_argument(
        "--turnback-mode",
        default="exp",
        choices=["exp", "ctrl", "exp_minus_ctrl"],
        help="For metrics with exp/ctrl variants (e.g. turnback, agarose, reward_lgturn_*, reward_lv): which curve to plot.",
    )
    p.add_argument(
        "--delta-vs",
        default=None,
        help=(
            "Optional baseline bundle (.npz). If set, plot Δ curves: (bundle - baseline) "
            "per training/bucket, aligned by video_ids when possible."
        ),
    )
    p.add_argument(
        "--delta-label",
        default=None,
        help="Optional label prefix for delta mode (e.g. 'Δ vs 0px'). If omitted, uses 'Δ vs <basename>'.",
    )
    p.add_argument(
        "--delta-ylabel",
        default=None,
        help=(
            "Optional y-axis label used in delta mode. "
            "Example: 'Δ(Δ turnback ratio)' or 'Contrast shift in turnback ratio'."
        ),
    )

    p.add_argument(
        "--delta-allow-unpaired",
        action="store_true",
        help="If video_ids do not overlap sufficiently, allow unpaired delta (mean(bundle)-mean(baseline)).",
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

    delta_label = args.delta_label
    if args.delta_vs and not delta_label:
        delta_label = f"Δ vs {os.path.basename(args.delta_vs)}"

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
        delta_vs_path=args.delta_vs,
        delta_label=delta_label,
        delta_ylabel=args.delta_ylabel,
        delta_allow_unpaired=args.delta_allow_unpaired,
    )


if __name__ == "__main__":
    main()
