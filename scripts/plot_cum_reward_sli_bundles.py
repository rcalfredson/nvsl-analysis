#!/usr/bin/env python3
import argparse
from types import SimpleNamespace

from src.plotting.cum_reward_sli_bundle_plotter import plot_cum_reward_sli_bundles


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bundles",
        required=True,
        help="Comma-separated list of cumulative-reward SLI bundle .npz files.",
    )
    p.add_argument(
        "--labels",
        default=None,
        help="Optional comma-separated labels overriding bundle labels.",
    )
    p.add_argument("--out", required=True, help="Output image filename (png/pdf).")
    p.add_argument(
        "--metric",
        default="sli",
        choices=["sli", "reward_pi", "reward_pi_exp", "reward_pi_yoked"],
        help=(
            "Y-axis metric to plot against cumulative rewards. "
            "'reward_pi' defaults to the experimental fly."
        ),
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
        "--ci-min-n",
        type=int,
        default=3,
        help="Only draw confidence-interval shading at ticks with at least this many flies in the subset.",
    )
    p.add_argument(
        "--show-n",
        action="store_true",
        help="Annotate per-tick sample sizes near the plotted curves.",
    )
    p.add_argument(
        "--individual-sample-n",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "If given and > 0, plot a random sample of this many individual-fly "
            "curves per plotted subset instead of subset means/CI."
        ),
    )
    p.add_argument(
        "--individual-seed",
        type=int,
        default=0,
        help="Random seed used with --individual-sample-n.",
    )
    p.add_argument(
        "--hide-legend",
        action="store_true",
        help="Hide the legend. Useful for dense sampled-individual plots.",
    )
    p.add_argument(
        "--individual-color-mode",
        choices=["gradient", "random"],
        default="gradient",
        help=(
            "Color style for sampled individual-fly traces. "
            "'gradient' uses the current single-hue shading; "
            "'random' assigns colors from a categorical palette."
        ),
    )
    p.add_argument(
        "--min-fly-pct",
        "--cum-reward-sli-min-fly-pct",
        dest="min_fly_pct",
        type=float,
        default=argparse.SUPPRESS,
        help=(
            "Restrict the plotted cumulative-reward range to ticks reached by at "
            "least this percent of flies in each plotted subset. Defaults to the "
            "value saved in each bundle, or 95 if unavailable. Use 0 to show the "
            "full tail."
        ),
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
    p.add_argument("--image-format", default="png")
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
    min_fly_pct = getattr(args, "min_fly_pct", None)
    if min_fly_pct is not None and not (0 <= float(min_fly_pct) <= 100):
        raise SystemExit("--min-fly-pct must be in the interval [0, 100].")
    individual_sample_n = getattr(args, "individual_sample_n", None)
    if individual_sample_n is not None and int(individual_sample_n) < 0:
        raise SystemExit("--individual-sample-n must be >= 0.")

    bundles = [s.strip() for s in args.bundles.split(",") if s.strip()]
    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]

    opts = SimpleNamespace(imageFormat=args.image_format)
    plot_cum_reward_sli_bundles(
        bundles,
        args.out,
        labels=labels,
        sli_extremes=args.sli_extremes,
        sli_fraction=shared_frac,
        sli_top_fraction=top_frac,
        sli_bottom_fraction=bottom_frac,
        standalone_extreme_labels=bool(args.standalone_extreme_labels),
        ci_min_n=max(1, int(args.ci_min_n)),
        show_n=bool(args.show_n),
        min_fly_pct=None if min_fly_pct is None else float(min_fly_pct),
        metric=args.metric,
        individual_sample_n=individual_sample_n,
        individual_seed=int(args.individual_seed),
        individual_color_mode=args.individual_color_mode,
        show_legend=not bool(args.hide_legend),
        opts=opts,
    )


if __name__ == "__main__":
    main()
