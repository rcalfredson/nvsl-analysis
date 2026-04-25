#!/usr/bin/env python3
from __future__ import annotations

import argparse
from types import SimpleNamespace

import matplotlib.pyplot as plt

from src.plotting.between_reward_tortuosity_distance_box import (
    BetweenRewardTortuosityDistanceBoxResult,
)
from src.plotting.between_reward_tortuosity_distance_box_plotter import (
    plot_box_results,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot between-reward tortuosity-by-max-distance bundles as grouped "
            "box-and-whisker plots."
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
    p.add_argument("--xlabel", default=None, help="X-axis label override.")
    p.add_argument("--ylabel", default=None, help="Y-axis label override.")
    p.add_argument("--ymax", type=float, default=None, help="Optional fixed y max.")
    p.add_argument("--showfliers", action="store_true", help="Draw outlier points.")
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
    labels = []
    results = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(
                f"--input must be of the form 'Label=path.npz' (got: {spec})"
            )
        label, path = spec.split("=", 1)
        labels.append(label.strip())
        results.append(BetweenRewardTortuosityDistanceBoxResult.load_npz(path.strip()))

    opts = SimpleNamespace(
        imageFormat=args.image_format,
        fontFamily=args.font_family,
        fontSize=args.font_size,
    )
    fig = plot_box_results(
        results,
        group_labels=labels,
        out_file=args.out,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        showfliers=bool(args.showfliers),
        ymax=args.ymax,
        opts=opts,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
