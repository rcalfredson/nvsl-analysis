#!/usr/bin/env python3
from __future__ import annotations

import argparse
from types import SimpleNamespace

import matplotlib.pyplot as plt

from src.plotting.wall_contact_episode_length_swarm import (
    load_episode_csvs,
    plot_wall_contact_episode_length_swarm,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot episode-level between-reward trajectory length, colored by "
            "whether the interval contained wall contact."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable path to an episode provenance CSV.",
    )
    parser.add_argument("--out", required=True, help="Output image path.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    parser.add_argument(
        "--ylabel",
        default="Trajectory length (mm)",
        help="Y-axis label.",
    )
    parser.add_argument("--ymax", type=float, default=None, help="Optional y maximum.")
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include episodes from flies excluded by the panel filters.",
    )
    parser.add_argument(
        "--imgFormat",
        "--image-format",
        dest="image_format",
        default="png",
        help="Output image format.",
    )
    parser.add_argument("--fs", dest="font_size", type=float, default=None)
    parser.add_argument("--fontFamily", dest="font_family", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, group_order = load_episode_csvs(
        args.input,
        include_excluded=bool(args.include_excluded),
    )
    fig = plot_wall_contact_episode_length_swarm(
        data,
        group_order=group_order,
        title=args.title,
        ylabel=args.ylabel,
        ymax=args.ymax,
        opts=SimpleNamespace(
            fontSize=args.font_size,
            fontFamily=args.font_family,
        ),
    )
    save_figure(fig, args.out, image_format=args.image_format)
    plt.close(fig)


if __name__ == "__main__":
    main()
