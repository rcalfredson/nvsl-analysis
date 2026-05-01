#!/usr/bin/env python3
from __future__ import annotations

import argparse
from types import SimpleNamespace

import matplotlib.pyplot as plt

from src.plotting.between_reward_tortuosity_wall_flycorr import (
    build_flycorr_exports,
    load_wall_scatter_bundle,
    write_flycorr_csv,
    write_flycorr_npz,
)
from src.plotting.scalar_swarm_bundle_plotter import plot_swarm_overlays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute one wall-contact/tortuosity correlation per fly from "
            "segment-level wall-scatter bundles, then plot those per-fly "
            "correlations as group swarms."
        )
    )
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable: 'GroupLabel=/path/to/tort_vs_wall_scatter_export.npz'",
    )
    p.add_argument("--out", required=True, help="Output image path.")
    p.add_argument(
        "--image-format",
        "--imgFormat",
        dest="image_format",
        default="png",
        help="Output image format, for example png, pdf, or svg.",
    )
    p.add_argument(
        "--corr-method",
        choices=("pearson", "spearman"),
        default="pearson",
        help="Within-fly correlation method.",
    )
    p.add_argument("--y-transform", choices=("log10", "none"), default="log10")
    p.add_argument(
        "--plot-value",
        choices=("fisher_z", "r"),
        default="fisher_z",
        help="Plot Fisher-transformed correlations or raw correlations.",
    )
    p.add_argument(
        "--min-segments-per-fly",
        type=int,
        default=5,
        help="Minimum finite segments required to compute a fly's correlation.",
    )
    p.add_argument(
        "--min-wall-range-pct",
        type=float,
        default=0.0,
        help=(
            "Optional minimum within-fly wall-contact percent range. Flies below "
            "this range are skipped."
        ),
    )
    p.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="Optional maximum segment wall-contact percent to include.",
    )
    p.add_argument("--title", default=None, help="Figure title override.")
    p.add_argument("--xlabel", default=None, help="X-axis label override.")
    p.add_argument("--ylabel", default=None, help="Y-axis label override.")
    p.add_argument("--ymin", type=float, default=None, help="Optional fixed y min.")
    p.add_argument("--ymax", type=float, default=None, help="Optional fixed y max.")
    p.add_argument(
        "--stats",
        action="store_true",
        help=(
            "Add group significance brackets using the shared independent "
            "Welch t-test / ANOVA + post-hoc annotation path."
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
        "--export-npz",
        default=None,
        help="Optional output .npz containing the per-fly correlation values.",
    )
    p.add_argument(
        "--export-csv",
        default=None,
        help="Optional output .csv containing one row per retained fly.",
    )
    p.add_argument("--fontFamily", dest="font_family", type=str, default=None)
    p.add_argument("--fs", dest="font_size", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundles = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(
                f"--input must be of the form 'Label=path.npz' (got: {spec})"
            )
        label, path = spec.split("=", 1)
        bundles.append(load_wall_scatter_bundle(label.strip(), path.strip()))

    xs, rows = build_flycorr_exports(
        bundles,
        corr_method=args.corr_method,
        y_transform=args.y_transform,
        plot_value=args.plot_value,
        min_segments_per_fly=int(args.min_segments_per_fly),
        min_wall_range_pct=float(args.min_wall_range_pct),
        xmax=args.xmax,
    )
    if args.export_npz:
        write_flycorr_npz(xs, args.export_npz)
        print(f"[btw_rwd_tortuosity_wall_flycorr] wrote {args.export_npz}")
    if args.export_csv:
        write_flycorr_csv(rows, args.export_csv)
        print(f"[btw_rwd_tortuosity_wall_flycorr] wrote {args.export_csv}")

    ylabel = args.ylabel or (xs[0].meta.get("y_label") if xs else None)
    opts = SimpleNamespace(fontSize=args.font_size, fontFamily=args.font_family)
    fig = plot_swarm_overlays(
        xs,
        out=args.out,
        image_format=args.image_format,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=ylabel,
        ymin=args.ymin,
        ymax=args.ymax,
        zero_line=True,
        stats=bool(args.stats),
        stats_alpha=float(args.stats_alpha),
        stats_debug=bool(args.stats_debug),
        log_tag="btw_rwd_tortuosity_wall_flycorr",
        opts=opts,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
