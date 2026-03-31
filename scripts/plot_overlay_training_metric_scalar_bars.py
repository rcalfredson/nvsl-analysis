#!/usr/bin/env python3
# scripts/plot_overlay_training_metric_scalar_bars.py

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt

from src.plotting.overlay_training_metric_scalar_bars import (
    load_export_npz,
    plot_overlays,
)


def _savefig(out_path: str, image_format: str) -> None:
    file_extension = "." + str(image_format).lstrip(".")
    base, ext = os.path.splitext(out_path)
    if ext.lower() != file_extension.lower():
        out_path = base + file_extension
        print(
            f"The file extension has been changed to {file_extension} to coincide with the specified format."
        )
    plt.savefig(out_path, bbox_inches="tight", format=image_format)
    print(f"[overlay_scalar_bars] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay training-panel scalar bar exports (NPZ), with optional stats."
    )
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable: 'GroupLabel=/path/to/export.npz'",
    )
    p.add_argument("--out", required=True, help="Output image path (png/pdf/etc).")
    p.add_argument(
        "--image-format",
        "--imgFormat",
        dest="image_format",
        default="png",
        help="Output image format (for example png, pdf, or svg).",
    )
    p.add_argument(
        "--fs",
        dest="font_size",
        type=float,
        default=None,
        help="Font size for plot text.",
    )
    p.add_argument(
        "--fontFamily",
        dest="font_family",
        type=str,
        default=None,
        help="Override the default font family for plots.",
    )
    p.add_argument("--title", default=None, help="Figure title override.")
    p.add_argument(
        "--suptitle",
        action="store_true",
        help=(
            "Show a figure suptitle. If --title is not provided, a default title is generated "
            "from the first input's base_title."
        ),
    )
    p.add_argument("--xlabel", default=None, help="X-axis label override.")
    p.add_argument("--ylabel", default=None, help="Y-axis label override.")
    p.add_argument("--ymax", type=float, default=None, help="Optional fixed y max.")
    p.add_argument(
        "--stats",
        action="store_true",
        help="Add per-panel one-way ANOVA + Holm posthoc stars (Welch or paired t-tests depending on --stats-paired).",
    )
    p.add_argument(
        "--stats-paired",
        action="store_true",
        help="When --stats is enabled, use paired t-tests (ttest_rel) when unit IDs overlap between groups.",
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
        label = label.strip()
        path = path.strip()
        xs.append(load_export_npz(label, path))

    # Reasonable defaults if not provided
    if args.xlabel is None:
        args.xlabel = "training"
    if args.ylabel is None:
        args.ylabel = xs[0].meta.get("y_label", "value")

    title = args.title
    if title is None and args.suptitle:
        title = xs[0].meta.get("base_title", "Overlay bars")

    opts = SimpleNamespace(
        imageFormat=args.image_format,
        fontSize=args.font_size,
        fontFamily=args.font_family,
    )

    fig = plot_overlays(
        xs,
        title=title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        ymax=args.ymax,
        stats=args.stats,
        stats_alpha=args.stats_alpha,
        stats_paired=args.stats_paired,
        debug=args.stats_debug,
        opts=opts,
    )
    _savefig(args.out, args.image_format)
    plt.close(fig)


if __name__ == "__main__":
    main()
