#!/usr/bin/env python3
from __future__ import annotations

import argparse
from types import SimpleNamespace

import matplotlib.pyplot as plt

from src.plotting.overlay_training_metric_filled_hist import (
    apply_distribution_operations,
    load_export_npz,
    parse_distribution_operation_spec,
    plot_filled_hist_overlays,
)
from src.utils.common import writeImage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot exported training histograms as transparent overlaid filled "
            "distributions, without CIs or statistical annotations."
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
    p.add_argument("--title", default=None, help="Figure title override.")
    p.add_argument("--xlabel", default=None, help="X-axis label override.")
    p.add_argument("--ylabel", default=None, help="Y-axis label override.")
    p.add_argument("--ymax", type=float, default=None, help="Optional fixed y max.")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.38,
        help="Fill transparency for each group distribution (default: %(default)s).",
    )
    p.add_argument(
        "--derive",
        action="append",
        default=[],
        help=(
            "Repeatable derived distribution spec: "
            "'Result label=Term 1 label,add,Term 2 label' or "
            "'Result label=Term 1 label,subtract,Term 2 label'. "
            "Terms must match --input labels."
        ),
    )
    p.add_argument(
        "--derived-only",
        action="store_true",
        help="Plot only distributions created with --derive, hiding original inputs.",
    )
    p.add_argument(
        "--renormalize-derived",
        action="store_true",
        help=(
            "After each --derive operation, divide each derived panel by its finite "
            "bin sum. Panels with zero finite sum are left unchanged."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    hists = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(
                f"--input must be of the form 'Label=path.npz' (got: {spec})"
            )
        label, path = spec.split("=", 1)
        hists.append(load_export_npz(label.strip(), path.strip()))

    derive_specs = []
    for spec in args.derive:
        try:
            derive_specs.append(parse_distribution_operation_spec(spec))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    try:
        hists = apply_distribution_operations(
            hists,
            derive_specs,
            include_inputs=not args.derived_only,
            renormalize=args.renormalize_derived,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not hists:
        raise SystemExit("--derived-only requires at least one --derive spec.")

    if args.xlabel is None:
        args.xlabel = hists[0].meta.get("x_label", None)
    if args.ylabel is None:
        args.ylabel = "Proportion"

    opts = SimpleNamespace(
        imageFormat=args.image_format,
        fontFamily=args.font_family,
        fontSize=args.font_size,
    )
    fig = plot_filled_hist_overlays(
        hists,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        ymax=args.ymax,
        alpha=args.alpha,
        opts=opts,
    )
    writeImage(args.out, format=args.image_format)
    plt.close(fig)


if __name__ == "__main__":
    main()
