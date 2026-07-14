#!/usr/bin/env python3
# scripts/plot_overlay_training_metric_scalar_bars.py

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt

from src.plotting.overlay_training_metric_scalar_bars import (
    OmnibusLearnerEntry,
    baseline_delta_exports,
    clustered_training_scalar_exports,
    load_export_npz,
    plot_omnibus_learner_overlays,
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
        help=(
            "Repeatable: 'GroupLabel=/path/to/export.npz'. With "
            "--omnibus-learner-layout or --clustered-layout, use "
            "'ClusterLabel|GroupLabel=/path/to/export.npz'."
        ),
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
    p.add_argument(
        "--baseline-delta-panel",
        default=None,
        help=(
            "Convert each export to per-unit target-minus-baseline deltas using "
            "this panel as baseline, for example 'Pre-training'."
        ),
    )
    p.add_argument(
        "--baseline-delta-target-panel",
        action="append",
        default=None,
        help=(
            "Panel to subtract --baseline-delta-panel from. Repeatable. "
            "Defaults to every non-baseline panel."
        ),
    )
    p.add_argument(
        "--ymax",
        "--max",
        dest="ymax",
        type=float,
        default=None,
        help="Optional fixed y max.",
    )
    p.add_argument(
        "--points",
        "--show-points",
        "--swarm",
        dest="show_points",
        action="store_true",
        help="Overlay an individual-value swarm on top of each bar.",
    )
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
    p.add_argument(
        "--omnibus-learner-layout",
        action="store_true",
        help=(
            "Plot one panel with learner clusters (for example top learners, then "
            "bottom learners) and genotype bars within each cluster. Stats are "
            "limited to genotype comparisons within each learner cluster and "
            "learner comparisons within each genotype."
        ),
    )
    p.add_argument(
        "--clustered-layout",
        action="store_true",
        help=(
            "Plot one cluster per label before '|' and one group bar per label "
            "after '|'. Input order determines cluster and group order."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.omnibus_learner_layout and args.clustered_layout:
        raise SystemExit(
            "Use only one of --omnibus-learner-layout and --clustered-layout."
        )
    clustered_layout = bool(
        args.omnibus_learner_layout or args.clustered_layout
    )

    xs = []
    omnibus_entries = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(
                f"--input must be of the form 'Label=path.npz' (got: {spec})"
            )
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if clustered_layout:
            if "|" not in label:
                raise SystemExit(
                    "clustered-layout input labels must be of the form "
                    f"'ClusterLabel|GroupLabel=path.npz' (got: {spec})"
                )
            learner, genotype = label.split("|", 1)
            learner = learner.strip()
            genotype = genotype.strip()
            if not learner or not genotype:
                raise SystemExit(
                    "clustered-layout input labels must include both "
                    f"cluster and group labels (got: {spec})"
                )
            omnibus_entries.append(
                OmnibusLearnerEntry(
                    learner=learner,
                    genotype=genotype,
                    export=load_export_npz(f"{learner}, {genotype}", path),
                )
            )
        else:
            xs.append(load_export_npz(label, path))

    if args.clustered_layout:
        xs = clustered_training_scalar_exports(omnibus_entries)

    if args.baseline_delta_panel:
        if clustered_layout:
            raise SystemExit(
                "--baseline-delta-panel is not currently supported with "
                "a clustered layout."
            )
        xs = baseline_delta_exports(
            xs,
            baseline_panel=args.baseline_delta_panel,
            target_panels=args.baseline_delta_target_panel,
        )

    # Reasonable defaults if not provided
    if args.xlabel is None:
        args.xlabel = None if clustered_layout else "training"
    if args.ylabel is None:
        first_export = omnibus_entries[0].export if args.omnibus_learner_layout else xs[0]
        args.ylabel = first_export.meta.get("y_label", "value")

    title = args.title
    if title is None and args.suptitle:
        first_export = omnibus_entries[0].export if args.omnibus_learner_layout else xs[0]
        title = first_export.meta.get("base_title", "Overlay bars")

    opts = SimpleNamespace(
        imageFormat=args.image_format,
        fontSize=args.font_size,
        fontFamily=args.font_family,
    )

    if args.omnibus_learner_layout:
        fig = plot_omnibus_learner_overlays(
            omnibus_entries,
            title=title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            ymax=args.ymax,
            ytick_step=0.2 if args.ymax == 1 else None,
            stats=args.stats,
            stats_alpha=args.stats_alpha,
            debug=args.stats_debug,
            show_points=args.show_points,
            opts=opts,
        )
    else:
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
            show_points=args.show_points,
            opts=opts,
        )
    _savefig(args.out, args.image_format)
    plt.close(fig)


if __name__ == "__main__":
    main()
