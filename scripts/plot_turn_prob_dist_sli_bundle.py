#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.turn_prob_dist_plotter import TurnProbabilityByDistancePlotter


def _scalar_string(bundle, key: str, default=""):
    if key not in bundle:
        return default
    value = bundle[key]
    try:
        return str(value.reshape(()).item())
    except Exception:
        return str(value)


def _bundle_to_vas(bundle):
    distances = [
        float(x) for x in np.asarray(bundle["turn_prob_distances_mm"], dtype=float)
    ]
    values = np.asarray(bundle["turn_prob_values"], dtype=float)
    group_indices = np.asarray(bundle["group_indices"], dtype=int).reshape(-1)
    video_ids = np.asarray(bundle["video_ids"], dtype=object).reshape(-1)

    vas = []
    for vi in range(values.shape[0]):
        by_distance = {}
        for di, dist in enumerate(distances):
            fly_values = []
            for fly_idx in range(values.shape[2]):
                fly_values.append(
                    [
                        (
                            float(values[vi, di, fly_idx, ti, 0]),
                            float(values[vi, di, fly_idx, ti, 1]),
                        )
                        for ti in range(values.shape[3])
                    ]
                )
            by_distance[dist] = fly_values
        vas.append(
            SimpleNamespace(
                fn=str(video_ids[vi]),
                gidx=int(group_indices[vi]) if vi < group_indices.size else 0,
                turn_prob_by_distance=by_distance,
            )
        )
    return vas


_IMAGE_FORMATS = {"png", "pdf", "svg", "jpg", "jpeg"}


def _resolve_image_format(out: str, requested: str | None) -> str:
    out_format = Path(out).suffix.lstrip(".").lower()
    image_format = (requested or out_format or "png").strip().lstrip(".").lower()
    if image_format not in _IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {image_format!r}")
    if out_format in _IMAGE_FORMATS and out_format != image_format:
        raise ValueError(
            f"Output extension .{out_format} does not match image format "
            f".{image_format}"
        )
    return image_format


def _selected_source(
    timeframe: str, direction: str, comparison: str, image_format: str
) -> str:
    if comparison == "exp_across_groups":
        return (
            f"imgs/turn_probability_{timeframe}_{direction}_exp_across_groups."
            f"{image_format}"
        )
    raise ValueError(f"Unsupported comparison={comparison!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="Turn-probability .npz bundle.")
    p.add_argument("--out", required=True, help="Output image filename.")
    p.add_argument(
        "--timeframe",
        default="t2_end",
        choices=["pre_trn", "t1_start", "t2_end", "t3_end"],
        help="Timeframe image to copy to --out after plotting.",
    )
    p.add_argument(
        "--direction",
        default="all",
        choices=["toward", "away", "all"],
        help="Turn direction image to copy to --out after plotting.",
    )
    p.add_argument(
        "--comparison",
        default="exp_across_groups",
        choices=["exp_across_groups"],
        help="Which across-group plot to copy to --out.",
    )
    p.add_argument(
        "--xlabel",
        "--turn-prob-dist-xlabel",
        dest="xlabel",
        default=None,
        help="Optional x-axis label override.",
    )
    p.add_argument(
        "--ylabel",
        "--turn-prob-dist-ylabel",
        dest="ylabel",
        default=None,
        help="Optional y-axis label override.",
    )
    p.add_argument(
        "--image-format",
        "--imgFormat",
        dest="image_format",
        default=None,
        help="Output image format (defaults to the --out extension, then PNG).",
    )
    p.add_argument("--fs", dest="font_size", type=float, default=None)
    p.add_argument("--fontFamily", dest="font_family", default=None)
    args = p.parse_args()
    image_format = _resolve_image_format(args.out, args.image_format)

    bundle = np.load(args.bundle, allow_pickle=True)
    vas = _bundle_to_vas(bundle)
    gls = [
        str(x) for x in np.asarray(bundle["group_labels"], dtype=object).reshape(-1)
    ]
    opts = SimpleNamespace(
        contact_geometry=_scalar_string(bundle, "contact_geometry", "circular"),
        imageFormat=image_format,
        fontSize=args.font_size,
        fontFamily=args.font_family,
        turn_prob_dist_xlabel=args.xlabel,
        turn_prob_dist_ylabel=args.ylabel,
        use_union_filter=bool(
            np.asarray(bundle.get("use_union_filter", False)).reshape(())
        ),
    )

    customizer = PlotCustomizer()
    if args.font_size is not None:
        customizer.update_font_size(args.font_size)
    if args.font_family:
        customizer.update_font_family(args.font_family)

    plotter = TurnProbabilityByDistancePlotter(vas, gls, opts, customizer)
    plotter.average_turn_probabilities()
    os.makedirs("imgs", exist_ok=True)
    plotter.plot_turn_probabilities()

    src = _selected_source(
        args.timeframe, args.direction, args.comparison, image_format
    )
    if not os.path.exists(src):
        raise FileNotFoundError(f"Expected plotter output was not created: {src}")
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(src, args.out)
    print(f"[plot] copied {src} -> {args.out}")


if __name__ == "__main__":
    main()
