"""Benchmark annotation layout without video inputs.

Run from the repository root::

    python -m scripts.benchmark_annotation_layout --profile

To compare positions and PNG/PDF/SVG exports against an earlier implementation::

    git show HEAD:src/plotting/annotation_layout.py > /tmp/annotation_layout_before.py
    python -m scripts.benchmark_annotation_layout --reference /tmp/annotation_layout_before.py

Timings cover the resolver only, including its initial figure draw. This is a
synthetic plotting workload, not an end-to-end postAnalyze benchmark.
"""

import argparse
import cProfile
import importlib.util
import io
import json
import pstats
from time import perf_counter
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plotting import annotation_layout


def make_figure(*, dpi=100, font_size=14, layout=None, panels=3, buckets=8):
    fig, axes = plt.subplots(
        1, panels, figsize=(5 * panels, 4), dpi=dpi, squeeze=False,
        **({"layout": layout} if layout else {}),
    )
    annotations = []
    for panel, ax in enumerate(axes.flat):
        xs = np.arange(1, buckets + 1, dtype=float)
        ys = 0.6 + 0.22 * np.sin(xs + panel)
        ax.set(xlim=(0, buckets + 1), ylim=(-0.5, 2.0), title=f"Training {panel + 1}")
        ax.axhline(0, color="gray")
        texts = []
        counts = []
        for group in range(2):
            values = ys + 0.12 * group
            ax.plot(xs, values, "o-", markersize=4, label=f"Group {group + 1}")
            ax.fill_between(xs, values - 0.08, values + 0.08, alpha=0.2)
            group_counts = []
            for x, y in zip(xs, values):
                count = ax.text(x, y + 0.1, str(80 + group), ha="center", fontsize=font_size)
                count._data_point_y_ = y
                count._data_marker_size_points_ = 4.0
                count._group_idx_ = group
                texts.append(count)
                group_counts.append(count)
            counts.append(group_counts)
        for bucket, x in enumerate(xs):
            stars = ax.text(x, ys[bucket] + 0.4, "***", ha="center", weight="bold", fontsize=font_size)
            stars._sample_size_texts_ = tuple(group[bucket] for group in counts)
            texts.append(stars)
        # Unlinked multiline/math labels exercise collisions and headroom.
        texts.append(ax.text(xs[-1], 1.95, "$p < 0.01$\nAUC", fontsize=font_size))
        texts.append(ax.text(xs[-1], 1.95, "82", fontsize=font_size))
        ax.legend(loc="upper left")
        annotations.append((ax, texts))
    return fig, annotations


def measure(module, *, exports=False, **figure_options):
    fig, annotations = make_figure(**figure_options)
    try:
        ylim = [-0.5, 2.0]
        with patch.object(fig.canvas, "draw", wraps=fig.canvas.draw) as draw:
            start = perf_counter()
            for ax, texts in annotations:
                module.resolve_annotation_text_overlaps(ax, texts, ylim)
            seconds = perf_counter() - start
            draws = draw.call_count
        state = np.array([t.get_position() for _, texts in annotations for t in texts])
        sides = [
            getattr(t, "_sample_size_side_", None)
            for _, texts in annotations for t in texts
        ]
        output = {}
        if exports:
            for fmt, metadata in (
                ("png", {}), ("pdf", {"CreationDate": None, "ModDate": None}),
                ("svg", {"Date": None}),
            ):
                buffer = io.BytesIO()
                fig.savefig(buffer, format=fmt, metadata=metadata)
                output[fmt] = buffer.getvalue()
        return {"seconds": seconds, "draws": draws, "ylim": ylim}, state, sides, output
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", help="Path to the previous annotation_layout.py")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--font-size", type=float, default=14)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()
    options = dict(
        panels=args.panels, buckets=args.buckets,
        font_size=args.font_size, dpi=args.dpi,
    )
    with matplotlib.rc_context({"svg.hashsalt": "annotation-layout-benchmark"}):
        profiler = cProfile.Profile()
        if args.profile:
            profiler.enable()
        current = measure(annotation_layout, exports=bool(args.reference), **options)
        profiler.disable()
        report = {"current": current[0]}
        if args.reference:
            spec = importlib.util.spec_from_file_location(
                "reference_annotation_layout", args.reference,
            )
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
            previous = measure(reference, exports=True, **options)
            report["reference"] = previous[0]
            report["speedup"] = previous[0]["seconds"] / current[0]["seconds"]
            report["identical_positions"] = np.array_equal(current[1], previous[1])
            report["identical_limits"] = current[0]["ylim"] == previous[0]["ylim"]
            report["identical_sides"] = current[2] == previous[2]
            report["identical_exports"] = {
                fmt: data == previous[3][fmt] for fmt, data in current[3].items()
            }
            if not all((
                report["identical_positions"], report["identical_limits"],
                report["identical_sides"], *report["identical_exports"].values(),
            )):
                raise AssertionError(report)
        print(json.dumps(report, indent=2))
        if args.profile:
            pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main()
