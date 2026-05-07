#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nvsl-analysis-matplotlib")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.agarose_time_summary import (
    DEFAULT_POST_COL,
    DEFAULT_PRE_COL,
    DEFAULT_SECTION,
    describe_values,
    paired_test,
    parse_group,
    reduction_anova_and_posthoc,
)
from src.plotting.palettes import (
    NEUTRAL_DARK,
    group_metric_edge_color,
    group_metric_fill_color,
)
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.stats_bars import draw_sig_bracket, format_sig_label


def _parse_group_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--group must be formatted as LABEL=CSV_PATH, for example Control=learning_stats.csv"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "--group must include both a non-empty label and a CSV path."
        )
    return label, path


def _savefig(out_path: str, image_format: str) -> None:
    file_extension = "." + str(image_format).lstrip(".")
    base, ext = os.path.splitext(out_path)
    if ext.lower() != file_extension.lower():
        out_path = base + file_extension
        print(
            "The file extension has been changed to "
            f"{file_extension} to coincide with the specified format."
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", format=image_format)
    print(f"[agarose_reduction_bars] wrote {out_path}")


def _errorbar_lengths(
    means: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray
) -> np.ndarray:
    lo = np.where(np.isfinite(ci_low), np.maximum(means - ci_low, 0.0), 0.0)
    hi = np.where(np.isfinite(ci_high), np.maximum(ci_high - means, 0.0), 0.0)
    return np.vstack([lo, hi])


def _blend_with_white(color: str, amount: float = 0.55) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * float(amount))


def _set_ylim_for_values(
    ax: plt.Axes,
    means: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    per_group_values: list[np.ndarray],
) -> None:
    raw_values = [
        values[np.isfinite(values)]
        for values in per_group_values
        if values is not None and np.any(np.isfinite(values))
    ]
    finite = np.concatenate(
        [
            means[np.isfinite(means)],
            ci_low[np.isfinite(ci_low)],
            ci_high[np.isfinite(ci_high)],
            *(raw_values if raw_values else []),
            np.asarray([0.0]),
        ]
    )
    if finite.size == 0:
        ax.set_ylim(-1.0, 1.0)
        return
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if y_min == y_max:
        pad = max(abs(y_max) * 0.2, 1.0)
    else:
        pad = 0.16 * float(y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + max(pad, 0.24 * float(y_max - y_min)))


def _draw_stats(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    y_base: float,
    posthoc_rows,
    group_names: list[str],
    alpha: float,
    fontsize: float,
    show_p_value: bool,
) -> None:
    sig_rows = [
        row
        for row in posthoc_rows
        if np.isfinite(row.p_value_holm) and row.p_value_holm < float(alpha)
    ]
    if not sig_rows:
        return

    idx_by_group = {name: i for i, name in enumerate(group_names)}
    ylim0, ylim1 = ax.get_ylim()
    y_rng = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0
    bracket_h = 0.018 * y_rng
    step = (0.12 if show_p_value else 0.085) * y_rng
    y = float(y_base + 0.055 * y_rng)

    max_top = y
    for level, row in enumerate(sig_rows):
        i = idx_by_group.get(row.group_a)
        j = idx_by_group.get(row.group_b)
        if i is None or j is None:
            continue
        stars = format_sig_label(row.p_value_holm, include_p_value=show_p_value)
        if not stars or stars == "ns":
            continue
        y_here = y + level * step
        draw_sig_bracket(
            ax,
            x1=float(x[i]),
            x2=float(x[j]),
            y=float(y_here),
            h=float(bracket_h),
            text=stars,
            fontsize=float(fontsize),
            color=NEUTRAL_DARK,
        )
        max_top = max(
            max_top,
            y_here + bracket_h + (0.10 if show_p_value else 0.06) * y_rng,
        )

    ylim0, ylim1 = ax.get_ylim()
    if max_top > ylim1:
        ax.set_ylim(ylim0, max_top + 0.05 * y_rng)


def _group_upper_tops(
    ci_high: np.ndarray,
    per_group_values: list[np.ndarray],
) -> np.ndarray:
    tops = np.asarray(ci_high, dtype=float).copy()
    for i, values in enumerate(per_group_values):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        value_top = float(np.max(finite))
        if i >= tops.size or not np.isfinite(tops[i]):
            tops[i] = value_top
        else:
            tops[i] = max(float(tops[i]), value_top)
    return tops


def build_figure(
    groups,
    *,
    section: str,
    pre_col: str,
    post_col: str,
    title: str | None,
    ylabel: str,
    control_group: str | None,
    posthoc_scope: str,
    posthoc_method: str,
    stats_alpha: float,
    show_stats: bool,
    show_stats_p_value: bool,
    show_points: bool,
    opts,
) -> plt.Figure:
    customizer = PlotCustomizer()
    if getattr(opts, "fontSize", None) is not None:
        customizer.update_font_size(getattr(opts, "fontSize"))
    customizer.update_font_family(getattr(opts, "fontFamily", None))

    parsed = [
        parse_group(label, path, section=section, numeric_cols=[pre_col, post_col])
        for label, path in groups
    ]
    paired = [paired_test(group, pre_col=pre_col, post_col=post_col) for group in parsed]
    group_names = [test.group for test in paired]
    desc = [describe_values(test.reductions) for test in paired]
    per_group_values = [
        np.asarray(test.reductions, dtype=float)[
            np.isfinite(np.asarray(test.reductions, dtype=float))
        ]
        for test in paired
    ]

    means = np.asarray([d.mean for d in desc], dtype=float)
    ci_low = np.asarray([d.ci95_low for d in desc], dtype=float)
    ci_high = np.asarray([d.ci95_high for d in desc], dtype=float)
    ns = np.asarray([d.n for d in desc], dtype=int)
    x = np.arange(len(paired), dtype=float)

    font_scale = max(float(customizer.increase_factor), 1.0)
    fig_w = max(4.8, 1.25 * len(paired) + 2.2) * min(1.0 + 0.10 * (font_scale - 1.0), 1.2)
    fig_h = 4.2 * min(1.0 + 0.12 * (font_scale - 1.0), 1.24)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    width = 0.68
    for i, mean in enumerate(means):
        ax.bar(
            x[i],
            mean,
            width=width,
            color=group_metric_fill_color(i, "between_reward_distance"),
            edgecolor=group_metric_edge_color(i, "between_reward_distance"),
            linewidth=1.2,
            alpha=0.92,
            zorder=2,
        )
    yerr = _errorbar_lengths(means, ci_low, ci_high)
    for i in range(len(means)):
        if not np.isfinite(means[i]):
            continue
        halo_color = _blend_with_white(
            group_metric_fill_color(i, "between_reward_distance"),
            amount=0.58,
        )
        ax.errorbar(
            [x[i]],
            [means[i]],
            yerr=yerr[:, [i]],
            fmt="none",
            ecolor=halo_color,
            elinewidth=2.3,
            capsize=5,
            capthick=2.3,
            alpha=0.5,
            zorder=5.8,
        )
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="none",
        ecolor=NEUTRAL_DARK,
        elinewidth=1.35,
        capsize=4,
        capthick=1.35,
        zorder=6,
    )

    if show_points:
        rng = np.random.default_rng(12345)
        for i, values in enumerate(per_group_values):
            if values.size == 0:
                continue
            jitter = rng.uniform(-0.11, 0.11, size=values.size)
            ax.scatter(
                np.full(values.size, x[i]) + jitter,
                values,
                s=22,
                color=group_metric_edge_color(i, "between_reward_distance"),
                alpha=0.72,
                linewidths=0,
                zorder=4,
            )

    ax.axhline(0, color="0.35", linewidth=0.9, linestyle="-", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    for tick in ax.get_xticklabels():
        tick.set_fontstyle("italic")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.8, zorder=0)

    _set_ylim_for_values(ax, means, ci_low, ci_high, per_group_values)

    text_fontsize = max(7.0, min(float(customizer.in_plot_font_size) - 3.0, 10.0))
    ylim0, ylim1 = ax.get_ylim()
    y_rng = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0
    group_tops = _group_upper_tops(ci_high, per_group_values)
    n_label_tops: list[float] = []
    for xi, y_top, n in zip(x, group_tops, ns):
        y = float(y_top) if np.isfinite(y_top) else float(ylim0)
        y_label = y + 0.055 * y_rng
        n_label_tops.append(y_label + 0.035 * y_rng)
        ax.text(
            float(xi),
            y_label,
            f"n={int(n)}",
            ha="center",
            va="bottom",
            fontsize=text_fontsize,
            color="0.25",
            clip_on=False,
            zorder=8,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 0.8,
            },
        )

    if show_stats and len(paired) >= 2:
        _anova, posthoc = reduction_anova_and_posthoc(
            paired,
            control_group=control_group,
            posthoc_scope=posthoc_scope,
            posthoc_method=posthoc_method,
        )
        _draw_stats(
            ax,
            x=x,
            y_base=max(n_label_tops) if n_label_tops else float(np.nanmax(group_tops)),
            posthoc_rows=posthoc,
            group_names=group_names,
            alpha=float(stats_alpha),
            fontsize=text_fontsize,
            show_p_value=bool(show_stats_p_value),
        )

    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot mean pre-to-post reduction in percent time spent on agarose from "
            "learning_stats.csv files."
        )
    )
    p.add_argument(
        "--group",
        action="append",
        type=_parse_group_arg,
        required=True,
        metavar="LABEL=CSV_PATH",
        help="Group label and learning_stats.csv path. Repeat once per group.",
    )
    p.add_argument(
        "--section", default=DEFAULT_SECTION, help="Exact section title to parse."
    )
    p.add_argument("--pre-col", default=DEFAULT_PRE_COL, help="Pre-training column.")
    p.add_argument("--post-col", default=DEFAULT_POST_COL, help="Post-training column.")
    p.add_argument("--out", required=True, help="Output image path.")
    p.add_argument(
        "--image-format",
        "--imgFormat",
        dest="image_format",
        default="png",
        help="Output image format (for example png, pdf, or svg).",
    )
    p.add_argument("--fs", dest="font_size", type=float, default=None)
    p.add_argument("--fontFamily", dest="font_family", type=str, default=None)
    p.add_argument("--title", default=None, help="Optional plot title.")
    p.add_argument(
        "--ylabel",
        default=(
            "Reduction in % time on agarose, $\\mathrm{pre} - \\mathrm{post}$\n"
            "(percentage points)"
        ),
        help="Y-axis label.",
    )
    p.add_argument(
        "--control-group",
        default=None,
        help="Control group label for post-hoc comparisons. Defaults to first group.",
    )
    p.add_argument(
        "--posthoc-scope",
        choices=["control", "all"],
        default="control",
        help=(
            "Post-hoc comparison family for Holm correction and plot stars. "
            "'control' compares only control-vs-other groups; 'all' compares all pairs."
        ),
    )
    p.add_argument(
        "--posthoc-method",
        choices=["holm-welch", "games-howell"],
        default="holm-welch",
        help=(
            "Post-hoc method for plot stars. 'holm-welch' runs pairwise Welch "
            "t-tests with Holm correction; 'games-howell' runs Games-Howell tests."
        ),
    )
    p.add_argument(
        "--stats-alpha",
        type=float,
        default=0.05,
        help="Alpha threshold for drawing Holm-adjusted post-hoc stars.",
    )
    p.add_argument(
        "--no-stats",
        action="store_true",
        help="Do not draw Holm-adjusted post-hoc stars.",
    )
    p.add_argument(
        "--show-stats-p-value",
        "--show-stats-p-values",
        action="store_true",
        help="Show the adjusted p-value below each significant post-hoc star label.",
    )
    p.add_argument(
        "--hide-points",
        action="store_true",
        help="Hide per-fly reduction points overlaid on the bars.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    opts = argparse.Namespace(
        imageFormat=args.image_format,
        fontSize=args.font_size,
        fontFamily=args.font_family,
    )
    fig = build_figure(
        args.group,
        section=args.section,
        pre_col=args.pre_col,
        post_col=args.post_col,
        title=args.title,
        ylabel=args.ylabel,
        control_group=args.control_group,
        posthoc_scope=args.posthoc_scope,
        posthoc_method=args.posthoc_method,
        stats_alpha=args.stats_alpha,
        show_stats=not args.no_stats,
        show_stats_p_value=args.show_stats_p_value,
        show_points=not args.hide_points,
        opts=opts,
    )
    _savefig(args.out, args.image_format)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
