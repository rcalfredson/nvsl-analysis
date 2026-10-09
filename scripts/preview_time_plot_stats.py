"""Preview time-plot statistics with synthetic curves and no fly analysis.

Run from the repository root:
    python -m scripts.preview_time_plot_stats --plot-font-size 27 --output-prefix tmp/time_plot_stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plotting.annotation_layout import place_auc_annotation
from src.plotting.plot_customizer import PlotCustomizer, compact_legend_spacing


def preview_font_context():
    """Use Arial for text and math, including artists created with the axes.

    Keep this context active when exporting figures returned by build_preview:
    mathtext resolves its symbol fonts at render time.
    """
    return plt.rc_context({
        "font.family": "Arial",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })


@preview_font_context()
def build_preview(*, panel_width_pt=436.0, font_size=24.0, dpi=100,
                  plot_font_size=None, sample_sizes=(17, 44)):
    """Return a figure, two axes and an AUC artist at manuscript-like sizes."""
    title_size = axis_label_size = font_size + 3
    annotation_size = font_size
    if plot_font_size is not None:
        customizer = PlotCustomizer()
        customizer.update_font_size(plot_font_size)
        font_size = float(plt.rcParams["xtick.labelsize"])
        title_size = float(plt.rcParams["axes.titlesize"])
        axis_label_size = float(plt.rcParams["axes.labelsize"])
        annotation_size = customizer.in_plot_font_size
    panel_height_pt = 327.0
    left, gap, right, bottom, top = 90.0, 32.0, 20.0, 85.0, 45.0
    width = left + 2 * panel_width_pt + gap + right
    height = bottom + panel_height_pt + top
    fig = plt.figure(figsize=(width / 72, height / 72), dpi=dpi)
    axes = [fig.add_axes([
        (left + i * (panel_width_pt + gap)) / width,
        bottom / height, panel_width_pt / width, panel_height_pt / height,
    ]) for i in range(2)]
    xs = np.arange(10, 60, 10)
    upper = ([0.89, 1.10, 1.05, 1.04, 1.11], [1.16, 1.18, 1.22, 1.19, 1.24])
    lower = ([0.05, 0.08, 0.12, 0.20, 0.17], [0.18, 0.20, 0.25, 0.25, 0.26])
    for i, ax in enumerate(axes):
        ax.set(xlim=(0, 60), ylim=(-0.2, 2.1), xticks=xs.tolist() + [60])
        ax.set_title(f"Training {i + 1}", fontsize=title_size)
        ax.tick_params(labelsize=font_size)
        for values, label, linestyle, band in (
            (upper[i], "Top 20% learners", "-", 0.13 if i == 0 else 0.07),
            (lower[i], "Bottom 50% learners", "--", 0.11),
        ):
            values = np.asarray(values)
            ax.plot(xs, values, linestyle, color="C0", marker=".", label=label)
            ax.fill_between(xs, values - band, values + band, color="C0", alpha=0.15)
        lower_counts = [sample_sizes[1] - 1] + [sample_sizes[1]] * 4
        for x, hi, lo, n_lo in zip(xs, upper[i], lower[i], lower_counts):
            ax.text(x, hi + 0.08, str(sample_sizes[0]), ha="center", fontsize=annotation_size)
            ax.text(x, hi + 0.27, "****", ha="center", fontsize=annotation_size, weight="bold")
            ax.text(x, lo + 0.07, str(n_lo if i == 0 else sample_sizes[1]),
                    ha="center", fontsize=annotation_size)
        ax.axhline(0, color="gray", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
    axes[0].set_ylabel("SLI", fontsize=axis_label_size)
    axes[1].tick_params(labelleft=False)
    axes[0].legend(loc="upper right", prop={"size": annotation_size, "style": "italic"},
                   **compact_legend_spacing(annotation_size))
    fig.text(0.55, 0.035, "10-min sync-bucket endpoint (min)",
             ha="center", fontsize=axis_label_size)
    text = axes[1].text(
        0.03, 0.97,
        f"AUC (n = {sample_sizes[0]}, {sample_sizes[1]}): **** "
        r"(p = $\mathregular{3.71 \times 10^{-28}}$)",
        transform=axes[1].transAxes, va="top", fontsize=font_size,
    )
    if not place_auc_annotation(axes[1], text):
        plt.close(fig)
        raise RuntimeError("The statistics annotation cannot fit in this panel.")
    return fig, axes, text


@preview_font_context()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-width-pt", type=float, default=436)
    fonts = parser.add_mutually_exclusive_group()
    fonts.add_argument("--font-size", type=float, default=24,
                       help="Literal stats and tick font size in points (default: 24).")
    fonts.add_argument("--plot-font-size", type=float,
                       help="Manuscript plot setting; stats match its proportional tick size.")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--sample-sizes", type=int, nargs=2, default=(17, 44),
                        metavar=("TOP", "BOTTOM"), help="Synthetic counts for both learner groups.")
    parser.add_argument("--output-prefix", type=Path, default=Path("tmp/time_plot_stats"))
    args = parser.parse_args()
    fig, axes, text = build_preview(
        panel_width_pt=args.panel_width_pt, font_size=args.font_size, dpi=args.dpi,
        plot_font_size=args.plot_font_size,
        sample_sizes=args.sample_sizes,
    )
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        print(f"Panel: {axes[1].get_window_extent(renderer).width * 72 / args.dpi:.1f} pt")
        print(f"Label: {text.get_window_extent(renderer).width * 72 / args.dpi:.1f} pt")
        print(f"Lines: {len(text.get_text().splitlines())}; font: {text.get_fontsize():g} pt")
        print(f"Stats family: {', '.join(text.get_fontfamily())}")
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            output = args.output_prefix.with_suffix(suffix)
            fig.savefig(output)
            print(output)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
