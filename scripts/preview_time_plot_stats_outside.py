"""Experimental outside-AUC layouts; does not change production placement.

    python -m scripts.preview_time_plot_stats_outside --sample-sizes 999 999
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.preview_time_plot_stats import build_preview, preview_font_context
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LAYOUTS = ("header", "panel-footer", "figure-footer")


@preview_font_context()
def build_outside_preview(layout, *, sample_sizes=(999, 999), plot_font_size=27,
                          panel_width_pt=436, dpi=100):
    """Keep the original panel geometry and reserve space for full-size Arial."""
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout: {layout}")
    fig, axes, original = build_preview(
        sample_sizes=sample_sizes, plot_font_size=plot_font_size,
        panel_width_pt=panel_width_pt, dpi=dpi,
    )
    wording = original._auc_original_text
    stats_size = original.get_fontsize()
    original.remove()

    left, gap, right, panel_height = 90, 32, 32, 327
    bottom = {"header": 85, "panel-footer": 140, "figure-footer": 168}[layout]
    headroom = 104 if layout == "header" else 45
    width = left + 2 * panel_width_pt + gap + right
    height = bottom + panel_height + headroom
    fig.set_size_inches(width / 72, height / 72)
    for i, ax in enumerate(axes):
        ax.set_position([
            (left + i * (panel_width_pt + gap)) / width, bottom / height,
            panel_width_pt / width, panel_height / height,
        ])
        ax.title.set_visible(False)
        title_y = bottom + panel_height + (62 if layout == "header" else 9)
        fig.text((left + i * (panel_width_pt + gap) + panel_width_pt / 2) / width,
                 title_y / height, f"Training {i + 1}",
                 ha="center", va="bottom", fontsize=plot_font_size * 30 / 27)

    xlabel = next(text for text in fig.texts if text.get_text().startswith("10-min"))
    center_x = (left + (2 * panel_width_pt + gap) / 2) / width
    xlabel_y = {"header": 40, "panel-footer": bottom - 104,
                "figure-footer": bottom - 60}[layout]
    xlabel.set_position((center_x, xlabel_y / height))
    xlabel.set_va("top")

    right_center = (left + panel_width_pt + gap + panel_width_pt / 2) / width
    if layout == "header":
        position = (right_center, (bottom + panel_height + 14) / height)
        va = "bottom"
    elif layout == "panel-footer":
        position = (right_center, (bottom - 52) / height)
        va = "top"
    else:
        position = (center_x, (bottom - 115) / height)
        va = "top"
        wording = "Training 2:  " + wording
        fig.add_artist(Line2D(
            [left / width, (width - right) / width],
            [(bottom - 101) / height] * 2,
            transform=fig.transFigure, color="0.8", linewidth=0.6,
        ))
    stats = fig.text(*position, wording, ha="center", va=va,
                     fontfamily="Arial", fontsize=stats_size)
    return fig, axes, stats


def check_preview(fig, axes, stats, *, panel_width_pt=436):
    """Check physical panel dimensions, font choice, clipping and collisions."""
    from matplotlib.text import Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = stats.get_window_extent(renderer)
    canvas = fig.get_window_extent(renderer)
    assert canvas.contains(bounds.x0, bounds.y0)
    assert canvas.contains(bounds.x1, bounds.y1)
    assert stats.get_fontfamily() == ["Arial"]
    assert "\n" not in stats.get_text()
    # Locators also create tick artists beyond the limits; Axis omits those
    # from painting even though the Text object's visible flag stays true.
    inactive_labels = {
        id(label) for ax in axes for axis in (ax.xaxis, ax.yaxis)
        for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks())
        if not min(axis.get_view_interval()) <= tick.get_loc() <= max(axis.get_view_interval())
        for label in (tick.label1, tick.label2)
    }
    for ax in axes:
        panel = ax.get_window_extent(renderer)
        assert abs(panel.width * 72 / fig.dpi - panel_width_pt) < 1e-8
        assert abs(panel.height * 72 / fig.dpi - 327) < 1e-8
        assert not bounds.overlaps(panel)
        assert all(abs(t.get_fontsize() - stats.get_fontsize()) < 1e-8
                   for t in ax.get_xticklabels())
    for other in fig.findobj(match=Text):
        if (other is stats or id(other) in inactive_labels
                or not other.get_visible() or not other.get_text()):
            continue
        assert other.get_fontfamily() == ["Arial"]
        assert not bounds.overlaps(other.get_window_extent(renderer)), other.get_text()
        other_bounds = other.get_window_extent(renderer)
        assert canvas.contains(other_bounds.x0, other_bounds.y0), other.get_text()
        assert canvas.contains(other_bounds.x1, other_bounds.y1), other.get_text()


@preview_font_context()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=(*LAYOUTS, "all"), default="all")
    parser.add_argument("--sample-sizes", type=int, nargs=2, default=(999, 999))
    parser.add_argument("--plot-font-size", type=float, default=27)
    parser.add_argument("--panel-width-pt", type=float, default=436)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/time_plot_stats_outside"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layouts = LAYOUTS if args.layout == "all" else (args.layout,)
    for layout in layouts:
        fig, axes, stats = build_outside_preview(
            layout, sample_sizes=args.sample_sizes, plot_font_size=args.plot_font_size,
            panel_width_pt=args.panel_width_pt, dpi=args.dpi,
        )
        try:
            check_preview(fig, axes, stats, panel_width_pt=args.panel_width_pt)
            for fmt in ("png", "pdf"):
                output = args.output_dir / f"{layout}.{fmt}"
                fig.savefig(output)
                check_preview(fig, axes, stats, panel_width_pt=args.panel_width_pt)
                print(output, flush=True)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
