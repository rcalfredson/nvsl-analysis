from __future__ import annotations
import numpy as np
from src.plotting.bin_edges import geom_from_edges


def grouped_bar_layout_from_edges(
    edges: np.ndarray,
    n_groups: int,
    *,
    categorical: bool = True,
    categorical_bin_ratio_max: float = 4.0,
    px_per_group: int = 12,
    px_per_bin: int = 70,
    dpi: int = 100,
    min_fig_w: float = 7.4,
    group_band_categorical: float = 0.80,
    group_band_frac: float = 0.95,
):
    """
    Returns geometry to draw grouped/dodged bars with consistent spacing.

    Output:
      - fig_w: suggested figure width (inches)
      - centers_x: x positions for bins (either categorical 0..B-1 or proportional centers)
      - bar_w: per-bin bar width (shape (B,))
      - offsets: per-group offsets (shape (G,B))
      - bin_ranges: list of (lo,hi) for tick labels
      - xlim: (xmin,xmax)
      - categorical_used: bool
    """
    widths, centers, bin_ranges, B, x0, x1 = geom_from_edges(edges)

    wpos = widths[np.isfinite(widths) & (widths > 0)]
    width_ratio = float(np.nanmax(wpos) / np.nanmin(wpos)) if wpos.size else 1.0

    categorical_used = bool(categorical) or (
        width_ratio > float(categorical_bin_ratio_max)
    )

    G = max(1, int(n_groups))

    if categorical_used:
        # width should scale with number of bins
        centers_x = np.arange(B, dtype=float)
        group_band = float(group_band_categorical)
        bar_w = np.full((B,), group_band / G, dtype=float)
        xlim = (-0.5, B - 0.5)
        fig_w = max(min_fig_w, (B * px_per_bin) / float(dpi))
    else:
        centers_x = centers
        group_band = group_band_frac * widths
        bar_w = group_band / G
        xlim = (x0, x1)
        fig_w = max(min_fig_w, (B * G * px_per_group) / float(dpi))

    gpos = (np.arange(G, dtype=float) - (G - 1) / 2.0)[:, None]  # (G,1)
    offsets = gpos * bar_w[None, :]  # (G,B)

    return fig_w, centers_x, bar_w, offsets, bin_ranges, xlim, categorical_used
