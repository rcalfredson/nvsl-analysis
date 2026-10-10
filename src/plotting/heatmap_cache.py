"""Portable, unclipped probability maps for post-hoc heatmap scale experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.plotting.heatmap_style import apply_heatmap_text_layout, heatmap_log_formatter


def save_heatmap_cache(path, rows, layout, cmap):
    """Save numeric arrays and JSON metadata without pickled Python objects."""
    arrays = {}
    metadata = {"version": 1, "layout": layout, "rows": []}
    for row in rows:
        entries = []
        for panel in row:
            panel = dict(panel)
            key = f"map_{len(arrays)}"
            arrays[key] = np.asarray(panel.pop("probability"))
            entries.append({**panel, "array": key})
        metadata["rows"].append(entries)
    arrays["cmap"] = cmap(np.linspace(0, 1, cmap.N))
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def load_heatmap_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["version"] != 1:
            raise ValueError("Unsupported heatmap cache version; rebuild the cache")
        rows = []
        for row in metadata["rows"]:
            entries = []
            for entry in row:
                panel = dict(entry)
                probability = archive[panel.pop("array")].copy()
                if (probability.ndim != 2 or not np.isfinite(probability).all()
                        or np.any(probability < 0) or not np.any(probability > 0)):
                    raise ValueError("Cache contains an invalid probability map")
                entries.append({**panel, "probability": probability})
            rows.append(entries)
        cmap = mpl.colors.ListedColormap(archive["cmap"].copy())
    return rows, metadata["layout"], cmap


def render_heatmap_cache(path, vmin, vmax, *, font_family="Arial", font_size=20):
    """Recreate a logarithmic heatmap figure; caller saves/closes the result.

    Only rendering uses the floor: the cached probabilities retain zeros and
    values below any previous vmin. No source recordings are accessed.
    """
    if not (np.isfinite(vmin) and np.isfinite(vmax) and 0 < vmin < vmax):
        raise ValueError("Logarithmic heatmaps require finite 0 < vmin < vmax")
    rows, layout, cmap = load_heatmap_cache(path)
    nc, nsr, nsc = (layout[key] for key in ("nc", "nsr", "nsc"))
    with mpl.rc_context({"font.family": font_family, "font.size": font_size}):
        fig = plt.figure(figsize=layout["figsize"])
        gs = fig.add_gridspec(
            len(rows), nc + 1, wspace=0.2, hspace=0.2 / nsr,
            width_ratios=[1] * nc + [0.07], top=0.9, bottom=0.05, left=0.05, right=0.95,
        )
        for row_index, row in enumerate(rows):
            cbar_ax = fig.add_subplot(gs[row_index, nc])
            axes, titles, subgrids = [], [], {}
            for panel in row:
                column = panel["column"]
                if column not in subgrids:
                    subgrids[column] = gs[row_index, column].subgridspec(
                        nsr, nsc, wspace=0.06 if nsc > 1 else 0,
                        hspace=0.045 if nsr > 1 else 0,
                    )
                ax = fig.add_subplot(subgrids[column][panel["fly"]])
                probability = panel["probability"]
                image = ax.imshow(
                    np.maximum(probability, vmin), cmap=cmap,
                    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
                    extent=[0, probability.shape[1], probability.shape[0], 0],
                )
                ax.set(xticks=[], yticks=[], aspect="equal")
                ax.axis("off")
                if not axes:
                    cb = fig.colorbar(
                        image, cax=cbar_ax, ax=ax,
                        ticks=mpl.ticker.LogLocator(subs=(1.0, 3.0)),
                        format=heatmap_log_formatter(),
                    )
                    cb.outline.set_linewidth(0)
                header = ax.set_title(panel["header"], loc="left") if panel["header"] else None
                sample = ax.set_title(panel["sample"], loc="right") if panel["sample"] else None
                if panel["circle"] is not None:
                    cx, cy, radius = panel["circle"]
                    ax.add_patch(mpl.patches.Circle(
                        (cx, cy), radius, color="k", fill=False, linewidth=0.8,
                    ))
                axes.append(ax)
                titles.append((header, sample))
            apply_heatmap_text_layout(axes, titles, cbar_ax, font_size)
    return fig
