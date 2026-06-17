from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.plotting.contactless_episode_maxdist_swarm import (
    STATUS_COLORS,
    STATUS_LABELS,
    STATUS_ORDER,
    _as_bool,
)
from src.plotting.palettes import NEUTRAL_DARK
from src.plotting.plot_customizer import PlotCustomizer


REQUIRED_COLUMNS = {
    "group",
    "unit_id",
    "contains_wall_contact",
    "included_in_metric",
    "trajectory_length_mm",
}


def load_episode_csvs(paths: list[str], *, include_excluded: bool = False):
    frames = []
    group_order = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for group in frame["group"].dropna().astype(str):
            if group not in group_order:
                group_order.append(group)
        frame["contains_wall_contact"] = _as_bool(frame["contains_wall_contact"])
        frame["included_in_metric"] = _as_bool(frame["included_in_metric"])
        frame["trajectory_length_mm"] = pd.to_numeric(
            frame["trajectory_length_mm"],
            errors="coerce",
        )
        frame["source_csv"] = str(path)
        frames.append(frame)

    if not frames:
        raise ValueError("no episode CSVs supplied")
    data = pd.concat(frames, ignore_index=True)
    if not include_excluded:
        data = data.loc[data["included_in_metric"]].copy()
    data = data.loc[np.isfinite(data["trajectory_length_mm"])].copy()
    return data, group_order


def plot_wall_contact_episode_length_swarm(
    data: pd.DataFrame,
    *,
    group_order: list[str],
    title: str | None = None,
    ylabel: str = "Trajectory length (mm)",
    ymax: float | None = None,
    opts=None,
):
    if opts is None:
        opts = SimpleNamespace(fontSize=None, fontFamily=None)

    customizer = PlotCustomizer()
    if getattr(opts, "fontSize", None) is not None:
        customizer.update_font_size(float(opts.fontSize))
    customizer.update_font_family(getattr(opts, "fontFamily", None))

    groups = [group for group in group_order if group in set(data["group"])]
    if not groups:
        raise ValueError("no finite episode values available to plot")

    fig_w = max(6.2, 1.65 * len(groups))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    rng = np.random.default_rng(0)
    centers = np.arange(len(groups), dtype=float)
    status_offset = {False: -0.16, True: 0.16}
    jitter_width = 0.13

    tick_labels = []
    for group_idx, group in enumerate(groups):
        group_data = data.loc[data["group"].astype(str) == group]
        n_episodes = int(len(group_data))
        n_flies = int(group_data["unit_id"].nunique())
        tick_labels.append(f"{group}\n({n_flies} flies, {n_episodes} episodes)")

        for status in STATUS_ORDER:
            values = group_data.loc[
                group_data["contains_wall_contact"] == status,
                "trajectory_length_mm",
            ].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            xpos = centers[group_idx] + status_offset[status]
            jitter = rng.uniform(-jitter_width, jitter_width, size=values.size)
            ax.scatter(
                np.full(values.size, xpos) + jitter,
                values,
                s=7,
                color=STATUS_COLORS[status],
                edgecolors="none",
                alpha=0.22,
                rasterized=True,
                zorder=2,
            )
            median = float(np.median(values))
            ax.plot(
                [xpos - 0.12, xpos + 0.12],
                [median, median],
                color=NEUTRAL_DARK,
                linewidth=1.6,
                zorder=4,
            )

    ax.set_xticks(centers)
    ax.set_xticklabels(tick_labels, rotation=24, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(bottom=0)
    if ymax is not None:
        ax.set_ylim(top=float(ymax))
    if title:
        ax.set_title(title)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=STATUS_COLORS[status],
            markeredgecolor="none",
            markersize=6,
            label=STATUS_LABELS[status],
        )
        for status in STATUS_ORDER
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=NEUTRAL_DARK,
            linewidth=1.6,
            label="Episode median",
        )
    )
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)

    if customizer.customized:
        customizer.adjust_padding_proportionally(
            wrap_legend_labels=False,
            wrap_y_axis_labels=True,
        )
    else:
        fig.tight_layout()
    return fig


def save_figure(fig, out: str, *, image_format: str) -> str:
    image_format = str(image_format).lstrip(".").lower()
    path = Path(out)
    if path.suffix.lower() != f".{image_format}":
        path = path.with_suffix(f".{image_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", format=image_format)
    print(f"[wall_contact_episode_length_swarm] wrote {path}")
    return str(path)
