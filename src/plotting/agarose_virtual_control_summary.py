from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import ttest_ind, ttest_rel

from scripts.stats_agarose_virtual_control import select_paired_values
from src.plotting.palettes import ACCENT_BLUE, ACCENT_ORANGE, NEUTRAL_DARK
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.stats_bars import draw_sig_bracket
from src.utils.util import meanConfInt


@dataclass(frozen=True)
class ChamberPlacementValues:
    label: str
    physical: np.ndarray
    virtual: np.ndarray

    @property
    def delta(self) -> np.ndarray:
        return self.physical - self.virtual


def load_chamber_placement_values(
    bundle_path,
    *,
    label,
    mode="exp",
    training_index_1based=2,
    bucket_index=-1,
    bucket_start_index_1based=None,
    bucket_end_index_1based=None,
) -> ChamberPlacementValues:
    training_idx = int(training_index_1based) - 1

    def _idx(value):
        if value is None or value < 0:
            return value
        if value == 0:
            raise ValueError("sync-bucket indices are 1-based; zero is invalid")
        return int(value) - 1

    with np.load(bundle_path, allow_pickle=True) as bundle:
        physical, virtual, paired, *_ = select_paired_values(
            bundle,
            mode=mode,
            training_idx=training_idx,
            bucket_idx=_idx(bucket_index),
            bucket_start_idx=_idx(bucket_start_index_1based),
            bucket_end_idx=_idx(bucket_end_index_1based),
        )
    return ChamberPlacementValues(
        label=str(label),
        physical=np.asarray(physical[paired], dtype=float),
        virtual=np.asarray(virtual[paired], dtype=float),
    )


def _p_text(p_value):
    if not np.isfinite(p_value):
        return "p=n/a"
    if p_value < 1e-3:
        return f"p={p_value:.1e}"
    return f"p={p_value:.3f}"


def _bar_ci(ax, xpos, values, color, *, width=0.34):
    mean, lo, hi, _n = meanConfInt(values)
    ax.bar(
        xpos,
        mean,
        width=width,
        color=color,
        edgecolor=NEUTRAL_DARK,
        linewidth=0.9,
        alpha=0.78,
        zorder=1,
    )
    ax.errorbar(
        [xpos],
        [mean],
        yerr=np.asarray([[mean - lo], [hi - mean]]),
        fmt="none",
        ecolor=NEUTRAL_DARK,
        capsize=3,
        capthick=1,
        linewidth=1,
        zorder=5,
    )
    return float(mean), float(hi)


def plot_agarose_virtual_control_summary(
    agarose: ChamberPlacementValues,
    flat: ChamberPlacementValues,
    *,
    out_path,
    title=None,
    image_format="png",
    dpi=220,
):
    """Create slide-ready ratio and physical-minus-virtual bar/swarm panels."""
    customizer = PlotCustomizer()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    rng = np.random.default_rng(0)
    chambers = (agarose, flat)
    colors = (ACCENT_ORANGE, ACCENT_BLUE)

    # Panel A: paired physical and virtual ratios within each chamber.
    ax = axes[0]
    centers = np.arange(2, dtype=float)
    offsets = (-0.20, 0.20)
    panel_top = 0.0
    for ci, chamber in enumerate(chambers):
        jitter = rng.uniform(-0.045, 0.045, size=chamber.physical.size)
        xs_physical = centers[ci] + offsets[0] + jitter
        xs_virtual = centers[ci] + offsets[1] + jitter
        for x1, x2, y1, y2 in zip(
            xs_physical, xs_virtual, chamber.physical, chamber.virtual
        ):
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="0.55",
                linewidth=0.45,
                alpha=0.22,
                zorder=2,
            )
        _, hi_physical = _bar_ci(
            ax, centers[ci] + offsets[0], chamber.physical, colors[0]
        )
        _, hi_virtual = _bar_ci(
            ax, centers[ci] + offsets[1], chamber.virtual, colors[1]
        )
        ax.scatter(
            xs_physical,
            chamber.physical,
            s=13,
            facecolor="white",
            edgecolor=colors[0],
            linewidth=0.65,
            alpha=0.74,
            zorder=4,
        )
        ax.scatter(
            xs_virtual,
            chamber.virtual,
            s=13,
            facecolor="white",
            edgecolor=colors[1],
            linewidth=0.65,
            alpha=0.74,
            zorder=4,
        )
        paired_test = ttest_rel(
            chamber.physical, chamber.virtual, nan_policy="omit"
        )
        bracket_y = max(
            hi_physical,
            hi_virtual,
            float(np.max(chamber.physical)),
            float(np.max(chamber.virtual)),
        ) + 0.025
        draw_sig_bracket(
            ax,
            x1=centers[ci] + offsets[0],
            x2=centers[ci] + offsets[1],
            y=bracket_y,
            h=0.012,
            text=_p_text(float(paired_test.pvalue)),
            fontsize=8.5,
        )
        panel_top = max(panel_top, bracket_y + 0.065)

    ax.set_xticks(centers)
    ax.set_xticklabels(
        [f"{x.label}\n(n={x.physical.size})" for x in chambers]
    )
    ax.set_ylabel("Dual-circle avoidance ratio")
    ax.set_ylim(0, max(0.55, panel_top))
    ax.set_title("Placement comparison")
    ax.legend(
        handles=[
            Line2D([0], [0], color=colors[0], lw=7, label="Physical positions"),
            Line2D([0], [0], color=colors[1], lw=7, label="45° virtual positions"),
        ],
        frameon=False,
        loc="upper right",
        fontsize=8.5,
    )

    # Panel B: per-video paired deltas and the chamber interaction test.
    ax = axes[1]
    delta_samples = (agarose.delta, flat.delta)
    ymax = 0.0
    for ci, (chamber, values) in enumerate(zip(chambers, delta_samples)):
        _mean, hi = _bar_ci(ax, centers[ci], values, colors[ci], width=0.48)
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        ax.scatter(
            centers[ci] + jitter,
            values,
            s=15,
            facecolor="white",
            edgecolor=colors[ci],
            linewidth=0.7,
            alpha=0.78,
            zorder=4,
        )
        ymax = max(ymax, hi, float(np.max(values)))
    interaction = ttest_ind(
        agarose.delta, flat.delta, equal_var=False, nan_policy="omit"
    )
    diff_in_diff = float(np.mean(agarose.delta) - np.mean(flat.delta))
    bracket_y = ymax + 0.035
    draw_sig_bracket(
        ax,
        x1=centers[0],
        x2=centers[1],
        y=bracket_y,
        h=0.012,
        text=f"ΔΔ={100 * diff_in_diff:.1f} pp; {_p_text(float(interaction.pvalue))}",
        fontsize=8.5,
    )
    ymin = min(-0.12, *(float(np.min(v)) for v in delta_samples))
    ax.set_ylim(ymin - 0.02, bracket_y + 0.075)
    ax.axhline(0, color="0.45", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [f"{x.label}\n(n={x.physical.size})" for x in chambers]
    )
    ax.set_ylabel("Physical − virtual avoidance ratio")
    ax.set_title("Orientation effect by chamber")

    for ax in axes:
        ax.grid(axis="y", alpha=0.18, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=customizer.font_size + 1)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=image_format)
    plt.close(fig)
    return {
        "interaction_difference": diff_in_diff,
        "interaction_p_value": float(interaction.pvalue),
        "out_path": str(out_path),
    }
