#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.sli_tools import select_fractional_groups


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot low-level count and PI distributions for one reward tick from "
            "a cumulative-reward SLI debug TSV."
        )
    )
    p.add_argument("--debug-tsv", required=True, help="Path to the debug TSV.")
    p.add_argument("--reward-tick", required=True, type=float, help="Reward tick to inspect.")
    p.add_argument("--out", required=True, help="Output image path.")
    p.add_argument(
        "--top-sli-fraction",
        type=float,
        default=0.2,
        help="Top-group fraction among rankable flies (default: %(default)s).",
    )
    p.add_argument(
        "--bottom-sli-fraction",
        type=float,
        default=0.8,
        help="Bottom-group fraction among rankable flies (default: %(default)s).",
    )
    p.add_argument(
        "--only-finite-sli",
        action="store_true",
        help="Restrict plotted rows to flies with finite SLI at the chosen tick.",
    )
    return p.parse_args()


def _assign_groups(df: pd.DataFrame, *, top_fraction: float, bottom_fraction: float) -> pd.DataFrame:
    per_fly = (
        df.groupby("video_index", as_index=False)
        .agg({"scalar_sli": "first"})
        .sort_values("video_index")
        .reset_index(drop=True)
    )
    bottom, top = select_fractional_groups(
        per_fly["scalar_sli"],
        top_fraction=top_fraction,
        bottom_fraction=bottom_fraction,
    )
    bottom = set(bottom or [])
    top = set(top or [])
    per_fly["group"] = "other"
    per_fly.loc[per_fly.index.isin(bottom), "group"] = "bottom"
    per_fly.loc[per_fly.index.isin(top), "group"] = "top"
    return per_fly[["video_index", "group"]]


def _group_arrays(df: pd.DataFrame, col: str) -> list[np.ndarray]:
    out = []
    for grp in ("bottom", "top"):
        vals = np.asarray(df.loc[df["group"] == grp, col], dtype=float)
        vals = vals[np.isfinite(vals)]
        out.append(vals)
    return out


def _box_strip(ax, arrays: list[np.ndarray], *, title: str, ylabel: str, colors: list[str]) -> None:
    positions = [1, 2]
    bp = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="0.35"),
        capprops=dict(color="0.35"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, arrays, colors):
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(
            np.full(vals.size, pos) + jitter,
            vals,
            s=26,
            alpha=0.8,
            color=color,
            edgecolors="none",
        )
    ax.set_xticks(positions, ["Bottom", "Top"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.debug_tsv, sep="\t")
    groups = _assign_groups(
        df,
        top_fraction=float(args.top_sli_fraction),
        bottom_fraction=float(args.bottom_sli_fraction),
    )
    df = df.merge(groups, on="video_index", how="left")
    tick = float(args.reward_tick)
    sub = df[(df["reward_tick"] == tick) & (df["group"].isin(["bottom", "top"]))].copy()
    if args.only_finite_sli:
        sub = sub[sub["sli_finite"] == 1].copy()
    if sub.empty:
        raise SystemExit(f"No rows available for reward tick {tick:g}.")

    colors = ["#2b6cb0", "#dd6b20"]
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 8))

    exp_count = _group_arrays(sub, "exp_calc_entries_exp_fly")
    ctrl_count = _group_arrays(sub, "ctrl_calc_entries_exp_fly")
    yoked_exp_count = _group_arrays(sub, "exp_calc_entries_yoked_fly")
    yoked_ctrl_count = _group_arrays(sub, "ctrl_calc_entries_yoked_fly")
    exp_pi = _group_arrays(sub, "exp_reward_pi")
    yoked_pi = _group_arrays(sub, "yoked_reward_pi")

    _box_strip(
        axs[0, 0],
        [exp_count[0], exp_count[1]],
        title="Experimental Fly Reward-Circle Entries",
        ylabel="Count",
        colors=colors,
    )
    _box_strip(
        axs[0, 1],
        [ctrl_count[0], ctrl_count[1]],
        title="Experimental Fly Control-Circle Entries",
        ylabel="Count",
        colors=colors,
    )
    _box_strip(
        axs[1, 0],
        [exp_pi[0], exp_pi[1]],
        title="Experimental Reward PI",
        ylabel="PI",
        colors=colors,
    )
    _box_strip(
        axs[1, 1],
        [yoked_pi[0], yoked_pi[1]],
        title="Yoked Reward PI",
        ylabel="PI",
        colors=colors,
    )

    mode = "finite-SLI only" if args.only_finite_sli else "all selected rows"
    fig.suptitle(
        f"Cumulative-reward SLI debug distributions at reward {tick:g}\n"
        f"Top {int(round(args.top_sli_fraction * 100))}% vs bottom "
        f"{int(round(args.bottom_sli_fraction * 100))}% ({mode})"
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(
        "n(bottom, top): "
        f"{int((sub['group'] == 'bottom').sum())}, "
        f"{int((sub['group'] == 'top').sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
