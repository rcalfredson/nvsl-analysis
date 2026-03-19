#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.sli_tools import select_fractional_groups


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Inspect one cumulative-reward tick from a cum_reward_sli debug TSV, "
            "including per-fly raw rows and reconstructed top/bottom SLI groups."
        )
    )
    p.add_argument(
        "--debug-tsv",
        required=True,
        help="Path to the cumulative reward SLI debug TSV.",
    )
    p.add_argument(
        "--reward-tick",
        required=True,
        type=float,
        help="Reward tick to inspect, e.g. 10.",
    )
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
        "--include-other",
        action="store_true",
        help="Include flies outside the selected top/bottom groups in the detailed table.",
    )
    p.add_argument(
        "--only-finite-sli",
        action="store_true",
        help="Restrict the detailed table to flies with finite SLI at the chosen tick.",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Optional CSV path for the detailed per-fly rows shown for the chosen tick.",
    )
    return p.parse_args()


def assign_groups(df: pd.DataFrame, *, top_fraction: float, bottom_fraction: float) -> pd.DataFrame:
    per_fly = (
        df.groupby("video_index", as_index=False)
        .agg({"video_id": "first", "scalar_sli": "first"})
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
    return per_fly


def summarize_pool(per_fly: pd.DataFrame) -> None:
    rankable = per_fly["scalar_sli"].notna()
    print(f"Total fly units: {len(per_fly)}")
    print(f"Rankable flies: {int(rankable.sum())}")
    print("Group counts:")
    print(per_fly["group"].value_counts().to_string())
    print()
    print("Scalar SLI summary by group:")
    print(
        per_fly.groupby("group")["scalar_sli"]
        .agg(["count", "min", "median", "max"])
        .to_string()
    )


def summarize_tick(sub: pd.DataFrame) -> None:
    print()
    print(f"Reward tick: {sub['reward_tick'].iloc[0]:g}")
    print("Overall support:")
    print(
        pd.Series(
            {
                "n_rows": len(sub),
                "n_reached": int(sub["reached_reward_tick"].sum()),
                "n_finite_exp_pi": int(sub["exp_reward_pi"].notna().sum()),
                "n_finite_yoked_pi": int(sub["yoked_reward_pi"].notna().sum()),
                "n_finite_sli": int(sub["sli_finite"].sum()),
            }
        ).to_string()
    )
    print()
    print("Per-group support and reconstructed means:")
    summary = (
        sub.groupby("group")
        .apply(
            lambda g: pd.Series(
                {
                    "n_rows": len(g),
                    "n_reached": int(g["reached_reward_tick"].sum()),
                    "n_finite_sli": int(g["sli_finite"].sum()),
                    "mean_sli": g.loc[g["sli_finite"] == 1, "sli"].mean(),
                    "mean_exp_pi": g.loc[g["sli_finite"] == 1, "exp_reward_pi"].mean(),
                    "mean_yoked_pi": g.loc[g["sli_finite"] == 1, "yoked_reward_pi"].mean(),
                    "n_missing_exp_pi": int(g["exp_reward_pi"].isna().sum()),
                    "n_missing_yoked_pi": int(g["yoked_reward_pi"].isna().sum()),
                    "n_missing_both_pi": int(
                        (g["exp_reward_pi"].isna() & g["yoked_reward_pi"].isna()).sum()
                    ),
                }
            )
        )
        .sort_index()
    )
    print(summary.to_string())


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.debug_tsv, sep="\t")
    per_fly = assign_groups(
        df,
        top_fraction=float(args.top_sli_fraction),
        bottom_fraction=float(args.bottom_sli_fraction),
    )
    summarize_pool(per_fly)

    merged = df.merge(per_fly[["video_index", "group"]], on="video_index", how="left")
    tick = float(args.reward_tick)
    sub = merged[merged["reward_tick"] == tick].copy()
    if sub.empty:
        raise SystemExit(f"No rows found for reward tick {tick:g}.")

    summarize_tick(sub)

    allowed_groups = {"bottom", "top"}
    if args.include_other:
        allowed_groups.add("other")
    detail = sub[sub["group"].isin(allowed_groups)].copy()
    if args.only_finite_sli:
        detail = detail[detail["sli_finite"] == 1].copy()

    cols = [
        "video_index",
        "video_id",
        "group",
        "scalar_sli",
        "reached_reward_tick",
        "exp_reward_pi",
        "yoked_reward_pi",
        "sli",
        "sli_finite",
        "exp_calc_entries_exp_fly",
        "ctrl_calc_entries_exp_fly",
        "exp_calc_entries_yoked_fly",
        "ctrl_calc_entries_yoked_fly",
        "total_actual_rewards",
        "cutoff_frame",
    ]
    detail = detail[cols].sort_values(
        ["group", "sli_finite", "scalar_sli"],
        ascending=[True, False, False],
    )

    print()
    print("Detailed per-fly rows:")
    print(detail.to_string(index=False, max_rows=200))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(out_path, index=False)
        print()
        print(f"Wrote detailed rows: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
