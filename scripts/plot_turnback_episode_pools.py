#!/usr/bin/env python3
"""Plot turnbacks in full crossed- and within-radius trajectory pools.

Inputs are the collated episode-audit CSVs produced by
``scripts.collate_episode_audit``. The two default figures are the audit panels
used to explain why a trajectory-level containment metric can diverge from an
episode-level turnback ratio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.palettes import group_metric_edge_color, group_metric_fill_color
from src.utils.util import meanConfInt


DEFAULT_PER_FLY = "tmp/episode_audit/collated/pfnd_vs_ar_episode_audit_per_fly.csv"
DEFAULT_EPISODES = "tmp/episode_audit/collated/pfnd_vs_ar_episode_audit_episodes.csv"
DEFAULT_CROSSED_OUT = (
    "tmp/episode_audit/collated/"
    "pfnd_vs_ar_turnback_success_failure_per_crossed_radius_trajectory.png"
)
DEFAULT_WITHIN_OUT = (
    "tmp/episode_audit/collated/"
    "pfnd_vs_ar_episode_matched_turnbacks_per_within_radius_trajectory.png"
)
DEFAULT_CROSSED_PER_FLY_OUT = (
    "tmp/episode_audit/collated/"
    "pfnd_vs_ar_turnback_episode_pool_crossed_per_fly.csv"
)
DEFAULT_WITHIN_PER_FLY_OUT = (
    "tmp/episode_audit/collated/"
    "pfnd_vs_ar_turnback_episode_pool_within_per_fly.csv"
)
DEFAULT_SUMMARY_OUT = (
    "tmp/episode_audit/collated/"
    "pfnd_vs_ar_turnback_episode_pool_group_summary.csv"
)

REGION_MATCHES = [
    {
        "region_label": "3/5 -> 6 mm",
        "turnback_pair": "3/5",
        "return_radius": "6",
        "inner_radius_mm": 3.0,
        "outer_radius_mm": 5.0,
        "matched_fraction_radius_mm": 6.0,
    },
    {
        "region_label": "8/10 -> 11 mm",
        "turnback_pair": "8/10",
        "return_radius": "11",
        "inner_radius_mm": 8.0,
        "outer_radius_mm": 10.0,
        "matched_fraction_radius_mm": 11.0,
    },
    {
        "region_label": "13/15 -> 16 mm",
        "turnback_pair": "13/15",
        "return_radius": "16",
        "inner_radius_mm": 13.0,
        "outer_radius_mm": 15.0,
        "matched_fraction_radius_mm": 16.0,
    },
]


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def _ci(values: pd.Series) -> tuple[float, float, float, int]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan, 0
    mean, lo, hi, n = meanConfInt(arr, conf=0.95)
    return float(mean), float(lo), float(hi), int(n)


def _valid_unit_keys(
    per_fly: pd.DataFrame,
    *,
    metric: str,
    radius_label: str,
    groups: list[str],
) -> set[tuple[str, str]]:
    sub = per_fly[
        per_fly["metric"].eq(metric)
        & per_fly["radius_label"].astype(str).eq(str(radius_label))
    ].copy()
    if groups:
        sub = sub[sub["group"].isin(groups)].copy()
    sub["passes_min_episodes"] = _as_bool(sub["passes_min_episodes"])
    sub = sub[sub["passes_min_episodes"]]
    return set(zip(sub["group"].astype(str), sub["unit_id"].astype(str)))


def build_episode_matched_pool_per_fly(
    episodes: pd.DataFrame,
    per_fly: pd.DataFrame,
    *,
    groups: list[str],
    require_min_episodes: bool,
    pool: str,
    return_prob_success: bool,
) -> pd.DataFrame:
    """Count turnbacks inside full reward-exit trajectories selected by max radius."""

    if pool not in {"crossed_matched_radius", "within_matched_radius"}:
        raise ValueError(f"Unknown pool: {pool!r}")

    required_full_cols = [
        "full_trajectory_start",
        "full_trajectory_stop",
        "full_trajectory_max_radius_mm",
    ]
    missing = [col for col in required_full_cols if col not in episodes.columns]
    if missing:
        raise ValueError(
            "Episode audit CSV is missing full-trajectory columns. Re-export "
            "return-probability debug CSVs and rerun scripts.collate_episode_audit. "
            f"Missing columns: {missing}"
        )

    rows = []
    episodes = episodes.copy()
    episodes["included_in_metric"] = _as_bool(episodes["included_in_metric"])
    episodes["success"] = _as_bool(episodes["success"])
    episodes["group"] = episodes["group"].astype(str)
    episodes["unit_id"] = episodes["unit_id"].astype(str)
    episodes["radius_label"] = episodes["radius_label"].astype(str)
    if groups:
        episodes = episodes[episodes["group"].isin(groups)].copy()

    for region_order, match in enumerate(REGION_MATCHES):
        ret = episodes[
            episodes["metric"].eq("return_prob")
            & episodes["radius_label"].eq(match["return_radius"])
            & episodes["included_in_metric"]
        ].copy()
        tb = episodes[
            episodes["metric"].eq("turnback")
            & episodes["radius_label"].eq(match["turnback_pair"])
            & episodes["included_in_metric"]
        ].copy()
        if require_min_episodes:
            valid_keys = _valid_unit_keys(
                per_fly,
                metric="return_prob",
                radius_label=match["return_radius"],
                groups=groups,
            ) & _valid_unit_keys(
                per_fly,
                metric="turnback",
                radius_label=match["turnback_pair"],
                groups=groups,
            )
            ret = ret[
                [
                    (group, unit) in valid_keys
                    for group, unit in zip(ret["group"], ret["unit_id"])
                ]
            ].copy()
            tb = tb[
                [
                    (group, unit) in valid_keys
                    for group, unit in zip(tb["group"], tb["unit_id"])
                ]
            ].copy()

        ret["full_trajectory_start"] = pd.to_numeric(
            ret["full_trajectory_start"], errors="coerce"
        )
        ret["full_trajectory_stop"] = pd.to_numeric(
            ret["full_trajectory_stop"], errors="coerce"
        )
        ret["full_trajectory_max_radius_mm"] = pd.to_numeric(
            ret["full_trajectory_max_radius_mm"], errors="coerce"
        )
        tb["event_frame"] = pd.to_numeric(tb["event_frame"], errors="coerce")
        ret = ret[
            np.isfinite(ret["full_trajectory_start"])
            & np.isfinite(ret["full_trajectory_stop"])
            & np.isfinite(ret["full_trajectory_max_radius_mm"])
        ].copy()
        ret = ret.drop_duplicates(
            [
                "group",
                "unit_id",
                "full_trajectory_start",
                "full_trajectory_stop",
            ]
        )
        tb = tb[np.isfinite(tb["event_frame"])].copy()

        for (group, unit_id), ret_unit in ret.groupby(["group", "unit_id"], sort=False):
            if return_prob_success:
                pool_trajectories = ret_unit[
                    ret_unit["full_trajectory_max_radius_mm"]
                    < match["matched_fraction_radius_mm"]
                ].copy()
            else:
                pool_trajectories = ret_unit[
                    ret_unit["full_trajectory_max_radius_mm"]
                    >= match["matched_fraction_radius_mm"]
                ].copy()
            denom = len(pool_trajectories)
            if denom == 0:
                continue

            tb_unit = tb[tb["group"].eq(group) & tb["unit_id"].eq(unit_id)]
            event_frames = tb_unit["event_frame"].to_numpy(dtype=float)
            event_success = tb_unit["success"].to_numpy(dtype=bool)
            n_success = 0
            n_failed = 0
            for traj in pool_trajectories.itertuples(index=False):
                in_traj = (event_frames >= float(traj.full_trajectory_start)) & (
                    event_frames < float(traj.full_trajectory_stop)
                )
                n_success += int((event_success & in_traj).sum())
                n_failed += int(((~event_success) & in_traj).sum())

            rows.append(
                {
                    "fly_id": unit_id,
                    "group": group,
                    "region_order": region_order,
                    "region_label": match["region_label"],
                    "pool": pool,
                    "inner_radius_mm": match["inner_radius_mm"],
                    "outer_radius_mm": match["outer_radius_mm"],
                    "matched_fraction_radius_mm": match[
                        "matched_fraction_radius_mm"
                    ],
                    "n_pool_trajectories": denom,
                    "n_successful_turnback_episodes": n_success,
                    "n_failed_turnback_episodes": n_failed,
                    "successful_turnbacks_per_pool_trajectory": n_success / denom,
                    "failed_turnbacks_per_pool_trajectory": n_failed / denom,
                    "total_turnbacks_per_pool_trajectory": (
                        n_success + n_failed
                    )
                    / denom,
                }
            )

    return pd.DataFrame(rows)


def summarize_pool(per_fly_pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "group",
        "region_order",
        "region_label",
        "pool",
        "inner_radius_mm",
        "outer_radius_mm",
        "matched_fraction_radius_mm",
    ]
    for keys, sub in per_fly_pool.groupby(group_cols, dropna=False, sort=True):
        success_mean, success_lo, success_hi, n = _ci(
            sub["successful_turnbacks_per_pool_trajectory"]
        )
        failed_mean, failed_lo, failed_hi, _ = _ci(
            sub["failed_turnbacks_per_pool_trajectory"]
        )
        total_mean, total_lo, total_hi, _ = _ci(
            sub["total_turnbacks_per_pool_trajectory"]
        )
        row = dict(zip(group_cols, keys))
        row.update(
            n_flies=n,
            mean_successful_turnbacks_per_pool_trajectory=success_mean,
            successful_ci_lo=success_lo,
            successful_ci_hi=success_hi,
            mean_failed_turnbacks_per_pool_trajectory=failed_mean,
            failed_ci_lo=failed_lo,
            failed_ci_hi=failed_hi,
            mean_total_turnbacks_per_pool_trajectory=total_mean,
            total_ci_lo=total_lo,
            total_ci_hi=total_hi,
            mean_pool_trajectories_per_fly=float(
                pd.to_numeric(sub["n_pool_trajectories"], errors="coerce").mean()
            ),
            mean_successful_turnback_episodes_per_fly=float(
                pd.to_numeric(
                    sub["n_successful_turnback_episodes"], errors="coerce"
                ).mean()
            ),
            mean_failed_turnback_episodes_per_fly=float(
                pd.to_numeric(sub["n_failed_turnback_episodes"], errors="coerce").mean()
            ),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pool_bars(
    summary: pd.DataFrame,
    *,
    pool: str,
    groups: list[str],
    title: str,
    ylabel: str,
    out: Path,
) -> None:
    sub_summary = summary[summary["pool"].eq(pool)].copy()
    if sub_summary.empty:
        raise ValueError(f"No rows available to plot for pool={pool!r}.")

    region_rows = (
        sub_summary[["region_order", "region_label"]]
        .drop_duplicates()
        .sort_values("region_order")
    )
    region_labels = region_rows["region_label"].astype(str).tolist()
    x = np.arange(len(region_labels), dtype=float)
    group_order = groups or list(dict.fromkeys(sub_summary["group"].astype(str)))
    n_groups = max(1, len(group_order))
    width = min(0.26, 0.75 / n_groups)
    offsets = (np.arange(n_groups, dtype=float) - (n_groups - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for gi, group in enumerate(group_order):
        sub = sub_summary[sub_summary["group"].eq(group)].set_index("region_label")
        success = np.asarray(
            [
                sub.loc[label, "mean_successful_turnbacks_per_pool_trajectory"]
                if label in sub.index
                else np.nan
                for label in region_labels
            ],
            dtype=float,
        )
        failure = np.asarray(
            [
                sub.loc[label, "mean_failed_turnbacks_per_pool_trajectory"]
                if label in sub.index
                else np.nan
                for label in region_labels
            ],
            dtype=float,
        )
        total = np.asarray(
            [
                sub.loc[label, "mean_total_turnbacks_per_pool_trajectory"]
                if label in sub.index
                else np.nan
                for label in region_labels
            ],
            dtype=float,
        )
        lo = np.asarray(
            [
                sub.loc[label, "total_ci_lo"] if label in sub.index else np.nan
                for label in region_labels
            ],
            dtype=float,
        )
        hi = np.asarray(
            [
                sub.loc[label, "total_ci_hi"] if label in sub.index else np.nan
                for label in region_labels
            ],
            dtype=float,
        )
        yerr = np.vstack([total - lo, hi - total])
        yerr = np.where(np.isfinite(yerr), np.maximum(yerr, 0.0), 0.0)
        fill = group_metric_fill_color(gi, "turnback_ratio")
        edge = group_metric_edge_color(gi, "turnback_ratio")
        xpos = x + offsets[gi]
        ax.bar(
            xpos,
            success,
            width=width,
            color=fill,
            edgecolor=edge,
            linewidth=0.8,
            label=f"{group}: successful",
        )
        ax.bar(
            xpos,
            failure,
            width=width,
            bottom=success,
            color=fill,
            edgecolor=edge,
            linewidth=0.8,
            alpha=0.35,
            hatch="//",
            label=f"{group}: failed",
        )
        ax.errorbar(
            xpos,
            total,
            yerr=yerr,
            fmt="none",
            ecolor=edge,
            elinewidth=0.9,
            capsize=3,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, rotation=18, ha="right")
    ax.set_xlabel("Turnback pair -> fraction radius")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=True,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Plot successful/failed turnback episodes per trajectory for crossed "
            "and within-radius reward-centered trajectory pools."
        )
    )
    p.add_argument("--per-fly", default=DEFAULT_PER_FLY, help="Collated per-fly CSV.")
    p.add_argument(
        "--episodes",
        default=DEFAULT_EPISODES,
        help="Collated episode-level CSV.",
    )
    p.add_argument(
        "--groups",
        default="AR Ctrl>Kir FLC,PFNd>Kir FLC",
        help="Comma-separated groups to plot. Use 'all' to include every group.",
    )
    p.add_argument(
        "--no-min-episode-filter",
        action="store_true",
        help="Keep fly-region rows even if one source metric fails its 5-episode minimum.",
    )
    p.add_argument(
        "--crossed-out",
        default=DEFAULT_CROSSED_OUT,
        help="Output plot for trajectories crossing the matched radius.",
    )
    p.add_argument(
        "--within-out",
        default=DEFAULT_WITHIN_OUT,
        help="Output plot for trajectories staying within the matched radius.",
    )
    p.add_argument(
        "--crossed-per-fly-out",
        default=DEFAULT_CROSSED_PER_FLY_OUT,
        help="Output crossed-radius per-fly CSV.",
    )
    p.add_argument(
        "--within-per-fly-out",
        default=DEFAULT_WITHIN_PER_FLY_OUT,
        help="Output within-radius per-fly CSV.",
    )
    p.add_argument(
        "--summary-out",
        default=DEFAULT_SUMMARY_OUT,
        help="Output combined group-summary CSV.",
    )
    args = p.parse_args(argv)

    groups = [] if str(args.groups).strip().lower() == "all" else _parse_csv_list(args.groups)
    require_min = not bool(args.no_min_episode_filter)
    per_fly = pd.read_csv(args.per_fly)
    episodes = pd.read_csv(args.episodes, low_memory=False)

    crossed = build_episode_matched_pool_per_fly(
        episodes,
        per_fly,
        groups=groups,
        require_min_episodes=require_min,
        pool="crossed_matched_radius",
        return_prob_success=False,
    )
    within = build_episode_matched_pool_per_fly(
        episodes,
        per_fly,
        groups=groups,
        require_min_episodes=require_min,
        pool="within_matched_radius",
        return_prob_success=True,
    )
    summary = summarize_pool(pd.concat([crossed, within], ignore_index=True, sort=False))

    crossed_per_fly_out = Path(args.crossed_per_fly_out)
    within_per_fly_out = Path(args.within_per_fly_out)
    summary_out = Path(args.summary_out)
    crossed_per_fly_out.parent.mkdir(parents=True, exist_ok=True)
    within_per_fly_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    crossed.to_csv(crossed_per_fly_out, index=False)
    within.to_csv(within_per_fly_out, index=False)
    summary.to_csv(summary_out, index=False)

    plot_pool_bars(
        summary,
        pool="crossed_matched_radius",
        groups=groups,
        title="Turnbacks in trajectories crossing matched radius",
        ylabel="Episodes per crossed trajectory",
        out=Path(args.crossed_out),
    )
    plot_pool_bars(
        summary,
        pool="within_matched_radius",
        groups=groups,
        title="Turnbacks in trajectories staying within matched radius",
        ylabel="Episodes per within-radius trajectory",
        out=Path(args.within_out),
    )

    print(f"Wrote {crossed_per_fly_out} rows={len(crossed)}")
    print(f"Wrote {within_per_fly_out} rows={len(within)}")
    print(f"Wrote {summary_out} rows={len(summary)}")
    print(f"Wrote {args.crossed_out}")
    print(f"Wrote {args.within_out}")
    print()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
