#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUTS = {
    "turnback_pfnd": (
        "turnback",
        "PFNd>Kir",
        "tmp/episode_audit/"
        "turnbackPairs_minEpSb5Filt_wall_intact_pfnKir_flatLgc_T2_"
        "p3-5_8-10_13-15_debug.csv",
        "exports/episode_audit/"
        "turnbackPairs_minEpSb5Filt_wall_intact_pfnKir_flatLgc_T2_"
        "p3-5_8-10_13-15_debug.npz",
    ),
    "turnback_ar": (
        "turnback",
        "Antennae removed",
        "tmp/episode_audit/"
        "turnbackPairs_minEpSb5Filt_wall_ar_ctrlKir_flatLgc_T2_"
        "p3-5_8-10_13-15_debug.csv",
        "exports/episode_audit/"
        "turnbackPairs_minEpSb5Filt_wall_ar_ctrlKir_flatLgc_T2_"
        "p3-5_8-10_13-15_debug.npz",
    ),
    "return_pfnd": (
        "return_prob",
        "PFNd>Kir",
        "tmp/episode_audit/"
        "returnProb_minEpSb5Filt_wall_intact_pfnKir_flatLgc_T2_"
        "r6_11_16_debug.csv",
        "exports/episode_audit/"
        "returnProb_minEpSb5Filt_wall_intact_pfnKir_flatLgc_T2_"
        "r6_11_16_debug.npz",
    ),
    "return_ar": (
        "return_prob",
        "Antennae removed",
        "tmp/episode_audit/"
        "returnProb_minEpSb5Filt_wall_ar_ctrlKir_flatLgc_T2_"
        "r6_11_16_debug.csv",
        "exports/episode_audit/"
        "returnProb_minEpSb5Filt_wall_ar_ctrlKir_flatLgc_T2_"
        "r6_11_16_debug.npz",
    ),
}


def _as_bool(series: pd.Series, *, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(default, index=pd.RangeIndex(0), dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    out = text.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
            "": default,
            "<na>": default,
            "nan": default,
            "none": default,
        }
    )
    return out.fillna(default).astype(bool)


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _str_col(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=object)
    return df[col].fillna(default).astype(str)


def _unit_ids(df: pd.DataFrame) -> pd.Series:
    video = _str_col(df, "video_id", default="")
    fly = pd.to_numeric(df.get("fly_idx"), errors="coerce")
    out = []
    for vid, fly_idx in zip(video, fly):
        if "::f" in vid:
            out.append(vid)
            continue
        if pd.isna(fly_idx):
            out.append(f"{vid}::f")
        else:
            out.append(f"{vid}::f{int(fly_idx)}")
    return pd.Series(out, index=df.index, dtype=object)


def _bundle_filter_meta(bundle_path: Path, *, metric: str) -> tuple[set[str], int]:
    with np.load(bundle_path, allow_pickle=True) as bundle:
        video_ids = np.asarray(bundle["video_ids"], dtype=object).astype(str)
        if "exp_target_sync_bucket_filter_eligible" not in bundle:
            eligible_ids = set(video_ids.tolist())
        else:
            eligible = np.asarray(
                bundle["exp_target_sync_bucket_filter_eligible"], dtype=bool
            )
            eligible_ids = set(video_ids[eligible].tolist())
        if metric == "turnback":
            min_episodes = int(np.asarray(bundle.get("min_turnback_episodes", 5)))
        else:
            min_episodes = int(
                np.asarray(bundle.get("btw_rwd_sync_bucket_min_trajectories", 5))
            )
        return eligible_ids, max(0, min_episodes)


def _default_min_episodes(metric: str) -> int:
    if metric in {"turnback", "return_prob"}:
        return 5
    return 0


def _filter_meta(
    bundle_path: Path | None,
    *,
    metric: str,
    eligibility_filter: bool,
) -> tuple[set[str] | None, int]:
    if bundle_path is None:
        return None, _default_min_episodes(metric)
    eligible_ids, min_episodes = _bundle_filter_meta(bundle_path, metric=metric)
    return (eligible_ids if eligibility_filter else None), min_episodes


def _load_one(
    path: Path,
    *,
    metric: str,
    group: str,
    bundle_path: Path | None,
    include_ctrl: bool,
    eligibility_filter: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["metric"] = metric
    df["group"] = group
    df["source_csv"] = str(path)

    if metric == "turnback":
        inner = _num(df, "requested_inner_mm")
        outer = _num(df, "requested_outer_mm")
        df["radius_order"] = pd.to_numeric(df.get("pair_idx"), errors="coerce")
        df["radius_label"] = [
            f"{i:g}/{o:g}" if np.isfinite(i) and np.isfinite(o) else ""
            for i, o in zip(inner, outer)
        ]
        df["radius_outer_mm"] = outer
        df["success"] = _as_bool(df.get("turns_back"), default=False)
        alignment = _str_col(df, "home_vector_alignment_pass")
        has_alignment_filter = alignment.str.strip().ne("").any()
        if has_alignment_filter:
            df["success"] = _as_bool(alignment, default=False)
        if "walking_fraction_pass" in df.columns:
            df["included_in_metric"] = _as_bool(
                df["walking_fraction_pass"], default=True
            )
        else:
            df["included_in_metric"] = True
        df["outcome"] = np.where(df["success"], "turns_back", "exit_outer")
    elif metric == "return_prob":
        outer = _num(df, "requested_outer_mm")
        df["radius_order"] = pd.to_numeric(df.get("radius_idx"), errors="coerce")
        df["radius_label"] = [
            f"{o:g}" if np.isfinite(o) else "" for o in outer
        ]
        df["radius_outer_mm"] = outer
        df["success"] = _as_bool(df.get("returns"), default=False)
        if "included_in_metric" in df.columns:
            df["included_in_metric"] = _as_bool(
                df["included_in_metric"], default=True
            )
        else:
            df["included_in_metric"] = True
        df["outcome"] = np.where(df["success"], "returns", "exit_outer")
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    df["wall_overlap"] = _as_bool(df.get("wall_overlap"), default=False)
    df["episode_duration_frames"] = _num(df, "stop") - _num(df, "start")
    df["event_frame"] = _num(df, "event_frame")
    df["fly_idx"] = pd.to_numeric(df.get("fly_idx"), errors="coerce")
    df["video_id"] = _str_col(df, "video_id", default="")
    df["fly_role"] = _str_col(df, "fly_role", default="")
    df["unit_id"] = _unit_ids(df)
    if not include_ctrl:
        df = df[df["fly_role"].eq("exp")].copy()
    eligible, min_episodes = _filter_meta(
        bundle_path, metric=metric, eligibility_filter=eligibility_filter
    )
    df["min_episodes"] = int(min_episodes)
    if eligible is not None:
        df = df[df["unit_id"].isin(eligible)].copy()
    return df


def _summarize_per_fly(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "metric",
        "group",
        "unit_id",
        "video_id",
        "fly_idx",
        "fly_role",
        "radius_order",
        "radius_label",
        "radius_outer_mm",
    ]
    for keys, sub in episodes.groupby(group_cols, dropna=False, sort=True):
        included = sub["included_in_metric"].astype(bool)
        denom = int(included.sum())
        successes = int((sub["success"].astype(bool) & included).sum())
        wall_overlap = int((sub["wall_overlap"].astype(bool) & included).sum())
        excluded = int((~included).sum())
        min_episodes = int(pd.to_numeric(sub["min_episodes"], errors="coerce").max())
        passes_min = denom >= max(0, min_episodes)
        duration = sub.loc[included, "episode_duration_frames"]
        row = dict(zip(group_cols, keys))
        row.update(
            min_episodes=min_episodes,
            passes_min_episodes=bool(passes_min),
            total=denom,
            success=successes,
            failure=denom - successes,
            ratio=(np.nan if denom == 0 or not passes_min else successes / denom),
            wall_overlap=wall_overlap,
            excluded=excluded,
            duration_mean_frames=float(duration.mean())
            if len(duration) > 0
            else np.nan,
            duration_median_frames=float(duration.median())
            if len(duration) > 0
            else np.nan,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_group(per_fly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["metric", "group", "radius_order", "radius_label", "radius_outer_mm"]
    for keys, sub in per_fly.groupby(group_cols, dropna=False, sort=True):
        ratios = pd.to_numeric(sub["ratio"], errors="coerce")
        finite = ratios[np.isfinite(ratios)]
        total = pd.to_numeric(sub["total"], errors="coerce").fillna(0)
        success = pd.to_numeric(sub["success"], errors="coerce").fillna(0)
        failure = pd.to_numeric(sub["failure"], errors="coerce").fillna(0)
        pooled_total = int(total.sum())
        pooled_success = int(success.sum())
        row = dict(zip(group_cols, keys))
        row.update(
            n_flies=int(finite.size),
            mean_ratio=float(finite.mean()) if finite.size else np.nan,
            median_ratio=float(finite.median()) if finite.size else np.nan,
            pooled_success=pooled_success,
            pooled_total=pooled_total,
            pooled_ratio=(
                np.nan if pooled_total == 0 else pooled_success / pooled_total
            ),
            mean_success_per_fly=float(success.mean()) if len(success) else np.nan,
            median_success_per_fly=float(success.median()) if len(success) else np.nan,
            mean_failure_per_fly=float(failure.mean()) if len(failure) else np.nan,
            median_failure_per_fly=float(failure.median()) if len(failure) else np.nan,
            median_total_per_fly=float(total.median()) if len(total) else np.nan,
            mean_total_per_fly=float(total.mean()) if len(total) else np.nan,
            total_episodes=pooled_total,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def collate(
    inputs: list[tuple[str, str, Path, Path | None]],
    *,
    include_ctrl: bool = False,
    eligibility_filter: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for metric, group, path, bundle_path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        if eligibility_filter and bundle_path is not None and not bundle_path.exists():
            raise FileNotFoundError(bundle_path)
        frames.append(
            _load_one(
                path,
                metric=metric,
                group=group,
                bundle_path=bundle_path,
                include_ctrl=include_ctrl,
                eligibility_filter=eligibility_filter,
            )
        )
    episodes = pd.concat(frames, ignore_index=True, sort=False)
    per_fly = _summarize_per_fly(episodes)
    group_summary = _summarize_group(per_fly)
    return episodes, per_fly, group_summary


def _inputs_from_args(args) -> list[tuple[str, str, Path, Path | None]]:
    return [
        (
            DEFAULT_INPUTS["turnback_pfnd"][0],
            DEFAULT_INPUTS["turnback_pfnd"][1],
            Path(args.turnback_pfnd),
            Path(args.turnback_pfnd_bundle),
        ),
        (
            DEFAULT_INPUTS["turnback_ar"][0],
            DEFAULT_INPUTS["turnback_ar"][1],
            Path(args.turnback_ar),
            Path(args.turnback_ar_bundle),
        ),
        (
            DEFAULT_INPUTS["return_pfnd"][0],
            DEFAULT_INPUTS["return_pfnd"][1],
            Path(args.return_pfnd),
            Path(args.return_pfnd_bundle),
        ),
        (
            DEFAULT_INPUTS["return_ar"][0],
            DEFAULT_INPUTS["return_ar"][1],
            Path(args.return_ar),
            Path(args.return_ar_bundle),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Collate turnback and return-probability per-episode audit CSVs "
            "into episode, per-fly, and group summaries."
        )
    )
    p.add_argument(
        "--turnback-pfnd",
        default=DEFAULT_INPUTS["turnback_pfnd"][2],
        help="PFNd>Kir turnback debug CSV.",
    )
    p.add_argument(
        "--turnback-pfnd-bundle",
        default=DEFAULT_INPUTS["turnback_pfnd"][3],
        help="PFNd>Kir turnback bundle used for target-eligibility filtering.",
    )
    p.add_argument(
        "--turnback-ar",
        default=DEFAULT_INPUTS["turnback_ar"][2],
        help="AR Ctrl>Kir turnback debug CSV.",
    )
    p.add_argument(
        "--turnback-ar-bundle",
        default=DEFAULT_INPUTS["turnback_ar"][3],
        help="AR Ctrl>Kir turnback bundle used for target-eligibility filtering.",
    )
    p.add_argument(
        "--return-pfnd",
        default=DEFAULT_INPUTS["return_pfnd"][2],
        help="PFNd>Kir return-probability debug CSV.",
    )
    p.add_argument(
        "--return-pfnd-bundle",
        default=DEFAULT_INPUTS["return_pfnd"][3],
        help="PFNd>Kir return-probability bundle used for target-eligibility filtering.",
    )
    p.add_argument(
        "--return-ar",
        default=DEFAULT_INPUTS["return_ar"][2],
        help="AR Ctrl>Kir return-probability debug CSV.",
    )
    p.add_argument(
        "--return-ar-bundle",
        default=DEFAULT_INPUTS["return_ar"][3],
        help="AR Ctrl>Kir return-probability bundle used for target-eligibility filtering.",
    )
    p.add_argument(
        "--include-ctrl",
        action="store_true",
        help="Keep yoked-control rows too. By default only experimental rows are used.",
    )
    p.add_argument(
        "--no-eligibility-filter",
        action="store_true",
        help="Do not filter rows using exp_target_sync_bucket_filter_eligible from bundles.",
    )
    p.add_argument(
        "--out-dir",
        default="tmp/episode_audit/collated",
        help="Output directory for collated CSVs.",
    )
    p.add_argument(
        "--prefix",
        default="pfnd_vs_ar_episode_audit",
        help="Output filename prefix.",
    )
    args = p.parse_args(argv)

    episodes, per_fly, group_summary = collate(
        _inputs_from_args(args),
        include_ctrl=bool(args.include_ctrl),
        eligibility_filter=not bool(args.no_eligibility_filter),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_out = out_dir / f"{args.prefix}_episodes.csv"
    per_fly_out = out_dir / f"{args.prefix}_per_fly.csv"
    group_out = out_dir / f"{args.prefix}_group_summary.csv"

    episodes.to_csv(episode_out, index=False)
    per_fly.to_csv(per_fly_out, index=False)
    group_summary.to_csv(group_out, index=False)

    print(f"Wrote {episode_out} rows={len(episodes)}")
    print(f"Wrote {per_fly_out} rows={len(per_fly)}")
    print(f"Wrote {group_out} rows={len(group_summary)}")
    print()
    print(group_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
