#!/usr/bin/env python3
"""
Build a stratified, threshold-biased review queue from large-turn exit event CSVs.

Usage examples:
  python build_turn_review_queue.py \
      --inputs per_fly_turn_exit_tables/*.csv \
      --out review_queue.csv \
      --seed 123 \
      --weaving-max-outside 1.5 \
      --tangent-thresh 30 \
      --backward-frac-thresh 0.6
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Targets:
    weaving: int = 30
    small_angle_reentry: int = 30
    backward_walking: int = 30
    large_turn: int = 30
    wall_contact: int = 20
    too_little_walking: int = 20
    low_displacement: int = 20


REVIEW_CLASSES = [
    "weaving",
    "small_angle_reentry",
    "backward_walking",
    "large_turn",
    "wall_contact",
    "too_little_walking",
    "low_displacement",
]


def _read_inputs(patterns: list[str]) -> pd.DataFrame:
    paths: list[str] = []
    for p in patterns:
        paths.extend(glob.glob(p))
    if not paths:
        raise SystemExit(f"No input files matched: {patterns}")
    dfs = []
    for fp in sorted(set(paths)):
        df = pd.read_csv(fp)
        df["source_csv"] = fp
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Some columns are floats with .0 (from Cython) or empty strings.
    for col in [
        "turn_start_idx",
        "turn_end_idx",
        "reject_turn_start_idx",
        "reject_turn_end_idx",
        "reentry_frame",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["max_outside_mm", "angle_to_tangent_deg", "frac_backward_frames"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "has_large_turn" in df.columns:
        # Can be True/False strings; normalize to bool.
        df["has_large_turn"] = df["has_large_turn"].astype(bool)
    return df


def _assign_review_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["reject_reason"] = df.get("reject_reason", pd.Series([None] * len(df))).astype(
        "object"
    )

    # large_turn: accepted turns
    df["review_class"] = np.where(
        df["has_large_turn"] == True, "large_turn", df["reject_reason"]
    )

    # Only keep rows that map to our classes
    df = df[df["review_class"].isin(REVIEW_CLASSES)].copy()
    return df


def _threshold_distance(
    df: pd.DataFrame,
    *,
    weaving_max_outside: float,
    tangent_thresh: float,
    backward_frac_thresh: float,
) -> pd.Series:
    """
    Return a per-row "distance-to-threshold" score used to rank borderline cases.
    Smaller = closer to thresholds = more ambiguous.
    """
    rc = df["review_class"].astype(str)

    # Default: inf (means "not rankable by a meaningful threshold")
    dist = pd.Series(np.inf, index=df.index, dtype="float64")

    # weaving: combine outside + angle distances (sum works fine; you can also use max())
    m_out = (df["max_outside_mm"] - weaving_max_outside).abs()
    m_ang = (df["angle_to_tangent_deg"] - tangent_thresh).abs()
    dist_weaving = m_out.fillna(np.inf) + m_ang.fillna(np.inf)
    dist.loc[rc == "weaving"] = dist_weaving.loc[rc == "weaving"]

    # backward walking: distance to frac threshold
    m_bw = (df["frac_backward_frames"] - backward_frac_thresh).abs()
    dist.loc[rc == "backward_walking"] = m_bw.fillna(np.inf).loc[
        rc == "backward_walking"
    ]

    # small_angle_reentry: closest to tangent threshold is the key ambiguity dial
    dist.loc[rc == "small_angle_reentry"] = m_ang.fillna(np.inf).loc[
        rc == "small_angle_reentry"
    ]

    # Other classes: leave as inf (they’ll be sampled mostly randomly)
    return dist


def _sample_class(
    df_c: pd.DataFrame,
    n_total: int,
    rng: np.random.Generator,
    *,
    near_metric: pd.Series,
) -> pd.DataFrame:
    if n_total <= 0 or df_c.empty:
        return df_c.head(0)

    n_total = min(n_total, len(df_c))
    n_near = n_total // 2
    n_rand = n_total - n_near

    # Near-threshold pool: sort ascending by distance metric
    df_sorted = df_c.assign(_near=near_metric.loc[df_c.index]).sort_values(
        ["_near", "exit_frame"], ascending=[True, True]
    )

    near = df_sorted.head(n_near)

    remaining = df_c.drop(index=near.index, errors="ignore")
    if n_rand > 0 and not remaining.empty:
        take = min(n_rand, len(remaining))
        rand_idx = rng.choice(remaining.index.to_numpy(), size=take, replace=False)
        rand = remaining.loc[rand_idx]
    else:
        rand = remaining.head(0)

    out = pd.concat([near, rand], ignore_index=False)
    return out


def build_review_queue(
    df: pd.DataFrame,
    *,
    targets: Targets,
    seed: int,
    weaving_max_outside: float,
    tangent_thresh: float,
    backward_frac_thresh: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _coerce_numeric(df)
    df = _assign_review_class(df)

    # Distance-to-threshold score
    near_metric = _threshold_distance(
        df,
        weaving_max_outside=weaving_max_outside,
        tangent_thresh=tangent_thresh,
        backward_frac_thresh=backward_frac_thresh,
    )

    rng = np.random.default_rng(seed)

    # Targets mapping
    tmap = {
        "weaving": targets.weaving,
        "small_angle_reentry": targets.small_angle_reentry,
        "backward_walking": targets.backward_walking,
        "large_turn": targets.large_turn,
        "wall_contact": targets.wall_contact,
        "too_little_walking": targets.too_little_walking,
        "low_displacement": targets.low_displacement,
    }

    sampled_parts = []
    for cls in REVIEW_CLASSES:
        df_c = df[df["review_class"] == cls]
        sampled = _sample_class(df_c, tmap[cls], rng, near_metric=near_metric)
        sampled_parts.append(sampled)

    queue = pd.concat(sampled_parts, ignore_index=True)

    # Make a nice review order: class blocks, within class: near first, then random, by frame
    queue["_near_score"] = _threshold_distance(
        queue,
        weaving_max_outside=weaving_max_outside,
        tangent_thresh=tangent_thresh,
        backward_frac_thresh=backward_frac_thresh,
    )
    queue = queue.sort_values(
        ["review_class", "_near_score", "video", "fly", "exit_frame"], ascending=True
    )

    # Add columns for human labeling
    for col in ["human_label", "is_wrong", "confidence", "notes"]:
        if col not in queue.columns:
            queue[col] = ""

    # Summary
    summary = (
        df.groupby("review_class")
        .size()
        .rename("available")
        .to_frame()
        .join(queue.groupby("review_class").size().rename("sampled"), how="left")
        .fillna(0)
        .astype({"sampled": int})
        .reset_index()
        .sort_values("review_class")
    )

    return queue, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="CSV glob(s), e.g. per_fly_turn_exit_tables/*.csv",
    )
    ap.add_argument(
        "--out", default="review_queue.csv", help="Output CSV for the review queue"
    )
    ap.add_argument(
        "--summary-out",
        default="review_queue_summary.csv",
        help="Output CSV for counts summary",
    )
    ap.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducible sampling"
    )

    ap.add_argument("--weaving-max-outside", type=float, default=1.5)
    ap.add_argument("--tangent-thresh", type=float, default=30.0)
    ap.add_argument("--backward-frac-thresh", type=float, default=0.6)

    args = ap.parse_args()

    df = _read_inputs(args.inputs)
    queue, summary = build_review_queue(
        df,
        targets=Targets(),
        seed=args.seed,
        weaving_max_outside=args.weaving_max_outside,
        tangent_thresh=args.tangent_thresh,
        backward_frac_thresh=args.backward_frac_thresh,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    queue.drop(columns=["_near_score"], errors="ignore").to_csv(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)

    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.summary_out}")


if __name__ == "__main__":
    main()
