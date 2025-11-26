import os
import pandas as pd
from pathlib import Path


def make_large_turn_exit_table(va, *, include_video_name=True, include_fly_index=True):
    """
    Convert the per-fly exit/turn records created in Cython
    (va.lg_turn_exit_events) into a single pandas DataFrame.

    Parameters
    ----------
    va : VideoAnalysis
        The VA object that contains lg_turn_exit_events (list-of-lists).
    include_video_name : bool
        If True, add a 'video' column (va.fn).
    include_fly_index : bool
        If True, add a 'fly' column (trajectory index).

    Returns
    -------
    df : pandas.DataFrame
        Columns:
            video              (optional)
            fly                (optional)
            trn_range_idx
            exit_idx
            exit_frame
            has_large_turn
            turn_start_idx
            turn_end_idx
            reject_reason
            reject_turn_start_idx
            reject_turn_end_idx
            reentry_frame
            strategy_weaving
            strategy_backward_walking
            max_outside_mm
            angle_to_tangent_deg
            frac_backward_frames
    """

    if not hasattr(va, "lg_turn_exit_events"):
        raise ValueError(
            "va.lg_turn_exit_events is missing — did you run calcLargeTurnsAfterCircleExit()?"
        )

    rows = []

    # Optional: per-exit rejection reasons populated by the Cython code
    # Shape: [fly_idx][trn_range_idx][exit_idx] -> (reason_str, (st_idx, end_idx))
    has_rejections = hasattr(va, "lg_turn_rejection_reasons")
    if has_rejections:
        rejections = va.lg_turn_rejection_reasons

    # iterate over flies
    for fly_idx, fly_records in enumerate(va.lg_turn_exit_events):

        for rec in fly_records:
            row = dict(rec)  # shallow copy of dict created by Cython

            # Attach rejection info, if available
            reject_reason = None
            reject_turn_start_idx = None
            reject_turn_end_idx = None

            if has_rejections:
                if fly_idx < len(rejections):
                    trn_range_idx = rec.get("trn_range_idx")
                    if isinstance(trn_range_idx, int) and 0 <= trn_range_idx < len(
                        rejections[fly_idx]
                    ):
                        per_range = rejections[fly_idx][trn_range_idx]
                        # per_range is a dict mapping exit_idx -> (reason, (st, end))
                        if isinstance(per_range, dict):
                            exit_idx = rec.get("exit_idx")
                            if isinstance(exit_idx, int) and exit_idx in per_range:
                                reason, indices = per_range[exit_idx]
                                reject_reason = reason
                                if (
                                    isinstance(indices, (tuple, list))
                                    and len(indices) == 2
                                ):
                                    reject_turn_start_idx, reject_turn_end_idx = indices
            row["reject_reason"] = reject_reason
            row["reject_turn_start_idx"] = reject_turn_start_idx
            row["reject_turn_end_idx"] = reject_turn_end_idx

            if include_fly_index:
                row["fly"] = fly_idx

            if include_video_name:
                row["video"] = va.fn

            rows.append(row)

    if not rows:
        # empty dataframe with correct schema
        cols = [
            *(["video"] if include_video_name else []),
            *(["fly"] if include_fly_index else []),
            "trn_range_idx",
            "exit_idx",
            "exit_frame",
            "has_large_turn",
            "turn_start_idx",
            "turn_end_idx",
            "reject_reason",
            "reject_turn_start_idx",
            "reject_turn_end_idx",
            "reentry_frame",
            "strategy_weaving",
            "strategy_backward_walking",
            "max_outside_mm",
            "angle_to_tangent_deg",
            "frac_backward_frames",
        ]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)

    # Reorder columns (nice, consistent, readable)
    col_order = []
    if include_video_name:
        col_order.append("video")
    if include_fly_index:
        col_order.append("fly")
    col_order += [
        "trn_range_idx",
        "exit_idx",
        "exit_frame",
        "has_large_turn",
        "turn_start_idx",
        "turn_end_idx",
        "reject_reason",
        "reject_turn_start_idx",
        "reject_turn_end_idx" "reentry_frame",
        "strategy_weaving",
        "strategy_backward_walking",
        "max_outside_mm",
        "angle_to_tangent_deg",
        "frac_backward_frames",
    ]

    df = df.reindex(columns=col_order + [c for c in df.columns if c not in col_order])

    return df


def save_large_turn_exit_table(va, out_path, *, per_fly=False):
    """
    Save the exit/turn table to CSV files.

    Parameters
    ----------
    va : VideoAnalysis
    out_path : str or Path
        If per_fly=False: single CSV path
        If per_fly=True: directory path (one file per fly)
    """
    out_path = Path(out_path)

    if per_fly:
        out_path.mkdir(parents=True, exist_ok=True)
        for fly_idx, fly_records in enumerate(va.lg_turn_exit_events):
            df_fly = make_large_turn_exit_table(va).query("fly == @fly_idx")
            df_fly.to_csv(
                out_path
                / f"{os.path.basename(va.fn)}_f{va.f}_{fly_idx:02d}_turn_exit_events.csv",
                index=False,
            )
    else:
        df = make_large_turn_exit_table(va)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
