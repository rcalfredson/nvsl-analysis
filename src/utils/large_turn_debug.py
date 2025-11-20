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
    """

    if not hasattr(va, "lg_turn_exit_events"):
        raise ValueError(
            "va.lg_turn_exit_events is missing — did you run calcLargeTurnsAfterCircleExit()?"
        )

    rows = []

    # iterate over flies
    for fly_idx, fly_records in enumerate(va.lg_turn_exit_events):

        for rec in fly_records:
            row = dict(rec)  # shallow copy of dict created by Cython

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
    ]

    df = df[col_order]

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
                out_path / f"fly_{fly_idx:02d}_turn_exit_events.csv", index=False
            )
    else:
        df = make_large_turn_exit_table(va)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
