import pandas as pd
from io import StringIO
from typing import Dict, Tuple, List

PI_TITLE = "__calculated__ reward PI by bucket"
COM_TITLE = (
    "median COM-to-center distance, by sync bucket (exp and yoked control), by training"
)
RPD_TITLE = "rewards per distance traveled [m⁻¹]"
NUM_RWD_TITLE = "number __calculated__ rewards by sync bucket"


def _extract_table(lines: List[str], start_idx: int) -> pd.DataFrame:
    """Read a CSV block that starts after a title line and ends at the next blank line."""
    block: List[str] = []
    for line in lines[start_idx + 1 :]:
        if line.strip() == "":
            break
        block.append(line)
    return pd.read_csv(StringIO("".join(block)))


def load_four_tables(csv_path: str) -> Dict[str, pd.DataFrame]:
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    titles = [PI_TITLE, COM_TITLE, RPD_TITLE, NUM_RWD_TITLE]
    idx = {t: i for t in titles for i, line in enumerate(lines) if t in line}
    return {t: _extract_table(lines, i) for t, i in idx.items()}


def build_mapping(
    reward_pi_cols: List[str], com_cols: List[str]
) -> Dict[str, Tuple[str, str, str]]:
    """
    Auto-map Reward PI SB columns to the corresponding COM/RPD/NUM_RWD bucket columns.
    Rules:
      - 'first SB' → b1
      - 'last SB'  → b5
      - Training-specific suffixes ('.1', '.2', …) for b2..b6 columns.
      - For b1: 'training {T} exp b1' and 'yok b1' ['.1', '.2' for T>1]
    """
    mapping: Dict[str, Tuple[str, str, str]] = {}

    for col in reward_pi_cols:
        if "training" not in col or "SB" not in col:
            continue

        T = int(col.split()[1])
        is_exp = "(exp)" in col
        is_yok = "(yok)" in col
        is_first = "first" in col
        is_last = "last" in col
        if not (is_first or is_last) or not (is_exp or is_yok):
            continue

        bucket = "b1" if is_first else "b5"

        if is_exp:
            if bucket == "b1":
                target = f"training {T} exp b1"
            else:
                target = f"exp {bucket}" if T == 1 else f"exp {bucket}.{T-1}"
        else:  # yok
            if bucket == "b1":
                target = "yok b1" if T == 1 else f"yok b1.{T-1}"
            else:
                target = f"yok {bucket}" if T == 1 else f"yok {bucket}.{T-1}"

        # Map to the three versions of the column in the merged table
        com_target = target
        rpd_target = f"{target}_rpd"
        numrwd_target = f"{target}_numrwd"

        mapping[col] = (com_target, rpd_target, numrwd_target)

    return mapping


def check_nan_alignment(csv_path: str):
    tables = load_four_tables(csv_path)
    pi = tables[PI_TITLE]
    com = tables[COM_TITLE]
    rpd = tables[RPD_TITLE]
    num_rwd = tables[NUM_RWD_TITLE]

    # Build mapping from Reward PI column names
    pi_cols = [c for c in pi.columns if "SB" in c]
    col_map = build_mapping(pi_cols, com.columns.tolist())

    # Align rows by (video, fly)
    merged = pi.merge(com, on=["video", "fly"], suffixes=("_pi", "_com"))
    merged = merged.merge(rpd, on=["video", "fly"], suffixes=("", "_rpd"))
    merged = merged.merge(num_rwd, on=["video", "fly"], suffixes=("", "_numrwd"))

    inconsistencies = []

    # --- Pass 1: cross-table row-wise NaN mismatches
    for pi_col, (com_col, rpd_col, numrwd_col) in col_map.items():
        # Sanity: ensure targets exist
        missing = [x for x in (com_col, rpd_col, numrwd_col) if x not in merged.columns]
        if missing:
            inconsistencies.append(
                ("__MISSING_COL__", None, pi_col, tuple(missing), None)
            )
            continue

        for _, row in merged.iterrows():
            pattern = (
                pd.isna(row[pi_col]),
                pd.isna(row[com_col]),
                pd.isna(row[rpd_col]),
                pd.isna(row[numrwd_col]),
            )
            if len(set(pattern)) > 1:
                inconsistencies.append(
                    (
                        row["video"],
                        row["fly"],
                        pi_col,
                        (com_col, rpd_col, numrwd_col),
                        pattern,
                    )
                )

    # --- Pass 2: exp/yok pairwise co-occurrence check
    pairwise_mismatches = []
    for T in [1, 2, 3]:  # trainings
        for bucket in ["b1", "b5"]:  # first/last
            exp_col = f"exp {bucket}" if T == 1 else f"exp {bucket}.{T-1}"
            yok_col = f"yok {bucket}" if T == 1 else f"yok {bucket}.{T-1}"

            counts = {}
            for tbl_name, tbl in zip(
                ["PI", "COM", "RPD", "NUM_RWD"], [pi, com, rpd, num_rwd]
            ):
                if exp_col in tbl.columns and yok_col in tbl.columns:
                    both_non_nan = tbl[[exp_col, yok_col]].dropna().shape[0]
                    counts[tbl_name] = both_non_nan

            if len(set(counts.values())) > 1:
                pairwise_mismatches.append((f"training {T} {bucket}", counts))

    return col_map, inconsistencies, pairwise_mismatches


# --- Example usage ---
csv_path = "learning_stats.csv"
col_map, mismatches, pairwise_mismatches = check_nan_alignment(csv_path)

print("Column mapping:")
for k, v in col_map.items():
    print(f"  {k} -> {v}")

print("\nRow-wise inconsistencies:", len(mismatches))
for m in mismatches[:10]:
    print(m)

print("\nPairwise exp/yok mismatches:", len(pairwise_mismatches))
for bucket, counts in pairwise_mismatches:
    print(f"  {bucket}: {counts}")
