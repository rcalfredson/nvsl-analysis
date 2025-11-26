"""
Script for processing per-fly large turn data raw data (obtained by running the analyze script using the --dump-large-turn-exits flag)
"""

import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pandas as pd

CATEGORY_ORDER = [
    "weaving",
    "backward_walking",
    "reward_circle_entry",
    "low_displacement",
    "too_little_walking",
    "wall_contact",
    "other",
]


def plot_category_bars(prefix, fly, role, category_dicts, outdir="plots"):
    """
    Create a bar chart of non-large-turn categories aggregated over all training blocks.

    category_dicts = list of dicts, one per training block
                     e.g. [{"weaving": 2, "reward_circle_entry": 10}, ...]
    """
    # Aggregate across training blocks
    agg_counts = defaultdict(int)
    for d in category_dicts:
        for k, v in d.items():
            agg_counts[k] += v

    # Restrict to known categories + ensure missing ones are 0
    cats = CATEGORY_ORDER
    values = [agg_counts.get(c, 0) for c in cats]

    # Skip empty plots
    if sum(values) == 0:
        print(f"No non-large-turn exits for {prefix}, fly {fly}, role {role}")
        return

    # Ensure output directory exists
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # Figure
    plt.figure(figsize=(8, 4))
    plt.bar(cats, values, color="#4a90e2")
    plt.title(f"{prefix} — fly {fly} — {role}")
    plt.ylabel("Count of non-large-turn exits")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    # Save
    fname = f"{prefix}_fly{fly}_{role}_categories.png"
    outfile = outdir / fname
    plt.savefig(outfile, dpi=150)
    plt.close()

    print(f"Saved plot: {outfile}")


def parse_file(path: Path) -> pd.DataFrame:
    """Read the exit-event CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path)

    # Normalize column names and booleans
    if "has_large_turn" not in df.columns:
        raise ValueError(f"Missing has_large_turn in {path}")

    df["has_large_turn"] = df["has_large_turn"].astype(str).str.lower() == "true"
    return df


def classify_non_large_turn(row):
    """Return category for exits that are not large turns."""
    # Override by weaving/backward-walking (weaving wins)
    weaving = str(row.get("strategy_weaving", "")).lower() == "true"
    bw = str(row.get("strategy_backward_walking", "")).lower() == "true"

    if weaving:
        return "weaving"
    if bw:
        return "backward_walking"

    # Fall back to reject_reason
    rr = str(row.get("reject_reason", "")).strip()

    if rr in (
        "low_displacement",
        "too_little_walking",
        "reward_circle_entry",
        "wall_contact",
    ):
        return rr

    # safety fallback (rare)
    return "other"


def compute_stats(df: pd.DataFrame):
    """
    Return:
      ratios[train_idx] = ratio
      counts[train_idx] = (num_large_turns, num_total_exits)
      categories[train_idx] = dict(category -> count)
    """
    ratios = {}
    counts = {}
    categories = {}

    grouped = df.groupby("trn_range_idx")

    for trn_idx, sub in grouped:
        total = len(sub)
        large_turns = sub["has_large_turn"].sum()

        # Basic counts
        counts[trn_idx] = (large_turns, total)

        if total == 0:
            ratios[trn_idx] = float("nan")
        else:
            ratios[trn_idx] = large_turns / total

        # --- NEW: Non-large-turn category counts ---
        non_large = sub[~sub["has_large_turn"]]

        cat_counts = defaultdict(int)
        for _, row in non_large.iterrows():
            cat = classify_non_large_turn(row)
            cat_counts[cat] += 1

        categories[trn_idx] = dict(cat_counts)

    return ratios, counts, categories


def extract_pair_key(path: Path):
    """
    From:
      c51__2025-07-10__20-09-48.avi_f0_00_turn_exit_events.csv
    Return:
      (video_base, fly_id), role
    where role = "exp" or "yok"
    """
    name = path.name

    if not name.endswith("_turn_exit_events.csv"):
        raise ValueError(f"Unexpected filename: {name}")

    core = name.replace("_turn_exit_events.csv", "")

    try:
        prefix, fpart, paircode = core.rsplit("_", 2)
    except ValueError:
        raise ValueError(f"Filename does not match expected pattern: {name}")

    if not fpart.startswith("f"):
        raise ValueError(f"Invalid fly segment in {name}")

    fly_id = int(fpart[1:])
    role = "exp" if paircode == "00" else "yok" if paircode == "01" else None

    if role is None:
        raise ValueError(f"Unrecognized pair code in {name}")

    return (prefix, fly_id), role


def filter_reward_circle_entry(df: pd.DataFrame) -> pd.DataFrame:
    """Return only reward-circle-entry non-large-turn exits, excluding weaving/backward walking."""
    df = df.copy()
    weaving = df["strategy_weaving"].astype(str).str.lower() == "true"
    bw = df["strategy_backward_walking"].astype(str).str.lower() == "true"
    non_large = ~df["has_large_turn"]

    # Base reject_reason
    rce = df["reject_reason"].astype(str).str.strip() == "reward_circle_entry"

    return df[non_large & (~weaving) & (~bw) & rce]


def plot_numeric_distribution(prefix, fly, role, df, column, outdir="plots_numeric"):
    """
    Plot histogram + KDE for a given numeric column in reward-circle-entry exits.
    """
    if df.empty:
        print(
            f"No reward-circle-entry exits for {prefix}, fly {fly}, role {role} ({column})"
        )
        return

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    values = df[column].dropna().astype(float)

    if values.empty:
        print(f"No numeric values for {column} in {prefix}, fly {fly}, role {role}")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(
        values, bins=30, alpha=0.7, density=True, color="#7c8cff", label="Histogram"
    )

    # Optional: simple KDE using pandas
    try:
        values.plot(kind="kde", color="black", linewidth=1.5, label="KDE")
    except Exception:
        pass

    plt.title(f"{prefix} — fly {fly} — {role}\n{column} distribution")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.tight_layout()

    fname = f"{prefix}_fly{fly}_{role}_{column}.png"
    outfile = outdir / fname
    plt.savefig(outfile, dpi=150)
    plt.close()

    print(f"Saved numeric plot: {outfile}")


def process_directory(dir_path: str, out_csv="large_turn_ratios.csv"):
    dir_path = Path(dir_path)

    # Collect files by (video_prefix, fly_id)
    pairs = defaultdict(dict)

    for csvfile in dir_path.glob("*_turn_exit_events.csv"):
        key, role = extract_pair_key(csvfile)
        pairs[key][role] = csvfile

    rows = []

    for (prefix, fly), d in pairs.items():
        if "exp" not in d or "yok" not in d:
            print(f"Skipping incomplete pair for {prefix}, fly {fly}")
            continue

        df_exp = parse_file(d["exp"])
        df_yok = parse_file(d["yok"])

        exp_plot_cats = []
        yok_plot_cats = []

        ratios_exp, counts_exp, cats_exp = compute_stats(df_exp)
        ratios_yok, counts_yok, cats_yok = compute_stats(df_yok)

        all_indices = set(ratios_exp) | set(ratios_yok)

        for trn_idx in sorted(all_indices):
            exp_ratio = ratios_exp.get(trn_idx, float("nan"))
            yok_ratio = ratios_yok.get(trn_idx, float("nan"))

            (exp_lt, exp_total) = counts_exp.get(trn_idx, (float("nan"), float("nan")))
            (yok_lt, yok_total) = counts_yok.get(trn_idx, (float("nan"), float("nan")))

            # Difference of ratios if both are valid
            diff = (
                exp_ratio - yok_ratio
                if pd.notna(exp_ratio) and pd.notna(yok_ratio)
                else float("nan")
            )

            exp_cats = cats_exp.get(trn_idx, {})
            yok_cats = cats_yok.get(trn_idx, {})
            exp_plot_cats.append(exp_cats)
            yok_plot_cats.append(yok_cats)

            row_dict = {
                "video": prefix,
                "fly": fly,
                "trn_range_idx": trn_idx,
                # Counts
                "exp_large_turns": exp_lt,
                "exp_total_exits": exp_total,
                "yok_large_turns": yok_lt,
                "yok_total_exits": yok_total,
                # Ratios
                "ratio_exp": exp_ratio,
                "ratio_yok": yok_ratio,
                "ratio_exp_minus_yok": diff,
            }

            # Add category counts with a prefix
            for cat in sorted(set(exp_cats) | set(yok_cats)):
                row_dict[f"exp_{cat}"] = exp_cats.get(cat, 0)
                row_dict[f"yok_{cat}"] = yok_cats.get(cat, 0)

            rows.append(row_dict)
        plot_category_bars(prefix, fly, "exp", exp_plot_cats)
        plot_category_bars(prefix, fly, "yok", yok_plot_cats)

        exp_rce = filter_reward_circle_entry(df_exp)
        yok_rce = filter_reward_circle_entry(df_yok)

        for col in ["max_outside_mm", "angle_to_tangent_deg"]:
            plot_numeric_distribution(prefix, fly, "exp", exp_rce, col)
            plot_numeric_distribution(prefix, fly, "yok", yok_rce, col)

    # Save
    out_path = Path(out_csv)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory containing *_turn_exit_events.csv files")
    parser.add_argument("--out", default="large_turn_ratios.csv")
    args = parser.parse_args()

    process_directory(args.dir, args.out)
