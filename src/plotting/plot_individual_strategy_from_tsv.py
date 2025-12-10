from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import f_oneway, ttest_ind

from src.plotting.individual_strategy_plotter import (
    IndividualStrategyConfig,
    _plot_overlays,
    _plot_multi_group_overlays,
)
from src.plotting.plot_customizer import PlotCustomizer

# Map metric name -> (title, y_label, filename_stub base)
METRIC_TITLES: dict[str, tuple[str, str, str]] = {
    "sharp_turn_away": (
        "Sharp-turn probability away from agarose",
        "Sharp-turn probability (exp − yoked)",
        "individual_sharp_turn_away_5mm_overlays",
    ),
    "median_dist_exp_minus_yok": (
        "Median distance to reward-circle center\n(exp − yoked)",
        "Median distance to reward (mm, exp − yoked)",
        "individual_median_distance_overlays",
    ),
    "large_turn_ratio": (
        "Large-turn probability after reward-circle exit\n(exp only)",
        "Large turns / exits (experimental fly)",
        "individual_large_turn_ratio_overlays",
    ),
    "weaving_ratio": (
        "Weaving re-entry probability after reward-circle exit\n"
        "(experimental fly only)",
        "Weaving re-entry / total exits\n(experimental fly)",
        "individual_weaving_ratio_overlays",
    ),
    "small_angle_ratio": (
        "Small-angle re-entry probability after reward-circle exit\n"
        "(experimental fly only)",
        "Small-angle re-entry / total exits\n(experimental fly)",
        "individual_small_angle_reentry_ratio_overlays",
    ),
}


@dataclass(frozen=True)
class FlyKey:
    video: str
    fly_idx: str  # keep as string to match TSV exactly


def _warn(msg: str) -> None:
    print(f"[individual_strategy_from_tsv] WARNING: {msg}")


def _load_tsv(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        _warn(f"No rows found in {path}")
    return rows


def _build_fly_index(
    rows: List[dict],
) -> Tuple[List[FlyKey], Dict[FlyKey, int], Sequence[int], Sequence[int]]:
    """
    From all rows, figure out:
      - ordered list of unique flies (video, fly_idx)
      - mapping FlyKey -> row index
      - indices of bottom learners
      - indices of top learners
    """
    fly_keys: List[FlyKey] = []
    fly_to_idx: Dict[FlyKey, int] = {}
    group_per_fly: Dict[FlyKey, str] = {}

    for r in rows:
        key = FlyKey(video=r["video"], fly_idx=r["fly_idx"])
        if key not in fly_to_idx:
            fly_to_idx[key] = len(fly_keys)
            fly_keys.append(key)

        grp = r.get("group", "")
        if grp:
            prev = group_per_fly.get(key)
            if prev is not None and prev != grp:
                _warn(
                    f"Inconsistent group for fly {key}: seen {prev!r} and {grp!r}; "
                    "using the first."
                )
            else:
                group_per_fly[key] = grp

    selected_bottom = [fly_to_idx[k] for k, g in group_per_fly.items() if g == "bottom"]
    selected_top = [fly_to_idx[k] for k, g in group_per_fly.items() if g == "top"]

    return fly_keys, fly_to_idx, selected_bottom, selected_top


def _build_metric_matrices(
    rows: List[dict], fly_to_idx: Dict[FlyKey, int]
) -> Dict[str, tuple[str, np.ndarray, list[str], np.ndarray]]:
    """
    Group rows by metric and construct matrices of shape (n_flies, n_timepoints).

    Returns:
        dict[metric] -> (axis_kind, x_values, x_labels, matrix)
    """
    by_metric: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        metric = r["metric"]
        if not metric:
            continue
        by_metric[metric].append(r)

    results: Dict[str, tuple[str, np.ndarray, list[str], np.ndarray]] = {}
    n_flies = len(fly_to_idx)

    for metric, m_rows in by_metric.items():
        axis_kinds = {r["axis_kind"] for r in m_rows}
        if len(axis_kinds) != 1:
            _warn(
                f"Metric {metric!r} has multiple axis_kinds {axis_kinds}; "
                "skipping this metric."
            )
            continue
        axis_kind = axis_kinds.pop()

        # Determine unique axis indices and their labels
        axis_indices = sorted({int(r["axis_index"]) for r in m_rows})
        idx_to_col = {ax: i for i, ax in enumerate(axis_indices)}
        x_labels: list[str] = [""] * len(axis_indices)

        # Initialize matrix with NaNs
        matrix = np.full((n_flies, len(axis_indices)), np.nan, dtype=float)

        for r in m_rows:
            try:
                fly_key = FlyKey(video=r["video"], fly_idx=r["fly_idx"])
                row_idx = fly_to_idx[fly_key]
            except KeyError:
                # Shouldn't happen, but be defensive
                continue

            ax_idx = int(r["axis_index"])
            col_idx = idx_to_col[ax_idx]

            # Keep the first non-empty label we see for each column
            lbl = r.get("axis_label", "")
            if lbl and not x_labels[col_idx]:
                x_labels[col_idx] = lbl

            try:
                val = float(r["value"])
            except (TypeError, ValueError):
                continue

            matrix[row_idx, col_idx] = val

        # Fill any empty labels with something generic
        for i, lbl in enumerate(x_labels):
            if not lbl:
                x_labels[i] = f"{axis_kind} {axis_indices[i]}"

        x_vals = np.arange(len(axis_indices))
        results[metric] = (axis_kind, x_vals, x_labels, matrix)

    return results


# --------------------------------------------------------------------------- #
# Existing single-TSV behavior                                                #
# --------------------------------------------------------------------------- #


def _plot_all_metrics_for_tsv(
    rows: List[dict],
    tsv_path: Path,
    *,
    out_dir: Path,
    image_format: str = "png",
) -> None:
    fly_keys, fly_to_idx, selected_bottom, selected_top = _build_fly_index(rows)

    if not selected_bottom and not selected_top:
        _warn(f"No top/bottom groups found in {tsv_path}; skipping plots.")
        return

    metric_data = _build_metric_matrices(rows, fly_to_idx)
    if not metric_data:
        _warn(f"No metric data found in {tsv_path}; nothing to plot.")
        return

    cfg = IndividualStrategyConfig(out_dir=out_dir, image_format=image_format)
    customizer = PlotCustomizer()

    for metric, (axis_kind, x_vals, x_labels, matrix) in metric_data.items():
        if metric not in METRIC_TITLES:
            _warn(f"Metric {metric!r} not recognized for plotting; skipping.")
            continue

        title, y_label, filename_stub = METRIC_TITLES[metric]

        _plot_overlays(
            title=title,
            y_label=y_label,
            x=x_vals,
            x_labels=x_labels,
            matrix=matrix,
            selected_bottom=selected_bottom,
            selected_top=selected_top,
            cfg=cfg,
            customizer=customizer,
            filename_stub=filename_stub,
        )


# --------------------------------------------------------------------------- #
# Multi-TSV (multi-group) statistics                                          #
# --------------------------------------------------------------------------- #


def _load_group_top_only(
    tsv_path: Path,
) -> Dict[str, tuple[str, np.ndarray, list[str], np.ndarray]]:
    """
    Load one TSV and return metric -> (axis_kind, x_vals, x_labels, matrix_top),
    where matrix_top contains only rows for flies in the 'top' group.
    """
    rows = _load_tsv(tsv_path)
    if not rows:
        return {}

    _, fly_to_idx, _selected_bottom, selected_top = _build_fly_index(rows)
    if not selected_top:
        _warn(f"No 'top' group flies found in {tsv_path}; skipping.")
        return {}

    metric_all = _build_metric_matrices(rows, fly_to_idx)
    metric_top: Dict[str, tuple[str, np.ndarray, list[str], np.ndarray]] = {}

    for metric, (axis_kind, x_vals, x_labels, matrix) in metric_all.items():
        matrix_top = matrix[selected_top, :] if matrix.size else matrix
        metric_top[metric] = (axis_kind, x_vals, x_labels, matrix_top)

    return metric_top


def _write_metric_pairwise_tsv(
    metric: str,
    axis_kind: str,
    x_vals: np.ndarray,
    x_labels: list[str],
    group_mats: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """
    For a single metric across >=2 groups, compute per-tick pairwise
    comparisons (Welch's t-test) and write them to a TSV.

    Bonferroni correction is applied across all pairwise comparisons
    for each axis_index (e.g. all genotype pairs for a given training).
    """
    group_labels = sorted(group_mats.keys())
    n_groups = len(group_labels)
    n_time = x_vals.size

    if n_groups < 2:
        return

    # Number of pairwise comparisons per axis (family size for Bonferroni)
    n_pairs = n_groups * (n_groups - 1) // 2
    if n_pairs == 0:
        return

    out_path = out_dir / f"{metric}__pairwise.tsv"

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        # Header:
        # one row per (axis_index, group_a, group_b)
        writer.writerow(
            [
                "axis_index",
                "axis_label",
                "group_a",
                "group_b",
                "group_a_n",
                "group_a_mean",
                "group_b_n",
                "group_b_mean",
                "t_stat",
                "p_raw",
                "p_bonf",
            ]
        )

        for j in range(n_time):
            ax_idx = int(x_vals[j])
            ax_label = x_labels[j]

            # Pre-compute cleaned samples, n, and mean per group for this axis
            per_group: Dict[str, tuple[np.ndarray, int, float]] = {}
            for label in group_labels:
                col = group_mats[label][:, j]
                col = col[np.isfinite(col)]
                n = int(col.size)
                if n > 0:
                    mean = float(np.mean(col))
                else:
                    mean = float("nan")
                per_group[label] = (col, n, mean)

            # Pairwise Welch t-tests with Bonferroni correction
            for g_a, g_b in combinations(group_labels, 2):
                a_vals, a_n, a_mean = per_group[g_a]
                b_vals, b_n, b_mean = per_group[g_b]

                if a_n == 0 or b_n == 0:
                    t_stat = float("nan")
                    p_raw = float("nan")
                    p_bonf = float("nan")
                else:
                    t_stat, p_raw = ttest_ind(a_vals, b_vals, equal_var=False)
                    t_stat = float(t_stat)
                    p_raw = float(p_raw)
                    # Bonferroni across all pairs for this axis
                    p_bonf = min(p_raw * n_pairs, 1.0)

                writer.writerow(
                    [
                        ax_idx,
                        ax_label,
                        g_a,
                        g_b,
                        a_n,
                        f"{a_mean:.6g}",
                        b_n,
                        f"{b_mean:.6g}",
                        f"{t_stat:.6g}",
                        f"{p_raw:.6g}",
                        f"{p_bonf:.6g}",
                    ]
                )


def _write_metric_stats_tsv(
    metric: str,
    axis_kind: str,
    x_vals: np.ndarray,
    x_labels: list[str],
    group_mats: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """
    For a single metric across >=2 groups, compute per-tick stats and
    write them to a TSV.

    If there are exactly 2 groups: Welch's t-test.
    If >=3 groups: one-way ANOVA.
    """
    group_labels = list(group_mats.keys())
    n_groups = len(group_labels)
    n_time = x_vals.size

    if n_groups < 2:
        return

    out_path = out_dir / f"{metric}__stats.tsv"

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        # Header
        header = ["axis_index", "axis_label"]
        for label in group_labels:
            header.extend([f"{label}_n", f"{label}_mean", f"{label}_sem"])
        stat_col = "t_stat" if n_groups == 2 else "F_stat"
        header.extend([stat_col, "p_value"])
        writer.writerow(header)

        for j in range(n_time):
            row = [int(x_vals[j]), x_labels[j]]

            samples = []
            for label in group_labels:
                col = group_mats[label][:, j]
                col = col[np.isfinite(col)]
                n = int(col.size)
                if n > 0:
                    mean = float(np.mean(col))
                    sem = (
                        float(np.std(col, ddof=1) / np.sqrt(n))
                        if n > 1
                        else float("nan")
                    )
                    samples.append(col)
                else:
                    mean = float("nan")
                    sem = float("nan")
                    samples.append(col)
                row.extend([n, f"{mean:.6g}", f"{sem:.6g}"])

            # Hypothesis test for this axis tick
            valid_samples = [s for s in samples if s.size > 0]
            if len(valid_samples) < 2:
                stat = float("nan")
                pval = float("nan")
            else:
                if n_groups == 2:
                    a, b = valid_samples[0], valid_samples[1]
                    t_stat, pval = ttest_ind(a, b, equal_var=False)
                    stat = float(t_stat)
                else:
                    F_stat, pval = f_oneway(*valid_samples)
                    stat = float(F_stat)

            row.extend([f"{stat:.6g}", f"{pval:.6g}"])
            writer.writerow(row)


def _compare_groups_from_tsvs(
    tsv_paths: List[Path],
    labels: List[str],
    *,
    out_dir: Path,
    image_format: str = "png",
) -> None:
    """
    Multi-group mode: each TSV is treated as one group of *top* learners.
    For each metric shared across all groups, compute per-tick stats and
    write a TSV with n/mean/SEM per group plus t/ANOVA results, and also
    generate multi-group overlay plots of individual flies.
    """
    assert len(tsv_paths) == len(labels)

    group_metric_data: Dict[
        str, Dict[str, tuple[str, np.ndarray, list[str], np.ndarray]]
    ] = {}
    metrics_per_group: Dict[str, set[str]] = {}

    for path, label in zip(tsv_paths, labels):
        metric_top = _load_group_top_only(path)
        if not metric_top:
            _warn(f"No usable metric data for group {label!r} from {path}; skipping.")
            continue
        group_metric_data[label] = metric_top
        metrics_per_group[label] = set(metric_top.keys())

    if not group_metric_data:
        _warn("No groups with usable data; nothing to compare.")
        return

    # Metrics common to all groups
    common_metrics = set.intersection(*metrics_per_group.values())
    if not common_metrics:
        _warn("No metrics shared across all groups; nothing to compare.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Config for plots
    cfg = IndividualStrategyConfig(out_dir=out_dir, image_format=image_format)
    customizer = PlotCustomizer()

    # Simple color palette for groups
    base_colors = ["#1f4da1", "#a00000", "#008b45", "#ff8c00", "#6a3d9a"]
    group_labels = list(group_metric_data.keys())
    group_colors = {
        label: base_colors[i % len(base_colors)] for i, label in enumerate(group_labels)
    }

    for metric in sorted(common_metrics):
        # Use the first group's axis as reference
        first_label = next(iter(group_metric_data.keys()))
        axis_kind, x_vals, x_labels, matrix_ref = group_metric_data[first_label][metric]

        # Collect matrices for all groups, checking consistency
        group_mats: Dict[str, np.ndarray] = {}
        consistent = True
        for label, metric_dict in group_metric_data.items():
            ak, xv, xl, mat = metric_dict[metric]
            if ak != axis_kind or xv.shape != x_vals.shape or xl != x_labels:
                _warn(
                    f"Metric {metric!r} axis mismatch for group {label!r}; "
                    "skipping this metric for multi-group comparison."
                )
                consistent = False
                break
            group_mats[label] = mat

        if not consistent:
            continue

        print("writing metric stats TSV:", metric)
        _write_metric_stats_tsv(
            metric=metric,
            axis_kind=axis_kind,
            x_vals=x_vals,
            x_labels=x_labels,
            group_mats=group_mats,
            out_dir=out_dir,
        )

        # Post-hoc pairwise tests with Bonferroni correction
        _write_metric_pairwise_tsv(
            metric=metric,
            axis_kind=axis_kind,
            x_vals=x_vals,
            x_labels=x_labels,
            group_mats=group_mats,
            out_dir=out_dir,
        )

        # Also create multi-group overlay plot of individual top learners
        if metric not in METRIC_TITLES:
            _warn(
                f"Metric {metric!r} not recognized in METRIC_TITLES; "
                "skipping multi-group plot."
            )
            continue

        title, y_label, base_stub = METRIC_TITLES[metric]
        # Make the filename stub explicit for multi-group plots
        filename_stub = f"{base_stub}__multi_group"

        _plot_multi_group_overlays(
            title=title,
            y_label=y_label,
            x=x_vals,
            x_labels=x_labels,
            group_matrices=group_mats,
            cfg=cfg,
            customizer=customizer,
            filename_stub=filename_stub,
            group_colors=group_colors,
        )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate individual-strategy overlay plots from exported "
            "individual_strategy_data TSVs (created by analyze.py with "
            "--export-individual-strategy-data), and optionally run "
            "multi-group statistics across multiple TSVs."
        )
    )
    parser.add_argument(
        "tsv",
        nargs="+",
        help="Path(s) to individual_strategy_data.tsv file(s).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help=(
            "Optional group labels corresponding to each TSV. "
            "Defaults to the TSV basenames."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Output directory for images / stats. "
            "Single-TSV mode defaults to <tsv_parent>/imgs_from_tsv. "
            "Multi-TSV mode defaults to <first_tsv_parent>/multi_group_stats."
        ),
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        help="Image format/extension to use in single-TSV mode (default: %(default)s).",
    )

    args = parser.parse_args(argv)
    tsv_paths = [Path(p) for p in args.tsv]

    for p in tsv_paths:
        if not p.is_file():
            raise SystemExit(f"TSV file not found: {p}")

    n_files = len(tsv_paths)

    # Derive labels if needed
    if args.labels and len(args.labels) != n_files:
        raise SystemExit(
            f"--labels must be omitted or have exactly {n_files} entries "
            f"(got {len(args.labels)})."
        )
    if args.labels:
        labels = list(args.labels)
    else:
        labels = [p.stem for p in tsv_paths]

    if n_files == 1:
        tsv_path = tsv_paths[0]
        if args.out_dir is None:
            out_dir = tsv_path.parent / "imgs_from_tsv"
        else:
            out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = _load_tsv(tsv_path)
        if not rows:
            return

        _plot_all_metrics_for_tsv(
            rows,
            tsv_path,
            out_dir=out_dir,
            image_format=args.image_format or "png",
        )
    else:
        # Multi-group stats mode
        if args.out_dir is None:
            out_dir = tsv_paths[0].parent / "multi_group_stats"
        else:
            out_dir = Path(args.out_dir)
        print("comparing groups from TSVs")
        _compare_groups_from_tsvs(
            tsv_paths, labels, out_dir=out_dir, image_format=args.image_format or "png"
        )


if __name__ == "__main__":
    main()
