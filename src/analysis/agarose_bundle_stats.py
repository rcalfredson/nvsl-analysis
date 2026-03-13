from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import sem, t, ttest_ind, ttest_rel

from src.analysis.sli_bundle_utils import load_sli_bundle


VALID_MODES = ("exp", "ctrl", "exp_minus_ctrl")


@dataclass(frozen=True)
class VectorSummary:
    n: int
    mean: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class BundleSelection:
    bundle_path: str
    bundle_label: str
    video_ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    delta: np.ndarray
    training_idx: int
    training_name: str
    bucket_start_idx: int
    bucket_end_idx: int
    bucket_start_min: float
    bucket_end_min: float
    bucket_label: str
    mode: str


@dataclass(frozen=True)
class BundleManifestEntry:
    bundle_path: str
    phenotype: str
    chamber: str
    bundle_label: str


def _validate_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    return mode


def _resolve_training_names(bundle: dict) -> list[str]:
    names = np.asarray(bundle.get("training_names", []), dtype=object).reshape(-1)
    return [str(x) for x in names]


def _resolve_bucket_index(n_buckets: int, bucket_index: int) -> int:
    idx = int(bucket_index)
    if idx < 0:
        idx = n_buckets + idx
    if idx < 0 or idx >= n_buckets:
        raise IndexError(
            f"sync bucket index {bucket_index} is out of range for {n_buckets} buckets"
        )
    return idx


def _resolve_bucket_window(
    n_buckets: int,
    *,
    bucket_index: int = -1,
    bucket_start_index: int | None = None,
    bucket_end_index: int | None = None,
) -> tuple[int, int]:
    if bucket_start_index is None and bucket_end_index is None:
        idx = _resolve_bucket_index(n_buckets, bucket_index)
        return idx, idx
    if bucket_start_index is None or bucket_end_index is None:
        raise ValueError(
            "bucket_start_index and bucket_end_index must be provided together."
        )
    start_idx = _resolve_bucket_index(n_buckets, bucket_start_index)
    end_idx = _resolve_bucket_index(n_buckets, bucket_end_index)
    if end_idx < start_idx:
        raise ValueError(
            f"bucket window is invalid: start={bucket_start_index}, end={bucket_end_index}"
        )
    return start_idx, end_idx


def _series_for_mode(bundle: dict, mode: str) -> np.ndarray:
    mode = _validate_mode(mode)
    if "agarose_ratio_exp" not in bundle:
        raise ValueError(
            f"Bundle {bundle.get('path', '<unknown>')} is missing agarose_ratio_exp."
        )
    exp = np.asarray(bundle["agarose_ratio_exp"], dtype=float)
    if mode == "exp":
        return exp
    if "agarose_ratio_ctrl" not in bundle:
        raise ValueError(
            f"Bundle {bundle.get('path', '<unknown>')} is missing agarose_ratio_ctrl."
        )
    ctrl = np.asarray(bundle["agarose_ratio_ctrl"], dtype=float)
    if mode == "ctrl":
        return ctrl
    return exp - ctrl


def _pre_for_mode(bundle: dict, mode: str) -> np.ndarray:
    mode = _validate_mode(mode)
    if "agarose_pre_ratio_exp" not in bundle:
        raise ValueError(
            f"Bundle {bundle.get('path', '<unknown>')} is missing pre-training agarose keys. "
            "Re-export with --agarose-sli-include-pre."
        )
    exp = np.asarray(bundle["agarose_pre_ratio_exp"], dtype=float)
    if mode == "exp":
        return exp
    if "agarose_pre_ratio_ctrl" not in bundle:
        raise ValueError(
            f"Bundle {bundle.get('path', '<unknown>')} is missing agarose_pre_ratio_ctrl. "
            "Re-export with --agarose-sli-include-pre."
        )
    ctrl = np.asarray(bundle["agarose_pre_ratio_ctrl"], dtype=float)
    if mode == "ctrl":
        return ctrl
    return exp - ctrl


def summarize_vector(x: np.ndarray) -> VectorSummary:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        m = lo = hi = np.nan
    else:
        m = float(np.mean(vals))
        if n == 1:
            lo = hi = np.nan
        else:
            half = float(t.ppf(0.975, df=n - 1) * sem(vals))
            lo = m - half
            hi = m + half
    return VectorSummary(
        n=int(n),
        mean=float(m),
        ci_low=float(lo),
        ci_high=float(hi),
    )


def paired_finite(pre: np.ndarray, post: np.ndarray, video_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = np.asarray(pre, dtype=float).reshape(-1)
    post = np.asarray(post, dtype=float).reshape(-1)
    ids = np.asarray(video_ids, dtype=object).reshape(-1)
    keep = np.isfinite(pre) & np.isfinite(post)
    return pre[keep], post[keep], ids[keep]


def select_agarose_pre_and_post(
    bundle_path: str,
    *,
    mode: str = "exp",
    training_index_1based: int = 2,
    bucket_index: int = -1,
    bucket_start_index: int | None = None,
    bucket_end_index: int | None = None,
    label: str | None = None,
) -> BundleSelection:
    bundle = load_sli_bundle(bundle_path)
    series = _series_for_mode(bundle, mode)
    pre = _pre_for_mode(bundle, mode)
    if series.ndim != 3:
        raise ValueError(
            f"Expected agarose series with shape (n_videos, n_trainings, n_buckets), got {series.shape}"
        )
    n_videos, n_trainings, n_buckets = series.shape
    if pre.ndim != 1 or pre.shape[0] != n_videos:
        raise ValueError(
            f"Expected pre-training agarose array with shape ({n_videos},), got {pre.shape}"
        )

    training_idx = int(training_index_1based) - 1
    if training_idx < 0 or training_idx >= n_trainings:
        raise IndexError(
            f"training index {training_index_1based} is out of range for {n_trainings} trainings"
        )
    bucket_start_idx, bucket_end_idx = _resolve_bucket_window(
        n_buckets,
        bucket_index=bucket_index,
        bucket_start_index=bucket_start_index,
        bucket_end_index=bucket_end_index,
    )
    post = np.nanmean(
        series[:, training_idx, bucket_start_idx : bucket_end_idx + 1], axis=1
    )
    pre_paired, post_paired, ids_paired = paired_finite(
        pre,
        post,
        bundle["video_ids"],
    )

    names = _resolve_training_names(bundle)
    training_name = (
        names[training_idx] if training_idx < len(names) else f"training {training_idx + 1}"
    )
    bucket_len_min = float(bundle["bucket_len_min"])
    bucket_start_min = float((bucket_start_idx + 1) * bucket_len_min)
    bucket_end_min = float((bucket_end_idx + 1) * bucket_len_min)
    bucket_label = (
        f"{bucket_start_idx}"
        if bucket_start_idx == bucket_end_idx
        else f"{bucket_start_idx}-{bucket_end_idx}"
    )

    return BundleSelection(
        bundle_path=bundle_path,
        bundle_label=label or str(bundle["group_label"]),
        video_ids=np.asarray(ids_paired, dtype=object),
        pre=np.asarray(pre_paired, dtype=float),
        post=np.asarray(post_paired, dtype=float),
        delta=np.asarray(post_paired - pre_paired, dtype=float),
        training_idx=training_idx,
        training_name=training_name,
        bucket_start_idx=bucket_start_idx,
        bucket_end_idx=bucket_end_idx,
        bucket_start_min=bucket_start_min,
        bucket_end_min=bucket_end_min,
        bucket_label=bucket_label,
        mode=mode,
    )


def paired_change_test(sel: BundleSelection) -> dict[str, Any]:
    n = int(sel.delta.size)
    if n < 2:
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat, p_value = ttest_rel(sel.pre, sel.post, nan_policy="omit")
    return {
        "n_pairs": n,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "pre": summarize_vector(sel.pre),
        "post": summarize_vector(sel.post),
        "delta": summarize_vector(sel.delta),
    }


def interaction_test_on_deltas(a: BundleSelection, b: BundleSelection) -> dict[str, Any]:
    xa = np.asarray(a.delta, dtype=float)
    xb = np.asarray(b.delta, dtype=float)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    if min(len(xa), len(xb)) < 2:
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat, p_value = ttest_ind(xa, xb, equal_var=False, nan_policy="omit")
    return {
        "n_a": int(len(xa)),
        "n_b": int(len(xb)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "delta_a": summarize_vector(xa),
        "delta_b": summarize_vector(xb),
        "mean_difference": float(np.nanmean(xa) - np.nanmean(xb)),
    }


def holm_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return []

    out = np.full_like(p, np.nan)
    finite_mask = np.isfinite(p)
    if not np.any(finite_mask):
        return out.tolist()

    pf = np.clip(p[finite_mask], 0.0, 1.0)
    m = pf.size
    order = np.argsort(pf)
    p_sorted = pf[order]

    adj_sorted = np.empty_like(p_sorted)
    running_max = 0.0
    for i, pv in enumerate(p_sorted):
        adj = float((m - i) * pv)
        running_max = max(running_max, adj)
        adj_sorted[i] = min(1.0, running_max)

    adj_f = np.empty_like(adj_sorted)
    adj_f[order] = adj_sorted
    out[finite_mask] = adj_f
    return out.tolist()


def read_bundle_manifest(path: str) -> list[BundleManifestEntry]:
    rows: list[BundleManifestEntry] = []
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"bundle_path", "phenotype", "chamber"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest {path} is missing required columns: {sorted(missing)}"
            )
        for i, row in enumerate(reader, start=2):
            bundle_path = str(row.get("bundle_path", "")).strip()
            phenotype = str(row.get("phenotype", "")).strip()
            chamber = str(row.get("chamber", "")).strip()
            bundle_label = str(row.get("bundle_label", "")).strip()
            if not bundle_path or not phenotype or not chamber:
                raise ValueError(
                    f"Manifest {path} has an incomplete row at line {i}: "
                    "bundle_path, phenotype, and chamber are required."
                )
            rows.append(
                BundleManifestEntry(
                    bundle_path=bundle_path,
                    phenotype=phenotype,
                    chamber=chamber,
                    bundle_label=bundle_label or Path(bundle_path).stem,
                )
            )
    if not rows:
        raise ValueError(f"Manifest {path} contains no bundle rows.")
    return rows


def build_delta_table(
    entries: list[BundleManifestEntry],
    *,
    mode: str = "exp",
    training_index_1based: int = 2,
    bucket_index: int = -1,
    bucket_start_index: int | None = None,
    bucket_end_index: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    long_rows_out: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    selections: list[tuple[BundleManifestEntry, BundleSelection]] = []
    reference = None

    for entry in entries:
        sel = select_agarose_pre_and_post(
            entry.bundle_path,
            mode=mode,
            training_index_1based=training_index_1based,
            bucket_index=bucket_index,
            bucket_start_index=bucket_start_index,
            bucket_end_index=bucket_end_index,
            label=entry.bundle_label,
        )
        if reference is None:
            reference = sel
        else:
            if (
                sel.training_idx != reference.training_idx
                or sel.bucket_start_idx != reference.bucket_start_idx
                or sel.bucket_end_idx != reference.bucket_end_idx
            ):
                raise ValueError("All manifest bundles must use the same selected training and bucket.")
            if abs(sel.bucket_start_min - reference.bucket_start_min) > 1e-9:
                raise ValueError("All manifest bundles must share the same bucket timing.")
            if abs(sel.bucket_end_min - reference.bucket_end_min) > 1e-9:
                raise ValueError("All manifest bundles must share the same bucket timing.")

        bundle_stats = paired_change_test(sel)
        bundle_rows.append(
            {
                "row_type": "bundle_summary",
                "bundle_label": entry.bundle_label,
                "bundle_path": entry.bundle_path,
                "phenotype": entry.phenotype,
                "chamber": entry.chamber,
                "mode": sel.mode,
                "training_index": sel.training_idx + 1,
                "training_name": sel.training_name,
                "pre_scope": "paired_to_selected_post",
                "bucket_label": (
                    f"{sel.bucket_start_idx + 1}"
                    if sel.bucket_start_idx == sel.bucket_end_idx
                    else f"{sel.bucket_start_idx + 1}-{sel.bucket_end_idx + 1}"
                ),
                "bucket_start_index": sel.bucket_start_idx + 1,
                "bucket_end_index": sel.bucket_end_idx + 1,
                "bucket_start_min": sel.bucket_start_min,
                "bucket_end_min": sel.bucket_end_min,
                "n_pairs": bundle_stats["n_pairs"],
                "pre_mean": bundle_stats["pre"].mean,
                "pre_ci_low": bundle_stats["pre"].ci_low,
                "pre_ci_high": bundle_stats["pre"].ci_high,
                "post_mean": bundle_stats["post"].mean,
                "post_ci_low": bundle_stats["post"].ci_low,
                "post_ci_high": bundle_stats["post"].ci_high,
                "delta_mean": bundle_stats["delta"].mean,
                "delta_ci_low": bundle_stats["delta"].ci_low,
                "delta_ci_high": bundle_stats["delta"].ci_high,
                "paired_t_stat": bundle_stats["t_stat"],
                "paired_t_p_value": bundle_stats["p_value"],
            }
        )
        for video_id, pre, post, delta in zip(sel.video_ids, sel.pre, sel.post, sel.delta):
            long_rows_out.append(
                {
                    "bundle_label": entry.bundle_label,
                    "bundle_path": entry.bundle_path,
                    "phenotype": entry.phenotype,
                    "chamber": entry.chamber,
                    "video_id": str(video_id),
                    "mode": sel.mode,
                    "training_index": sel.training_idx + 1,
                    "training_name": sel.training_name,
                    "pre_scope": "paired_to_selected_post",
                    "bucket_label": (
                        f"{sel.bucket_start_idx + 1}"
                        if sel.bucket_start_idx == sel.bucket_end_idx
                        else f"{sel.bucket_start_idx + 1}-{sel.bucket_end_idx + 1}"
                    ),
                    "bucket_start_index": sel.bucket_start_idx + 1,
                    "bucket_end_index": sel.bucket_end_idx + 1,
                    "bucket_start_min": sel.bucket_start_min,
                    "bucket_end_min": sel.bucket_end_min,
                    "pre": float(pre),
                    "post": float(post),
                    "delta": float(delta),
                }
            )
        selections.append((entry, sel))

    assert reference is not None
    meta = {
        "mode": reference.mode,
        "training_index": reference.training_idx + 1,
        "training_name": reference.training_name,
        "pre_scope": "paired_to_selected_post",
        "bucket_label": (
            f"{reference.bucket_start_idx + 1}"
            if reference.bucket_start_idx == reference.bucket_end_idx
            else f"{reference.bucket_start_idx + 1}-{reference.bucket_end_idx + 1}"
        ),
        "bucket_start_index": reference.bucket_start_idx + 1,
        "bucket_end_index": reference.bucket_end_idx + 1,
        "bucket_start_min": reference.bucket_start_min,
        "bucket_end_min": reference.bucket_end_min,
    }
    return long_rows_out, bundle_rows, meta


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def two_way_anova_from_rows(
    rows: list[dict[str, Any]],
    *,
    factor_a: str = "phenotype",
    factor_b: str = "chamber",
    value_col: str = "delta",
) -> dict[str, Any]:
    clean_rows = []
    for row in rows:
        val = _safe_float(row.get(value_col))
        if np.isfinite(val):
            clean_rows.append(
                {
                    factor_a: str(row[factor_a]),
                    factor_b: str(row[factor_b]),
                    value_col: val,
                }
            )
    if not clean_rows:
        raise ValueError("No finite delta values available for ANOVA.")

    a_levels = sorted({row[factor_a] for row in clean_rows})
    b_levels = sorted({row[factor_b] for row in clean_rows})
    if len(a_levels) < 2 or len(b_levels) < 2:
        raise ValueError(
            "Two-way ANOVA requires at least two levels for each factor across the manifest."
        )

    cells: dict[tuple[str, str], np.ndarray] = {}
    for a_level in a_levels:
        for b_level in b_levels:
            vals = np.asarray(
                [
                    row[value_col]
                    for row in clean_rows
                    if row[factor_a] == a_level and row[factor_b] == b_level
                ],
                dtype=float,
            )
            if vals.size == 0:
                raise ValueError(
                    "Two-way ANOVA requires at least one observation in every phenotype/chamber cell. "
                    f"Missing cell: {factor_a}={a_level!r}, {factor_b}={b_level!r}."
                )
            cells[(a_level, b_level)] = vals

    all_vals = np.concatenate(list(cells.values()))
    grand_mean = float(np.mean(all_vals))
    n_total = int(all_vals.size)
    df_a = len(a_levels) - 1
    df_b = len(b_levels) - 1
    df_ab = df_a * df_b
    df_error = n_total - len(a_levels) * len(b_levels)
    if df_error <= 0:
        raise ValueError(
            "Two-way ANOVA requires replication within cells (N must exceed number of phenotype/chamber cells)."
        )

    mean_a = {}
    mean_b = {}
    count_a = {}
    count_b = {}
    for a_level in a_levels:
        vals = np.concatenate([cells[(a_level, b)] for b in b_levels])
        mean_a[a_level] = float(np.mean(vals))
        count_a[a_level] = int(vals.size)
    for b_level in b_levels:
        vals = np.concatenate([cells[(a, b_level)] for a in a_levels])
        mean_b[b_level] = float(np.mean(vals))
        count_b[b_level] = int(vals.size)

    ss_a = sum(
        count_a[a_level] * (mean_a[a_level] - grand_mean) ** 2 for a_level in a_levels
    )
    ss_b = sum(
        count_b[b_level] * (mean_b[b_level] - grand_mean) ** 2 for b_level in b_levels
    )
    ss_ab = 0.0
    ss_error = 0.0
    cell_means = {}
    cell_counts = {}
    for key, vals in cells.items():
        cell_mean = float(np.mean(vals))
        cell_means[key] = cell_mean
        cell_counts[key] = int(vals.size)
        a_level, b_level = key
        ss_ab += cell_counts[key] * (
            cell_mean - mean_a[a_level] - mean_b[b_level] + grand_mean
        ) ** 2
        ss_error += float(np.sum((vals - cell_mean) ** 2))
    ss_total = float(np.sum((all_vals - grand_mean) ** 2))

    ms_a = ss_a / df_a if df_a > 0 else np.nan
    ms_b = ss_b / df_b if df_b > 0 else np.nan
    ms_ab = ss_ab / df_ab if df_ab > 0 else np.nan
    ms_error = ss_error / df_error
    f_a = ms_a / ms_error if df_a > 0 and ms_error > 0 else np.nan
    f_b = ms_b / ms_error if df_b > 0 and ms_error > 0 else np.nan
    f_ab = ms_ab / ms_error if df_ab > 0 and ms_error > 0 else np.nan

    def _pval(f_stat: float, dfn: int, dfd: int) -> float:
        if not np.isfinite(f_stat):
            return np.nan
        from scipy.stats import f

        return float(f.sf(f_stat, dfn, dfd))

    terms = [
        {
            "term": factor_a,
            "df": df_a,
            "sum_sq": float(ss_a),
            "mean_sq": float(ms_a),
            "F": float(f_a),
            "p_value": _pval(f_a, df_a, df_error),
        },
        {
            "term": factor_b,
            "df": df_b,
            "sum_sq": float(ss_b),
            "mean_sq": float(ms_b),
            "F": float(f_b),
            "p_value": _pval(f_b, df_b, df_error),
        },
        {
            "term": f"{factor_a}:{factor_b}",
            "df": df_ab,
            "sum_sq": float(ss_ab),
            "mean_sq": float(ms_ab),
            "F": float(f_ab),
            "p_value": _pval(f_ab, df_ab, df_error),
        },
        {
            "term": "residual",
            "df": df_error,
            "sum_sq": float(ss_error),
            "mean_sq": float(ms_error),
            "F": np.nan,
            "p_value": np.nan,
        },
        {
            "term": "total",
            "df": n_total - 1,
            "sum_sq": float(ss_total),
            "mean_sq": np.nan,
            "F": np.nan,
            "p_value": np.nan,
        },
    ]
    return {
        "terms": terms,
        "levels_a": a_levels,
        "levels_b": b_levels,
        "n_total": n_total,
        "cell_counts": {
            f"{a_level}|{b_level}": cell_counts[(a_level, b_level)]
            for a_level in a_levels
            for b_level in b_levels
        },
    }


def pairwise_welch_by_slice(
    rows: list[dict[str, Any]],
    *,
    slice_factor: str,
    compare_factor: str,
    value_col: str = "delta",
) -> list[dict[str, Any]]:
    slice_levels = sorted({str(row[slice_factor]) for row in rows})
    results: list[dict[str, Any]] = []
    raw_ps: list[float] = []
    for slice_level in slice_levels:
        sub = [row for row in rows if str(row[slice_factor]) == slice_level]
        compare_levels = sorted({str(row[compare_factor]) for row in sub})
        for i in range(len(compare_levels)):
            for j in range(i + 1, len(compare_levels)):
                level_a = compare_levels[i]
                level_b = compare_levels[j]
                xa = np.asarray(
                    [
                        _safe_float(row.get(value_col))
                        for row in sub
                        if str(row[compare_factor]) == level_a
                    ],
                    dtype=float,
                )
                xb = np.asarray(
                    [
                        _safe_float(row.get(value_col))
                        for row in sub
                        if str(row[compare_factor]) == level_b
                    ],
                    dtype=float,
                )
                xa = xa[np.isfinite(xa)]
                xb = xb[np.isfinite(xb)]
                if min(len(xa), len(xb)) < 2:
                    t_stat = np.nan
                    p_value = np.nan
                else:
                    t_stat, p_value = ttest_ind(
                        xa, xb, equal_var=False, nan_policy="omit"
                    )
                results.append(
                    {
                        "slice_factor": slice_factor,
                        "slice_level": slice_level,
                        "compare_factor": compare_factor,
                        "level_a": level_a,
                        "level_b": level_b,
                        "n_a": int(len(xa)),
                        "n_b": int(len(xb)),
                        "mean_a": float(np.mean(xa)) if len(xa) else np.nan,
                        "mean_b": float(np.mean(xb)) if len(xb) else np.nan,
                        "mean_diff": float(np.mean(xa) - np.mean(xb))
                        if len(xa) and len(xb)
                        else np.nan,
                        "t_stat": float(t_stat),
                        "p_value": float(p_value),
                    }
                )
                raw_ps.append(float(p_value))
    adj = holm_adjust(raw_ps)
    for row, p_adj in zip(results, adj):
        row["p_value_holm"] = float(p_adj)
    return results


def selective_followups(
    rows: list[dict[str, Any]],
    anova: dict[str, Any],
    *,
    alpha: float = 0.05,
    always_run: bool = False,
) -> list[dict[str, Any]]:
    term_p = {row["term"]: row["p_value"] for row in anova["terms"]}
    run_chamber_within_phenotype = always_run or (
        np.isfinite(term_p.get("phenotype:chamber", np.nan))
        and term_p["phenotype:chamber"] < alpha
    ) or (
        np.isfinite(term_p.get("chamber", np.nan)) and term_p["chamber"] < alpha
    )
    run_phenotype_within_chamber = always_run or (
        np.isfinite(term_p.get("phenotype:chamber", np.nan))
        and term_p["phenotype:chamber"] < alpha
    ) or (
        np.isfinite(term_p.get("phenotype", np.nan)) and term_p["phenotype"] < alpha
    )

    out: list[dict[str, Any]] = []
    if run_chamber_within_phenotype:
        for row in pairwise_welch_by_slice(
            rows, slice_factor="phenotype", compare_factor="chamber"
        ):
            row["triggered_by"] = (
                "interaction"
                if np.isfinite(term_p.get("phenotype:chamber", np.nan))
                and term_p["phenotype:chamber"] < alpha
                else "chamber_main_effect"
            )
            out.append(row)
    if run_phenotype_within_chamber:
        for row in pairwise_welch_by_slice(
            rows, slice_factor="chamber", compare_factor="phenotype"
        ):
            row["triggered_by"] = (
                "interaction"
                if np.isfinite(term_p.get("phenotype:chamber", np.nan))
                and term_p["phenotype:chamber"] < alpha
                else "phenotype_main_effect"
            )
            out.append(row)
    return out


def compare_agarose_bundles(
    bundle_a_path: str,
    bundle_b_path: str,
    *,
    mode: str = "exp",
    training_index_1based: int = 2,
    bucket_index: int = -1,
    bucket_start_index: int | None = None,
    bucket_end_index: int | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
) -> dict[str, Any]:
    sel_a = select_agarose_pre_and_post(
        bundle_a_path,
        mode=mode,
        training_index_1based=training_index_1based,
        bucket_index=bucket_index,
        bucket_start_index=bucket_start_index,
        bucket_end_index=bucket_end_index,
        label=label_a,
    )
    sel_b = select_agarose_pre_and_post(
        bundle_b_path,
        mode=mode,
        training_index_1based=training_index_1based,
        bucket_index=bucket_index,
        bucket_start_index=bucket_start_index,
        bucket_end_index=bucket_end_index,
        label=label_b,
    )
    if (
        sel_a.training_idx != sel_b.training_idx
        or sel_a.bucket_start_idx != sel_b.bucket_start_idx
        or sel_a.bucket_end_idx != sel_b.bucket_end_idx
    ):
        raise ValueError("Selected bundle comparisons do not refer to the same training/bucket.")
    if not np.isfinite(sel_a.bucket_start_min) or not np.isfinite(sel_b.bucket_start_min):
        raise ValueError("Selected bundles are missing finite bucket timing metadata.")
    if abs(sel_a.bucket_start_min - sel_b.bucket_start_min) > 1e-9:
        raise ValueError(
            "Selected bundles disagree on bucket timing; cannot compare different sync-bucket definitions."
        )
    if not np.isfinite(sel_a.bucket_end_min) or not np.isfinite(sel_b.bucket_end_min):
        raise ValueError("Selected bundles are missing finite bucket timing metadata.")
    if abs(sel_a.bucket_end_min - sel_b.bucket_end_min) > 1e-9:
        raise ValueError(
            "Selected bundles disagree on bucket timing; cannot compare different sync-bucket definitions."
        )

    within_a = paired_change_test(sel_a)
    within_b = paired_change_test(sel_b)
    interaction = interaction_test_on_deltas(sel_a, sel_b)
    return {
        "bundle_a": sel_a,
        "bundle_b": sel_b,
        "within_a": within_a,
        "within_b": within_b,
        "interaction": interaction,
    }


def summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, sel_key in (("within_a", "bundle_a"), ("within_b", "bundle_b")):
        sel = result[sel_key]
        stats = result[key]
        for phase_name, summary in (
            ("pre", stats["pre"]),
            ("post", stats["post"]),
            ("delta", stats["delta"]),
        ):
            rows.append(
                {
                    "row_type": "bundle_summary",
                    "bundle_label": sel.bundle_label,
                    "bundle_path": sel.bundle_path,
                    "mode": sel.mode,
                    "training_index": sel.training_idx + 1,
                    "training_name": sel.training_name,
                    "pre_scope": "paired_to_selected_post",
                    "bucket_label": (
                        f"{sel.bucket_start_idx + 1}"
                        if sel.bucket_start_idx == sel.bucket_end_idx
                        else f"{sel.bucket_start_idx + 1}-{sel.bucket_end_idx + 1}"
                    ),
                    "bucket_start_index": sel.bucket_start_idx + 1,
                    "bucket_end_index": sel.bucket_end_idx + 1,
                    "bucket_start_min": sel.bucket_start_min,
                    "bucket_end_min": sel.bucket_end_min,
                    "phase": phase_name,
                    "n": summary.n,
                    "mean": summary.mean,
                    "ci_low": summary.ci_low,
                    "ci_high": summary.ci_high,
                    "test": "paired_t" if phase_name == "delta" else "",
                    "statistic": stats["t_stat"] if phase_name == "delta" else np.nan,
                    "p_value": stats["p_value"] if phase_name == "delta" else np.nan,
                }
            )

    interaction = result["interaction"]
    rows.append(
        {
            "row_type": "interaction",
            "bundle_label": f"{result['bundle_a'].bundle_label} vs {result['bundle_b'].bundle_label}",
            "bundle_path": "",
            "mode": result["bundle_a"].mode,
            "training_index": result["bundle_a"].training_idx + 1,
            "training_name": result["bundle_a"].training_name,
            "pre_scope": "paired_to_selected_post",
            "bucket_label": (
                f"{result['bundle_a'].bucket_start_idx + 1}"
                if result["bundle_a"].bucket_start_idx == result["bundle_a"].bucket_end_idx
                else f"{result['bundle_a'].bucket_start_idx + 1}-{result['bundle_a'].bucket_end_idx + 1}"
            ),
            "bucket_start_index": result["bundle_a"].bucket_start_idx + 1,
            "bucket_end_index": result["bundle_a"].bucket_end_idx + 1,
            "bucket_start_min": result["bundle_a"].bucket_start_min,
            "bucket_end_min": result["bundle_a"].bucket_end_min,
            "phase": "delta_difference",
            "n": interaction["n_a"] + interaction["n_b"],
            "mean": interaction["mean_difference"],
            "ci_low": np.nan,
            "ci_high": np.nan,
            "test": "welch_t_on_delta",
            "statistic": interaction["t_stat"],
            "p_value": interaction["p_value"],
        }
    )
    return rows


def long_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sel_key in ("bundle_a", "bundle_b"):
        sel = result[sel_key]
        for video_id, pre, post, delta in zip(sel.video_ids, sel.pre, sel.post, sel.delta):
            rows.append(
                {
                    "bundle_label": sel.bundle_label,
                    "bundle_path": sel.bundle_path,
                    "video_id": str(video_id),
                    "mode": sel.mode,
                    "training_index": sel.training_idx + 1,
                    "training_name": sel.training_name,
                    "pre_scope": "paired_to_selected_post",
                    "bucket_label": (
                        f"{sel.bucket_start_idx + 1}"
                        if sel.bucket_start_idx == sel.bucket_end_idx
                        else f"{sel.bucket_start_idx + 1}-{sel.bucket_end_idx + 1}"
                    ),
                    "bucket_start_index": sel.bucket_start_idx + 1,
                    "bucket_end_index": sel.bucket_end_idx + 1,
                    "bucket_start_min": sel.bucket_start_min,
                    "bucket_end_min": sel.bucket_end_min,
                    "pre": float(pre),
                    "post": float(post),
                    "delta": float(delta),
                }
            )
    return rows
