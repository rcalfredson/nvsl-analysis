from __future__ import annotations

import copy
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_ind

from src.analysis.episode_filters import (
    EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY,
    EPISODE_TYPE_INNER_EXIT_REENTRY,
    min_episode_count_for_type,
)
from src.analysis.multiple_comparisons import holm_adjust
from src.analysis.sli_tools import select_fractional_groups
from src.analysis.sync_bucket_presence_filters import (
    exp_target_sync_bucket_filter_result,
)
from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
)
from src.exporting.turnback_home_vector_alignment_sli_bundle import (
    _collect_per_fly_values as _collect_alignment_values,
)
from src.plotting.between_reward_tortuosity_mean_swarm import (
    BetweenRewardTortuosityMeanSwarmConfig,
    BetweenRewardTortuosityMeanSwarmPlotter,
)
from src.plotting.between_reward_segment_binning import (
    sync_bucket_window,
    wall_contact_mask,
)
from src.plotting.btw_rwd_return_leg_dist_collectors import (
    ReturnLegDistPerFlyCollector,
)
from src.plotting.plot_customizer import PlotCustomizer


METRIC_ORDER = (
    "dual_circle_turnback_ratio",
    "home_vector_alignment",
    "between_reward_tortuosity",
    "com_distance_to_reward_center",
    "return_leg_distance",
)

METRIC_LABELS = {
    "dual_circle_turnback_ratio": "Dual-circle turnback ratio",
    "home_vector_alignment": "Home-vector alignment, cos(theta)",
    "between_reward_tortuosity": "Between-reward tortuosity",
    "com_distance_to_reward_center": "COM distance to reward center (mm)",
    "return_leg_distance": "Return-leg distance traveled (mm)",
}


@dataclass(frozen=True)
class MetricObservation:
    unit_id: str
    sli: float
    cohort: str
    metric: str
    value: float
    n_events: int
    eligible: bool
    exclusion_reason: str


def _canonical_unit_id(va) -> str:
    fn = str(getattr(va, "fn", "unknown_video"))
    try:
        fly_id = int(getattr(va, "f", 0) or 0)
    except (TypeError, ValueError):
        fly_id = 0
    return f"{fn}::f{fly_id}"


def _mean_ci(values, conf: float = 0.95) -> tuple[float, float, float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(np.mean(x))
    if n < 2:
        return mean, math.nan, math.nan, n
    sem = float(np.std(x, ddof=1) / math.sqrt(n))
    crit = float(student_t.ppf((1.0 + float(conf)) / 2.0, n - 1))
    return mean, mean - crit * sem, mean + crit * sem, n


def _welch_difference(
    strong_values, weak_values, conf: float = 0.95
) -> dict[str, float | str]:
    strong = np.asarray(strong_values, dtype=float)
    weak = np.asarray(weak_values, dtype=float)
    strong = strong[np.isfinite(strong)]
    weak = weak[np.isfinite(weak)]
    out: dict[str, float | str] = {
        "difference_strong_minus_weak": math.nan,
        "difference_ci_lo": math.nan,
        "difference_ci_hi": math.nan,
        "test": "Welch t-test",
        "statistic": math.nan,
        "p_value": math.nan,
    }
    if strong.size == 0 or weak.size == 0:
        return out

    difference = float(np.mean(strong) - np.mean(weak))
    out["difference_strong_minus_weak"] = difference
    if strong.size < 2 or weak.size < 2:
        return out

    result = ttest_ind(strong, weak, equal_var=False, nan_policy="omit")
    out["statistic"] = float(result.statistic)
    out["p_value"] = float(result.pvalue)

    var_strong = float(np.var(strong, ddof=1))
    var_weak = float(np.var(weak, ddof=1))
    term_strong = var_strong / strong.size
    term_weak = var_weak / weak.size
    se2 = term_strong + term_weak
    if se2 <= 0 or not np.isfinite(se2):
        return out
    df_den = (
        (term_strong**2) / (strong.size - 1)
        + (term_weak**2) / (weak.size - 1)
    )
    if df_den <= 0 or not np.isfinite(df_den):
        return out
    df = (se2**2) / df_den
    crit = float(student_t.ppf((1.0 + float(conf)) / 2.0, df))
    half_width = crit * math.sqrt(se2)
    out["difference_ci_lo"] = difference - half_width
    out["difference_ci_hi"] = difference + half_width
    return out


def summarize_observations(
    observations: list[MetricObservation], *, ci_conf: float = 0.95
) -> list[dict]:
    rows = []
    for metric in METRIC_ORDER:
        strong = [
            row.value
            for row in observations
            if row.metric == metric and row.cohort == "strong" and row.eligible
        ]
        weak = [
            row.value
            for row in observations
            if row.metric == metric and row.cohort == "weak" and row.eligible
        ]
        strong_mean, strong_lo, strong_hi, strong_n = _mean_ci(
            strong, conf=ci_conf
        )
        weak_mean, weak_lo, weak_hi, weak_n = _mean_ci(weak, conf=ci_conf)
        comparison = _welch_difference(strong, weak, conf=ci_conf)
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "strong_n": strong_n,
                "strong_mean": strong_mean,
                "strong_ci_lo": strong_lo,
                "strong_ci_hi": strong_hi,
                "weak_n": weak_n,
                "weak_mean": weak_mean,
                "weak_ci_lo": weak_lo,
                "weak_ci_hi": weak_hi,
                **comparison,
            }
        )

    adjusted = holm_adjust([float(row["p_value"]) for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["p_value_holm"] = adjusted_p
    return rows


def _record_metric(
    observations,
    *,
    va,
    sli,
    cohort,
    metric,
    value,
    n_events,
    minimum_events,
    target_result,
):
    reason = ""
    eligible = bool(np.isfinite(value) and int(n_events) >= int(minimum_events))
    if not target_result.eligible:
        eligible = False
        reason = str(target_result.reason)
    elif int(n_events) < int(minimum_events):
        reason = "insufficient_episodes"
    elif not np.isfinite(value):
        reason = "metric_unavailable"
    observations.append(
        MetricObservation(
            unit_id=_canonical_unit_id(va),
            sli=float(sli),
            cohort=str(cohort),
            metric=str(metric),
            value=float(value) if np.isfinite(value) else math.nan,
            n_events=int(n_events),
            eligible=eligible,
            exclusion_reason=reason,
        )
    )


def _selected_vas_and_sli(vas, opts):
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    sli, _sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    sli = np.asarray(sli, dtype=float)
    top_fraction = float(getattr(opts, "top_sli_fraction", 0.2))
    bottom_fraction = float(getattr(opts, "bottom_sli_fraction", 0.5))
    bottom, top = select_fractional_groups(
        pd.Series(sli),
        top_fraction=top_fraction,
        bottom_fraction=bottom_fraction,
    )
    top = [] if top is None else list(top)
    bottom = [] if bottom is None else list(bottom)
    if not top or not bottom:
        rankable_n = int(np.count_nonzero(np.isfinite(sli)))
        raise ValueError(
            "learner metric table requires non-empty strong and weak cohorts "
            f"(rankable flies: {rankable_n})"
        )
    selected = [
        (idx, vas_ok[idx], "strong") for idx in top
    ] + [
        (idx, vas_ok[idx], "weak") for idx in bottom
    ]
    return vas_ok, sli, selected


def _metric_window(opts) -> tuple[int, int, int, list[int]]:
    training = max(1, int(getattr(opts, "learner_metric_table_training", 2)))
    skip = max(
        0, int(getattr(opts, "learner_metric_table_skip_first_sync_buckets", 1))
    )
    keep = max(
        0, int(getattr(opts, "learner_metric_table_keep_first_sync_buckets", 4))
    )
    indices = list(range(skip, skip + keep)) if keep > 0 else []
    return training, training - 1, skip, indices


def _collect_array_metrics(selected, sli, opts, observations):
    training, training_idx, _skip, bucket_indices = _metric_window(opts)
    min_turnback = min_episode_count_for_type(opts, EPISODE_TYPE_INNER_EXIT_REENTRY)
    min_between = min_episode_count_for_type(
        opts, EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY
    )
    warned_missing_wc = [False]

    for idx, va, cohort in selected:
        target = exp_target_sync_bucket_filter_result(va, opts)

        counts = getattr(va, "reward_turnback_dual_circle_counts", {}) or {}
        turn = np.asarray(counts.get("turnback", []))
        total = np.asarray(counts.get("total", []))
        n_events = 0
        value = math.nan
        if (
            turn.ndim == 3
            and total.shape == turn.shape
            and training_idx < turn.shape[0]
        ):
            success = 0
            for b_idx in bucket_indices:
                if b_idx >= turn.shape[2]:
                    continue
                success += int(turn[training_idx, 0, b_idx])
                n_events += int(total[training_idx, 0, b_idx])
            if n_events > 0:
                value = float(success / n_events)
        _record_metric(
            observations,
            va=va,
            sli=sli[idx],
            cohort=cohort,
            metric="dual_circle_turnback_ratio",
            value=value,
            n_events=n_events,
            minimum_events=min_turnback,
            target_result=target,
        )

        value = math.nan
        n_events = 0
        trns = getattr(va, "trns", [])
        trx = getattr(va, "trx", [])
        if training_idx < len(trns) and trx and not trx[0].bad():
            trn = trns[training_idx]
            fi, df, n_buckets, complete = sync_bucket_window(
                va,
                trn,
                t_idx=training_idx,
                f=0,
                skip_first=bucket_indices[0] if bucket_indices else 0,
                keep_first=len(bucket_indices),
                use_exclusion_mask=False,
            )
            wc = wall_contact_mask(
                opts,
                va,
                0,
                fi=fi,
                n_frames=max(1, n_buckets * df),
                log_tag="learner_metric_table_com",
                warned_missing_wc=warned_missing_wc,
            )
            values = [
                float(seg.mag_mm)
                for seg in va._iter_between_reward_segment_com(
                    trn,
                    0,
                    fi=fi,
                    df=df,
                    n_buckets=n_buckets,
                    complete=complete,
                    relative_to_reward=True,
                    per_segment_min_meddist_mm=float(
                        getattr(opts, "com_per_segment_min_meddist_mm", 0.0) or 0.0
                    ),
                    exclude_wall=bool(
                        getattr(opts, "com_exclude_wall_contact", False)
                    ),
                    wc=wc,
                    exclude_reward_endpoints=bool(
                        getattr(opts, "btw_rwd_com_exclude_reward_endpoints", False)
                    ),
                    debug=False,
                    yield_skips=False,
                )
                if np.isfinite(seg.mag_mm)
            ]
            n_events = len(values)
            if values:
                value = float(np.mean(values))
        _record_metric(
            observations,
            va=va,
            sli=sli[idx],
            cohort=cohort,
            metric="com_distance_to_reward_center",
            value=value,
            n_events=n_events,
            minimum_events=min_between,
            target_result=target,
        )


def _collect_return_leg(selected, sli_by_id, opts, observations):
    training, training_idx, skip, bucket_indices = _metric_window(opts)
    keep = len(bucket_indices)
    min_between = min_episode_count_for_type(
        opts, EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY
    )
    for cohort in ("strong", "weak"):
        cohort_vas = [va for _idx, va, name in selected if name == cohort]
        eligible_vas = [
            va
            for va in cohort_vas
            if exp_target_sync_bucket_filter_result(va, opts).eligible
        ]
        collector = ReturnLegDistPerFlyCollector()
        collector.vas = eligible_vas
        collector.opts = opts
        collector.cfg = SimpleNamespace(
            skip_first_sync_buckets=skip,
            keep_first_sync_buckets=keep,
        )
        panels = collector._collect_return_leg_aggregates_by_training_per_fly()
        values = {}
        if training_idx < len(panels):
            values = {
                str(uid): (total, count)
                for uid, total, count in panels[training_idx]
            }
        for va in cohort_vas:
            target = exp_target_sync_bucket_filter_result(va, opts)
            key = collector._unit_id(va, f=0)
            total, count = values.get(str(key), (math.nan, 0))
            value = (
                float(total / count)
                if count > 0 and np.isfinite(total)
                else math.nan
            )
            _record_metric(
                observations,
                va=va,
                sli=sli_by_id[_canonical_unit_id(va)],
                cohort=cohort,
                metric="return_leg_distance",
                value=value,
                n_events=count,
                minimum_events=min_between,
                target_result=target,
            )


def _collect_tortuosity(selected, sli_by_id, opts, gls, observations):
    training, training_idx, skip, bucket_indices = _metric_window(opts)
    keep = len(bucket_indices)
    min_between = min_episode_count_for_type(
        opts, EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY
    )
    for cohort in ("strong", "weak"):
        cohort_vas = [va for _idx, va, name in selected if name == cohort]
        eligible_vas = [
            va
            for va in cohort_vas
            if exp_target_sync_bucket_filter_result(va, opts).eligible
        ]
        cfg = BetweenRewardTortuosityMeanSwarmConfig(
            out_file="",
            trainings=[training],
            skip_first_sync_buckets=skip,
            keep_first_sync_buckets=keep,
            metric_mode="path_over_max_radius",
            segment_scope="full",
            min_segments_per_fly=min_between,
        )
        plotter = BetweenRewardTortuosityMeanSwarmPlotter(
            eligible_vas, opts, gls, PlotCustomizer(), cfg
        )
        panels = plotter._collect_tortuosity_aggregates_by_training_per_fly()
        values = {}
        if training_idx < len(panels):
            values = {
                str(uid): (total, count)
                for uid, total, count in panels[training_idx]
            }
        for va in cohort_vas:
            target = exp_target_sync_bucket_filter_result(va, opts)
            key = plotter._unit_id(va, f=0)
            total, count = values.get(str(key), (math.nan, 0))
            value = (
                float(total / count)
                if count > 0 and np.isfinite(total)
                else math.nan
            )
            _record_metric(
                observations,
                va=va,
                sli=sli_by_id[_canonical_unit_id(va)],
                cohort=cohort,
                metric="between_reward_tortuosity",
                value=value,
                n_events=count,
                minimum_events=min_between,
                target_result=target,
            )


def _collect_alignment(selected, sli_by_id, opts, observations):
    training, _training_idx, skip, bucket_indices = _metric_window(opts)
    keep = len(bucket_indices)
    min_turnback = min_episode_count_for_type(opts, EPISODE_TYPE_INNER_EXIT_REENTRY)
    for cohort in ("strong", "weak"):
        cohort_vas = [va for _idx, va, name in selected if name == cohort]
        targets = np.asarray(
            [
                exp_target_sync_bucket_filter_result(va, opts).eligible
                for va in cohort_vas
            ],
            dtype=bool,
        )
        ids, values, counts, _meta = _collect_alignment_values(
            cohort_vas,
            opts,
            selected_trainings=[training - 1],
            skip_first=skip,
            keep_first=keep,
            last_sync_buckets=0,
            target_sync_bucket_eligible=targets,
            apply_min_episode_filter=False,
        )
        collected = {
            str(uid): (float(value), int(count))
            for uid, value, count in zip(ids, values, counts)
        }
        for i, va in enumerate(cohort_vas):
            target = exp_target_sync_bucket_filter_result(va, opts)
            value, count = collected.get(_alignment_unit_id(va, i), (math.nan, 0))
            _record_metric(
                observations,
                va=va,
                sli=sli_by_id[_canonical_unit_id(va)],
                cohort=cohort,
                metric="home_vector_alignment",
                value=value,
                n_events=count,
                minimum_events=min_turnback,
                target_result=target,
            )


def _alignment_unit_id(va, local_index: int) -> str:
    import os.path

    video_id = os.path.splitext(
        os.path.basename(str(getattr(va, "fn", f"video_{local_index}")))
    )[0]
    try:
        fly_id = int(getattr(va, "f"))
    except (TypeError, ValueError):
        fly_id = 0
    return f"{video_id}:fly{fly_id}"


def collect_learner_metric_observations(vas, opts, gls) -> tuple[list, dict]:
    vas_ok, sli, selected = _selected_vas_and_sli(vas, opts)
    if not selected:
        raise ValueError("learner metric table selected no strong or weak learners")

    report_opts = copy.copy(opts)
    report_opts.require_exp_target_sync_bucket = bool(
        getattr(opts, "learner_metric_table_require_sb5", True)
    )
    training, _training_idx, skip, bucket_indices = _metric_window(report_opts)
    report_opts.exp_target_sync_bucket_filter_training = training
    report_opts.exp_target_sync_bucket_filter_sync_bucket = (
        skip + len(bucket_indices) if bucket_indices else skip + 1
    )
    report_opts.turnback_home_vector_alignment_inner_radius_mm = float(
        getattr(opts, "learner_metric_table_alignment_inner_radius_mm", 3.0)
    )
    report_opts.turnback_home_vector_alignment_outer_radius_mm = float(
        getattr(opts, "learner_metric_table_alignment_outer_radius_mm", 5.0)
    )
    report_opts.turnback_home_vector_alignment_inner_delta_mm = None
    report_opts.turnback_home_vector_alignment_outer_delta_mm = None

    observations: list[MetricObservation] = []
    _collect_array_metrics(selected, sli, report_opts, observations)
    sli_by_id = {
        _canonical_unit_id(va): float(sli[idx]) for idx, va, _cohort in selected
    }
    _collect_alignment(selected, sli_by_id, report_opts, observations)
    _collect_tortuosity(selected, sli_by_id, report_opts, gls, observations)
    _collect_return_leg(selected, sli_by_id, report_opts, observations)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rankable_n": int(np.count_nonzero(np.isfinite(sli))),
        "strong_selected_n": sum(1 for _idx, _va, c in selected if c == "strong"),
        "weak_selected_n": sum(1 for _idx, _va, c in selected if c == "weak"),
        "cohort_definition": {
            "training": int(getattr(opts, "best_worst_trn", 2)),
            "use_training_mean": bool(getattr(opts, "sli_use_training_mean", False)),
            "skip_first_sync_buckets": int(
                getattr(opts, "sli_select_skip_first_sync_buckets", 0) or 0
            ),
            "keep_first_sync_buckets": int(
                getattr(opts, "sli_select_keep_first_sync_buckets", 0) or 0
            ),
            "min_valid_sync_buckets": int(
                getattr(opts, "sli_min_valid_sync_buckets", 3)
            ),
            "top_fraction": float(getattr(opts, "top_sli_fraction", 0.2)),
            "bottom_fraction": float(getattr(opts, "bottom_sli_fraction", 0.5)),
        },
        "metric_window": {
            "training": training,
            "skip_first_sync_buckets": skip,
            "keep_first_sync_buckets": len(bucket_indices),
            "sync_buckets_1_based": [idx + 1 for idx in bucket_indices],
        },
        "eligibility": {
            "require_final_selected_sync_bucket": bool(
                report_opts.require_exp_target_sync_bucket
            ),
            "min_between_reward_trajectories": min_episode_count_for_type(
                report_opts, EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY
            ),
            "min_turnback_episodes": min_episode_count_for_type(
                report_opts, EPISODE_TYPE_INNER_EXIT_REENTRY
            ),
        },
        "metric_definitions": {
            "dual_circle_turnback_ratio": {
                "inner_delta_mm": float(
                    getattr(opts, "learner_metric_table_dctr_inner_delta_mm", 4.0)
                ),
                "outer_delta_mm": float(
                    getattr(opts, "learner_metric_table_dctr_outer_delta_mm", 8.0)
                ),
                "aggregation": "sum(successes) / sum(qualifying exits)",
            },
            "home_vector_alignment": {
                "inner_radius_mm": float(
                    report_opts.turnback_home_vector_alignment_inner_radius_mm
                ),
                "outer_radius_mm": float(
                    report_opts.turnback_home_vector_alignment_outer_radius_mm
                ),
                "aggregation": "mean cos(theta) across successful re-entry episodes",
            },
            "between_reward_tortuosity": {
                "mode": "path_over_max_radius",
                "scope": "full",
                "aggregation": "episode-pooled mean per fly",
            },
            "com_distance_to_reward_center": {
                "aggregation": "mean across pooled between-reward episodes"
            },
            "return_leg_distance": {
                "scope": "maximum-distance frame through reward re-entry",
                "aggregation": "episode-pooled mean per fly",
            },
        },
        "statistics": {
            "unit": "fly",
            "difference": "strong minus weak",
            "test": "independent-group Welch t-test",
            "multiple_comparisons": "Holm adjustment across five metrics",
            "ci_confidence": 0.95,
        },
    }
    return observations, metadata


def _output_prefix(raw_path: str) -> str:
    path = str(raw_path)
    for suffix in ("_summary.csv", ".csv", ".md", ".json"):
        if path.lower().endswith(suffix):
            return path[: -len(suffix)]
    return path


def _fmt(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.6g}"


def export_learner_metric_table(vas, opts, gls, out_path: str) -> dict[str, str]:
    observations, metadata = collect_learner_metric_observations(vas, opts, gls)
    summary = summarize_observations(observations)
    prefix = _output_prefix(out_path)
    paths = {
        "summary_csv": f"{prefix}_summary.csv",
        "per_fly_csv": f"{prefix}_per_fly.csv",
        "metadata_json": f"{prefix}_metadata.json",
        "summary_md": f"{prefix}_summary.md",
    }
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    with open(paths["summary_csv"], "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    per_fly_fields = list(asdict(observations[0]).keys())
    with open(paths["per_fly_csv"], "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=per_fly_fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in observations)

    metadata["outputs"] = paths
    with open(paths["metadata_json"], "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
        fh.write("\n")

    with open(paths["summary_md"], "w", encoding="utf-8") as fh:
        fh.write("| Metric | Strong learners | Weak learners | Strong vs. weak |\n")
        fh.write("|---|---:|---:|---:|\n")
        for row in summary:
            strong = (
                f"{_fmt(row['strong_mean'])} "
                f"[{_fmt(row['strong_ci_lo'])}, {_fmt(row['strong_ci_hi'])}], "
                f"n={row['strong_n']}"
            )
            weak = (
                f"{_fmt(row['weak_mean'])} "
                f"[{_fmt(row['weak_ci_lo'])}, {_fmt(row['weak_ci_hi'])}], "
                f"n={row['weak_n']}"
            )
            difference = (
                f"{_fmt(row['difference_strong_minus_weak'])} "
                f"[{_fmt(row['difference_ci_lo'])}, {_fmt(row['difference_ci_hi'])}]; "
                f"Welch p={_fmt(row['p_value'])}; "
                f"Holm p={_fmt(row['p_value_holm'])}"
            )
            fh.write(
                f"| {row['metric_label']} | {strong} | {weak} | {difference} |\n"
            )

    print(
        "[learner-metric-table] wrote "
        + ", ".join(paths.values())
    )
    return paths
