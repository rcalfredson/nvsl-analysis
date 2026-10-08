from __future__ import annotations

import copy
import csv
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
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
from src.analysis.sli_tools import select_fractional_groups
from src.analysis.learner_metric_stats import (
    analyze_repeated_metrics,
    adjust_report_families,
    COHORT_AUDIT_FIELDS,
    MODEL_INPUT_FIELDS,
    MODEL_FORMULA,
)
from src.analysis.sync_bucket_presence_filters import (
    exp_target_sync_bucket_filter_result,
)
from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
)
from src.exporting.turnback_home_vector_alignment_sli_bundle import (
    _collect_per_fly_values as _collect_alignment_values,
)
from src.exporting.turnback_excursion_bin_sli_bundle import _compute_pair_curves
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

DEFAULT_CIRCLE_PAIRS_MM = ((3.0, 5.0), (8.0, 10.0), (13.0, 15.0))

REPEATED_METRICS = (
    "dual_circle_turnback_ratio",
    "home_vector_alignment",
)

STANDALONE_METRICS = (
    "between_reward_tortuosity",
    "com_distance_to_reward_center",
    "return_leg_distance",
)

METRIC_ORDER = REPEATED_METRICS + STANDALONE_METRICS

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
    inner_radius_mm: float
    outer_radius_mm: float
    value: float
    n_events: int
    successes: float
    eligible: bool
    exclusion_reason: str


def parse_circle_pairs_mm(raw) -> tuple[tuple[float, float], ...]:
    if raw is None or not str(raw).strip():
        return DEFAULT_CIRCLE_PAIRS_MM
    pairs = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                "--learner-metric-table-circle-pairs-mm entries must use "
                "inner:outer format, e.g. '3:5,8:10,13:15'"
            )
        inner_raw, outer_raw = token.split(":", 1)
        try:
            inner = float(inner_raw.strip())
            outer = float(outer_raw.strip())
        except ValueError as exc:
            raise ValueError(
                "--learner-metric-table-circle-pairs-mm values must be numeric"
            ) from exc
        if not (np.isfinite(inner) and np.isfinite(outer)):
            raise ValueError(
                "--learner-metric-table-circle-pairs-mm values must be finite"
            )
        if inner < 0.0 or outer <= inner:
            raise ValueError(
                "learner-table circle pairs require 0 <= inner radius < outer radius"
            )
        pairs.append((float(inner), float(outer)))
    if not pairs:
        raise ValueError(
            "--learner-metric-table-circle-pairs-mm must contain at least one pair"
        )
    if len(set(pairs)) != len(pairs):
        raise ValueError(
            "--learner-metric-table-circle-pairs-mm must not contain duplicate pairs"
        )
    return tuple(pairs)


def _circle_pair_label(inner: float, outer: float) -> str:
    return f"{inner:g}/{outer:g} mm"


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
    df_den = (term_strong**2) / (strong.size - 1) + (term_weak**2) / (weak.size - 1)
    if df_den <= 0 or not np.isfinite(df_den):
        return out
    df = (se2**2) / df_den
    crit = float(student_t.ppf((1.0 + float(conf)) / 2.0, df))
    half_width = crit * math.sqrt(se2)
    out["difference_ci_lo"] = difference - half_width
    out["difference_ci_hi"] = difference + half_width
    return out


def _summary_specs(circle_pairs):
    for metric in REPEATED_METRICS:
        for inner, outer in circle_pairs:
            yield metric, float(inner), float(outer)
    for metric in STANDALONE_METRICS:
        yield metric, math.nan, math.nan


def summarize_observations(
    observations: list[MetricObservation],
    *,
    circle_pairs=DEFAULT_CIRCLE_PAIRS_MM,
    ci_conf: float = 0.95,
) -> list[dict]:
    rows = []
    for metric, inner, outer in _summary_specs(circle_pairs):

        def matches(obs):
            if obs.metric != metric:
                return False
            if metric in REPEATED_METRICS:
                return bool(
                    np.isclose(obs.inner_radius_mm, inner)
                    and np.isclose(obs.outer_radius_mm, outer)
                )
            return True

        strong = [
            row.value
            for row in observations
            if matches(row) and row.cohort == "strong" and row.eligible
        ]
        weak = [
            row.value
            for row in observations
            if matches(row) and row.cohort == "weak" and row.eligible
        ]
        strong_mean, strong_lo, strong_hi, strong_n = _mean_ci(strong, conf=ci_conf)
        weak_mean, weak_lo, weak_hi, weak_n = _mean_ci(weak, conf=ci_conf)
        label = METRIC_LABELS[metric]
        if metric in REPEATED_METRICS:
            label = f"{label} ({_circle_pair_label(inner, outer)})"
            comp = {
                "difference_strong_minus_weak": (
                    strong_mean - weak_mean
                    if np.isfinite(strong_mean) and np.isfinite(weak_mean)
                    else math.nan
                ),
                "difference_ci_lo": math.nan,
                "difference_ci_hi": math.nan,
                "test": "Not performed",
                "statistic": math.nan,
                "p_value": math.nan,
            }
        else:
            comp = _welch_difference(strong, weak, conf=ci_conf)
        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "inner_radius_mm": inner,
                "outer_radius_mm": outer,
                "strong_n": strong_n,
                "strong_mean": strong_mean,
                "strong_ci_lo": strong_lo,
                "strong_ci_hi": strong_hi,
                "weak_n": weak_n,
                "weak_mean": weak_mean,
                "weak_ci_lo": weak_lo,
                "weak_ci_hi": weak_hi,
                **comp,
            }
        )
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
    inner_radius_mm=math.nan,
    outer_radius_mm=math.nan,
    successes=math.nan,
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
            inner_radius_mm=float(inner_radius_mm),
            outer_radius_mm=float(outer_radius_mm),
            value=float(value) if np.isfinite(value) else math.nan,
            n_events=int(n_events),
            successes=float(successes) if np.isfinite(successes) else math.nan,
            eligible=eligible,
            exclusion_reason=reason,
        )
    )


def _cohort_options(opts):
    """Resolve report selection without modifying global plotting options."""
    local = copy.copy(opts)
    mode = getattr(opts, "learner_metric_table_sli_mode", "mean")
    if mode not in ("mean", "final"):
        raise ValueError(f"unknown learner metric SLI mode: {mode}")
    local.sli_use_training_mean = mode == "mean"
    if mode == "final":
        local.best_worst_trn = 2
        local.sli_select_bucket = "5"
        local.sli_select_skip_first_sync_buckets = 0
        local.sli_select_keep_first_sync_buckets = 0
    else:
        local.best_worst_trn = int(getattr(opts, "best_worst_trn", 2))
        local.sli_select_bucket = None
        for name, default in (
            ("sli_select_skip_first_sync_buckets", 1),
            ("sli_select_keep_first_sync_buckets", 4),
        ):
            raw = getattr(opts, name, None)
            setattr(local, name, default if raw is None else max(0, int(raw)))
    return local


def _select_cohorts(vas, opts):
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    unit_ids = [_canonical_unit_id(va) for va in vas_ok]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("learner report requires unique fly unit IDs")
    cohort_opts = _cohort_options(opts)
    sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, cohort_opts)
    sli = np.asarray(sli, dtype=float)
    sli[~np.isfinite(sli)] = np.nan
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
    selected = [(idx, vas_ok[idx], "strong") for idx in top] + [
        (idx, vas_ok[idx], "weak") for idx in bottom
    ]
    ranks = pd.Series(sli).sort_values(kind="mergesort").dropna().index
    rank_by_idx = {idx: rank for rank, idx in enumerate(ranks, 1)}
    cohort_by_idx = {idx: cohort for idx, _va, cohort in selected}
    sli_ts = np.asarray(sli_ts, dtype=float)
    final = np.full(len(vas_ok), np.nan)
    mean = np.full(len(vas_ok), np.nan)
    counts = np.zeros(len(vas_ok), dtype=int)
    if sli_ts.ndim == 3 and sli_ts.shape[1] >= 2:
        if sli_ts.shape[2] >= 5:
            final = sli_ts[:, 1, 4]
        minimum = min(4, max(0, sli_ts.shape[2] - 1),
                      int(getattr(opts, "sli_min_valid_sync_buckets", 3)))
        for idx, values in enumerate(sli_ts[:, 1, 1:5]):
            valid = values[np.isfinite(values)]
            counts[idx] = len(valid)
            if len(valid) and len(valid) >= minimum:
                mean[idx] = float(np.mean(valid))
    audit = [
        {
            "unit_id": uid,
            "sli_t2_sb5": float(final[idx]),
            "sli_t2_sb2_sb5_mean": float(mean[idx]),
            "sli_t2_sb2_sb5_valid_bucket_count": int(counts[idx]),
            "selection_score": float(sli[idx]),
            "rank_ascending": rank_by_idx.get(idx, ""),
            "cohort": cohort_by_idx.get(idx, ""),
            "exclusion_reason": (
                "sli_unavailable" if not np.isfinite(sli[idx])
                else "middle_rank" if idx not in cohort_by_idx else ""
            ),
        }
        for idx, uid in enumerate(unit_ids)
    ]
    return vas_ok, sli, selected, audit


def _selected_vas_and_sli(vas, opts):
    # Preserve the collector-facing helper contract.
    return _select_cohorts(vas, opts)[:3]


def _metric_window(opts) -> tuple[int, int, int, list[int]]:
    training = max(1, int(getattr(opts, "learner_metric_table_training", 2)))
    skip = max(0, int(getattr(opts, "learner_metric_table_skip_first_sync_buckets", 1)))
    keep = max(0, int(getattr(opts, "learner_metric_table_keep_first_sync_buckets", 4)))
    indices = list(range(skip, skip + keep)) if keep > 0 else []
    return training, training - 1, skip, indices


def _collect_dctr(selected, sli, opts, observations, circle_pairs):
    training, _training_idx, skip, bucket_indices = _metric_window(opts)
    min_turnback = min_episode_count_for_type(opts, EPISODE_TYPE_INNER_EXIT_REENTRY)
    selected_vas = [va for _idx, va, _cohort in selected]
    inner = np.asarray([pair[0] for pair in circle_pairs], dtype=float)
    outer = np.asarray([pair[1] for pair in circle_pairs], dtype=float)
    ratios, _ratio_ctrl, successes, _success_ctrl, totals, _total_ctrl, _windows = (
        _compute_pair_curves(
            selected_vas,
            inner_deltas_mm=inner,
            outer_deltas_mm=outer,
            legacy_pair_deltas=False,
            border_width_mm=float(
                getattr(opts, "turnback_border_width_mm", 0.1) or 0.1
            ),
            radius_offset_px=float(
                getattr(opts, "turnback_inner_radius_offset_px", 0.0) or 0.0
            ),
            selected_trainings=[training - 1],
            skip_first=skip,
            keep_first=len(bucket_indices),
            last_sync_buckets=0,
            debug=False,
            min_episodes=0,
            exclude_wall_contact=False,
            min_walking_fraction=0.0,
        )
    )
    for local_idx, (idx, va, cohort) in enumerate(selected):
        target = exp_target_sync_bucket_filter_result(va, opts)
        for pair_idx, (inner_mm, outer_mm) in enumerate(circle_pairs):
            _record_metric(
                observations,
                va=va,
                sli=sli[idx],
                cohort=cohort,
                metric="dual_circle_turnback_ratio",
                value=float(ratios[local_idx, pair_idx]),
                n_events=int(totals[local_idx, pair_idx]),
                minimum_events=min_turnback,
                target_result=target,
                inner_radius_mm=inner_mm,
                outer_radius_mm=outer_mm,
                successes=float(successes[local_idx, pair_idx]),
            )


def _collect_com(selected, sli, opts, observations):
    _training, training_idx, _skip, bucket_indices = _metric_window(opts)
    min_between = min_episode_count_for_type(
        opts, EPISODE_TYPE_BETWEEN_REWARD_TRAJECTORY
    )
    warned_missing_wc = [False]
    for idx, va, cohort in selected:
        target = exp_target_sync_bucket_filter_result(va, opts)
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
                    exclude_wall=bool(getattr(opts, "com_exclude_wall_contact", False)),
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
                str(uid): (total, count) for uid, total, count in panels[training_idx]
            }
        for va in cohort_vas:
            target = exp_target_sync_bucket_filter_result(va, opts)
            key = collector._unit_id(va, f=0)
            total, count = values.get(str(key), (math.nan, 0))
            value = (
                float(total / count) if count > 0 and np.isfinite(total) else math.nan
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
                str(uid): (total, count) for uid, total, count in panels[training_idx]
            }
        for va in cohort_vas:
            target = exp_target_sync_bucket_filter_result(va, opts)
            key = plotter._unit_id(va, f=0)
            total, count = values.get(str(key), (math.nan, 0))
            value = (
                float(total / count) if count > 0 and np.isfinite(total) else math.nan
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


def _collect_alignment(selected, sli_by_id, opts, observations, circle_pairs):
    training, _training_idx, skip, bucket_indices = _metric_window(opts)
    keep = len(bucket_indices)
    min_turnback = min_episode_count_for_type(opts, EPISODE_TYPE_INNER_EXIT_REENTRY)
    for inner_mm, outer_mm in circle_pairs:
        pair_opts = copy.copy(opts)
        pair_opts.turnback_home_vector_alignment_inner_radius_mm = float(inner_mm)
        pair_opts.turnback_home_vector_alignment_outer_radius_mm = float(outer_mm)
        pair_opts.turnback_home_vector_alignment_inner_delta_mm = None
        pair_opts.turnback_home_vector_alignment_outer_delta_mm = None
        for cohort in ("strong", "weak"):
            cohort_vas = [va for _idx, va, name in selected if name == cohort]
            targets = np.asarray(
                [
                    exp_target_sync_bucket_filter_result(va, pair_opts).eligible
                    for va in cohort_vas
                ],
                dtype=bool,
            )
            ids, values, counts, _meta = _collect_alignment_values(
                cohort_vas,
                pair_opts,
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
                target = exp_target_sync_bucket_filter_result(va, pair_opts)
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
                    inner_radius_mm=inner_mm,
                    outer_radius_mm=outer_mm,
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


def build_prism_wide_rows(observations, circle_pairs=DEFAULT_CIRCLE_PAIRS_MM):
    units = {}
    for row in observations:
        base = units.setdefault(
            row.unit_id, {"unit_id": row.unit_id, "cohort": row.cohort, "sli": row.sli}
        )
        if row.metric in REPEATED_METRICS:
            key = f"{row.metric}_{row.inner_radius_mm:g}_{row.outer_radius_mm:g}_mm"
        else:
            key = row.metric
        base[key] = row.value if row.eligible else math.nan
    return [units[key] for key in sorted(units)]


def collect_learner_metric_observations(vas, opts, gls) -> tuple[list, dict]:
    vas_ok, sli, selected, cohort_audit = _select_cohorts(vas, opts)
    cohort_opts = _cohort_options(opts)
    if not selected:
        raise ValueError("learner metric table selected no strong or weak learners")

    circle_pairs = parse_circle_pairs_mm(
        getattr(opts, "learner_metric_table_circle_pairs_mm", None)
    )
    report_opts = copy.copy(opts)
    report_opts.require_exp_target_sync_bucket = bool(
        getattr(opts, "learner_metric_table_require_sb5", True)
    )
    training, _training_idx, skip, bucket_indices = _metric_window(report_opts)
    report_opts.exp_target_sync_bucket_filter_training = training
    report_opts.exp_target_sync_bucket_filter_sync_bucket = (
        skip + len(bucket_indices) if bucket_indices else skip + 1
    )

    observations: list[MetricObservation] = []
    _collect_dctr(selected, sli, report_opts, observations, circle_pairs)
    _collect_com(selected, sli, report_opts, observations)
    sli_by_id = {
        _canonical_unit_id(va): float(sli[idx]) for idx, va, _cohort in selected
    }
    _collect_alignment(selected, sli_by_id, report_opts, observations, circle_pairs)
    _collect_tortuosity(selected, sli_by_id, report_opts, gls, observations)
    _collect_return_leg(selected, sli_by_id, report_opts, observations)

    metadata = {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_unit_ids": [_canonical_unit_id(va) for va in vas],
        "group_labels": list(gls) if gls is not None else [],
        "skipped_input_n": len(vas) - len(vas_ok),
        "cohort_audit": cohort_audit,
        "rankable_n": int(np.count_nonzero(np.isfinite(sli))),
        "strong_selected_n": sum(1 for _idx, _va, c in selected if c == "strong"),
        "weak_selected_n": sum(1 for _idx, _va, c in selected if c == "weak"),
        "cohort_definition": {
            "mode": getattr(opts, "learner_metric_table_sli_mode", "mean"),
            "training": int(cohort_opts.best_worst_trn),
            "sync_bucket_1_based": 5 if not cohort_opts.sli_use_training_mean else None,
            "use_training_mean": cohort_opts.sli_use_training_mean,
            "skip_first_sync_buckets": int(
                cohort_opts.sli_select_skip_first_sync_buckets
            ),
            "keep_first_sync_buckets": int(
                cohort_opts.sli_select_keep_first_sync_buckets
            ),
            "min_valid_sync_buckets": int(
                getattr(opts, "sli_min_valid_sync_buckets", 3)
            ) if cohort_opts.sli_use_training_mean else None,
            "ties": "stable input order; floor fractional counts with minimum one",
            "top_fraction": float(getattr(opts, "top_sli_fraction", 0.2)),
            "bottom_fraction": float(getattr(opts, "bottom_sli_fraction", 0.5)),
        },
        "metric_window": {
            "training": training,
            "skip_first_sync_buckets": skip,
            "keep_first_sync_buckets": len(bucket_indices),
            "sync_buckets_1_based": [idx + 1 for idx in bucket_indices],
        },
        "circle_pairs_mm": [
            {"inner_radius_mm": inner, "outer_radius_mm": outer}
            for inner, outer in circle_pairs
        ],
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
                "radius_mode": "absolute",
                "circle_pairs_mm": [list(pair) for pair in circle_pairs],
                "aggregation": "sum(successes) / sum(qualifying exits)",
                "audit_counts": ["successes", "n_events"],
            },
            "home_vector_alignment": {
                "radius_mode": "absolute",
                "circle_pairs_mm": [list(pair) for pair in circle_pairs],
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
            "repeated_radius_metrics": (
                "descriptive only; no hypothesis test is performed for "
                "dual-circle turnback ratio or home-vector alignment"
            ),
            "standalone_metrics": (
                "independent-group Welch t-test for tortuosity, COM distance, "
                "and return-leg distance"
            ),
            "multiple_comparisons": "none; no p-value adjustment is applied",
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


def _json_safe(value):
    """Keep JSON portable: unavailable numeric diagnostics are null, not NaN."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _code_provenance():
    try:
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=2, check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            capture_output=True, text=True, timeout=2, check=False,
        )
        return {
            "code_revision": result.stdout.strip() if result.returncode == 0 else None,
            "working_tree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except (OSError, subprocess.TimeoutExpired):
        return {"code_revision": None, "working_tree_dirty": None}


def _methods_text(metadata, mixed):
    definition = metadata.get("cohort_definition", {})
    mode = definition.get("mode", "mean")
    if mode == "final":
        selection = (
            "Cohorts ranked by experimental minus yoked reward PI at T2 SB5; "
            "both component PIs must be finite, with no mean-window bucket minimum."
        )
    else:
        skip = definition.get("skip_first_sync_buckets", 1)
        keep = definition.get("keep_first_sync_buckets", 4)
        window = f"SB{skip + 1}–SB{skip + keep}" if keep else f"SB{skip + 1} onward"
        selection = (
            f"Cohorts ranked by mean paired bucket-level SLI at T{definition.get('training', 2)} "
            f"{window}, requiring at least {definition.get('min_valid_sync_buckets', 3)} "
            "valid buckets (all selected buckets if the window is shorter)."
        )
    selection += (
        f" Strong learners: top {100 * definition.get('top_fraction', 0.2):g}%; "
        f"weak learners: bottom {100 * definition.get('bottom_fraction', 0.5):g}%. "
        "Cohorts are fixed before metric exclusions. Fractional counts are floored "
        "with a minimum of one; ties follow input order."
    )
    window = metadata.get("metric_window", {})
    eligibility = metadata.get("eligibility", {})
    pooling = (
        f"Metrics pool episodes at T{window.get('training', 2)}, sync buckets "
        f"{window.get('sync_buckets_1_based', [2, 3, 4, 5])}. "
        f"Minimum episodes: {eligibility.get('min_turnback_episodes', 5)} for turnback/alignment, "
        f"{eligibility.get('min_between_reward_trajectories', 5)} for between-reward metrics. "
        f"Final selected experimental bucket presence required: "
        f"{eligibility.get('require_final_selected_sync_bucket', True)}. "
        "Observed values are fly-level means with two-sided t-based 95% CIs; "
        "sample sizes vary by metric and radius."
    )
    inference = (
        "Turnback and alignment use separate REML Gaussian mixed models with cohort, "
        "categorical radius pair, their interaction, and a fly random intercept. "
        "Only eligible finite rows enter each model; other available radii from partially "
        "observed flies are retained without imputation or episode-count weighting. "
        "Primary cohort contrasts average equally across radii; radius contrasts and "
        "cohort-by-radius interaction tests use asymptotic Wald inference. Modeled "
        "contrasts may differ from observed strong-minus-weak differences. Missing-data "
        "inference assumes an ignorable missingness mechanism; episode exclusions can "
        "still bias results. Standalone metrics use independent-group Welch tests. "
        "Holm correction is applied separately to the radius contrasts within each "
        "metric (three per metric by default). Standalone Welch tests are assessed "
        "individually without multiple-comparison adjustment. "
        "Each metric's across-radius cohort and interaction tests are singleton "
        "families, with unchanged p-values; turnback and alignment never share "
        "a correction family. "
        "Incomplete families have unavailable adjusted p-values. All CIs are pointwise "
        "95% intervals, not simultaneous or small-sample-corrected intervals."
    ) if mixed else (
        "Repeated-radius metrics are descriptive only. Standalone metrics use "
        "independent-group Welch tests with unadjusted p-values."
    )
    return "\n\n".join((selection, pooling, inference))


def export_learner_metric_table(vas, opts, gls, out_path: str) -> dict[str, str]:
    observations, metadata = collect_learner_metric_observations(vas, opts, gls)
    circle_pairs = tuple(
        (row["inner_radius_mm"], row["outer_radius_mm"])
        for row in metadata["circle_pairs_mm"]
    )
    summary = summarize_observations(observations, circle_pairs=circle_pairs)
    stats_mode = getattr(opts, "learner_metric_table_stats", "legacy")
    if stats_mode not in ("legacy", "mixed"):
        raise ValueError(f"unknown learner metric statistics mode: {stats_mode}")
    mixed = stats_mode == "mixed"
    if mixed:
        model_input, model_tests, contrasts, diagnostics = analyze_repeated_metrics(
            observations, circle_pairs
        )
        for row in summary:
            row.update(
                model_difference_strong_minus_weak=math.nan,
                model_difference_ci_lo=math.nan, model_difference_ci_hi=math.nan,
                model_standard_error=math.nan, test_status="ok" if np.isfinite(row["p_value"]) else "unavailable",
                test_reason="" if np.isfinite(row["p_value"]) else "insufficient data for Welch test",
            )
            if row["metric"] in REPEATED_METRICS:
                contrast = contrasts[(row["metric"], row["inner_radius_mm"], row["outer_radius_mm"])]
                row.update(
                    model_difference_strong_minus_weak=contrast["estimate"],
                    model_difference_ci_lo=contrast["ci_lo"],
                    model_difference_ci_hi=contrast["ci_hi"],
                    model_standard_error=contrast["standard_error"],
                    test=contrast["test"], statistic=contrast["statistic"],
                    p_value=contrast["p_value"], test_status=contrast["status"],
                    test_reason=contrast["reason"],
                )
        families = adjust_report_families(summary, model_tests)
        import statsmodels

        metadata["statistics"].update(
            mode="mixed", model_formula=MODEL_FORMULA, fit_method="REML",
            statsmodels_version=statsmodels.__version__,
            repeated_radius_metrics="Gaussian mixed models; asymptotic Wald inference",
            multiple_comparisons="Holm within each repeated metric's radius contrasts; unadjusted standalone Welch tests and model cohort/interaction tests",
            comparison_families=families,
            ci_type="pointwise 95%; asymptotic Wald for models, t-based for observed means/Welch",
            primary_cohort_contrast="equal-weight average of strong minus weak across radius pairs",
        )
    else:
        metadata["statistics"]["mode"] = "legacy"
    metadata["methods_text"] = _methods_text(metadata, mixed)
    metadata.update(_code_provenance())
    prefix = _output_prefix(out_path)
    paths = {
        "summary_csv": f"{prefix}_summary.csv",
        "per_fly_csv": f"{prefix}_per_fly.csv",
        "prism_wide_csv": f"{prefix}_prism_wide.csv",
        "metadata_json": f"{prefix}_metadata.json",
        "summary_md": f"{prefix}_summary.md",
        "cohort_audit_csv": f"{prefix}_cohort_audit.csv",
    }
    if mixed:
        paths.update(
            model_input_csv=f"{prefix}_model_input.csv",
            model_tests_csv=f"{prefix}_model_tests.csv",
            model_diagnostics_json=f"{prefix}_model_diagnostics.json",
        )
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    def write_rows(path, rows, fieldnames=None):
        fields = fieldnames or list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_rows(paths["summary_csv"], summary)
    write_rows(paths["cohort_audit_csv"], metadata.pop("cohort_audit", []), COHORT_AUDIT_FIELDS)
    if mixed:
        write_rows(paths["model_input_csv"], model_input, MODEL_INPUT_FIELDS)
        write_rows(paths["model_tests_csv"], model_tests)
        with open(paths["model_diagnostics_json"], "w", encoding="utf-8") as fh:
            json.dump(_json_safe(diagnostics), fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
    write_rows(paths["per_fly_csv"], [asdict(row) for row in observations])
    wide_rows = build_prism_wide_rows(observations, circle_pairs)
    wide_fields = ["unit_id", "cohort", "sli"]
    for metric in REPEATED_METRICS:
        wide_fields.extend(
            f"{metric}_{inner:g}_{outer:g}_mm" for inner, outer in circle_pairs
        )
    wide_fields.extend(STANDALONE_METRICS)
    write_rows(paths["prism_wide_csv"], wide_rows, wide_fields)

    metadata["outputs"] = paths
    with open(paths["metadata_json"], "w", encoding="utf-8") as fh:
        json.dump(_json_safe(metadata), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")

    with open(paths["summary_md"], "w", encoding="utf-8") as fh:
        fh.write(metadata["methods_text"] + "\n\n")
        fh.write("| Metric | Strong learners | Weak learners | Observed strong − weak |")
        fh.write(" Model strong − weak [95% CI] | Raw p | Holm p | Status |\n" if mixed else "\n")
        fh.write("|---|---:|---:|---:|" + ("---:|---:|---:|---|\n" if mixed else "\n"))
        for row in summary:
            strong = f"{_fmt(row['strong_mean'])} [{_fmt(row['strong_ci_lo'])}, {_fmt(row['strong_ci_hi'])}], n={row['strong_n']}"
            weak = f"{_fmt(row['weak_mean'])} [{_fmt(row['weak_ci_lo'])}, {_fmt(row['weak_ci_hi'])}], n={row['weak_n']}"
            if mixed:
                difference = _fmt(row["difference_strong_minus_weak"])
                model_difference = (
                    f"{_fmt(row['model_difference_strong_minus_weak'])} "
                    f"[{_fmt(row['model_difference_ci_lo'])}, {_fmt(row['model_difference_ci_hi'])}]"
                    if row["metric"] in REPEATED_METRICS else
                    f"Welch CI [{_fmt(row['difference_ci_lo'])}, {_fmt(row['difference_ci_hi'])}]"
                )
                status = row["test_status"] + (": " + row["test_reason"] if row["test_reason"] else "")
                difference += f" | {model_difference} | {_fmt(row['p_value'])} | {_fmt(row['p_value_holm'])} | {status}"
            elif row["metric"] in REPEATED_METRICS:
                difference = (
                    f"{_fmt(row['difference_strong_minus_weak'])} "
                    "(descriptive); no test performed"
                )
            else:
                difference = (
                    f"{_fmt(row['difference_strong_minus_weak'])} "
                    f"[{_fmt(row['difference_ci_lo'])}, {_fmt(row['difference_ci_hi'])}]; "
                    f"Welch p={_fmt(row['p_value'])}"
                )
            fh.write(f"| {row['metric_label']} | {strong} | {weak} | {difference} |\n")
        if mixed:
            fh.write("\nPrimary cohort and interaction tests:\n\n")
            fh.write("| Metric | Comparison | Estimate [95% CI] | Statistic | df | Raw p | Holm p | Status |\n")
            fh.write("|---|---|---:|---:|---:|---:|---:|---|\n")
            for row in model_tests:
                estimate = (
                    f"{_fmt(row['estimate'])} [{_fmt(row['ci_lo'])}, {_fmt(row['ci_hi'])}]"
                    if row["comparison"] == "cohort_average" else "NA"
                )
                fh.write(
                    f"| {METRIC_LABELS[row['metric']]} | {row['comparison']} | {estimate} | "
                    f"{_fmt(row['statistic'])} | {row['df']} | {_fmt(row['p_value'])} | "
                    f"{_fmt(row['p_value_holm'])} | {row['status']} {row['reason']} |\n"
                )
            incomplete = [name for name, info in families.items() if not info["complete"]]
            if incomplete:
                fh.write("\nIncomplete Holm families (adjusted p-values unavailable): " + ", ".join(incomplete) + ".\n")

    print("[learner-metric-table] wrote " + ", ".join(paths.values()))
    return paths
