import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.video_analysis import VideoAnalysis
from src.exporting import learner_metric_table as report
from src.analysis.sli_tools import compute_sli_per_fly


def _observation(
    metric,
    cohort,
    value,
    *,
    eligible=True,
    unit_id="fly",
    inner=np.nan,
    outer=np.nan,
    n_events=8,
    successes=np.nan,
):
    return report.MetricObservation(
        unit_id=unit_id,
        sli=1.0 if cohort == "strong" else -1.0,
        cohort=cohort,
        metric=metric,
        inner_radius_mm=float(inner),
        outer_radius_mm=float(outer),
        value=float(value),
        n_events=int(n_events),
        successes=float(successes),
        eligible=eligible,
        exclusion_reason="" if eligible else "insufficient_episodes",
    )


def _complete_observations():
    observations = []
    for metric_idx, metric in enumerate(report.REPEATED_METRICS):
        for pair_idx, (inner, outer) in enumerate(report.DEFAULT_CIRCLE_PAIRS_MM):
            for fly_idx, value in enumerate((5.0, 6.0, 7.0)):
                observations.append(
                    _observation(
                        metric, "strong", value + metric_idx + pair_idx,
                        unit_id=f"strong-{fly_idx}", inner=inner, outer=outer,
                    )
                )
            for fly_idx, value in enumerate((1.0, 2.0, 3.0)):
                observations.append(
                    _observation(
                        metric, "weak", value + metric_idx + pair_idx,
                        unit_id=f"weak-{fly_idx}", inner=inner, outer=outer,
                    )
                )
    for metric_idx, metric in enumerate(report.STANDALONE_METRICS):
        for fly_idx, value in enumerate((5.0, 6.0, 7.0)):
            observations.append(
                _observation(metric, "strong", value + metric_idx, unit_id=f"strong-{fly_idx}")
            )
        for fly_idx, value in enumerate((1.0, 2.0, 3.0)):
            observations.append(
                _observation(metric, "weak", value + metric_idx, unit_id=f"weak-{fly_idx}")
            )
    return observations


def test_parse_circle_pairs_mm_defaults_and_validates_absolute_pairs():
    assert report.parse_circle_pairs_mm(None) == report.DEFAULT_CIRCLE_PAIRS_MM
    assert report.parse_circle_pairs_mm("3:5, 8:10,13:15") == (
        (3.0, 5.0), (8.0, 10.0), (13.0, 15.0)
    )
    for bad in ("3", "5:3", "-1:3", "3:inf", "3:5,3:5", ""):
        if bad == "":
            assert report.parse_circle_pairs_mm(bad) == report.DEFAULT_CIRCLE_PAIRS_MM
        else:
            with pytest.raises(ValueError):
                report.parse_circle_pairs_mm(bad)


def test_summary_has_nine_rows_with_descriptive_repeated_metrics_and_raw_welch():
    rows = report.summarize_observations(_complete_observations())
    assert len(rows) == 9
    assert [(r["metric"], r["inner_radius_mm"], r["outer_radius_mm"]) for r in rows[:6]] == [
        (metric, inner, outer)
        for metric in report.REPEATED_METRICS
        for inner, outer in report.DEFAULT_CIRCLE_PAIRS_MM
    ]
    assert [r["metric"] for r in rows[6:]] == list(report.STANDALONE_METRICS)
    assert all(r["test"] == "Not performed" for r in rows[:6])
    assert all(np.isnan(r["p_value"]) for r in rows[:6])
    assert all(r["difference_strong_minus_weak"] == pytest.approx(4.0) for r in rows[:6])
    assert all(r["test"] == "Welch t-test" for r in rows[6:])
    assert all(np.isfinite(r["p_value"]) for r in rows[6:])


def test_dctr_absolute_pairs_preserve_success_and_total_counts(monkeypatch):
    class Target:
        eligible = True
        reason = ""

    selected = [
        (0, SimpleNamespace(fn="a.avi", f=0), "strong"),
        (1, SimpleNamespace(fn="b.avi", f=0), "weak"),
    ]
    ratios = np.asarray([[0.5, 0.25, 0.75], [0.2, 0.4, 0.6]])
    successes = np.asarray([[4, 2, 6], [1, 2, 3]], dtype=float)
    totals = np.asarray([[8, 8, 8], [5, 5, 5]], dtype=int)
    seen = {}

    def fake_pair_curves(vas, **kwargs):
        seen.update(kwargs)
        zf = np.zeros_like(ratios)
        zi = np.zeros_like(totals)
        return ratios, zf, successes, zf, totals, zi, []

    monkeypatch.setattr(report, "_compute_pair_curves", fake_pair_curves)
    monkeypatch.setattr(report, "exp_target_sync_bucket_filter_result", lambda va, opts: Target())
    monkeypatch.setattr(report, "min_episode_count_for_type", lambda *args: 1)
    opts = SimpleNamespace(
        learner_metric_table_training=2,
        learner_metric_table_skip_first_sync_buckets=1,
        learner_metric_table_keep_first_sync_buckets=4,
        turnback_border_width_mm=0.1,
        turnback_inner_radius_offset_px=0.0,
    )
    observations = []
    report._collect_dctr(
        selected, np.asarray([0.8, -0.5]), opts, observations,
        report.DEFAULT_CIRCLE_PAIRS_MM,
    )
    assert seen["legacy_pair_deltas"] is False
    assert list(seen["inner_deltas_mm"]) == [3.0, 8.0, 13.0]
    assert list(seen["outer_deltas_mm"]) == [5.0, 10.0, 15.0]
    assert len(observations) == 6
    assert [(r.successes, r.n_events) for r in observations[:3]] == [(4.0, 8), (2.0, 8), (6.0, 8)]


def test_alignment_is_collected_once_per_absolute_pair_and_missingness_is_per_radius(monkeypatch):
    class Target:
        eligible = True
        reason = ""

    vas = [SimpleNamespace(fn="a.avi", f=0), SimpleNamespace(fn="b.avi", f=0)]
    selected = [(0, vas[0], "strong"), (1, vas[1], "weak")]
    sli_by_id = {report._canonical_unit_id(vas[0]): 0.8, report._canonical_unit_id(vas[1]): -0.5}
    calls = []

    def fake_collect(cohort_vas, opts, **kwargs):
        inner = opts.turnback_home_vector_alignment_inner_radius_mm
        outer = opts.turnback_home_vector_alignment_outer_radius_mm
        calls.append((inner, outer, len(cohort_vas)))
        va = cohort_vas[0]
        if inner == 8.0 and "b.avi" in va.fn:
            return np.asarray([], object), np.asarray([]), np.asarray([], int), {}
        return (
            np.asarray([report._alignment_unit_id(va, 0)], object),
            np.asarray([inner / outer]),
            np.asarray([4], int),
            {},
        )

    monkeypatch.setattr(report, "_collect_alignment_values", fake_collect)
    monkeypatch.setattr(report, "exp_target_sync_bucket_filter_result", lambda va, opts: Target())
    monkeypatch.setattr(report, "min_episode_count_for_type", lambda *args: 1)
    opts = SimpleNamespace(
        learner_metric_table_training=2,
        learner_metric_table_skip_first_sync_buckets=1,
        learner_metric_table_keep_first_sync_buckets=4,
    )
    observations = []
    report._collect_alignment(selected, sli_by_id, opts, observations, report.DEFAULT_CIRCLE_PAIRS_MM)
    assert calls == [
        (3.0, 5.0, 1), (3.0, 5.0, 1),
        (8.0, 10.0, 1), (8.0, 10.0, 1),
        (13.0, 15.0, 1), (13.0, 15.0, 1),
    ]
    weak_mid = next(
        r for r in observations
        if r.cohort == "weak" and r.inner_radius_mm == 8.0
    )
    assert weak_mid.eligible is False
    assert weak_mid.exclusion_reason == "insufficient_episodes"
    rows = report.summarize_observations(observations)
    alignment_rows = [r for r in rows if r["metric"] == "home_vector_alignment"]
    assert [r["weak_n"] for r in alignment_rows] == [1, 0, 1]


def test_fixed_cohort_assignments_are_reused_across_all_metric_rows():
    observations = _complete_observations()
    by_metric = {}
    for row in observations:
        if row.eligible:
            by_metric.setdefault((row.metric, row.inner_radius_mm, row.outer_radius_mm), set()).add((row.unit_id, row.cohort))
    expected = {(f"strong-{i}", "strong") for i in range(3)} | {(f"weak-{i}", "weak") for i in range(3)}
    assert all(assignments == expected for assignments in by_metric.values())


def test_export_writes_all_report_outputs(tmp_path, monkeypatch):
    observations = _complete_observations()
    metadata = {
        "cohort_definition": {"top_fraction": 0.2},
        "circle_pairs_mm": [
            {"inner_radius_mm": inner, "outer_radius_mm": outer}
            for inner, outer in report.DEFAULT_CIRCLE_PAIRS_MM
        ],
        "statistics": {},
    }
    monkeypatch.setattr(report, "collect_learner_metric_observations", lambda vas, opts, gls: (observations, metadata))
    paths = report.export_learner_metric_table([], SimpleNamespace(), None, str(tmp_path / "learner_metrics.csv"))
    assert set(paths) == {
        "summary_csv", "per_fly_csv", "prism_wide_csv",
        "metadata_json", "summary_md", "cohort_audit_csv",
    }
    with open(paths["summary_csv"], newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert len(summary_rows) == 9
    with open(paths["per_fly_csv"], newline="", encoding="utf-8") as fh:
        per_fly_rows = list(csv.DictReader(fh))
    assert "inner_radius_mm" in per_fly_rows[0]
    assert "outer_radius_mm" in per_fly_rows[0]
    assert "successes" in per_fly_rows[0]
    with open(paths["metadata_json"], encoding="utf-8") as fh:
        written_metadata = json.load(fh)
    assert written_metadata["outputs"] == paths


def test_export_prevents_trajectory_slimming():
    va = SimpleNamespace(opts=SimpleNamespace(export_learner_metric_table="learner_metrics"))
    assert VideoAnalysis.memory_saver_can_slim_trajectories(va) is False


def _raw_selection_fixture(monkeypatch):
    raw = np.zeros((10, 2, 2, 6))
    for idx in range(10):
        raw[idx, 1, 0, 1:5] = idx / 10
    raw[0, 1, 0, 1:4] = np.nan
    raw[0, 1, 0, 4] = 0.99  # Final-only, highest final score.
    raw[2, 1, 1, 4] = np.nan  # Experimental SB5 exists, yoked SB5 unavailable.
    raw[3, 1, 0, 1] = np.nan  # Three valid paired buckets: mean eligible.
    raw[9, 1, 0, 1:4] = 0.1  # Final rank differs from mean rank.
    seen = []

    def compute(vas, opts):
        seen.append(opts)
        score = compute_sli_per_fly(
            raw, opts.best_worst_trn - 1,
            bucket_idx=4 if not opts.sli_use_training_mean else None,
            average_over_buckets=opts.sli_use_training_mean,
            skip_first_sync_buckets=opts.sli_select_skip_first_sync_buckets,
            keep_first_sync_buckets=opts.sli_select_keep_first_sync_buckets,
            min_valid_buckets=getattr(opts, "sli_min_valid_sync_buckets", 3),
        )
        return np.asarray(score), raw[:, :, 0] - raw[:, :, 1]

    monkeypatch.setattr(report, "_compute_sli_scalar_and_timeseries_from_rpid", compute)
    vas = [SimpleNamespace(fn=f"video_{idx}.avi", f=0) for idx in range(10)]
    return vas, seen, raw


def test_final_ranks_sb5_and_mean_ranks_three_of_four_without_mutating_options(monkeypatch):
    vas, seen, _ = _raw_selection_fixture(monkeypatch)
    opts = SimpleNamespace(learner_metric_table_sli_mode="mean", sli_use_training_mean=False)
    _, mean, selected_mean, audit_mean = report._select_cohorts(vas, opts)
    assert np.isnan(mean[0])
    assert np.isfinite(mean[2]) and np.isfinite(mean[3])
    assert audit_mean[3]["sli_t2_sb2_sb5_valid_bucket_count"] == 3
    assert [idx for idx, _, cohort in selected_mean if cohort == "strong"] == [8]
    assert opts.sli_use_training_mean is False
    assert not hasattr(opts, "sli_select_skip_first_sync_buckets")

    opts.learner_metric_table_sli_mode = "final"
    opts.sli_min_valid_sync_buckets = 4  # Must not affect final eligibility.
    opts.best_worst_trn = 1
    opts.sli_select_skip_first_sync_buckets = 1
    opts.sli_select_keep_first_sync_buckets = 3
    _, final, selected_final, audit_final = report._select_cohorts(vas, opts)
    assert np.isfinite(final[0]) and np.isnan(final[2])
    assert [idx for idx, _, cohort in selected_final if cohort == "strong"] == [0]
    assert audit_final[2]["exclusion_reason"] == "sli_unavailable"
    assert seen[-1].sli_select_bucket == "5"
    assert seen[-1].best_worst_trn == 2
    assert seen[-1].sli_select_skip_first_sync_buckets == 0
    assert seen[-1].sli_select_keep_first_sync_buckets == 0
    assert opts.best_worst_trn == 1 and opts.sli_select_keep_first_sync_buckets == 3


def test_collection_keeps_metric_window_independent_from_final_ranking(monkeypatch):
    vas, _, _ = _raw_selection_fixture(monkeypatch)
    windows = []

    def collect(selected, sli, opts, observations, *args):
        windows.append(report._metric_window(opts))
        assert opts.exp_target_sync_bucket_filter_sync_bucket == 5

    for name in ("_collect_dctr", "_collect_com", "_collect_alignment", "_collect_tortuosity", "_collect_return_leg"):
        monkeypatch.setattr(report, name, collect)
    opts = SimpleNamespace(learner_metric_table_sli_mode="final")
    _, metadata = report.collect_learner_metric_observations(vas, opts, None)
    assert all(window == (2, 1, 1, [1, 2, 3, 4]) for window in windows)
    assert metadata["cohort_definition"]["sync_bucket_1_based"] == 5
    assert metadata["cohort_definition"]["min_valid_sync_buckets"] is None
    assert len(metadata["cohort_audit"]) == 10


def test_real_sli_helper_selects_numeric_sb5_from_padded_arrays(monkeypatch):
    # analyze parses argv at import time; isolate it from pytest arguments.
    monkeypatch.setattr(sys, "argv", ["analyze.py"])
    import analyze
    from src.exporting.com_sli_bundle import _compute_sli_scalar_and_timeseries_from_rpid

    vas, _, raw = _raw_selection_fixture(monkeypatch)
    for idx, va in enumerate(vas):
        va.flies = [0, 1]
        va.test_index = idx
    monkeypatch.setattr(analyze, "typeCalc", lambda _name: ("rpid", None))
    monkeypatch.setattr(analyze, "trnsForType", lambda *args: [None, None])
    monkeypatch.setattr(analyze, "vaVarForType", lambda va, *args: raw[va.test_index])
    opts = report._cohort_options(SimpleNamespace(
        learner_metric_table_sli_mode="final", sli_min_valid_sync_buckets=4,
        sli_select_skip_first_sync_buckets=1, sli_select_keep_first_sync_buckets=4,
    ))
    sli, _ = _compute_sli_scalar_and_timeseries_from_rpid(vas, opts)
    assert sli[0] == pytest.approx(0.99)
    assert np.isnan(sli[2])
    assert sli[9] == pytest.approx(0.9)


def test_cli_report_normalization_preserves_global_sli_options(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["analyze.py"])
    from analyze import p, _normalize_learner_metric_table_options

    opts = p.parse_args([
        "--export-learner-metric-table", "unused",
        "--learner-metric-table-sli-mode", "final",
        "--learner-metric-table-stats", "mixed",
        "--sli-use-training-mean", "--best-worst-trn", "1",
        "--sli-select-skip-first-sync-buckets", "2",
        "--sli-select-keep-first-sync-buckets", "2",
    ])
    _normalize_learner_metric_table_options(opts)
    assert opts.sli_use_training_mean
    assert opts.best_worst_trn == 1
    assert opts.sli_select_skip_first_sync_buckets == 2
    assert opts.sli_select_keep_first_sync_buckets == 2
    assert opts.learner_metric_table_stats == "mixed"
    assert report._cohort_options(opts).best_worst_trn == 2


def test_mixed_export_keeps_descriptives_and_exposes_unavailable_fits(tmp_path, monkeypatch):
    # These legacy synthetic values have no radius residual variance; do not
    # pretend such a singular fit produces usable inference.
    observations = _complete_observations()
    metadata = {
        "cohort_definition": {"mode": "final"},
        "circle_pairs_mm": [
            {"inner_radius_mm": inner, "outer_radius_mm": outer}
            for inner, outer in report.DEFAULT_CIRCLE_PAIRS_MM
        ],
        "statistics": {},
    }
    monkeypatch.setattr(report, "collect_learner_metric_observations", lambda *args: (observations, metadata))
    paths = report.export_learner_metric_table(
        [], SimpleNamespace(learner_metric_table_stats="mixed"), None,
        str(tmp_path / "mixed"),
    )
    assert len(paths) == 9
    with open(paths["summary_csv"]) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 9
    assert float(rows[0]["difference_strong_minus_weak"]) == pytest.approx(4)
    assert "model_difference_strong_minus_weak" in rows[0]
    with open(paths["model_tests_csv"]) as fh:
        assert len(list(csv.DictReader(fh))) == 4
    with open(paths["model_diagnostics_json"]) as fh:
        diagnostics = json.load(fh)
    assert len(diagnostics) == 2
    with open(paths["summary_md"]) as fh:
        markdown = fh.read()
    assert "T2 SB5" in markdown and "Primary cohort and interaction" in markdown
