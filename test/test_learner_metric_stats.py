from dataclasses import replace
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import chi2, norm

from src.analysis import learner_metric_stats as stats
from src.analysis.multiple_comparisons import holm_adjust
from src.exporting.learner_metric_table import (
    DEFAULT_CIRCLE_PAIRS_MM, MetricObservation, summarize_observations,
)


def observations():
    rng = np.random.default_rng(4721)
    rows = []
    for metric in stats.REPEATED_METRICS:
        for cohort in ("weak", "strong"):
            for fly in range(24):
                intercept = rng.normal(0, 0.035)
                for idx, (inner, outer) in enumerate(DEFAULT_CIRCLE_PAIRS_MM):
                    value = (0.3 + 0.1 * (cohort == "strong") + 0.06 * idx
                             + 0.02 * idx * (cohort == "strong") + intercept
                             + rng.normal(0, 0.025))
                    rows.append(MetricObservation(
                        unit_id=f"{cohort}-{fly}", sli=0.5, cohort=cohort,
                        metric=metric, inner_radius_mm=inner, outer_radius_mm=outer,
                        value=value, n_events=10, successes=np.nan,
                        eligible=True, exclusion_reason="",
                    ))
    return rows


def test_mixed_model_recovers_effects_and_covariance_based_contrasts():
    inputs, tests, contrasts, diagnostics = stats.analyze_repeated_metrics(
        observations(), DEFAULT_CIRCLE_PAIRS_MM
    )
    assert len(inputs) == 288
    assert len(tests) == 4 and len(contrasts) == 6
    for metric in stats.REPEATED_METRICS:
        diag = diagnostics[metric]
        assert diag["status"] == "ok" and diag["converged"]
        assert diag["fly_counts"] == {"strong": 24, "weak": 24}
        assert diag["random_intercept_variance"] > 0
        primary = next(row for row in tests if row["metric"] == metric and row["comparison"] == "cohort_average")
        assert primary["estimate"] == pytest.approx(0.12, abs=0.03)
        beta = np.array(diag["fixed_effect_estimates"])
        covariance = np.array(diag["fixed_effect_covariance"])
        vector = np.array([0, 1, 0, 0, 1 / 3, 1 / 3])
        assert primary["estimate"] == pytest.approx(vector @ beta)
        assert primary["standard_error"] == pytest.approx(np.sqrt(vector @ covariance @ vector))
        assert primary["p_value"] == pytest.approx(2 * norm.sf(abs(primary["statistic"])))
        for idx, pair in enumerate(DEFAULT_CIRCLE_PAIRS_MM):
            assert contrasts[(metric, *pair)]["estimate"] == pytest.approx(0.1 + idx * 0.02, abs=0.03)


def test_reml_fit_and_wald_contrasts_match_independent_r_nlme_reference():
    reference = json.loads((Path(__file__).parent / "reference/learner_metric_mixed_nlme.json").read_text())
    _, tests, contrasts, diagnostics = stats.analyze_repeated_metrics(observations(), DEFAULT_CIRCLE_PAIRS_MM)
    for metric, expected in reference["metrics"].items():
        actual = diagnostics[metric]
        np.testing.assert_allclose(actual["fixed_effect_estimates"], expected["fixed_effect_estimates"], atol=1e-8)
        np.testing.assert_allclose(actual["fixed_effect_covariance"], expected["fixed_effect_covariance"], rtol=5e-5, atol=1e-10)
        beta = np.array(expected["fixed_effect_estimates"])
        covariance = np.array(expected["fixed_effect_covariance"])
        vectors = [np.array([0, 1, 0, 0, 0, 0]),
                   np.array([0, 1, 0, 0, 1, 0]),
                   np.array([0, 1, 0, 0, 0, 1])]
        for pair, vector in zip(DEFAULT_CIRCLE_PAIRS_MM, vectors):
            contrast = contrasts[(metric, *pair)]
            expected_se = np.sqrt(vector @ covariance @ vector)
            assert contrast["standard_error"] == pytest.approx(expected_se, rel=5e-5)
            assert contrast["ci_lo"] == pytest.approx(vector @ beta - norm.ppf(0.975) * expected_se, abs=1e-6)
        primary = next(row for row in tests if row["metric"] == metric and row["comparison"] == "cohort_average")
        vector = np.mean(vectors, axis=0)
        assert primary["standard_error"] == pytest.approx(np.sqrt(vector @ covariance @ vector), rel=5e-5)
        interaction = next(row for row in tests if row["metric"] == metric and row["comparison"] == "cohort_by_radius")
        expected_statistic = beta[4:] @ np.linalg.solve(covariance[4:, 4:], beta[4:])
        assert interaction["statistic"] == pytest.approx(expected_statistic, rel=5e-5)
        assert interaction["p_value"] == pytest.approx(chi2.sf(expected_statistic, 2), rel=5e-4)


@pytest.mark.parametrize("n_pairs", [2, 4])
def test_categorical_contrasts_support_configurable_radius_counts(n_pairs):
    # Deliberately reverse physical radii: the first configured pair is the
    # reference, and radius must remain categorical rather than numeric.
    pairs = tuple((float(3 + 5 * idx), float(5 + 5 * idx)) for idx in reversed(range(n_pairs)))
    rng = np.random.default_rng(341)
    rows = []
    for metric in stats.REPEATED_METRICS:
        for cohort in ("weak", "strong"):
            for fly in range(16):
                offset = rng.normal(0, 0.04)
                for idx, (inner, outer) in enumerate(pairs):
                    rows.append(MetricObservation(
                        unit_id=f"{cohort}-{fly}", sli=0.5, cohort=cohort,
                        metric=metric, inner_radius_mm=inner, outer_radius_mm=outer,
                        value=0.4 + idx * 0.03 + (cohort == "strong") * (0.1 + idx * 0.02)
                              + offset + rng.normal(0, 0.02),
                        n_events=10, successes=np.nan, eligible=True, exclusion_reason="",
                    ))
    inputs, tests, contrasts, diagnostics = stats.analyze_repeated_metrics(rows, pairs)
    for metric in stats.REPEATED_METRICS:
        assert diagnostics[metric]["status"] == "ok"
        beta = np.asarray(diagnostics[metric]["fixed_effect_estimates"])
        assert len(beta) == 2 * n_pairs
        estimates = []
        for idx, pair in enumerate(pairs):
            actual = contrasts[(metric, *pair)]["estimate"]
            # Balanced design gives the directly observed cohort difference.
            means = {
                cohort: np.mean([row["value"] for row in inputs
                                 if row["metric"] == metric and row["cohort"] == cohort
                                 and row["radius_pair"] == idx])
                for cohort in ("strong", "weak")
            }
            assert actual == pytest.approx(means["strong"] - means["weak"], abs=1e-10)
            estimates.append(actual)
        average = next(row for row in tests if row["metric"] == metric and row["comparison"] == "cohort_average")
        assert average["estimate"] == pytest.approx(np.mean(estimates))
        interaction = next(row for row in tests if row["metric"] == metric and row["comparison"] == "cohort_by_radius")
        assert interaction["df"] == n_pairs - 1
        assert np.isfinite(interaction["p_value"])


def test_partial_radius_missingness_retains_fly_and_missingness_audit():
    rows = observations()
    rows = [replace(row, eligible=False) if row.unit_id == "strong-0" and row.inner_radius_mm != 3 else row for row in rows]
    inputs, tests, _, diagnostics = stats.analyze_repeated_metrics(rows, DEFAULT_CIRCLE_PAIRS_MM)
    assert len(inputs) == 284
    assert sum(row["unit_id"] == "strong-0" for row in inputs) == 2
    assert all(row["status"] == "ok" for row in tests)
    for diag in diagnostics.values():
        assert diag["fly_counts"]["strong"] == 24
        assert diag["missingness_patterns"]["strong"] == {"0": 1, "0,1,2": 23}


def test_duplicate_keys_and_inconsistent_cohorts_are_rejected():
    rows = observations()
    with pytest.raises(ValueError, match="duplicate"):
        stats.analyze_repeated_metrics(rows + [rows[0]], DEFAULT_CIRCLE_PAIRS_MM)
    with pytest.raises(ValueError, match="inconsistent cohort"):
        stats.analyze_repeated_metrics(rows + [replace(rows[0], cohort="strong")], DEFAULT_CIRCLE_PAIRS_MM)


def test_missing_cell_produces_explicit_unavailable_results():
    rows = [replace(row, eligible=False) if row.cohort == "strong" and row.inner_radius_mm == 13 else row for row in observations()]
    inputs, tests, contrasts, diagnostics = stats.analyze_repeated_metrics(rows, DEFAULT_CIRCLE_PAIRS_MM)
    assert len(inputs) == 240
    assert all(row["status"] == "unavailable" and np.isnan(row["p_value"]) for row in tests)
    assert all(np.isnan(row["estimate"]) for row in contrasts.values())
    assert all("missing cohort/radius cell" in diag["reason"] for diag in diagnostics.values())


def test_failed_fit_retries_are_recorded_without_fallback_p_values(monkeypatch):
    methods = []

    def fail(self, **kwargs):
        methods.append(kwargs["method"])
        raise np.linalg.LinAlgError("test singular covariance")

    monkeypatch.setattr(stats.MixedLM, "fit", fail)
    _, tests, _, diagnostics = stats.analyze_repeated_metrics(observations(), DEFAULT_CIRCLE_PAIRS_MM)
    assert methods == ["lbfgs", "bfgs", "powell"] * 2
    assert all(row["status"] == "unavailable" and np.isnan(row["p_value"]) for row in tests)
    assert all(len(diag["fit_attempts"]) == 3 for diag in diagnostics.values())


def test_predeclared_holm_families_and_incomplete_family_policy():
    summary = summarize_observations(observations())
    for idx, row in enumerate(summary):
        row["p_value"] = (idx + 1) / 100
    tests = []
    for idx, metric in enumerate(stats.REPEATED_METRICS):
        tests.extend((
            dict(metric=metric, comparison="cohort_average", p_value=0.01 + 0.01 * idx),
            dict(metric=metric, comparison="cohort_by_radius", p_value=0.03 + 0.01 * idx),
        ))
    info = stats.adjust_report_families(summary, tests)
    assert {key: val["size"] for key, val in info.items()} == {
        "between_reward_tortuosity_standalone": 1,
        "com_distance_to_reward_center_standalone": 1,
        "return_leg_distance_standalone": 1,
        "dual_circle_turnback_ratio_radius_contrasts": 3,
        "dual_circle_turnback_ratio_cohort_average": 1,
        "dual_circle_turnback_ratio_cohort_by_radius": 1,
        "home_vector_alignment_radius_contrasts": 3,
        "home_vector_alignment_cohort_average": 1,
        "home_vector_alignment_cohort_by_radius": 1,
    }
    for rows in (summary[:3], summary[3:6]):
        assert [row["p_value_holm"] for row in rows] == pytest.approx(holm_adjust([row["p_value"] for row in rows]))
    assert all(row["p_value_holm"] == row["p_value"] for row in summary[6:])
    assert all(row["p_value_holm"] == row["p_value"] for row in tests)

    summary[0]["p_value"] = np.nan
    tests[0]["p_value"] = np.nan
    info = stats.adjust_report_families(summary, tests)
    assert not info["dual_circle_turnback_ratio_radius_contrasts"]["complete"]
    assert not info["dual_circle_turnback_ratio_cohort_average"]["complete"]
    assert all(np.isnan(row["p_value_holm"]) for row in summary[:3])
    assert all(np.isfinite(row["p_value_holm"]) for row in summary[3:])
    assert all(np.isfinite(row["p_value_holm"]) for row in tests[1:])

    summary[6]["p_value"] = np.nan
    info = stats.adjust_report_families(summary, tests)
    assert not info["between_reward_tortuosity_standalone"]["complete"]
    assert np.isnan(summary[6]["p_value_holm"])
    assert all(row["p_value_holm"] == row["p_value"] for row in summary[7:])
    # Changing one metric's tests cannot change the other metric's adjustment.
    before = [row["p_value_holm"] for row in summary[3:6]]
    summary[0]["p_value"] = 1e-8
    summary[1]["p_value"] = 0.99
    stats.adjust_report_families(summary, tests)
    assert [row["p_value_holm"] for row in summary[3:6]] == before


def test_one_pair_is_not_a_repeated_measures_design():
    pair = DEFAULT_CIRCLE_PAIRS_MM[0]
    rows = [row for row in observations() if row.inner_radius_mm == pair[0]]
    _, tests, _, diagnostics = stats.analyze_repeated_metrics(rows, (pair,))
    assert all(row["status"] == "unavailable" for row in tests)
    assert all("repeated observations" in diag["reason"] for diag in diagnostics.values())


def test_complete_mixed_report_exports_adjusted_inference_and_exact_inputs(tmp_path, monkeypatch):
    from src.exporting import learner_metric_table as report

    rows = observations()
    rng = np.random.default_rng(672)
    for metric in report.STANDALONE_METRICS:
        for cohort in ("strong", "weak"):
            for fly in range(24):
                rows.append(MetricObservation(
                    unit_id=f"{cohort}-{fly}", sli=0.5, cohort=cohort, metric=metric,
                    inner_radius_mm=np.nan, outer_radius_mm=np.nan,
                    value=float(1 + (cohort == "strong") + rng.normal(0, 0.5)),
                    n_events=10, successes=np.nan, eligible=True, exclusion_reason="",
                ))
    metadata = {
        "cohort_definition": {"mode": "mean", "min_valid_sync_buckets": 3},
        "circle_pairs_mm": [{"inner_radius_mm": inner, "outer_radius_mm": outer}
                            for inner, outer in DEFAULT_CIRCLE_PAIRS_MM],
        "statistics": {},
    }
    monkeypatch.setattr(report, "collect_learner_metric_observations", lambda *args: (rows, metadata))
    paths = report.export_learner_metric_table(
        [], SimpleNamespace(learner_metric_table_stats="mixed"), None,
        str(tmp_path / "complete"),
    )
    with open(paths["summary_csv"]) as fh:
        summary = list(csv.DictReader(fh))
    assert len(summary) == 9
    assert all(row["test_status"] == "ok" for row in summary)
    assert all(np.isfinite(float(row["p_value_holm"])) for row in summary)
    with open(paths["model_input_csv"]) as fh:
        assert len(list(csv.DictReader(fh))) == 288
    with open(paths["metadata_json"]) as fh:
        exported_metadata = json.load(fh)
    assert all(family["complete"] for family in exported_metadata["statistics"]["comparison_families"].values())
    assert [int(row["family_size"]) for row in summary] == [3] * 6 + [1] * 3
    assert all(row["p_value_holm"] == row["p_value"] for row in summary[6:])
    assert "unadjusted standalone Welch tests" in exported_metadata["statistics"]["multiple_comparisons"]
    assert summary[0]["comparison_family"] != summary[3]["comparison_family"]
    assert exported_metadata["statistics"]["statsmodels_version"]
    markdown = Path(paths["summary_md"]).read_text()
    assert "at least 3 valid buckets" in markdown
    assert "three per metric by default" in markdown
