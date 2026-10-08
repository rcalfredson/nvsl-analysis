"""Fly-level repeated-radius mixed models and explicitly defined test families.

Observations are pooled fly/radius readouts, never individual episodes. Missing
radius measurements are omitted row by row. Inference uses REML estimates and
asymptotic Wald tests; all confidence intervals are pointwise.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.stats import skew
from statsmodels.regression.mixed_linear_model import MixedLM

from src.analysis.multiple_comparisons import holm_adjust

REPEATED_METRICS = ("dual_circle_turnback_ratio", "home_vector_alignment")
MODEL_FORMULA = "value ~ cohort * radius_pair + (1 | unit_id)"
FIXED_EFFECT_FORMULA = "cohort * radius_pair"
MODEL_INPUT_FIELDS = (
    "metric", "unit_id", "cohort", "inner_radius_mm", "outer_radius_mm",
    "radius_pair", "value", "n_events", "successes",
)
COHORT_AUDIT_FIELDS = (
    "unit_id", "sli_t2_sb5", "sli_t2_sb2_sb5_mean",
    "sli_t2_sb2_sb5_valid_bucket_count", "selection_score", "rank_ascending",
    "cohort", "exclusion_reason",
)


def _contrast_template(metric, comparison, inner=np.nan, outer=np.nan):
    return {
        "metric": metric,
        "comparison": comparison,
        "inner_radius_mm": float(inner),
        "outer_radius_mm": float(outer),
        "estimate": np.nan,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "standard_error": np.nan,
        "statistic": np.nan,
        "df": 1,
        "p_value": np.nan,
        "test": "asymptotic Wald z",
        "status": "unavailable",
        "reason": "",
    }


def _wald_contrast(result, contrast):
    """Adapt Statsmodels' normal-reference contrast result to the report schema."""
    test = result.t_test(np.asarray(contrast)[None, :], use_t=False)
    se = float(np.asarray(test.sd).item())
    if not np.isfinite(se) or se <= 0:
        raise ValueError("nonpositive contrast variance")
    lo, hi = np.asarray(test.conf_int(alpha=0.05))[0]
    return {
        "estimate": float(np.asarray(test.effect).item()),
        "ci_lo": float(lo), "ci_hi": float(hi), "standard_error": se,
        "statistic": float(np.asarray(test.statistic).item()),
        "p_value": float(np.asarray(test.pvalue).item()),
        "status": "ok", "reason": "",
    }


def _design(cohorts, pair_indices, n_pairs):
    """Let Patsy encode categorical effects, with weak/first-radius references.

    Declare all levels even when a cell is missing so validation can detect a
    nonestimable full model rather than silently fitting a reduced design.
    """
    return dmatrix(FIXED_EFFECT_FORMULA, {
        "cohort": pd.Categorical(cohorts, categories=["weak", "strong"]),
        "radius_pair": pd.Categorical(pair_indices, categories=range(n_pairs)),
    }, NA_action="raise")


def _radius_contrasts(design_info, n_pairs):
    """Subtract weak from strong prediction rows using the fitted encoding."""
    prediction_design = build_design_matrices([design_info], {
        "cohort": ["weak", "strong"] * n_pairs,
        "radius_pair": np.repeat(np.arange(n_pairs), 2),
    }, NA_action="raise")[0]
    return np.asarray(prediction_design[1::2] - prediction_design[::2])


def _fit_model(data, design, diagnostics):
    for optimizer in ("lbfgs", "bfgs", "powell"):
        attempt = {"optimizer": optimizer, "warnings": []}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = MixedLM(data["value"].to_numpy(), design,
                                groups=data["unit_id"].to_numpy())
                result = model.fit(reml=True, method=optimizer, maxiter=1000, disp=False)
                beta = np.asarray(result.fe_params)
                covariance = np.asarray(result.cov_params())[:len(beta), :len(beta)]
                if not result.converged:
                    raise ValueError("model did not converge")
                if not np.isfinite(result.llf):
                    raise ValueError("nonfinite model log likelihood")
                if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(covariance))):
                    raise ValueError("nonfinite fixed-effect estimates or covariance")
                if not np.allclose(covariance, covariance.T):
                    raise ValueError("asymmetric fixed-effect covariance")
                np.linalg.cholesky(covariance)
                if (not np.isfinite(result.scale) or result.scale <= 0
                        or result.scale < 1e-12 * np.var(data["value"])):
                    raise ValueError("invalid residual variance")
                random_variance = float(np.asarray(result.cov_re)[0, 0])
                if not np.isfinite(random_variance) or random_variance < 0:
                    raise ValueError("invalid fly-intercept variance")
                if any("Hessian" in str(w.message) for w in caught):
                    raise ValueError("non-positive-definite Hessian")
                attempt["status"] = "ok"
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError,
                    FloatingPointError, RuntimeError) as exc:
                attempt.update(status="failed", reason=str(exc))
                result = None
            attempt["warnings"] = [str(w.message) for w in caught]
        diagnostics["fit_attempts"].append(attempt)
        if result is not None:
            diagnostics.update(
                optimizer=optimizer, converged=True,
                random_intercept_variance=random_variance,
                residual_variance=float(result.scale),
                log_likelihood=float(result.llf),
                boundary_fit=bool(random_variance < 1e-6 * result.scale),
                fixed_effect_estimates=beta.tolist(),
                fixed_effect_covariance=covariance.tolist(),
            )
            marginal_residuals = data["value"].to_numpy() - design @ beta
            try:
                residuals = np.asarray(result.resid)
                residual_type = "conditional"
            except (ValueError, np.linalg.LinAlgError):
                residuals = marginal_residuals
                residual_type = "marginal (random effects unavailable)"
            diagnostics["residuals"] = {
                "type": residual_type,
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "skewness": float(skew(residuals)),
                "by_radius_pair": {
                    str(pair): {
                        "n": int(len(indices)),
                        "mean": float(np.mean(residuals[indices])),
                        "sd": float(np.std(residuals[indices], ddof=1)) if len(indices) > 1 else np.nan,
                    }
                    for pair, indices in data.groupby("radius_pair").indices.items()
                },
            }
            diagnostics["observed_range"] = [float(data["value"].min()), float(data["value"].max())]
            fitted = data["value"].to_numpy() - residuals
            diagnostics["fitted_range"] = [float(np.min(fitted)), float(np.max(fitted))]
            return result
    raise ValueError("all optimizer attempts failed; see model diagnostics")


def analyze_repeated_metrics(observations, circle_pairs):
    """Return exact model rows, primary/interaction tests, radius contrasts, diagnostics."""
    pairs = tuple((float(a), float(b)) for a, b in circle_pairs)
    pair_lookup = {pair: idx for idx, pair in enumerate(pairs)}
    if not pairs or len(pair_lookup) != len(pairs):
        raise ValueError("mixed models require distinct configured circle pairs")
    seen = set()
    cohorts = {}
    inputs = []
    for row in observations:
        if row.cohort not in ("strong", "weak"):
            raise ValueError(f"unknown cohort for {row.unit_id}: {row.cohort}")
        if row.unit_id in cohorts and cohorts[row.unit_id] != row.cohort:
            raise ValueError(f"inconsistent cohort for {row.unit_id}")
        cohorts[row.unit_id] = row.cohort
        if row.metric not in REPEATED_METRICS:
            continue
        pair = (row.inner_radius_mm, row.outer_radius_mm)
        if pair not in pair_lookup:
            raise ValueError(f"unconfigured radius pair: {pair}")
        key = (row.metric, row.unit_id, pair)
        if key in seen:
            raise ValueError(f"duplicate fly/metric/radius observation: {key}")
        seen.add(key)
        if row.eligible and np.isfinite(row.value):
            inputs.append({
                "metric": row.metric, "unit_id": row.unit_id,
                "cohort": row.cohort, "inner_radius_mm": pair[0],
                "outer_radius_mm": pair[1], "radius_pair": pair_lookup[pair],
                "value": float(row.value), "n_events": int(row.n_events),
                "successes": float(row.successes),
            })

    tests, contrasts, diagnostics = [], {}, {}
    for metric in REPEATED_METRICS:
        data = pd.DataFrame([row for row in inputs if row["metric"] == metric],
                            columns=MODEL_INPUT_FIELDS)
        primary = _contrast_template(metric, "cohort_average")
        interaction = _contrast_template(metric, "cohort_by_radius")
        interaction.update(test="asymptotic Wald chi-square", df=len(pairs) - 1)
        per_radius = [_contrast_template(metric, "cohort_at_radius", *pair)
                      for pair in pairs]
        counts = data.groupby("cohort")["unit_id"].nunique().to_dict()
        for test in (primary, interaction):
            test.update(n_observations=len(data), strong_n=int(counts.get("strong", 0)),
                        weak_n=int(counts.get("weak", 0)))
        diag = {
            "formula": MODEL_FORMULA, "method": "REML", "status": "unavailable",
            "converged": False, "fit_attempts": [], "n_observations": len(data),
            "max_iterations": 1000,
            "fly_counts": {str(key): int(val) for key, val in counts.items()},
            "radius_pairs_mm": [list(pair) for pair in pairs],
            "fixed_effect_order": ["Intercept", "strong"] +
                [f"radius_{idx}" for idx in range(1, len(pairs))] +
                [f"strong:radius_{idx}" for idx in range(1, len(pairs))],
            "missingness_patterns": {},
            "available_by_radius": {},
        }
        for cohort in ("strong", "weak"):
            metric_units = [uid for uid, name in cohorts.items() if name == cohort]
            patterns = {}
            for uid in metric_units:
                available = sorted(data.loc[data["unit_id"] == uid, "radius_pair"].tolist())
                pattern = ",".join(str(idx) for idx in available) or "none"
                patterns[pattern] = patterns.get(pattern, 0) + 1
            diag["missingness_patterns"][cohort] = patterns
            diag["available_by_radius"][cohort] = [
                int(((data["cohort"] == cohort) & (data["radius_pair"] == idx)).sum())
                for idx in range(len(pairs))
            ]
        try:
            if min(counts.get("strong", 0), counts.get("weak", 0)) < 2:
                raise ValueError("requires at least two observed flies in each cohort")
            design = _design(data["cohort"], data["radius_pair"], len(pairs))
            if np.linalg.matrix_rank(design) != design.shape[1]:
                raise ValueError("nonestimable design: missing cohort/radius cell")
            if len(data) <= design.shape[1]:
                raise ValueError("insufficient residual degrees of freedom")
            repeats = data.groupby("unit_id").size()
            if int((repeats > 1).sum()) < 2:
                raise ValueError("requires repeated observations in at least two flies")
            diag["design_column_names"] = design.design_info.column_names
            result = _fit_model(data, design, diag)
            radius_vectors = _radius_contrasts(design.design_info, len(pairs))
            for test, vector in zip(per_radius, radius_vectors):
                test.update(_wald_contrast(result, vector))
            primary.update(_wald_contrast(result,
                                         np.mean(radius_vectors, axis=0)))
            if len(pairs) > 1:
                term = design.design_info.term_name_slices["cohort:radius_pair"]
                indices = np.arange(term.start, term.stop)
                # wald_test uses the full parameter vector, including variance
                # parameters; restrict only the fixed interaction coefficients.
                restrictions = np.zeros((len(indices), len(result.params)))
                restrictions[np.arange(len(indices)), indices] = 1
                joint = result.wald_test(restrictions, use_f=False, scalar=True)
                interaction.update(statistic=float(joint.statistic),
                                   p_value=float(joint.pvalue),
                                   status="ok")
            else:
                interaction["reason"] = "interaction requires more than one radius pair"
            diag["status"] = "ok"
        except (ValueError, np.linalg.LinAlgError) as exc:
            reason = str(exc)
            diag.update(status="unavailable", reason=reason)
            for test in (primary, interaction, *per_radius):
                test.update(status="unavailable", reason=reason,
                            estimate=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                            standard_error=np.nan, statistic=np.nan, p_value=np.nan)
        tests.extend((primary, interaction))
        for pair, test in zip(pairs, per_radius):
            contrasts[(metric, *pair)] = test
        diagnostics[metric] = diag
    return inputs, tests, contrasts, diagnostics


def adjust_report_families(summary, tests):
    """Keep repeated-metric analyses separate, including their Holm families.

    Radius contrasts are adjusted within each metric. Each metric's overall
    cohort and interaction tests are singleton families (p-values unchanged).
    Each standalone Welch test is a singleton (p-value unchanged). An
    incomplete family has no adjusted p-values, without affecting other families.
    """
    families = {
        f"{row['metric']}_standalone": [row]
        for row in summary if row["metric"] not in REPEATED_METRICS
    }
    for metric in REPEATED_METRICS:
        families[f"{metric}_radius_contrasts"] = [
            row for row in summary if row["metric"] == metric
        ]
        for comparison in ("cohort_average", "cohort_by_radius"):
            families[f"{metric}_{comparison}"] = [
                test for test in tests
                if test["metric"] == metric and test["comparison"] == comparison
            ]
    payload = {}
    for name, rows in families.items():
        complete = all(np.isfinite(row["p_value"]) for row in rows)
        adjusted = holm_adjust([row["p_value"] for row in rows]) if complete else [np.nan] * len(rows)
        for row, p_adjusted in zip(rows, adjusted):
            row.update(p_value_holm=float(p_adjusted), comparison_family=name,
                       family_complete=complete, family_size=len(rows))
        payload[name] = {"size": len(rows), "complete": complete,
                         "available": int(sum(np.isfinite(row["p_value"]) for row in rows))}
    return payload
