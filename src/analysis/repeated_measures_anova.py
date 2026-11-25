"""Two-way repeated-measures ANOVA with independent groups and one repeated factor."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import f, studentized_range, t

from src.analysis.posthoc_tests import PairwisePosthocResult


@dataclass(frozen=True)
class RepeatedMeasuresEffect:
    effect: str
    sum_squares: float
    error_sum_squares: float
    df_numerator: int
    df_denominator: int
    statistic: float
    p_value: float


def repeated_measures_anova_tukey(groups):
    """Analyze finite subject-by-level matrices under sphericity.

    Uses Type III sums of squares for the fixed effects. Tukey–Kramer
    compares independent groups within each repeated level, one family per
    level. Its pooled within-cell error combines subjects and residuals as
    described in GraphPad FAQ 1265.
    """
    names = tuple(groups)
    matrices = [np.asarray(groups[name], dtype=float) for name in names]
    if len(matrices) < 2 or any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("Expected at least two subject-by-level matrices")
    n_levels = matrices[0].shape[1]
    if n_levels < 2 or any(
        matrix.shape[1] != n_levels or matrix.shape[0] < 2 or not np.isfinite(matrix).all()
        for matrix in matrices
    ):
        raise ValueError("Groups must have complete observations at the same repeated levels")

    ns = np.array([len(matrix) for matrix in matrices])
    means = np.array([matrix.mean(axis=0) for matrix in matrices])
    group_means = means.mean(axis=1)
    group_profiles = means - group_means[:, None]
    group_ss = n_levels * np.dot(ns, (group_means - np.average(group_means, weights=ns)) ** 2)
    # The Type III repeated-factor main effect averages the group profiles equally.
    repeated_ss = np.sum(group_profiles.mean(axis=0) ** 2) / np.sum(1 / ns) * len(names) ** 2
    interaction_ss = np.dot(
        ns, np.sum((group_profiles - np.average(group_profiles, axis=0, weights=ns)) ** 2, axis=1)
    )
    cell_residuals = [matrix - mean for matrix, mean in zip(matrices, means, strict=True)]
    subjects_ss = n_levels * sum(np.sum(residual.mean(axis=1) ** 2) for residual in cell_residuals)
    residual_ss = sum(
        np.sum((residual - residual.mean(axis=1, keepdims=True)) ** 2)
        for residual in cell_residuals
    )
    subjects_df = int(ns.sum()) - len(names)
    residual_df = subjects_df * (n_levels - 1)
    if subjects_ss <= 0 or residual_ss <= 0:
        raise ValueError("ANOVA requires positive between-subject and within-subject error variance")

    effects = []
    for name, ss, error, df, error_df in (
        ("group", group_ss, subjects_ss, len(names) - 1, subjects_df),
        ("repeated factor", repeated_ss, residual_ss, n_levels - 1, residual_df),
        ("group × repeated factor", interaction_ss, residual_ss,
         (len(names) - 1) * (n_levels - 1), residual_df),
    ):
        statistic = (ss / df) / (error / error_df)
        effects.append(RepeatedMeasuresEffect(name, ss, error, df, error_df, statistic,
                                              float(f.sf(statistic, df, error_df))))

    pooled_df = subjects_df + residual_df
    pooled_ms = (subjects_ss + residual_ss) / pooled_df
    critical = studentized_range.ppf(.95, len(names), pooled_df)
    comparisons = []
    for level in range(n_levels):
        level_results = []
        for a, b in combinations(range(len(names)), 2):
            difference = means[a, level] - means[b, level]
            se = np.sqrt(pooled_ms * (1 / ns[a] + 1 / ns[b]))
            statistic = np.sqrt(2) * abs(difference) / se
            half_width = critical * se / np.sqrt(2)
            level_results.append(PairwisePosthocResult(
                names[a], names[b], int(ns[a]), int(ns[b]), means[a, level], means[b, level],
                difference, difference - half_width, difference + half_width, statistic, pooled_df,
                float(2 * t.sf(abs(difference) / se, pooled_df)),
                float(studentized_range.sf(statistic, len(names), pooled_df)),
                "Tukey–Kramer",
            ))
        comparisons.append(level_results)
    return effects, comparisons
