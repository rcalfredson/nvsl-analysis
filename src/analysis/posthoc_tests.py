from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PairwisePosthocResult:
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    mean_difference_a_minus_b: float
    ci95_low: float
    ci95_high: float
    statistic: float
    df: float
    p_value: float
    p_value_adjusted: float
    test: str


def clean_sample(values) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def holm_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return []
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not np.any(finite):
        return out.tolist()

    pf = np.clip(p[finite], 0.0, 1.0)
    order = np.argsort(pf)
    p_sorted = pf[order]
    adj_sorted = np.empty_like(p_sorted)
    running_max = 0.0
    m = int(p_sorted.size)
    for idx, pv in enumerate(p_sorted):
        running_max = max(running_max, float((m - idx) * pv))
        adj_sorted[idx] = min(1.0, running_max)
    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    out[finite] = adj
    return out.tolist()


def welch_t_pair(
    group_a: str,
    sample_a,
    group_b: str,
    sample_b,
    *,
    p_value_adjusted: float = np.nan,
) -> PairwisePosthocResult:
    xa = clean_sample(sample_a)
    xb = clean_sample(sample_b)
    n_a = int(xa.size)
    n_b = int(xb.size)
    mean_a = float(np.mean(xa)) if n_a else np.nan
    mean_b = float(np.mean(xb)) if n_b else np.nan
    mean_diff = mean_a - mean_b
    if n_a < 2 or n_b < 2:
        return PairwisePosthocResult(
            group_a,
            group_b,
            n_a,
            n_b,
            mean_a,
            mean_b,
            mean_diff,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            float(p_value_adjusted),
            "Welch independent-samples t-test",
        )

    var_a = float(np.var(xa, ddof=1))
    var_b = float(np.var(xb, ddof=1))
    se2 = var_a / n_a + var_b / n_b
    if se2 <= 0:
        df = statistic = p_value = ci_low = ci_high = np.nan
    else:
        df = float(
            se2**2
            / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
        )
        se = float(np.sqrt(se2))
        statistic, p_value = stats.ttest_ind(xa, xb, equal_var=False)
        half = float(stats.t.ppf(0.975, df=df) * se)
        ci_low = mean_diff - half
        ci_high = mean_diff + half

    return PairwisePosthocResult(
        group_a=group_a,
        group_b=group_b,
        n_a=n_a,
        n_b=n_b,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_difference_a_minus_b=float(mean_diff),
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        statistic=float(statistic),
        df=float(df),
        p_value=float(p_value),
        p_value_adjusted=float(p_value_adjusted),
        test="Welch independent-samples t-test",
    )


def games_howell_pair(
    group_a: str,
    sample_a,
    group_b: str,
    sample_b,
    *,
    n_groups: int,
) -> PairwisePosthocResult:
    xa = clean_sample(sample_a)
    xb = clean_sample(sample_b)
    n_a = int(xa.size)
    n_b = int(xb.size)
    mean_a = float(np.mean(xa)) if n_a else np.nan
    mean_b = float(np.mean(xb)) if n_b else np.nan
    mean_diff = mean_a - mean_b

    if n_a < 2 or n_b < 2 or int(n_groups) < 2:
        q_stat = df = p_value = ci_low = ci_high = np.nan
    else:
        var_a = float(np.var(xa, ddof=1))
        var_b = float(np.var(xb, ddof=1))
        se2 = var_a / n_a + var_b / n_b
        if se2 <= 0:
            q_stat = df = p_value = ci_low = ci_high = np.nan
        else:
            df = float(
                se2**2
                / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
            )
            q_se = float(np.sqrt(0.5 * se2))
            q_stat = abs(float(mean_diff)) / q_se
            p_value = float(stats.studentized_range.sf(q_stat, int(n_groups), df))
            q_crit = float(stats.studentized_range.ppf(0.95, int(n_groups), df))
            half = q_crit * q_se
            ci_low = mean_diff - half
            ci_high = mean_diff + half

    return PairwisePosthocResult(
        group_a=group_a,
        group_b=group_b,
        n_a=n_a,
        n_b=n_b,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_difference_a_minus_b=float(mean_diff),
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        statistic=float(q_stat),
        df=float(df),
        p_value=float(p_value),
        p_value_adjusted=float(p_value),
        test="Games-Howell post-hoc test",
    )
