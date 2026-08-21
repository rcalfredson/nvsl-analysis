from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.oneway import anova_oneway


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


@dataclass(frozen=True)
class WelchAnovaResult:
    groups: tuple[str, ...]
    ns: tuple[int, ...]
    statistic: float
    df_numerator: float
    df_denominator: float
    p_value: float
    test: str = "Welch's one-way ANOVA"


def clean_sample(values) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def welch_anova(
    group_samples,
    *,
    group_names: list[str] | tuple[str, ...] | None = None,
    min_n_per_group: int = 2,
) -> WelchAnovaResult:
    """Welch's heteroscedastic one-way ANOVA via Statsmodels."""
    clean = [clean_sample(sample) for sample in group_samples]
    if group_names is None:
        names = tuple(f"group_{idx + 1}" for idx in range(len(clean)))
    else:
        names = tuple(str(name) for name in group_names)
    if len(names) != len(clean):
        raise ValueError("group_names must have the same length as group_samples")

    keep = [
        idx
        for idx, sample in enumerate(clean)
        if sample.size >= int(min_n_per_group)
    ]
    samples = [clean[idx] for idx in keep]
    kept_names = tuple(names[idx] for idx in keep)
    ns = tuple(int(sample.size) for sample in samples)
    k = len(samples)
    if k < 2:
        return WelchAnovaResult(
            kept_names, ns, np.nan, np.nan, np.nan, np.nan
        )

    try:
        result = anova_oneway(
            samples,
            use_var="unequal",
            welch_correction=True,
        )
    except Exception:
        return WelchAnovaResult(
            kept_names, ns, np.nan, float(k - 1), np.nan, np.nan
        )
    df_num, df_den = result.df
    return WelchAnovaResult(
        kept_names,
        ns,
        float(result.statistic),
        float(df_num),
        float(df_den),
        float(result.pvalue),
    )


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


def games_howell_all_pairs(
    group_samples,
    *,
    group_names: list[str] | tuple[str, ...],
    alpha: float = 0.05,
    min_n_per_group: int = 2,
) -> list[PairwisePosthocResult]:
    """All-pairs Games-Howell comparisons via Pingouin."""
    names = tuple(str(name) for name in group_names)
    clean = [clean_sample(sample) for sample in group_samples]
    if len(names) != len(clean):
        raise ValueError("group_names must have the same length as group_samples")
    if len(set(names)) != len(names):
        raise ValueError("group_names must be unique")
    keep = [
        idx
        for idx, sample in enumerate(clean)
        if sample.size >= int(min_n_per_group)
    ]
    clean = [clean[idx] for idx in keep]
    names = tuple(names[idx] for idx in keep)
    if len(clean) < 2:
        return []

    values = np.concatenate(clean)
    labels = np.concatenate(
        [np.repeat(name, sample.size) for name, sample in zip(names, clean)]
    )
    frame = pd.DataFrame({"value": values, "group": labels})
    table = pg.pairwise_gameshowell(
        data=frame,
        dv="value",
        between="group",
    )

    samples_by_name = dict(zip(names, clean))
    results = []
    for row in table.itertuples(index=False):
        group_a = str(getattr(row, "A"))
        group_b = str(getattr(row, "B"))
        xa = samples_by_name[group_a]
        xb = samples_by_name[group_b]
        mean_a = float(np.mean(xa))
        mean_b = float(np.mean(xb))
        diff = mean_a - mean_b
        se = float(getattr(row, "se"))
        df = float(getattr(row, "df"))
        q_crit = float(stats.studentized_range.ppf(1.0 - alpha, len(clean), df))
        half = q_crit * se / np.sqrt(2.0)
        t_stat = float(getattr(row, "T"))
        results.append(
            PairwisePosthocResult(
                group_a=group_a,
                group_b=group_b,
                n_a=int(xa.size),
                n_b=int(xb.size),
                mean_a=mean_a,
                mean_b=mean_b,
                mean_difference_a_minus_b=diff,
                ci95_low=diff - half,
                ci95_high=diff + half,
                statistic=abs(t_stat) * np.sqrt(2.0),
                df=df,
                p_value=float(getattr(row, "pval")),
                p_value_adjusted=float(getattr(row, "pval")),
                test="Games-Howell post-hoc test",
            )
        )
    return results
