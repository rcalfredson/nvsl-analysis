from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CorrelationSummary:
    r: float
    p: float
    n: int


@dataclass(frozen=True)
class FisherIndependentCorrelationResult:
    r1: float
    r2: float
    n1: int
    n2: int
    z1: float
    z2: float
    se: float
    z_stat: float
    p_two_sided: float


def pearson_correlation_summary(x: np.ndarray, y: np.ndarray) -> CorrelationSummary:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 3 or np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return CorrelationSummary(r=np.nan, p=np.nan, n=n)
    r, p = stats.pearsonr(x, y)
    return CorrelationSummary(r=float(r), p=float(p), n=n)


def fisher_independent_correlation_test(
    r1: float,
    n1: int,
    r2: float,
    n2: int,
) -> FisherIndependentCorrelationResult:
    """
    Fisher r-to-z test for comparing two independent Pearson correlations.
    """
    r1 = float(r1)
    r2 = float(r2)
    n1 = int(n1)
    n2 = int(n2)
    if n1 <= 3 or n2 <= 3:
        raise ValueError("Fisher independent-correlation test requires n > 3 per group.")
    if not np.isfinite(r1) or not np.isfinite(r2):
        raise ValueError("correlations must be finite.")

    r1_clip = float(np.clip(r1, -0.999999, 0.999999))
    r2_clip = float(np.clip(r2, -0.999999, 0.999999))
    z1 = float(np.arctanh(r1_clip))
    z2 = float(np.arctanh(r2_clip))
    se = float(np.sqrt((1.0 / (n1 - 3)) + (1.0 / (n2 - 3))))
    z_stat = float((z1 - z2) / se)
    p_two_sided = float(2.0 * stats.norm.sf(abs(z_stat)))
    return FisherIndependentCorrelationResult(
        r1=r1,
        r2=r2,
        n1=n1,
        n2=n2,
        z1=z1,
        z2=z2,
        se=se,
        z_stat=z_stat,
        p_two_sided=p_two_sided,
    )
