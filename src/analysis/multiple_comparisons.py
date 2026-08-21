from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests


def holm_adjust(pvals: list[float]) -> list[float]:
    """Adjust finite p-values with Statsmodels' Holm procedure."""
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return []

    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not np.any(finite):
        return out.tolist()

    finite_pvals = np.clip(p[finite], 0.0, 1.0)
    out[finite] = multipletests(finite_pvals, method="holm")[1]
    return out.tolist()
