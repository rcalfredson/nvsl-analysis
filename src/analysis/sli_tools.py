import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


def compute_sli_per_fly(
    perf4: np.ndarray, training_idx: int, bucket_idx: Optional[int] = None
) -> pd.Series:
    if bucket_idx is None:
        bucket_idx = perf4.shape[3] - 2
    sli = {
        vid: perf4[vid, training_idx, 0, bucket_idx]
        - perf4[vid, training_idx, 1, bucket_idx]
        for vid in range(perf4.shape[0])
    }
    return pd.Series(sli, name="SLI").astype(float)


def select_extremes(
    sli_series: pd.Series, fraction: float = 0.1
) -> Tuple[List[int], List[int]]:
    n = len(sli_series)
    k = max(1, int(n * fraction))
    bottom = sli_series.nsmallest(k).index.tolist()
    top = sli_series.nlargest(k).index.tolist()
    return bottom, top
