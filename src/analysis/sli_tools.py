from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class SLISelectionSpec:
    training_idx: int  # 0-based
    bucket_idx: Optional[int]  # 0-based or None (use default in compute_sli_per_fly)
    average_over_buckets: bool = False


def compute_sli_per_fly(
    perf4: np.ndarray,
    training_idx: int,
    bucket_idx: Optional[int] = None,
    average_over_buckets: bool = False,
    *,
    skip_first_sync_buckets: int = 0,
) -> pd.Series:
    """
    Compute SLI per fly.

    If average_over_buckets is False:
        Use a single sync bucket (bucket_idx or default = final sync bucket - 2).
    If average_over_buckets is True:
        Use the mean across *all* sync buckets for the given training_idx.
    """
    n_vids = perf4.shape[0]
    nb = perf4.shape[3]

    k = int(skip_first_sync_buckets or 0)
    k = max(0, min(k, nb))  # clamp

    if training_idx < 0 or training_idx >= perf4.shape[1]:
        return pd.Series({vid: np.nan for vid in range(n_vids)}, name="SLI").astype(
            float
        )

    # Buckets eligible for use after skip
    start = k
    if start >= nb:
        # No buckets left; return all-NaN series (keeps downstream logic simple)
        return pd.Series({vid: np.nan for vid in range(n_vids)}, name="SLI").astype(
            float
        )

    if average_over_buckets:
        # Mean over all buckets (optionally skipping first k) in this training for each fly
        sli = {
            vid: (
                np.nanmean(perf4[vid, training_idx, 0, start:])
                - np.nanmean(perf4[vid, training_idx, 1, start:])
            )
            for vid in range(n_vids)
        }
    else:
        if bucket_idx is None:
            # Preserve the old "near-end" intent, but do it within [start:].
            # Old default was nb - 2. Now use the same offset relative to the end,
            # but ensure we never go before `start`.
            default_idx = nb - 2
            bucket_idx = max(start, default_idx)

        # If user explicitly requested a bucket in the skipped region, that's probably a bug.
        if bucket_idx < start:
            print(
                f"[SLI] WARNING: requested bucket_idx={bucket_idx} is within skipped region "
                f"(<{start}); returning NaNs"
            )
            return pd.Series({vid: np.nan for vid in range(n_vids)}, name="SLI").astype(
                float
            )

        if bucket_idx >= nb:
            return pd.Series({vid: np.nan for vid in range(n_vids)}, name="SLI").astype(
                float
            )

        sli = {
            vid: perf4[vid, training_idx, 0, bucket_idx]
            - perf4[vid, training_idx, 1, bucket_idx]
            for vid in range(n_vids)
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


def compute_sli_set_groups(
    perf4: np.ndarray,
    pos_spec: SLISelectionSpec,
    neg_spec: Optional[SLISelectionSpec],
    fraction: float,
) -> Dict[str, List[int]]:
    """
    Compute top-fraction groups for a positive and (optional) negative SLI spec,
    and return basic set operations over their *top* groups.

    Returns indices into the first axis of perf4 (i.e., aligned with vas).
    Keys:
        - 'pos_top'
        - 'neg_top' (if neg_spec provided)
        - 'intersection'
        - 'pos_minus_neg'
        - 'neg_minus_pos'
        - 'union'
    """
    # Positive selection
    sli_pos = compute_sli_per_fly(
        perf4,
        training_idx=pos_spec.training_idx,
        bucket_idx=pos_spec.bucket_idx,
        average_over_buckets=pos_spec.average_over_buckets,
    )
    _, pos_top = select_extremes(sli_pos, fraction=fraction)
    pos_set = set(pos_top)

    groups: Dict[str, List[int]] = {
        "pos_top": sorted(pos_set),
    }

    if neg_spec is not None:
        sli_neg = compute_sli_per_fly(
            perf4,
            training_idx=neg_spec.training_idx,
            bucket_idx=neg_spec.bucket_idx,
            average_over_buckets=neg_spec.average_over_buckets,
        )
        _, neg_top = select_extremes(sli_neg, fraction=fraction)
        neg_set = set(neg_top)

        groups.update(
            {
                "neg_top": sorted(neg_set),
                "intersection": sorted(pos_set & neg_set),
                "pos_minus_neg": sorted(pos_set - neg_set),
                "neg_minus_pos": sorted(neg_set - pos_set),
                "union": sorted(pos_set | neg_set),
            }
        )

    return groups
