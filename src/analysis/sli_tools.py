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
    keep_first_sync_buckets: int = 0,
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

    kskip = max(0, min(int(skip_first_sync_buckets or 0), nb))
    kkeep = int(keep_first_sync_buckets or 0)
    if kkeep < 0:
        kkeep = 0

    # Buckets eligible for use after skip
    start = kskip
    end = nb if kkeep == 0 else min(nb, start + kkeep)

    if training_idx < 0 or training_idx >= perf4.shape[1]:
        return pd.Series({vid: np.nan for vid in range(n_vids)}, name="SLI").astype(
            float
        )

    if end <= start:
        return pd.Series({vid: np.nan for vid in range(n_vids)}, name="SLI").astype(
            float
        )

    if average_over_buckets:
        # Mean over all buckets (optionally skipping first k) in this training for each fly
        sli = {
            vid: (
                np.nanmean(perf4[vid, training_idx, 0, start:end])
                - np.nanmean(perf4[vid, training_idx, 1, start:end])
            )
            for vid in range(n_vids)
        }
    else:
        if bucket_idx is None:
            default_idx = end - 2
            bucket_idx = max(start, default_idx)

        # Disallow buckets outside the window
        if bucket_idx < start or bucket_idx >= end:
            print(
                f"[SLI] WARNING: requested bucket_idx={bucket_idx} outside included window "
                f"[{start}, {end}); returning NaNs"
            )
            return pd.Series({vid: np.nan for vid in range(n_vids)}, name="SLI").astype(
                float
            )

        sli = {
            vid: perf4[vid, training_idx, 0, bucket_idx]
            - perf4[vid, training_idx, 1, bucket_idx]
            for vid in range(n_vids)
        }

    return pd.Series(sli, name="SLI").astype(float)


def _validate_fraction(name: str, fraction: Optional[float]) -> None:
    if fraction is None:
        return
    if not (0 < float(fraction) <= 1):
        raise ValueError(f"{name} must be in the interval (0, 1], got {fraction!r}")


def _count_from_fraction(n: int, fraction: float) -> int:
    """
    Convert a fraction to a number of flies, keeping historical behavior:
    use floor via int(n * fraction), but always select at least 1 fly if the
    fraction is positive and n > 0.
    """
    if n <= 0:
        return 0
    return max(1, int(n * fraction))


def select_fractional_groups(
    sli_series: pd.Series,
    *,
    top_fraction: Optional[float] = None,
    bottom_fraction: Optional[float] = None,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Select bottom and/or top groups from an SLI series using potentially
    different fractions.

    Returns
    -------
    (bottom, top)
        Each element is either a list of fly indices or None if that side
        was not requested.

    Notes
    -----
    - This function allows overlap between top and bottom selections when the
      requested fractions imply it.
    - NaNs are ignored automatically by pandas nsmallest/nlargest.
    """
    _validate_fraction("bottom_fraction", bottom_fraction)
    _validate_fraction("top_fraction", top_fraction)

    n = len(sli_series)

    bottom = None
    top = None

    if bottom_fraction is not None:
        k_bottom = _count_from_fraction(n, float(bottom_fraction))
        bottom = sli_series.nsmallest(k_bottom).index.tolist()

    if top_fraction is not None:
        k_top = _count_from_fraction(n, float(top_fraction))
        top = sli_series.nlargest(k_top).index.tolist()

    return bottom, top


def select_extremes(
    sli_series: pd.Series, fraction: float = 0.1
) -> Tuple[List[int], List[int]]:
    """
    Backward-compatible wrapper: select the same fraction from the bottom and top.
    """
    bottom, top = select_fractional_groups(
        sli_series, top_fraction=fraction, bottom_fraction=fraction
    )
    # Historical contract: always return lists, never None
    bottom = [] if bottom is None else bottom
    top = [] if top is None else top
    return bottom, top


def compute_sli_set_groups(
    perf4: np.ndarray,
    pos_spec: SLISelectionSpec,
    neg_spec: Optional[SLISelectionSpec],
    fraction: float,
    skip_first_sync_buckets: int = 0,
    keep_first_sync_buckets: int = 0,
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
        skip_first_sync_buckets=skip_first_sync_buckets,
        keep_first_sync_buckets=keep_first_sync_buckets,
    )
    _, pos_top = select_fractional_groups(
        sli_pos,
        top_fraction=fraction,
        bottom_fraction=None,
    )
    pos_top = [] if pos_top is None else pos_top
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
            skip_first_sync_buckets=skip_first_sync_buckets,
            keep_first_sync_buckets=keep_first_sync_buckets,
        )
        _, neg_top = select_fractional_groups(
            sli_neg, top_fraction=fraction, bottom_fraction=None
        )
        neg_top = [] if neg_top is None else neg_top
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
