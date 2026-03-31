from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class SLISelectionSpec:
    training_idx: int  # 0-based
    bucket_idx: Optional[int]  # 0-based or None (use default in compute_sli_per_fly)
    average_over_buckets: bool = False


def default_single_bucket_idx(start: int, end: int) -> int:
    """
    Historical single-bucket SLI selection.

    The sync-bucket arrays may include a final theoretical slot that is not
    practically occupiable for reward-anchored buckets, so the legacy code uses
    ``end - 2`` rather than ``end - 1`` whenever the window spans at least two
    slots.
    """
    start = int(start)
    end = int(end)
    if end <= start:
        return start
    if end - start <= 1:
        return start
    return end - 2


def resolve_sync_bucket_selector(
    bucket_selector: Optional[str],
    *,
    nb: int,
    skip_first_sync_buckets: int = 0,
    keep_first_sync_buckets: int = 0,
) -> Optional[int]:
    """
    Resolve a user-facing sync-bucket selector to a 0-based bucket index.

    Parameters
    ----------
    bucket_selector
        Either ``None`` (no explicit bucket), ``"first"``, ``"last"``, or a
        1-based bucket index encoded as a string.
    nb
        Total number of sync buckets available for the training.
    skip_first_sync_buckets, keep_first_sync_buckets
        Optional windowing constraints. The resolved bucket must lie inside the
        included window after these constraints are applied.
    """
    if bucket_selector is None:
        return None

    token = str(bucket_selector).strip().lower()
    if not token:
        return None

    start = max(0, min(int(skip_first_sync_buckets or 0), nb))
    keep = max(0, int(keep_first_sync_buckets or 0))
    end = nb if keep == 0 else min(nb, start + keep)
    if end <= start:
        return None

    if token == "first":
        return start
    if token == "last":
        return default_single_bucket_idx(start, end)

    try:
        bucket_1based = int(token)
    except Exception as exc:
        raise ValueError(
            "sync-bucket selector must be 'first', 'last', or a 1-based integer"
        ) from exc

    bucket_idx = bucket_1based - 1
    if bucket_idx < start or bucket_idx >= end:
        raise ValueError(
            f"requested sync bucket SB{bucket_1based} lies outside the included "
            f"window SB{start + 1}-SB{end}"
        )
    return bucket_idx


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
            bucket_idx = default_single_bucket_idx(start, end)

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


def _fractional_group_counts(
    n: int,
    *,
    bottom_fraction: Optional[float],
    top_fraction: Optional[float],
) -> Tuple[int, int]:
    """
    Convert requested fractions to bottom/top counts.

    Historical behavior uses floor on each side. When both fractions are
    requested and sum to 1, we instead force an exhaustive partition by
    assigning any rounding remainder to the larger-fraction side. Ties go to
    the bottom group for determinism.
    """
    k_bottom = (
        _count_from_fraction(n, float(bottom_fraction))
        if bottom_fraction is not None
        else 0
    )
    k_top = (
        _count_from_fraction(n, float(top_fraction))
        if top_fraction is not None
        else 0
    )

    if (
        n > 0
        and bottom_fraction is not None
        and top_fraction is not None
        and np.isclose(float(bottom_fraction) + float(top_fraction), 1.0, atol=1e-12)
    ):
        assigned = k_bottom + k_top
        if assigned < n:
            remainder = n - assigned
            if float(bottom_fraction) >= float(top_fraction):
                k_bottom += remainder
            else:
                k_top += remainder

    if top_fraction is not None and bottom_fraction is not None:
        max_bottom = max(0, n - k_top)
        k_bottom = min(k_bottom, max_bottom)

    return k_bottom, k_top


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
    - Fractions are computed among flies with finite SLI only.
    - When both top and bottom selections are requested, they are made disjoint.
    - If top_fraction + bottom_fraction == 1, the finite flies are partitioned
      exhaustively; any rounding remainder is assigned to the larger-fraction
      side (bottom on ties).
    - Requests with top_fraction + bottom_fraction > 1 are rejected.
    """
    _validate_fraction("bottom_fraction", bottom_fraction)
    _validate_fraction("top_fraction", top_fraction)

    finite = sli_series.dropna()
    n_finite = len(finite)

    if (
        top_fraction is not None
        and bottom_fraction is not None
        and float(top_fraction) + float(bottom_fraction) > 1.0 + 1e-12
    ):
        raise ValueError(
            "top_fraction + bottom_fraction must be <= 1 for disjoint selections "
            f"(got top_fraction={top_fraction!r}, "
            f"bottom_fraction={bottom_fraction!r})"
        )

    bottom = None
    top = None

    if n_finite == 0:
        return [] if bottom_fraction is not None else None, [] if top_fraction is not None else None

    order = finite.sort_values(kind="mergesort")

    k_bottom, k_top = _fractional_group_counts(
        n_finite,
        bottom_fraction=bottom_fraction,
        top_fraction=top_fraction,
    )

    if bottom_fraction is not None:
        bottom = order.index[:k_bottom].tolist() if k_bottom > 0 else []

    if top_fraction is not None:
        if bottom_fraction is not None and k_bottom > 0:
            top_pool = order.index[k_bottom:]
            top = top_pool[-k_top:].tolist() if k_top > 0 else []
        else:
            top = order.index[-k_top:].tolist() if k_top > 0 else []

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
