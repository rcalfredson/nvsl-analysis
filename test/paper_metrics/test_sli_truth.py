import numpy as np
import pandas as pd
import pytest

from src.analysis.sli_tools import (
    compute_sli_per_fly,
    resolve_sync_bucket_selector,
    select_fractional_groups,
)


def _perf4(exp, ctrl):
    """Build a minimal (video, training, fly, sync-bucket) SLI fixture."""
    return np.stack(
        [np.asarray(exp, dtype=float), np.asarray(ctrl, dtype=float)], axis=2
    )


def test_sli_scalar_defaults_to_historical_penultimate_available_bucket():
    perf4 = _perf4(
        exp=[
            [[10, 20, 30, 40, 50, np.nan]],
            [[1, 2, 3, 4, 5, np.nan]],
        ],
        ctrl=[
            [[5, 6, 7, 8, 9, np.nan]],
            [[10, 20, 30, 40, 50, np.nan]],
        ],
    )

    sli = compute_sli_per_fly(perf4, training_idx=0)

    # With five finite buckets (six total) and no explicit selector,
    # legacy behavior uses end - 2: bucket index 4, i.e. SB5 in
    # user-facing 1-based language.
    np.testing.assert_allclose(sli.to_numpy(), [50 - 9, 5 - 50])


def test_sli_scalar_training_mean_honors_skip_keep_and_nan_handling():
    perf4 = _perf4(
        exp=[[[100, 2, 4, np.nan]], [[100, np.nan, 30, 40]]],
        ctrl=[[[100, 1, np.nan, 10]], [[100, 10, 20, np.nan]]],
    )

    sli = compute_sli_per_fly(
        perf4,
        training_idx=0,
        average_over_buckets=True,
        skip_first_sync_buckets=1,
        keep_first_sync_buckets=3,
    )

    # The first bucket is skipped. Means are computed separately for exp and ctrl
    # over SB2-SB4, then subtracted.
    np.testing.assert_allclose(
        sli.to_numpy(),
        [
            np.nanmean([2, 4, np.nan]) - np.nanmean([1, np.nan, 10]),
            np.nanmean([np.nan, 30, 40]) - np.nanmean([10, 20, np.nan]),
        ],
    )


def test_sli_scalar_explicit_bucket_must_fall_inside_selection_window():
    perf4 = _perf4(exp=[[[1, 2, 3, 4]]], ctrl=[[[0, 0, 0, 0]]])

    sli = compute_sli_per_fly(
        perf4, training_idx=0, bucket_idx=0, skip_first_sync_buckets=1
    )

    assert np.isnan(sli.iloc[0])


def test_sli_scalar_out_of_range_training_returns_nan_per_video():
    perf4 = _perf4(exp=[[[1, 2, 3]], [[4, 5, 6]]], ctrl=[[[0, 0, 0]], [[1, 1, 1]]])

    sli = compute_sli_per_fly(perf4, training_idx=3)

    assert list(sli.index) == [0, 1]
    assert sli.isna().all()


def test_sync_bucket_selector_resolves_user_facing_windowed_tokens():
    assert resolve_sync_bucket_selector("first", nb=6, skip_first_sync_buckets=2) == 2
    assert (
        resolve_sync_bucket_selector(
            "last", nb=6, skip_first_sync_buckets=1, keep_first_sync_buckets=4
        )
        == 3
    )
    assert (
        resolve_sync_bucket_selector(
            "4", nb=6, skip_first_sync_buckets=1, keep_first_sync_buckets=4
        )
        == 3
    )


def test_sync_bucket_selector_rejects_explicit_bucket_outside_window():
    with pytest.raises(ValueError, match="outside the included window"):
        resolve_sync_bucket_selector(
            "1", nb=6, skip_first_sync_buckets=1, keep_first_sync_buckets=4
        )


def test_fractional_sli_groups_ignore_nan_and_select_disjoint_extremes():
    sli = pd.Series(
        {
            "nan_fly": np.nan,
            "lowest": -0.2,
            "low": 0.1,
            "middle": 0.3,
            "high": 0.7,
            "highest": 0.9,
        }
    )

    bottom, top = select_fractional_groups(
        sli, bottom_fraction=0.4, top_fraction=0.4
    )

    assert bottom == ['lowest', 'low']
    assert top == ['high', 'highest']

def test_fractional_sli_groups_partition_when_fractions_sum_to_one():
    sli = pd.Series({0: -2.0, 1: -1.0, 2: 0.0, 3: 1.0, 4: 2.0})

    bottom, top = select_fractional_groups(
        sli,
        bottom_fraction=0.8,
        top_fraction=0.2
    )

    assert bottom == [0, 1, 2, 3]
    assert top == [4]

def test_fractional_sli_groups_reject_overlapping_fraction_request():
    sli = pd.Series([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="must be <= 1"):
        select_fractional_groups(sli, bottom_fraction=0.6, top_fraction=0.6)