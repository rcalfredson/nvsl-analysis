from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.sync_bucket_presence_filters import (
    DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_SYNC_BUCKET,
    DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_TRAINING,
    exp_target_sync_bucket_eligibility_mask,
    exp_target_sync_bucket_filter_payload,
    exp_target_sync_bucket_filter_result,
    mask_by_exp_target_sync_bucket_filter,
)


def _ranges(n_buckets, *, start=0, width=10):
    return [(start + i * width, start + (i + 1) * width) for i in range(n_buckets)]


def _va_with_sync_ranges(n_t1, n_t2=None):
    if n_t2 is None:
        n_t2 = DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_SYNC_BUCKET
    return SimpleNamespace(sync_bucket_ranges=[_ranges(n_t1), _ranges(n_t2)])


def _va_with_buckets(edges_by_training):
    return SimpleNamespace(buckets=edges_by_training)


def test_exp_target_sync_bucket_filter_is_noop_when_disabled():
    result = exp_target_sync_bucket_filter_result(
        _va_with_sync_ranges(0, 0),
        SimpleNamespace(require_exp_target_sync_bucket=False, piTh=10),
    )

    assert result.eligible
    assert result.reason == "disabled"
    assert result.training == 2
    assert result.sync_bucket == 5
    assert result.available_sync_buckets == 0


def test_exp_target_sync_bucket_filter_passes_when_target_range_exists():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True, piTh=10)

    result = exp_target_sync_bucket_filter_result(
        _va_with_sync_ranges(1, DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_SYNC_BUCKET),
        opts,
    )

    assert result.eligible
    assert result.reason == "passes"
    assert result.available_sync_buckets == 5
    assert result.target_bucket_start == 40
    assert result.target_bucket_stop == 50


def test_exp_target_sync_bucket_filter_fails_when_target_range_is_missing():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True, piTh=10)

    result = exp_target_sync_bucket_filter_result(_va_with_sync_ranges(1, 4), opts)

    assert not result.eligible
    assert result.reason == "target_sync_bucket_missing"
    assert result.available_sync_buckets == 4
    assert np.isnan(result.target_bucket_start)
    assert np.isnan(result.target_bucket_stop)


def test_exp_target_sync_bucket_filter_ignores_pi_threshold_counts():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True, piTh=10)
    va = _va_with_sync_ranges(1, 5)
    va.flies = [0]
    va.numRewardsTot = [
        [[]],
        [
            [tuple([0, 0, 0, 0, 1])],
            [tuple([0, 0, 0, 0, 1])],
        ],
    ]
    va.reward_exclusion_mask = [
        [[False]],
        [[False, False, False, False, True]],
    ]

    result = exp_target_sync_bucket_filter_result(va, opts)

    assert result.eligible
    assert result.reason == "passes"
    assert result.available_sync_buckets == 5


def test_exp_target_sync_bucket_filter_can_use_custom_training_and_bucket():
    opts = SimpleNamespace(
        require_exp_target_sync_bucket=True,
        exp_target_sync_bucket_filter_training=1,
        exp_target_sync_bucket_filter_sync_bucket=2,
    )

    result = exp_target_sync_bucket_filter_result(_va_with_sync_ranges(2, 0), opts)

    assert result.eligible
    assert result.reason == "passes"
    assert result.training == 1
    assert result.sync_bucket == 2
    assert result.available_sync_buckets == 2


def test_exp_target_sync_bucket_filter_ignores_pi_threshold_option_names():
    opts = SimpleNamespace(
        require_exp_pi_threshold_bucket=True,
        exp_pi_threshold_filter_training=1,
        exp_pi_threshold_filter_sync_bucket=2,
    )

    result = exp_target_sync_bucket_filter_result(_va_with_sync_ranges(1, 0), opts)

    assert result.eligible
    assert result.reason == "disabled"
    assert result.training == 2
    assert result.sync_bucket == 5


def test_exp_target_sync_bucket_filter_falls_back_to_finite_bucket_edges():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True)
    va = _va_with_buckets(
        [
            [0, 10, 20],
            [100, 110, 120, 130, 140, 150, np.nan],
        ]
    )

    result = exp_target_sync_bucket_filter_result(va, opts)

    assert result.eligible
    assert result.reason == "passes"
    assert result.available_sync_buckets == 5
    assert result.target_bucket_start == 140
    assert result.target_bucket_stop == 150


def test_exp_target_sync_bucket_filter_fails_when_finite_bucket_edges_are_short():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True)
    va = _va_with_buckets(
        [
            [0, 10, 20],
            [100, 110, 120, 130, 140, np.nan],
        ]
    )

    result = exp_target_sync_bucket_filter_result(va, opts)

    assert not result.eligible
    assert result.reason == "target_sync_bucket_missing"
    assert result.available_sync_buckets == 4


def test_exp_target_sync_bucket_filter_reports_missing_data():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True)

    missing_data = exp_target_sync_bucket_filter_result(SimpleNamespace(), opts)
    missing_training = exp_target_sync_bucket_filter_result(
        SimpleNamespace(sync_bucket_ranges=[_ranges(2)]),
        opts,
    )

    assert not missing_data.eligible
    assert missing_data.reason == "missing_sync_bucket_data"
    assert not missing_training.eligible
    assert missing_training.reason == "target_training_missing"


def test_exp_target_sync_bucket_filter_reports_invalid_target_range():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True)
    va = SimpleNamespace(sync_bucket_ranges=[_ranges(2), _ranges(4) + [(140, 140)]])

    result = exp_target_sync_bucket_filter_result(va, opts)

    assert not result.eligible
    assert result.reason == "target_sync_bucket_invalid"
    assert result.available_sync_buckets == 5
    assert result.target_bucket_start == 140
    assert result.target_bucket_stop == 140


def test_exp_target_sync_bucket_eligibility_mask_and_payload_report_per_video_results():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True, piTh=7)
    vas = [
        _va_with_sync_ranges(1, 5),
        _va_with_sync_ranges(1, 4),
    ]

    eligible = exp_target_sync_bucket_eligibility_mask(vas, opts)
    payload = exp_target_sync_bucket_filter_payload(
        vas, opts, prefix="exp_target_sync_bucket_filter"
    )

    np.testing.assert_array_equal(eligible, [True, False])
    assert bool(payload["exp_target_sync_bucket_filter_enabled"])
    np.testing.assert_array_equal(
        payload["exp_target_sync_bucket_filter_eligible"], [True, False]
    )
    np.testing.assert_array_equal(
        payload["exp_target_sync_bucket_filter_reason"],
        ["passes", "target_sync_bucket_missing"],
    )
    np.testing.assert_array_equal(
        payload["exp_target_sync_bucket_filter_available_sync_buckets"], [5, 4]
    )
    assert "exp_target_sync_bucket_filter_target_bucket_start" in payload
    assert "exp_pi_threshold_filter_pi_threshold" in payload
    assert "exp_pi_threshold_filter_target_count_sum" in payload


def test_mask_by_exp_target_sync_bucket_filter_masks_first_dimension_rows():
    values = np.asarray(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ]
    )

    masked = mask_by_exp_target_sync_bucket_filter(values, [True, False])

    np.testing.assert_allclose(masked[0], [[1.0, 2.0]])
    assert np.isnan(masked[1]).all()


def test_mask_by_exp_target_sync_bucket_filter_rejects_length_mismatch():
    with pytest.raises(ValueError, match="target sync-bucket eligibility mask"):
        mask_by_exp_target_sync_bucket_filter(np.ones((2, 3)), [True])
