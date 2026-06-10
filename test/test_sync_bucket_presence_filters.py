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


def _va_with_mask(mask):
    return SimpleNamespace(reward_exclusion_mask=mask)


def _va_with_counts(exp_counts, ctrl_counts, *, n_flies=1):
    prefix_rows = [tuple() for _ in range(n_flies)]
    return SimpleNamespace(
        flies=list(range(n_flies)),
        numRewardsTot=[
            [[]],
            [
                prefix_rows + [tuple(exp_counts)],
                prefix_rows + [tuple(ctrl_counts)],
            ],
        ],
    )


def _default_mask(*, excluded=False):
    mask = [
        [[False] * DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_SYNC_BUCKET],
        [[False] * DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_SYNC_BUCKET],
    ]
    mask[DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_TRAINING - 1][0][
        DEFAULT_EXP_TARGET_SYNC_BUCKET_FILTER_SYNC_BUCKET - 1
    ] = bool(excluded)
    return mask


def test_exp_target_sync_bucket_filter_is_noop_when_disabled():
    result = exp_target_sync_bucket_filter_result(
        _va_with_mask(_default_mask(excluded=True)),
        SimpleNamespace(require_exp_target_sync_bucket=False, piTh=10),
    )

    assert result.eligible
    assert result.reason == "disabled"
    assert result.training == 2
    assert result.sync_bucket == 5
    assert result.pi_threshold == 10


def test_exp_target_sync_bucket_filter_uses_target_experimental_bucket():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True, piTh=10)

    passing = exp_target_sync_bucket_filter_result(
        _va_with_mask(_default_mask(excluded=False)), opts
    )
    failing = exp_target_sync_bucket_filter_result(
        _va_with_mask(_default_mask(excluded=True)), opts
    )

    assert passing.eligible
    assert passing.reason == "passes"
    assert not failing.eligible
    assert failing.reason == "pi_threshold_failed"


def test_exp_target_sync_bucket_filter_uses_target_count_sum_when_available():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True, piTh=10)

    passing = exp_target_sync_bucket_filter_result(
        _va_with_counts([0, 0, 0, 0, 6], [0, 0, 0, 0, 4]),
        opts,
    )
    failing = exp_target_sync_bucket_filter_result(
        _va_with_counts([0, 0, 0, 0, 5], [0, 0, 0, 0, 4]),
        opts,
    )

    assert passing.eligible
    assert passing.reason == "passes"
    assert passing.target_count_sum == 10
    assert not failing.eligible
    assert failing.reason == "pi_threshold_failed"
    assert failing.target_count_sum == 9


def test_exp_target_sync_bucket_filter_reports_nan_target_bucket_separately():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True, piTh=10)

    result = exp_target_sync_bucket_filter_result(
        _va_with_counts([0, 0, 0, 0, np.nan], [0, 0, 0, 0, np.nan]),
        opts,
    )

    assert not result.eligible
    assert result.reason == "target_sync_bucket_nan"
    assert np.isnan(result.target_count_sum)


def test_exp_target_sync_bucket_filter_reports_short_target_training_separately():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True, piTh=10)

    result = exp_target_sync_bucket_filter_result(
        _va_with_counts([0], [2]),
        opts,
    )

    assert not result.eligible
    assert result.reason == "target_sync_bucket_missing"
    assert np.isnan(result.target_count_sum)


def test_exp_target_sync_bucket_filter_can_use_custom_training_and_bucket():
    mask = [[[False, True]]]
    opts = SimpleNamespace(
        require_exp_target_sync_bucket=True,
        exp_target_sync_bucket_filter_training=1,
        exp_target_sync_bucket_filter_sync_bucket=2,
    )

    result = exp_target_sync_bucket_filter_result(_va_with_mask(mask), opts)

    assert not result.eligible
    assert result.reason == "pi_threshold_failed"
    assert result.training == 1
    assert result.sync_bucket == 2


def test_exp_target_sync_bucket_filter_accepts_legacy_option_names():
    mask = [[[False, True]]]
    opts = SimpleNamespace(
        require_exp_pi_threshold_bucket=True,
        exp_pi_threshold_filter_training=1,
        exp_pi_threshold_filter_sync_bucket=2,
    )

    result = exp_target_sync_bucket_filter_result(_va_with_mask(mask), opts)

    assert not result.eligible
    assert result.reason == "pi_threshold_failed"
    assert result.training == 1
    assert result.sync_bucket == 2


def test_exp_target_sync_bucket_filter_fails_closed_when_required_data_is_missing():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True)

    missing_mask = exp_target_sync_bucket_filter_result(SimpleNamespace(), opts)
    missing_bucket = exp_target_sync_bucket_filter_result(
        _va_with_mask([[[False]], [[False]]]), opts
    )

    assert not missing_mask.eligible
    assert missing_mask.reason == "missing_reward_exclusion_mask"
    assert not missing_bucket.eligible
    assert missing_bucket.reason == "target_sync_bucket_missing"


def test_exp_target_sync_bucket_eligibility_mask_and_payload_report_per_video_results():
    opts = SimpleNamespace(require_exp_target_sync_bucket=True, piTh=7)
    vas = [
        _va_with_mask(_default_mask(excluded=False)),
        _va_with_mask(_default_mask(excluded=True)),
    ]

    eligible = exp_target_sync_bucket_eligibility_mask(vas, opts)
    payload = exp_target_sync_bucket_filter_payload(
        vas, opts, prefix="exp_pi_threshold_filter"
    )

    np.testing.assert_array_equal(eligible, [True, False])
    assert bool(payload["exp_pi_threshold_filter_enabled"])
    assert int(payload["exp_pi_threshold_filter_pi_threshold"]) == 7
    np.testing.assert_array_equal(
        payload["exp_pi_threshold_filter_eligible"], [True, False]
    )
    np.testing.assert_array_equal(
        payload["exp_pi_threshold_filter_reason"],
        ["passes", "pi_threshold_failed"],
    )
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
    with pytest.raises(ValueError, match="eligibility mask length"):
        mask_by_exp_target_sync_bucket_filter(np.ones((2, 3)), [True])
