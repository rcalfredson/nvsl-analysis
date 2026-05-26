import numpy as np
import pytest

from src.analysis.sli_bundle_utils import (
    normalize_sli_bundle,
    validate_between_reward_maxdist_bundle,
    validate_between_reward_return_leg_dist_bundle,
)


def _base_bundle(**overrides):
    bundle = {
        "sli": np.asarray([0.1, np.nan], dtype=float),
        "sli_ts": np.asarray(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[np.nan, 0.1, 0.2], [0.3, np.nan, 0.5]],
            ],
            dtype=float,
        ),
        "group_label": np.asarray("Control", dtype=object),
        "bucket_len_min": np.array(10.0, dtype=float),
        "training_names": np.asarray(["T1", "T2"], dtype=object),
        "video_ids": np.asarray(["video_a", "video_b"], dtype=object),
        "sli_training_idx": np.array(1, dtype=int),
        "sli_use_training_mean": np.array(True),
        "sli_select_skip_first_sync_buckets": np.array(1, dtype=int),
        "sli_select_keep_first_sync_buckets": np.array(0, dtype=int),
    }
    bundle.update(overrides)
    return bundle


def _maxdist_bundle(**overrides):
    exp = np.asarray(
        [
            [[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]],
            [[np.nan, 1.5, 2.5], [2.0, np.nan, 3.0]],
        ],
        dtype=float,
    )
    ctrl = np.asarray(
        [
            [[1.0, np.nan, 2.0], [3.0, 4.0, np.nan]],
            [[np.nan, np.nan, 1.0], [2.0, 2.5, 3.0]],
        ],
        dtype=float,
    )
    bundle = _base_bundle(
        between_reward_maxdist_exp=exp,
        between_reward_maxdist_ctrl=ctrl,
        between_reward_maxdistN_exp=np.asarray(np.isfinite(exp), dtype=int),
        between_reward_maxdistN_ctrl=np.asarray(np.isfinite(ctrl), dtype=int),
    )
    bundle.update(overrides)
    return bundle


def _return_leg_bundle(**overrides):
    exp = np.asarray(
        [
            [[0.5, 1.0, np.nan], [1.5, 2.0, 2.5]],
            [[np.nan, 0.5, 1.0], [1.0, np.nan, 1.5]],
        ],
        dtype=float,
    )
    ctrl = np.asarray(
        [
            [[0.5, np.nan, 1.0], [1.5, 2.0, np.nan]],
            [[np.nan, np.nan, 0.5], [1.0, 1.25, 1.5]],
        ],
        dtype=float,
    )
    bundle = _base_bundle(
        between_reward_return_leg_dist_exp=exp,
        between_reward_return_leg_dist_ctrl=ctrl,
        between_reward_return_leg_distN_exp=np.asarray(np.isfinite(exp), dtype=int),
        between_reward_return_leg_distN_ctrl=np.asarray(np.isfinite(ctrl), dtype=int),
    )
    bundle.update(overrides)
    return bundle


def test_validate_between_reward_maxdist_bundle_accepts_valid_shapes_and_metadata():
    validate_between_reward_maxdist_bundle(_maxdist_bundle())

    normalized = normalize_sli_bundle(_maxdist_bundle())
    assert normalized["between_reward_maxdist_exp"].shape == (2, 2, 3)


def test_validate_between_reward_return_leg_dist_bundle_accepts_valid_shapes_and_metadata():
    validate_between_reward_return_leg_dist_bundle(_return_leg_bundle())

    normalized = normalize_sli_bundle(_return_leg_bundle())
    assert normalized["between_reward_return_leg_dist_exp"].shape == (2, 2, 3)


def test_validate_between_reward_distance_bundle_rejects_missing_metric_key():
    bundle = _maxdist_bundle()
    del bundle["between_reward_maxdist_ctrl"]

    with pytest.raises(ValueError, match="missing between-reward max-distance keys"):
        validate_between_reward_maxdist_bundle(bundle)


def test_validate_between_reward_distance_bundle_rejects_metric_shape_mismatch():
    with pytest.raises(
        ValueError, match=r"between_reward_maxdist_exp\.shape=\(2, 2, 2\)"
    ):
        validate_between_reward_maxdist_bundle(
            _maxdist_bundle(
                between_reward_maxdist_exp=np.ones((2, 2, 2), dtype=float)
            )
        )


def test_validate_between_reward_distance_bundle_rejects_bad_counts():
    with pytest.raises(ValueError, match="negative values"):
        validate_between_reward_return_leg_dist_bundle(
            _return_leg_bundle(
                between_reward_return_leg_distN_exp=np.asarray(
                    [[[1, -1, 0], [1, 1, 1]], [[0, 1, 1], [1, 0, 1]]]
                )
            )
        )

    with pytest.raises(ValueError, match="non-integer values"):
        validate_between_reward_return_leg_dist_bundle(
            _return_leg_bundle(
                between_reward_return_leg_distN_exp=np.asarray(
                    [[[1, 1.5, 0], [1, 1, 1]], [[0, 1, 1], [1, 0, 1]]]
                )
            )
        )


def test_validate_between_reward_distance_bundle_rejects_bad_distances():
    with pytest.raises(ValueError, match="negative values"):
        validate_between_reward_maxdist_bundle(
            _maxdist_bundle(
                between_reward_maxdist_exp=np.asarray(
                    [
                        [[1.0, -2.0, np.nan], [3.0, 4.0, 5.0]],
                        [[np.nan, 1.5, 2.5], [2.0, np.nan, 3.0]],
                    ]
                )
            )
        )

    with pytest.raises(ValueError, match="infinite values"):
        validate_between_reward_maxdist_bundle(
            _maxdist_bundle(
                between_reward_maxdist_exp=np.asarray(
                    [
                        [[1.0, np.inf, np.nan], [3.0, 4.0, 5.0]],
                        [[np.nan, 1.5, 2.5], [2.0, np.nan, 3.0]],
                    ]
                )
            )
        )


def test_validate_between_reward_distance_bundle_rejects_finite_value_with_zero_count():
    with pytest.raises(ValueError, match="finite between_reward_return_leg_dist_exp"):
        validate_between_reward_return_leg_dist_bundle(
            _return_leg_bundle(
                between_reward_return_leg_dist_exp=np.asarray(
                    [
                        [[0.5, 1.0, 2.0], [1.5, 2.0, 2.5]],
                        [[np.nan, 0.5, 1.0], [1.0, np.nan, 1.5]],
                    ]
                ),
                between_reward_return_leg_distN_exp=np.asarray(
                    [[[1, 1, 0], [1, 1, 1]], [[0, 1, 1], [1, 0, 1]]]
                ),
            )
        )
