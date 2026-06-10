import numpy as np
import pytest

from src.analysis.sli_bundle_utils import (
    normalize_sli_bundle,
    validate_agarose_sli_bundle,
)


def _bundle(**overrides):
    bundle = {
        "sli": np.asarray([0.1, np.nan], dtype=float),
        "sli_ts": np.asarray(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, np.nan]],
                [[np.nan, 0.1, 0.2], [0.3, np.nan, 0.5]],
            ]
        ),
        "group_label": np.asarray("Intact Control>Kir", dtype=object),
        "bucket_len_min": np.array(10, dtype=float),
        "training_names": np.asarray(["T1", "T2"], dtype=object),
        "video_ids": np.asarray(["video_a", "video_b"], dtype=object),
        "sli_training_idx": np.array(1, dtype=int),
        "sli_use_training_mean": np.array(True),
        "sli_select_skip_first_sync_buckets": np.array(1, dtype=int),
        "sli_select_keep_first_sync_buckets": np.array(0, dtype=int),
        "min_agarose_episodes": np.array(2, dtype=int),
        "agarose_dual_circle_min_total": np.array(2, dtype=int),
        "agarose_ratio_exp": np.asarray(
            [
                [[1.0, 0.5, np.nan], [0.0, 2.0 / 3.0, np.nan]],
                [[np.nan, 1.0, 0.25], [0.5, np.nan, 1.0]],
            ],
            dtype=float,
        ),
        "agarose_ratio_ctrl": np.asarray(
            [
                [[0.0, 0.25, np.nan], [0.5, np.nan, 1.0]],
                [[np.nan, 0.0, 0.75], [1.0, 0.5, np.nan]],
            ],
            dtype=float,
        ),
        "agarose_total_exp": np.asarray(
            [[[2, 2, 1], [2, 3, 0]], [[0, 2, 4], [2, 1, 2]]], dtype=int
        ),
        "agarose_total_ctrl": np.asarray(
            [[[2, 4, 0], [2, 1, 2]], [[0, 2, 4], [2, 2, 1]]], dtype=int
        ),
        "agarose_avoid_exp": np.asarray(
            [[[2, 1, 1], [0, 2, 0]], [[0, 2, 1], [1, 0, 2]]], dtype=int
        ),
        "agarose_avoid_ctrl": np.asarray(
            [[[0, 1, 0], [1, 0, 2]], [[0, 0, 3], [2, 1, 0]]], dtype=int
        ),
        "agarose_pre_ratio_exp": np.asarray([0.5, np.nan]),
        "agarose_pre_ratio_ctrl": np.asarray([1.0, 0.0]),
        "agarose_pre_total_exp": np.asarray([2, 1], dtype=int),
        "agarose_pre_total_ctrl": np.asarray([2, 2], dtype=int),
        "agarose_pre_avoid_exp": np.asarray([1, 1], dtype=int),
        "agarose_pre_avoid_ctrl": np.asarray([2, 0], dtype=int),
        "agarose_pre_window_min": np.array(10.0, dtype=float),
        "agarose_training_pre_ratio_exp": np.asarray([[0.5, np.nan], [1.0, 0.0]]),
        "agarose_training_pre_ratio_ctrl": np.asarray([[1.0, 0.0], [np.nan, 0.5]]),
        "agarose_training_pre_total_exp": np.asarray([[2, 1], [2, 2]], dtype=int),
        "agarose_training_pre_total_ctrl": np.asarray([[2, 2], [1, 2]], dtype=int),
        "agarose_training_pre_avoid_exp": np.asarray([[1, 1], [2, 0]], dtype=int),
        "agarose_training_pre_avoid_ctrl": np.asarray([[2, 0], [1, 1]], dtype=int),
        "agarose_training_pre_window_min": np.asarray([10.0, 10.0]),
    }
    bundle.update(overrides)
    return bundle


def test_validate_agarose_bundle_accepts_valid_shapes_counts_and_metadata():
    validate_agarose_sli_bundle(_bundle())

    normalized = normalize_sli_bundle(_bundle())
    assert normalized["agarose_ratio_exp"].shape == (2, 2, 3)


def test_validate_agarose_bundle_accepts_legacy_min_total_metadata():
    bundle = _bundle()
    del bundle["min_agarose_episodes"]

    validate_agarose_sli_bundle(bundle)


def test_validate_agarose_bundle_allows_target_sync_bucket_filtered_exp_ratios():
    validate_agarose_sli_bundle(
        _bundle(
            exp_target_sync_bucket_filter_eligible=np.asarray([False, True]),
            agarose_ratio_exp=np.asarray(
                [
                    [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                    [[np.nan, 1.0, 0.25], [0.5, np.nan, 1.0]],
                ],
                dtype=float,
            ),
            agarose_pre_ratio_exp=np.asarray([np.nan, np.nan]),
            agarose_training_pre_ratio_exp=np.asarray(
                [[np.nan, np.nan], [1.0, 0.0]]
            ),
        )
    )


def test_validate_agarose_bundle_prefers_primary_min_episode_metadata():
    with pytest.raises(ValueError, match="where total is below reporting threshold"):
        validate_agarose_sli_bundle(
            _bundle(
                min_agarose_episodes=np.array(3, dtype=int),
                agarose_dual_circle_min_total=np.array(2, dtype=int),
            )
        )


def test_validate_agarose_bundle_rejects_missing_metric_key():
    bundle = _bundle()
    del bundle["agarose_avoid_ctrl"]

    with pytest.raises(ValueError, match="missing agarose ratio keys"):
        validate_agarose_sli_bundle(bundle)


def test_validate_agarose_bundle_rejects_metric_shape_mismatch():
    with pytest.raises(ValueError, match=r"agarose_ratio_exp\.shape=\(2, 2, 2\)"):
        validate_agarose_sli_bundle(
            _bundle(agarose_ratio_exp=np.ones((2, 2, 2), dtype=float))
        )


def test_validate_agarose_bundle_rejects_bad_counts():
    with pytest.raises(ValueError, match="negative values"):
        validate_agarose_sli_bundle(
            _bundle(
                agarose_total_exp=np.asarray(
                    [[[2, -1, 1], [2, 3, 0]], [[0, 2, 4], [2, 1, 2]]]
                )
            )
        )

    with pytest.raises(ValueError, match="non-integer values"):
        validate_agarose_sli_bundle(
            _bundle(
                agarose_avoid_ctrl=np.asarray(
                    [[[0, 1.5, 0], [1, 0, 2]], [[0, 0, 3], [2, 1, 0]]]
                )
            )
        )

    with pytest.raises(ValueError, match="values greater than totals"):
        validate_agarose_sli_bundle(
            _bundle(
                agarose_avoid_exp=np.asarray(
                    [[[2, 3, 1], [0, 2, 0]], [[0, 2, 1], [1, 0, 2]]]
                )
            )
        )


def test_validate_agarose_bundle_rejects_bad_probabilities_and_threshold_semantics():
    with pytest.raises(ValueError, match="out-of-range probabilities"):
        validate_agarose_sli_bundle(
            _bundle(
                agarose_ratio_exp=np.asarray(
                    [
                        [[1.2, 0.5, np.nan], [0.0, 2.0 / 3.0, np.nan]],
                        [[np.nan, 1.0, 0.25], [0.5, np.nan, 1.0]],
                    ]
                )
            )
        )

    with pytest.raises(ValueError, match="where total is below reporting threshold"):
        validate_agarose_sli_bundle(
            _bundle(
                agarose_ratio_exp=np.asarray(
                    [
                        [[1.0, 0.5, 1.0], [0.0, 2.0 / 3.0, np.nan]],
                        [[np.nan, 1.0, 0.25], [0.5, np.nan, 1.0]],
                    ]
                )
            )
        )


def test_validate_agarose_bundle_rejects_ratio_inconsistent_with_counts():
    with pytest.raises(ValueError, match="inconsistent agarose_ratio_ctrl values"):
        validate_agarose_sli_bundle(
            _bundle(
                agarose_ratio_ctrl=np.asarray(
                    [
                        [[0.0, 0.3, np.nan], [0.5, np.nan, 1.0]],
                        [[np.nan, 0.0, 0.75], [1.0, 0.5, np.nan]],
                    ]
                )
            )
        )


def test_validate_agarose_bundle_rejects_bad_pre_shapes_and_semantics():
    with pytest.raises(ValueError, match=r"agarose_pre_ratio_exp\.shape=\(1,\)"):
        validate_agarose_sli_bundle(_bundle(agarose_pre_ratio_exp=np.asarray([0.5])))

    with pytest.raises(
        ValueError, match=r"agarose_training_pre_ratio_ctrl\.shape=\(2, 1\)"
    ):
        validate_agarose_sli_bundle(
            _bundle(agarose_training_pre_ratio_ctrl=np.asarray([[1.0], [0.5]]))
        )

    with pytest.raises(ValueError, match="where total is below reporting threshold"):
        validate_agarose_sli_bundle(
            _bundle(agarose_pre_ratio_exp=np.asarray([0.5, 1.0]))
        )
