import numpy as np

from scripts.stats_agarose_virtual_control import paired_summary, select_paired_values


def test_select_paired_values_preserves_video_pairing_and_masks_missing():
    bundle = {
        "min_agarose_episodes": np.array(2),
        "agarose_avoid_exp": np.asarray([[[4, 9]], [[3, 0]]]),
        "agarose_total_exp": np.asarray([[[5, 10]], [[5, 1]]]),
        "agarose_avoid_ctrl": np.asarray([[[1, 3]], [[1, 0]]]),
        "agarose_total_ctrl": np.asarray([[[5, 10]], [[10, 1]]]),
        "agarose_virtual_avoid_exp": np.asarray([[[2, 5]], [[3, 1]]]),
        "agarose_virtual_total_exp": np.asarray([[[5, 10]], [[5, 1]]]),
        "agarose_virtual_avoid_ctrl": np.asarray([[[1, 2]], [[2, 0]]]),
        "agarose_virtual_total_ctrl": np.asarray([[[10, 10]], [[10, 1]]]),
    }

    actual, virtual, paired, trn_idx, start_idx, end_idx = select_paired_values(
        bundle, mode="exp_minus_ctrl", training_idx=0, bucket_idx=-1
    )

    np.testing.assert_allclose(actual, [0.6, np.nan])
    np.testing.assert_allclose(virtual, [0.3, np.nan])
    np.testing.assert_array_equal(paired, [True, False])
    assert (trn_idx, start_idx, end_idx) == (0, 1, 1)


def test_select_paired_values_pools_counts_across_bucket_window():
    bundle = {
        "min_agarose_episodes": np.array(5),
        "agarose_avoid_exp": np.asarray([[[1, 4, 0]]]),
        "agarose_total_exp": np.asarray([[[1, 9, 1]]]),
        "agarose_virtual_avoid_exp": np.asarray([[[0, 2, 1]]]),
        "agarose_virtual_total_exp": np.asarray([[[1, 4, 1]]]),
    }

    actual, virtual, paired, _, start_idx, end_idx = select_paired_values(
        bundle,
        mode="exp",
        training_idx=0,
        bucket_start_idx=0,
        bucket_end_idx=1,
    )

    # Pooled ratios are 5/10 and 2/5, not the mean of per-bucket ratios.
    np.testing.assert_allclose(actual, [0.5])
    np.testing.assert_allclose(virtual, [0.4])
    np.testing.assert_array_equal(paired, [True])
    assert (start_idx, end_idx) == (0, 1)


def test_paired_summary_tests_physical_minus_virtual():
    actual = np.asarray([0.9, 0.8, 0.7])
    virtual = np.asarray([0.4, 0.5, 0.3])
    result = paired_summary(actual, virtual, np.ones(3, dtype=bool))

    assert result["n"] == 3
    np.testing.assert_allclose(result["mean_actual_minus_virtual"], 0.4)
    assert result["ci_low"] < 0.4 < result["ci_high"]
    assert result["p_value"] < 0.05
