from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.sli_bundle_utils import (
    validate_return_leg_tortuosity_excursion_bin_bundle,
)
from src.exporting.return_leg_tortuosity_excursion_bin_sli_bundle import (
    _aggregate_quartile_records,
    _binning_mode,
    _equal_count_bin_indices,
    _off_wall_path_over_max_radius,
    _top_fraction,
    _validate_binning_options,
)
from src.plotting.return_leg_tortuosity_excursion_bin_sli_bundle_plotter import (
    _bundle_to_exported,
    _panel_labels,
)


def test_equal_count_quartiles_are_rank_based_and_stable_with_ties():
    distances = np.asarray([8, 1, 4, 4, 9, 2, 7, 6, 3, 5], dtype=float)

    assignment = _equal_count_bin_indices(distances)

    assert np.bincount(assignment, minlength=4).tolist() == [3, 3, 2, 2]
    ordered = np.argsort(distances, kind="stable")
    assert assignment[ordered].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]

    tied = _equal_count_bin_indices(np.ones(8, dtype=float))
    assert tied.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_quartile_aggregation_exports_counts_means_and_physical_distance_stats():
    exp_records = [(float(distance), float(distance * 10)) for distance in range(1, 9)]
    records = [[exp_records, []]]

    (
        mean_exp,
        mean_ctrl,
        count_exp,
        count_ctrl,
        selected_exp,
        selected_ctrl,
        distance_stats,
    ) = _aggregate_quartile_records(records, min_segments=2)

    np.testing.assert_array_equal(count_exp, [[2, 2, 2, 2]])
    np.testing.assert_array_equal(selected_exp, count_exp)
    np.testing.assert_allclose(mean_exp, [[15, 35, 55, 75]])
    np.testing.assert_allclose(distance_stats["min"][0], [[1, 3, 5, 7]])
    np.testing.assert_allclose(distance_stats["max"][0], [[2, 4, 6, 8]])
    np.testing.assert_allclose(distance_stats["mean"][0], [[1.5, 3.5, 5.5, 7.5]])
    assert np.isnan(mean_ctrl).all()
    assert not count_ctrl.any()
    assert not selected_ctrl.any()


def test_quartile_minimum_is_applied_separately_to_each_quartile():
    records = [[[(float(i), 1.0) for i in range(19)], []]]

    mean_exp, _mean_ctrl, count_exp, *_rest = _aggregate_quartile_records(
        records, min_segments=5
    )

    np.testing.assert_array_equal(count_exp, [[5, 5, 5, 4]])
    assert np.isfinite(mean_exp[0, :3]).all()
    assert np.isnan(mean_exp[0, 3])


def test_quartile_binning_rejects_tortuosity_top_fraction():
    opts = SimpleNamespace(
        return_leg_tortuosity_excursion_bin_binning_mode="per_fly_quartile",
        return_leg_tortuosity_excursion_bin_top_fraction=0.25,
    )

    with pytest.raises(ValueError, match="must be 1.0"):
        _top_fraction(opts)


def test_quartile_binning_rejects_absolute_distance_options():
    opts = SimpleNamespace(
        return_leg_tortuosity_excursion_bin_binning_mode="per_fly_quartile",
        return_leg_tortuosity_excursion_bin_radius_pairs_mm="3:5,8:10",
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        _validate_binning_options(opts, _binning_mode(opts))


def test_quartile_panel_labels_are_percentile_ranges():
    bundle = {
        "return_leg_tortuosity_excursion_bin_binning_mode": np.asarray(
            "per_fly_quartile"
        )
    }

    assert _panel_labels(bundle) == [
        "Q1 (0-25%)",
        "Q2 (25-50%)",
        "Q3 (50-75%)",
        "Q4 (75-100%)",
    ]


def test_off_wall_tortuosity_excludes_wall_steps_but_keeps_global_radius():
    traj = SimpleNamespace(
        x=np.asarray([10.0, 9.0, 8.0, 7.0, 6.0]),
        y=np.zeros(5),
        d=np.asarray([1.0, 1.0, 1.0, 1.0]),
    )
    wall_mask = np.asarray([False, True, True, False, False])

    value = _off_wall_path_over_max_radius(
        traj=traj,
        s=0,
        e=5,
        fi=0,
        wall_mask=wall_mask,
        nonwalk_mask=None,
        exclude_nonwalk=False,
        px_per_mm=1.0,
        reward_center_xy=(0.0, 0.0),
        min_keep_frames=2,
        min_radius_mm=0.0,
    )

    # Only the final 7 -> 6 step is wholly off-wall; denominator remains 10.
    assert value == pytest.approx(0.1)


def test_quartile_bundle_validation_accepts_distance_metadata():
    counts = np.asarray([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=int)
    values = np.asarray([[1.0, 1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 1.7]])
    ctrl_counts = np.zeros_like(counts)
    ctrl_values = np.full_like(values, np.nan)
    percentile_edges = np.asarray([0.0, 25.0, 50.0, 75.0, 100.0])
    bundle = {
        "sli": np.asarray([0.1, 0.2]),
        "return_leg_tortuosity_excursion_bin_exp": values,
        "return_leg_tortuosity_excursion_bin_ctrl": ctrl_values,
        "return_leg_tortuosity_excursion_binN_exp": counts,
        "return_leg_tortuosity_excursion_binN_ctrl": ctrl_counts,
        "return_leg_tortuosity_excursion_bin_selectedN_exp": counts.copy(),
        "return_leg_tortuosity_excursion_bin_selectedN_ctrl": ctrl_counts.copy(),
        "return_leg_tortuosity_excursion_bin_edges_mm": percentile_edges,
        "return_leg_tortuosity_excursion_bin_requested_edges_mm": percentile_edges,
        "return_leg_tortuosity_excursion_bin_percentile_edges": percentile_edges,
        "return_leg_tortuosity_excursion_bin_pair_mode": np.asarray(False),
        "return_leg_tortuosity_excursion_bin_binning_mode": np.asarray(
            "per_fly_quartile"
        ),
        "return_leg_tortuosity_excursion_bin_top_fraction": np.asarray(1.0),
        "return_leg_tortuosity_excursion_bin_return_start_mode": np.asarray(
            "global_max"
        ),
        "return_leg_tortuosity_excursion_bin_exclude_wall_contact": np.asarray(False),
        "btw_rwd_sync_bucket_min_trajectories": np.asarray(1),
    }
    for stat, stat_values in {
        "min": [[1, 3, 5, 7], [2, 4, 6, 8]],
        "max": [[2, 4, 6, 8], [3, 5, 7, 9]],
        "mean": [[1.5, 3.5, 5.5, 7.5], [2.5, 4.5, 6.5, 8.5]],
        "median": [[1.5, 3.5, 5.5, 7.5], [2.5, 4.5, 6.5, 8.5]],
    }.items():
        bundle[
            f"return_leg_tortuosity_excursion_bin_distance_{stat}_mm_exp"
        ] = np.asarray(stat_values, dtype=float)
        bundle[
            f"return_leg_tortuosity_excursion_bin_distance_{stat}_mm_ctrl"
        ] = np.full((2, 4), np.nan)

    validate_return_leg_tortuosity_excursion_bin_bundle(bundle)

    bundle["video_ids"] = np.asarray(["fly-a", "fly-b"])
    exported = _bundle_to_exported(
        bundle,
        label="control",
        mode="exp",
        sub_idx=np.asarray([0, 1]),
    )
    assert exported.panel_labels == [
        "Q1 (0-25%)",
        "Q2 (25-50%)",
        "Q3 (50-75%)",
        "Q4 (75-100%)",
    ]
    assert exported.meta["binning_mode"] == "per_fly_quartile"
    assert "quartile" in exported.meta["base_title"].lower()


def test_bundle_and_plotter_report_frame_level_wall_exclusion():
    bundle = {
        "sli": np.asarray([0.1]),
        "video_ids": np.asarray(["fly-a"]),
        "return_leg_tortuosity_excursion_bin_exp": np.asarray([[1.2]]),
        "return_leg_tortuosity_excursion_bin_ctrl": np.asarray([[np.nan]]),
        "return_leg_tortuosity_excursion_binN_exp": np.asarray([[5]]),
        "return_leg_tortuosity_excursion_binN_ctrl": np.asarray([[0]]),
        "return_leg_tortuosity_excursion_bin_selectedN_exp": np.asarray([[5]]),
        "return_leg_tortuosity_excursion_bin_selectedN_ctrl": np.asarray([[0]]),
        "return_leg_tortuosity_excursion_bin_edges_mm": np.asarray([3.0, 5.0]),
        "return_leg_tortuosity_excursion_bin_requested_edges_mm": np.asarray(
            [3.0, 5.0]
        ),
        "return_leg_tortuosity_excursion_bin_pair_mode": np.asarray(False),
        "return_leg_tortuosity_excursion_bin_binning_mode": np.asarray(
            "absolute_distance"
        ),
        "return_leg_tortuosity_excursion_bin_top_fraction": np.asarray(1.0),
        "return_leg_tortuosity_excursion_bin_return_start_mode": np.asarray(
            "global_max"
        ),
        "return_leg_tortuosity_excursion_bin_exclude_wall_contact": np.asarray(
            False
        ),
        "return_leg_tortuosity_excursion_bin_exclude_wall_contact_frames": (
            np.asarray(True)
        ),
        "return_leg_tortuosity_excursion_bin_metric_mode": np.asarray(
            "path_over_max_radius"
        ),
        "btw_rwd_sync_bucket_min_trajectories": np.asarray(1),
    }

    validate_return_leg_tortuosity_excursion_bin_bundle(bundle)
    exported = _bundle_to_exported(
        bundle,
        label="control",
        mode="exp",
        sub_idx=np.asarray([0]),
    )

    assert exported.meta["exclude_wall_contact_frames"] is True
    assert "Off-wall" in exported.meta["y_label"]
