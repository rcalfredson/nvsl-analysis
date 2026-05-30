import json

import numpy as np
import pytest

from src.plotting.overlay_training_metric_scalar_bars import load_export_npz
from src.validation.between_reward_tortuosity import (
    validate_between_reward_tortuosity_distance_box_result,
    validate_between_reward_tortuosity_graphpad_columns,
    validate_between_reward_tortuosity_scalar_export,
    validate_between_reward_tortuosity_wall_scatter_export,
)


def _scalar_export(**overrides):
    payload = {
        "panel_labels": np.asarray(["T2"], dtype=object),
        "per_unit_values_panel": np.asarray(
            [np.asarray([1.0, 2.0, np.nan], dtype=float)], dtype=object
        ),
        "per_unit_ids_panel": np.asarray(
            [np.asarray(["fly_a", "fly_b", "fly_c"], dtype=object)], dtype=object
        ),
        "mean": np.asarray([1.5], dtype=float),
        "ci_lo": np.asarray([np.nan], dtype=float),
        "ci_hi": np.asarray([np.nan], dtype=float),
        "n_units_panel": np.asarray([2], dtype=int),
        "meta_json": json.dumps(
            {
                "metric": "between_reward_tortuosity_mean",
                "metric_mode": "path_over_max_radius",
                "segment_scope": "full",
                "min_radius_mm": 0.0,
            },
            sort_keys=True,
        ),
    }
    payload.update(overrides)
    return payload


def test_validate_tortuosity_scalar_export_accepts_nan_and_nonnegative_values(tmp_path):
    path = tmp_path / "tort.npz"
    np.savez_compressed(path, **_scalar_export())

    validate_between_reward_tortuosity_scalar_export(_scalar_export())
    loaded = load_export_npz("Ctrl>Kir FLC", str(path))
    assert loaded.n_units_panel.tolist() == [2]


def test_validate_tortuosity_scalar_export_rejects_inf_negative_and_bad_counts():
    with pytest.raises(ValueError, match="infinite values"):
        validate_between_reward_tortuosity_scalar_export(
            _scalar_export(mean=np.asarray([np.inf]))
        )

    with pytest.raises(ValueError, match="negative values"):
        validate_between_reward_tortuosity_scalar_export(
            _scalar_export(
                per_unit_values_panel=np.asarray(
                    [np.asarray([1.0, -0.5], dtype=float)], dtype=object
                ),
                per_unit_ids_panel=np.asarray(
                    [np.asarray(["fly_a", "fly_b"], dtype=object)], dtype=object
                ),
                mean=np.asarray([0.25]),
                n_units_panel=np.asarray([2]),
            )
        )

    with pytest.raises(ValueError, match="n_units_panel=3"):
        validate_between_reward_tortuosity_scalar_export(
            _scalar_export(n_units_panel=np.asarray([3]))
        )


def test_validate_tortuosity_graphpad_columns_rejects_inf_and_negative_values():
    validate_between_reward_tortuosity_graphpad_columns(
        ["Ctrl>Kir FLC", "PFN>Kir FLC"],
        [np.asarray([1.0, 2.0]), np.asarray([np.nan, 3.0])],
    )

    with pytest.raises(ValueError, match="negative values"):
        validate_between_reward_tortuosity_graphpad_columns(
            ["Ctrl>Kir FLC"], [np.asarray([1.0, -1.0])]
        )

    with pytest.raises(ValueError, match="infinite values"):
        validate_between_reward_tortuosity_graphpad_columns(
            ["Ctrl>Kir FLC"], [np.asarray([np.inf])]
        )


class _BoxResult:
    x_edges_mm = np.asarray([0.0, 5.0, 10.0], dtype=float)
    bin_labels = np.asarray(["[0, 5)", "[5, 10)"], dtype=object)
    values_by_bin = np.asarray(
        [np.asarray([1.0, 1.5]), np.asarray([], dtype=float)], dtype=object
    )
    n_segments = np.asarray([2, 0], dtype=int)
    n_units = np.asarray([2, 0], dtype=int)
    unit_stat = "median"
    segment_level = False
    q1 = np.asarray([1.125, np.nan])
    median = np.asarray([1.25, np.nan])
    q3 = np.asarray([1.375, np.nan])
    whisker_low = np.asarray([1.0, np.nan])
    whisker_high = np.asarray([1.5, np.nan])
    meta = {"metric": "between_reward_tortuosity_by_max_radius"}


def test_validate_tortuosity_distance_box_result_accepts_consistent_bins():
    validate_between_reward_tortuosity_distance_box_result(_BoxResult())


def test_validate_tortuosity_distance_box_result_rejects_bad_values():
    class Bad(_BoxResult):
        values_by_bin = np.asarray(
            [np.asarray([1.0]), np.asarray([np.inf])], dtype=object
        )
        n_segments = np.asarray([1, 1], dtype=int)
        n_units = np.asarray([1, 1], dtype=int)

    with pytest.raises(ValueError, match="infinite values"):
        validate_between_reward_tortuosity_distance_box_result(Bad())


def _wall_scatter(**overrides):
    payload = {
        "wall_frac": np.asarray([0.0, 0.5], dtype=float),
        "wall_pct": np.asarray([0.0, 50.0], dtype=float),
        "tortuosity": np.asarray([1.0, 2.0], dtype=float),
        "s": np.asarray([0, 10], dtype=int),
        "e": np.asarray([5, 15], dtype=int),
        "metric_s": np.asarray([0, 10], dtype=int),
        "metric_e": np.asarray([5, 15], dtype=int),
        "n_wall_frames": np.asarray([0, 2], dtype=int),
        "n_metric_frames": np.asarray([5, 4], dtype=int),
        "b_idx": np.asarray([0, 1], dtype=int),
        "video_id": np.asarray(["v1", "v2"], dtype=object),
        "unit_id": np.asarray(["u1", "u2"], dtype=object),
        "fly_id": np.asarray([7, 8], dtype=int),
        "trx_idx": np.asarray([0, 0], dtype=int),
        "role_idx": np.asarray([0, 0], dtype=int),
        "fly_role": np.asarray(["exp", "exp"], dtype=object),
        "meta_json": json.dumps({"metric": "between_reward_tortuosity_wall_scatter"}),
    }
    payload.update(overrides)
    return payload


def test_validate_tortuosity_wall_scatter_export_accepts_consistent_rows():
    validate_between_reward_tortuosity_wall_scatter_export(_wall_scatter())


def test_validate_tortuosity_wall_scatter_export_rejects_bad_rows():
    with pytest.raises(ValueError, match="wall_frac values above 1"):
        validate_between_reward_tortuosity_wall_scatter_export(
            _wall_scatter(wall_frac=np.asarray([0.0, 1.2]))
        )

    with pytest.raises(ValueError, match="n_wall_frames greater"):
        validate_between_reward_tortuosity_wall_scatter_export(
            _wall_scatter(n_wall_frames=np.asarray([0, 5]))
        )
