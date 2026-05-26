import numpy as np
import pytest

from src.plotting.overlay_training_metric_hist import ExportedTrainingHistogram


def _per_fly_hist(**overrides):
    payload = {
        "group": "Control",
        "panel_labels": ["T2"],
        "counts": np.asarray(None, dtype=object),
        "mean": np.asarray([[0.25, 0.75, np.nan]], dtype=float),
        "ci_lo": np.asarray([[0.20, 0.70, np.nan]], dtype=float),
        "ci_hi": np.asarray([[0.30, 0.80, np.nan]], dtype=float),
        "n_units": np.asarray([[2, 2, 0]], dtype=int),
        "n_units_panel": np.asarray([2], dtype=int),
        "per_unit_panel": np.asarray(
            [np.asarray([[0.2, 0.8, np.nan], [0.3, 0.7, np.nan]])],
            dtype=object,
        ),
        "per_unit_ids_panel": np.asarray(
            [np.asarray(["fly_a", "fly_b"], dtype=object)], dtype=object
        ),
        "bin_edges": np.asarray([[0.0, 60.0, 200.0, 1600.0]], dtype=float),
        "n_raw": np.asarray([20], dtype=int),
        "n_used": np.asarray([18], dtype=int),
        "n_dropped": np.asarray([2], dtype=int),
        "meta": {"bins": 3, "per_fly": True, "normalize": True},
    }
    payload.update(overrides)
    return ExportedTrainingHistogram(**payload)


def _pooled_hist(**overrides):
    payload = {
        "group": "Control",
        "panel_labels": ["T1", "T2"],
        "counts": np.asarray([[1, 2], [3, 0]], dtype=int),
        "mean": None,
        "ci_lo": None,
        "ci_hi": None,
        "n_units": None,
        "n_units_panel": None,
        "per_unit_panel": None,
        "per_unit_ids_panel": None,
        "bin_edges": np.asarray([[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]]),
        "n_raw": np.asarray([4, 3], dtype=int),
        "n_used": np.asarray([3, 3], dtype=int),
        "n_dropped": np.asarray([1, 0], dtype=int),
        "meta": {"bins": 2, "per_fly": False, "normalize": False},
    }
    payload.update(overrides)
    return ExportedTrainingHistogram(**payload)


def test_exported_training_histogram_validate_accepts_per_fly_panel_payload():
    _per_fly_hist().validate()


def test_exported_training_histogram_validate_accepts_pooled_counts_payload():
    _pooled_hist().validate()


def test_exported_training_histogram_validate_rejects_bad_edges():
    with pytest.raises(ValueError, match="strictly increasing"):
        _per_fly_hist(bin_edges=np.asarray([[0.0, 60.0, 60.0, 1600.0]])).validate()


def test_exported_training_histogram_validate_rejects_raw_used_dropped_mismatch():
    with pytest.raises(ValueError, match="n_raw must equal n_used \\+ n_dropped"):
        _per_fly_hist(n_raw=np.asarray([19])).validate()


def test_exported_training_histogram_validate_rejects_bad_summary_values():
    with pytest.raises(ValueError, match="mean must be nonnegative"):
        _per_fly_hist(mean=np.asarray([[-0.1, 0.75, np.nan]])).validate()

    with pytest.raises(ValueError, match="ci_lo must be <= mean"):
        _per_fly_hist(ci_lo=np.asarray([[0.30, 0.70, np.nan]])).validate()

    with pytest.raises(ValueError, match="mean must be NaN where n_units == 0"):
        _per_fly_hist(mean=np.asarray([[0.25, 0.75, 0.1]])).validate()


def test_exported_training_histogram_validate_rejects_bad_per_unit_payload():
    with pytest.raises(ValueError, match="length must match per_unit rows"):
        _per_fly_hist(
            per_unit_ids_panel=np.asarray(
                [np.asarray(["fly_a"], dtype=object)], dtype=object
            )
        ).validate()

    with pytest.raises(ValueError, match="n_units panel 0 must match"):
        _per_fly_hist(n_units=np.asarray([[1, 2, 0]], dtype=int)).validate()


def test_exported_training_histogram_validate_rejects_bad_pooled_counts():
    with pytest.raises(ValueError, match="counts must be nonnegative"):
        _pooled_hist(counts=np.asarray([[1, -2], [3, 0]])).validate()

    with pytest.raises(ValueError, match="pooled counts must sum to n_used"):
        _pooled_hist(n_raw=np.asarray([5, 3]), n_used=np.asarray([4, 3])).validate()
