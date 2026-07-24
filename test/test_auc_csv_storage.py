from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.common import flatten_auc_entries, store_auc_entries


def _va(n_flies=2):
    return SimpleNamespace(saved_auc={}, flies=tuple(range(n_flies)))


def test_store_rpid_auc_entries_for_single_default_group():
    vas = [_va(), _va()]
    values = np.array(
        [
            [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]],
            [[2.0, 2.0, 2.0], [0.0, -1.0, -2.0]],
        ]
    )

    store_auc_entries(vas, "rpid", values, n_flies=1)

    assert flatten_auc_entries(vas[0], "rpid")[0] == pytest.approx([2.0, 2.0])
    assert flatten_auc_entries(vas[1], "rpid")[0] == pytest.approx([4.0, -2.0])


def test_store_auc_entries_replaces_previous_plot_records():
    va = _va()
    va.saved_auc["rpid"] = [{"training": 99, "exp": 99.0, "ctrl": np.nan}]

    store_auc_entries(
        [va],
        "rpid",
        np.array([[[1.0, 2.0, 3.0]]]),
        n_flies=1,
    )

    assert flatten_auc_entries(va, "rpid")[0] == pytest.approx([4.0])


def test_single_fly_auc_export_matches_single_exp_column():
    va = _va(n_flies=1)

    store_auc_entries(
        [va],
        "rpd",
        np.array([[[1.0, 2.0, 3.0]]]),
        n_flies=1,
    )

    assert flatten_auc_entries(va, "rpd")[0] == pytest.approx([4.0])


def test_two_fly_auc_export_splits_exp_and_control_buckets():
    va = _va()

    store_auc_entries(
        [va],
        "rpd",
        np.array([[[0.0, 1.0, 2.0, 4.0]]]),
        n_flies=2,
    )

    assert flatten_auc_entries(va, "rpd")[0] == pytest.approx([0.5, 3.0])
