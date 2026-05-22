import json

import numpy as np
import pytest

from src.validation.bundle_digest import (
    SLI_REGRESSION_KEYS,
    check_manifest,
    digest_npz,
    write_manifest,
)


def _write_sli_bundle(path, *, sli, unchecked=None):
    payload = {
        "sli": np.array(sli, dtype=float),
        "sli_ts": np.asarray([[[1.0, np.nan, 3.0]]], dtype=float),
        "video_ids": np.asarray(["video_a"], dtype=object),
        "training_names": np.asarray(["T1"], dtype=object),
        "sli_training_idx": np.array(0, dtype=int),
        "sli_use_training_mean": np.array(True),
        "sli_select_skip_first_sync_buckets": np.array(1, dtype=int),
        "sli_select_keep_first_sync_buckets": np.array(2, dtype=int),
        "group_label": np.asarray("Control", dtype=object),
    }
    if unchecked is not None:
        payload["unchecked_metric"] = np.asarray(unchecked, dtype=float)
    np.savez_compressed(path, **payload)


def test_npz_digest_is_stable_across_npz_key_write_order(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    payload = {
        "sli": np.asarray([0.25, np.nan]),
        "video_ids": np.asarray(["v1", "v2"], dtype=object),
    }
    np.savez_compressed(a, **payload)
    np.savez_compressed(b, video_ids=payload["video_ids"], sli=payload["sli"])

    assert digest_npz(a)["sha256"] == digest_npz(b)["sha256"]


def test_npz_digest_can_be_limited_to_sli_regression_keys(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    _write_sli_bundle(a, sli=[0.1], unchecked=[1.0])
    _write_sli_bundle(b, sli=[0.1], unchecked=[999.0])

    assert (
        digest_npz(a, keys=SLI_REGRESSION_KEYS)["sha256"]
        == digest_npz(b, keys=SLI_REGRESSION_KEYS)["sha256"]
    )
    assert digest_npz(a)["sha256"] != digest_npz(b)["sha256"]


def test_npz_digest_records_shape_counts_and_per_key_hash(tmp_path):
    path = tmp_path / "bundle.npz"
    _write_sli_bundle(path, sli=[0.1, np.nan])

    digest = digest_npz(path, keys=["sli", "video_ids"])

    assert digest["schema_version"] == 1
    assert digest["keys"] == ["sli", "video_ids"]
    assert digest["arrays"]["sli"]["shape"] == [2]
    assert digest["arrays"]["sli"]["finite_count"] == 1
    assert digest["arrays"]["sli"]["nan_count"] == 1
    assert len(digest["arrays"]["sli"]["sha256"]) == 64


def test_npz_digest_rejects_missing_requested_key(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez_compressed(path, sli=np.asarray([0.1]))

    with pytest.raises(KeyError, match="missing digest keys"):
        digest_npz(path, keys=["sli", "sli_ts"])


def test_manifest_check_report_digest_drift(tmp_path):
    path = tmp_path / "bundle.npz"
    manifest_path = tmp_path / "manifest.json"
    _write_sli_bundle(path, sli=[0.1])
    write_manifest(
        [{"name": "example", "path": "bundle.npz", "keys": SLI_REGRESSION_KEYS}],
        manifest_path,
        project_root=tmp_path,
    )
    _write_sli_bundle(path, sli=[0.2])

    failures = check_manifest(manifest_path, project_root=tmp_path)

    assert len(failures) == 1
    assert "example: digest mismatch" in failures[0]


def test_manifest_check_can_allow_missing_external_bundles(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundles": [
                    {
                        "name": "external",
                        "path": "exports/not_present.npz",
                        "keys": ["sli"],
                        "sha256": "0" * 64,
                        "arrays": {},
                    }
                ],
            }
        )
    )

    assert check_manifest(manifest_path, project_root=tmp_path, require_all=False) == []
    failures = check_manifest(manifest_path, project_root=tmp_path, require_all=True)
    assert "external: missing bundle exports/not_present.npz" in failures
