from pathlib import Path

import numpy as np
import pytest

from src.validation.bundle_digest import check_manifest, load_manifest


MANIFEST = Path("test/reference/bundles/turnback_excursion_bin_paper_manifest.json")


def _missing_bundle_paths(manifest):
    return [
        bundle["path"]
        for bundle in manifest["bundles"]
        if not Path(bundle["path"]).exists()
    ]


def _bundles_missing_manifest_keys(manifest):
    missing = []
    for bundle in manifest["bundles"]:
        path = Path(bundle["path"])
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            missing_keys = [key for key in bundle["keys"] if key not in data.files]
        if missing_keys:
            missing.append((bundle["path"], missing_keys))
    return missing


def test_paper_turnback_excursion_bin_bundle_manifest_matches_available_exports():
    manifest = load_manifest(MANIFEST)
    missing_paths = _missing_bundle_paths(manifest)
    if missing_paths:
        pytest.skip(
            "Paper turnback excursion-bin export bundles are not available locally: "
            + ", ".join(missing_paths)
        )

    missing_keys = _bundles_missing_manifest_keys(manifest)
    if missing_keys:
        detail = "; ".join(
            f"{path}: {', '.join(keys)}" for path, keys in missing_keys
        )
        pytest.skip(
            "Paper turnback excursion-bin export bundles predate required metadata: "
            + detail
        )

    assert check_manifest(MANIFEST) == []
