from pathlib import Path

import pytest

from src.plotting.overlay_training_metric_scalar_bars import load_export_npz
from src.validation.bundle_digest import check_manifest, load_manifest


MANIFEST = Path("test/reference/bundles/between_reward_tortuosity_paper_manifest.json")


def test_between_reward_tortuosity_manifest_matches_available_panel_26_exports():
    manifest = load_manifest(MANIFEST)
    missing = [
        bundle["path"]
        for bundle in manifest["bundles"]
        if not Path(bundle["path"]).exists()
    ]
    if missing:
        pytest.skip(
            "Paper between-reward tortuosity exports are not available locally: "
            + ", ".join(missing)
        )

    for bundle in manifest["bundles"]:
        if bundle.get("type", "npz") == "npz":
            load_export_npz(bundle["name"], bundle["path"])

    assert check_manifest(MANIFEST) == []
