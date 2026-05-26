from pathlib import Path

import pytest

from src.validation.bundle_digest import check_manifest, load_manifest


MANIFEST = Path("test/reference/bundles/return_prob_excursion_bin_paper_manifest.json")


def test_paper_return_prob_excursion_bin_bundle_manifest_matches_available_exports():
    manifest = load_manifest(MANIFEST)
    missing = [
        bundle["path"]
        for bundle in manifest["bundles"]
        if not Path(bundle["path"]).exists()
    ]
    if missing:
        pytest.skip(
            "Paper return-probability excursion-bin export bundles are not "
            "available locally: "
            + ", ".join(missing)
        )

    assert check_manifest(MANIFEST) == []
