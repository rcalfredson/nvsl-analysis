from pathlib import Path

import pytest

from src.plotting.first_n_reward_diagnostics import validate_first_n_reward_diagnostics_csv
from src.validation.bundle_digest import check_manifest, load_manifest


MANIFEST = Path("test/reference/bundles/first_n_reward_diagnostics_paper_manifest.json")


def test_paper_first_n_reward_diagnostic_manifest_matches_available_exports():
    manifest = load_manifest(MANIFEST)
    missing = [
        bundle["path"]
        for bundle in manifest["bundles"]
        if not Path(bundle["path"]).exists()
    ]
    if missing:
        pytest.skip(
            "Paper first-N reward diagnostic CSV exports are not available locally: "
            + ", ".join(missing)
        )

    for bundle in manifest["bundles"]:
        validate_first_n_reward_diagnostics_csv(bundle["path"])

    assert check_manifest(MANIFEST) == []
