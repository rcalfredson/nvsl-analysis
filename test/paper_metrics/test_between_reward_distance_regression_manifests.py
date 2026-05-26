from pathlib import Path

import pytest

from src.validation.bundle_digest import check_manifest, load_manifest


MANIFESTS = (
    (
        Path("test/reference/bundles/between_reward_distance_hist_paper_manifest.json"),
        "Paper between-reward distance histogram export bundles",
    ),
    (
        Path(
            "test/reference/bundles/"
            "between_reward_conditioned_disttrav_paper_manifest.json"
        ),
        "Paper between-reward conditioned distance-traveled export bundles",
    ),
    (
        Path("test/reference/bundles/between_reward_maxdist_paper_manifest.json"),
        "Paper between-reward max-distance SLI export bundles",
    ),
    (
        Path(
            "test/reference/bundles/"
            "between_reward_return_leg_dist_paper_manifest.json"
        ),
        "Paper between-reward return-leg distance SLI export bundles",
    ),
)


@pytest.mark.parametrize("manifest_path, description", MANIFESTS)
def test_between_reward_distance_family_manifests_match_available_exports(
    manifest_path, description
):
    manifest = load_manifest(manifest_path)
    missing = [
        bundle["path"]
        for bundle in manifest["bundles"]
        if not Path(bundle["path"]).exists()
    ]
    if missing:
        pytest.skip(
            f"{description} are not available locally: " + ", ".join(missing)
        )

    assert check_manifest(manifest_path) == []
