import json

import numpy as np
import pytest

from src.caching.wall_contact_cache import (
    load_wall_cache_npz,
    save_wall_cache_npz,
    wall_payload_has_required_regions,
)


def _wall_payload(regions=None):
    if regions is None:
        regions = np.empty((0, 2), dtype=np.int32)
    return {"wall": {"all": {"edge": {"regions": regions}}}}


def test_required_wall_regions_accepts_an_empty_region_array():
    assert wall_payload_has_required_regions(_wall_payload())
    assert not wall_payload_has_required_regions({"wall": {}})


def test_wall_cache_round_trip_preserves_fly_with_no_contact_bouts(tmp_path):
    cache_path = tmp_path / "wall.npz"
    save_wall_cache_npz(
        str(cache_path),
        {"va_id": 0},
        wall_orientations=["all"],
        per_fly_payload={1: _wall_payload()},
    )

    manifest, per_fly, orientations = load_wall_cache_npz(str(cache_path))

    assert manifest["payload"]["flies"] == [1]
    assert orientations == ["all"]
    assert wall_payload_has_required_regions(per_fly[1])
    assert per_fly[1]["wall"]["all"]["edge"]["regions"].shape == (0, 2)


def test_wall_cache_writer_rejects_missing_required_regions(tmp_path):
    with pytest.raises(ValueError, match="incomplete.*fly IDs.*1"):
        save_wall_cache_npz(
            str(tmp_path / "wall.npz"),
            {"va_id": 0},
            wall_orientations=None,
            per_fly_payload={1: {"wall": {}}},
        )


def test_wall_cache_loader_rejects_manifest_only_fly_payload(tmp_path):
    cache_path = tmp_path / "wall.npz"
    manifest = {"va_id": 0, "payload": {"va_id": 0, "flies": [1]}}
    manifest_bytes = np.frombuffer(
        json.dumps(manifest).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(cache_path, manifest_json=manifest_bytes)

    with pytest.raises(ValueError, match="missing required regions.*fly IDs.*1"):
        load_wall_cache_npz(str(cache_path))
