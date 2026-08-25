import numpy as np

from src.analysis.training import Training
from src.plotting.agarose_virtual_control_summary import (
    ChamberPlacementValues,
    plot_agarose_virtual_control_summary,
)
from scripts.plot_agarose_virtual_control import (
    _bundle_variant_metadata,
    _farthest_center_indices,
    _large_chamber_geometry_from_protocol,
    _rotated_centers,
)


def test_bundle_variant_metadata_reads_new_settings_and_legacy_defaults(tmp_path):
    current = tmp_path / "current.npz"
    legacy = tmp_path / "legacy.npz"
    np.savez(
        current,
        agarose_farthest_from_reward_only=np.array(False),
        agarose_wall_facing_entry_only=np.array(True),
        agarose_wall_facing_reference=np.array("reward", dtype=object),
        agarose_dual_circle_center_shift_mm=np.array(1.0),
    )
    np.savez(legacy, agarose_ratio_exp=np.asarray([0.5]))

    assert _bundle_variant_metadata(current) == {
        "farthest_only": False,
        "wall_facing_only": True,
        "wall_reference": "reward",
        "center_shift_mm": 1.0,
    }
    assert _bundle_variant_metadata(legacy) == {
        "farthest_only": False,
        "wall_facing_only": False,
        "wall_reference": "arena",
        "center_shift_mm": 0.0,
    }


def test_geometry_annotation_preserves_center_distance_and_draws():
    image = np.zeros((220, 220, 3), dtype=np.uint8)
    physical = ((50, 110), (110, 50), (170, 110), (110, 170))

    geometry = Training.annotateAgaroseVirtualControlGeometry(
        image,
        reward_circle=(110, 110, 8),
        physical_centers=physical,
        arena_center=(110, 110),
        inner_radius_px=18,
        outer_radius_px=22,
        rotation_deg=45,
    )

    p = np.asarray(geometry["physical_centers"], dtype=float)
    v = np.asarray(geometry["virtual_centers"], dtype=float)
    center = np.asarray(geometry["arena_center"], dtype=float)
    np.testing.assert_allclose(
        np.linalg.norm(p - center, axis=1), np.linalg.norm(v - center, axis=1)
    )
    assert np.count_nonzero(image) > 0


def test_farthest_subset_indices_match_upper_right_reward_schematic():
    center = (100.0, 100.0)
    physical = ((40, 100), (100, 40), (160, 100), (100, 160))
    reward = (115.0, 85.0)
    virtual = _rotated_centers(physical, center, 45.0)

    assert _farthest_center_indices(
        physical, reward, tie_tolerance_px=0.1
    ) == (0, 3)
    assert _farthest_center_indices(
        virtual, reward, tie_tolerance_px=0.1
    ) == (3,)

    image = np.zeros((220, 220, 3), dtype=np.uint8)
    geometry = Training.annotateAgaroseVirtualControlGeometry(
        image,
        reward_circle=reward + (8,),
        physical_centers=physical,
        arena_center=center,
        inner_radius_px=12,
        outer_radius_px=15,
        rotation_deg=45,
        physical_indices=(0, 3),
        virtual_indices=(3,),
    )
    assert geometry["physical_indices"] == (0, 3)
    assert geometry["virtual_indices"] == (3,)


def test_summary_plot_writes_slide_ready_figure(tmp_path):
    agarose = ChamberPlacementValues(
        "Agarose large",
        physical=np.asarray([0.4, 0.3, 0.5, 0.2]),
        virtual=np.asarray([0.2, 0.2, 0.3, 0.1]),
    )
    flat = ChamberPlacementValues(
        "Flat large",
        physical=np.asarray([0.3, 0.2, 0.4, 0.1]),
        virtual=np.asarray([0.2, 0.1, 0.3, 0.1]),
    )
    out = tmp_path / "summary.png"

    result = plot_agarose_virtual_control_summary(
        agarose, flat, out_path=out
    )

    assert out.exists()
    assert out.stat().st_size > 0
    assert result["out_path"] == str(out)


def test_background_geometry_uses_floor_center_not_offset_reward(monkeypatch):
    payload = {
        "protocol": {
            "tm": {"x": 44.0, "y": 19.0, "fctr": 1.0},
            "info": [
                {"cPos": [(221, 147), (221, 147)], "r": [22, 10]},
                {"cPos": [(507, 147), (507, 147)], "r": [22, 10]},
            ],
        }
    }
    trx_payload = {"x": [object(), object(), object(), object()]}

    def fake_unpickle(path):
        return trx_payload if str(path).endswith(".trx") else payload

    monkeypatch.setattr(
        "scripts.plot_agarose_virtual_control.util.unpickle", fake_unpickle
    )

    geometry = _large_chamber_geometry_from_protocol(
        "representative.avi",
        frame=np.zeros((720, 720, 3), dtype=np.uint8),
        protocol_index=0,
        training_index_1based=2,
    )

    assert geometry["chamber_type"].name == "large2"
    assert geometry["arena_center"] == (196.5, 171.5)
    assert geometry["floor_bounds"] == (49.0, 24.0, 344.0, 319.0)
    assert geometry["inner_radius_px"] == 4.0 * 7.56
    assert geometry["reward_circle"] == (221.0, 147.0, 10.0)
    assert geometry["arena_center"] != geometry["reward_circle"][:2]
