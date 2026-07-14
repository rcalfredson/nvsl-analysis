import zipfile

import numpy as np
import pytest

from src.exporting.bundle_metadata import edit_group_labels, read_group_labels


def _bundle(path):
    np.savez_compressed(
        path,
        group_label=np.array("combined"),
        group_labels=np.asarray(["Control", "Treatment"]),
        group_indices=np.asarray([0, 1, 1]),
        data=np.arange(12).reshape(3, 4),
    )


def test_edits_group_labels_without_changing_other_npy_members(tmp_path):
    path = tmp_path / "bundle.npz"
    _bundle(path)
    with zipfile.ZipFile(path) as archive:
        original_members = {
            name: archive.read(name)
            for name in archive.namelist()
            if name != "group_labels.npy"
        }

    edit_group_labels(path, ["Control>Kir", "MBKC-1>Kir"])

    assert read_group_labels(path)["group_labels"] == [
        "Control>Kir",
        "MBKC-1>Kir",
    ]
    with zipfile.ZipFile(path) as archive:
        assert all(archive.read(name) == value for name, value in original_members.items())


def test_rejects_label_count_change_and_preserves_source(tmp_path):
    path = tmp_path / "bundle.npz"
    _bundle(path)
    original = path.read_bytes()

    with pytest.raises(ValueError, match="contains 2 label"):
        edit_group_labels(path, ["only one"])

    assert path.read_bytes() == original


def test_can_edit_scalar_label_to_a_new_output(tmp_path):
    source = tmp_path / "bundle.npz"
    output = tmp_path / "edited.npz"
    _bundle(source)

    edit_group_labels(source, ["new label"], field="group_label", output=output)

    assert read_group_labels(source)["group_label"] == ["combined"]
    assert read_group_labels(output)["group_label"] == ["new label"]


def test_rejects_non_allowlisted_field(tmp_path):
    path = tmp_path / "bundle.npz"
    _bundle(path)
    with pytest.raises(ValueError, match="not editable"):
        edit_group_labels(path, ["nope"], field="data")
