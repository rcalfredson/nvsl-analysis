"""Exercise cumulative ODS imports without real videos or mounted recordings."""
import csv
import hashlib
import json
from types import SimpleNamespace
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest

from scripts.import_tracking_gap_audit import import_audit
from src.utils.tracking_gap_review_io import write_csv


TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"


def make_bundle(path, cases):
    path.mkdir()
    episodes = [dict(case_number=n, review_id=f"case-{n}", video_path="/data/video.avi",
                     fly=1, role="exp", circle="reward", source="gap_review",
                     review_start_frame=10, review_stop_frame_exclusive=20,
                     scored_missing_frames=1, calculated_entry_frames="[12]") for n in cases]
    write_csv(path / "episodes.csv", episodes)
    write_csv(path / "candidates.csv", [], fields=["video_path", "fly", "role",
              "gap_start_frame", "gap_stop_frame_exclusive"])
    (path / "manifest.json").write_text(json.dumps(dict(
        episodes_sha256=hashlib.sha256((path / "episodes.csv").read_bytes()).hexdigest())))
    return path


def make_sheet(path, observations):
    root = ET.Element("root")
    table = ET.SubElement(root, f"{{{TABLE}}}table")
    for n, frames, notes in observations:
        row = ET.SubElement(table, f"{{{TABLE}}}table-row")
        for value in (n, "video.avi", 1, "exp", "reward", "[10, 20)", "G", frames, notes):
            cell = ET.SubElement(row, f"{{{TABLE}}}table-cell")
            ET.SubElement(cell, f"{{{TEXT}}}p").text = str(value)
    with ZipFile(path, "w") as archive:
        archive.writestr("content.xml", ET.tostring(root))
    return path


def args(tmp_path, bundles, observations):
    return SimpleNamespace(bundle=bundles,
                           spreadsheet=make_sheet(tmp_path / "audit.ods", observations),
                           output_dir=tmp_path / "output", sheet=None, case_start=None,
                           case_end=None, allow_partial=False)


def records(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_two_bundles_imported_with_source_and_intervals_unchanged(tmp_path):
    bundles = [make_bundle(tmp_path / "pilot", [1]), make_bundle(tmp_path / "next", [53])]
    settings = args(tmp_path, bundles, [(1, "12", ""), (53, "13", "")])
    before = {p: p.read_bytes() for b in bundles for p in b.iterdir()}
    source_bytes = settings.spreadsheet.read_bytes()
    result = import_audit(settings)
    assert [r["case"] for r in result] == [1, 53]
    assert all(p.read_bytes() == original for p, original in before.items())
    assert (settings.output_dir / "source_audit.ods").read_bytes() == source_bytes
    provenance = json.loads((settings.output_dir / "provenance.json").read_text())
    assert provenance["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert len(provenance["bundles"]) == 2
    assert provenance["selection"]["included_cases"] == [1, 53]
    with pytest.raises(ValueError, match="preserved"):
        import_audit(settings)


def test_partial_continuation_reports_excluded_and_missing_cases(tmp_path):
    bundle = make_bundle(tmp_path / "next", [53, 54])
    settings = args(tmp_path, [bundle], [(1, "12", ""), (53, "12", "")])
    with pytest.raises(ValueError, match="missing selected"):
        import_audit(settings)
    assert not settings.output_dir.exists()
    settings.allow_partial = True
    result = import_audit(settings)
    assert [r["case"] for r in result] == [53]
    selection = json.loads((settings.output_dir / "selection.json").read_text())
    assert selection["excluded_spreadsheet_cases"] == [1]
    assert selection["missing_cases"] == [54]
    assert "54" in (settings.output_dir / "report.md").read_text()


def test_all_unscorable_import_has_no_division_by_zero_or_false_agreements(tmp_path):
    settings = args(tmp_path, [make_bundle(tmp_path / "pilot", [1])], [(1, "unknown", "unclear")])
    result = import_audit(settings)
    assert result[0]["category"] == "unscorable"
    assert result[0]["observed_count"] is None
    report = (settings.output_dir / "report.md").read_text()
    assert "no reviewed cases" in report
    assert "Matching counts and times: 0" in report
    assert "three-fly" not in report


def test_temporal_mismatch_and_circle_specific_note_are_in_recheck_export(tmp_path):
    bundle = make_bundle(tmp_path / "pilot", [1])
    episodes = records(bundle / "episodes.csv")
    episodes[0]["calculated_entry_frames"] = "[12, 18]"
    write_csv(bundle / "episodes.csv", episodes)
    (bundle / "manifest.json").write_text(json.dumps(dict(episodes_sha256=
        hashlib.sha256((bundle / "episodes.csv").read_bytes()).hexdigest())))
    settings = args(tmp_path, [bundle], [(1, "11, 15", "no reward circle interactions during scored frames")])
    import_audit(settings)
    row, = records(settings.output_dir / "recheck_cases.csv")
    assert row["category"] == "matching_count_temporal_mismatch"
    assert "nonempty" in row["annotation_flags"]


def test_tampered_bundle_fails_before_output_creation(tmp_path):
    bundle = make_bundle(tmp_path / "pilot", [1])
    settings = args(tmp_path, [bundle], [(1, "12", "")])
    with (bundle / "episodes.csv").open("a") as handle:
        handle.write("changed\n")
    with pytest.raises(ValueError, match="episodes.csv changed"):
        import_audit(settings)
    assert not settings.output_dir.exists()


def test_explicit_case_range_ignores_other_rows_but_reports_them(tmp_path):
    bundle = make_bundle(tmp_path / "next", [53, 54])
    settings = args(tmp_path, [bundle], [(1, "12", ""), (53, "12", ""), (54, "13", "")])
    settings.case_start = settings.case_end = 54
    result = import_audit(settings)
    assert [r["case"] for r in result] == [54]
    selection = json.loads((settings.output_dir / "selection.json").read_text())
    assert selection["excluded_spreadsheet_cases"] == [1, 53]
    assert selection["excluded_bundle_cases"] == [53]


def test_duplicate_bundle_numbers_fail_before_output_creation(tmp_path):
    bundles = [make_bundle(tmp_path / "a", [1]), make_bundle(tmp_path / "b", [1])]
    settings = args(tmp_path, bundles, [(1, "12", "")])
    with pytest.raises(ValueError, match="duplicate global"):
        import_audit(settings)
    assert not settings.output_dir.exists()
