"""Exercise the helper's manifest stage without running video analysis."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_film_slide_paired_heatmaps.sh"


def record(date, camera="c51", *, after=False, suffix="a"):
    folder = f"/data/{date}/" + ("2nd round/" if after else "")
    return {"video": f"{folder}{camera}_{suffix}.avi", "fly": "0", "reason": ""}


def run_manifest(tmp_path, before, after):
    for side, records in (("before", before), ("after", after)):
        (tmp_path / f"{side}.json").write_text(json.dumps({"records": records}))
    return subprocess.run(
        ["bash", str(SCRIPT), "manifest"], capture_output=True, text=True,
        env={**os.environ, "FILM_PYTHON": sys.executable, "FILM_PAIR_DIR": str(tmp_path)},
    )


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def run_export(tmp_path, cohort):
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
from pathlib import Path
import sys

args = sys.argv[1:]
report = Path(args[args.index('--hm-pair-export') + 1])
side = report.stem
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps({'version': 2, 'records': []}))
(report.parent / f'{side}.args.json').write_text(json.dumps(args))
"""
    )
    fake_python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT), cohort, "export"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "FILM_PYTHON": str(fake_python),
            "FILM_PAIR_DIR": str(tmp_path / "output"),
        },
    )
    return result, tmp_path / "output"


def test_shared_pairs_and_two_after_only_flies(tmp_path):
    before = [record("2025-07-19")]
    after = [record(date, after=True) for date in ("2025-07-19", "2025-07-25", "2025-07-28")]
    # Matched-but-ineligible flies must still reach the rendering audit.
    before[0]["reason"] = "empty_heatmap_window"
    result = run_manifest(tmp_path, before, after)
    assert result.returncode == 0, result.stderr
    assert len(rows(tmp_path / "pairs.csv")) == 1
    unmatched = rows(tmp_path / "unmatched.csv")
    assert [row["date"] for row in unmatched] == ["2025-07-25", "2025-07-28"]
    assert all(row["reason"] == "missing_before_identity" for row in unmatched)


def test_before_only_and_no_shared_pairs_clear_old_manifest(tmp_path):
    (tmp_path / "pairs.csv").write_text("old manifest\n")
    result = run_manifest(tmp_path, [record("2025-07-19")], [])
    assert result.returncode != 0
    assert rows(tmp_path / "pairs.csv") == []
    assert rows(tmp_path / "unmatched.csv")[0]["reason"] == "missing_after_identity"


def test_duplicate_identity_still_fails(tmp_path):
    result = run_manifest(tmp_path, [record("2025-07-19"), record("2025-07-19", suffix="b")], [])
    assert result.returncode != 0
    assert "Ambiguous before match" in result.stderr
    assert not (tmp_path / "pairs.csv").exists()


def test_fully_matched_inputs_write_empty_unmatched_report(tmp_path):
    result = run_manifest(tmp_path, [record("2025-07-19")], [record("2025-07-19", after=True)])
    assert result.returncode == 0, result.stderr
    assert len(rows(tmp_path / "pairs.csv")) == 1
    assert rows(tmp_path / "unmatched.csv") == []


def test_export_uses_exact_video_patterns_for_each_cohort(tmp_path):
    expected = {
        "film-slide": (15, ("c31", "c41", "c51", "c61")),
        "mock-slide": (16, ("c32", "c42", "c52", "c62")),
        "antennae-removed": (8, ("c31", "c41", "c51", "c61")),
    }
    for cohort, (expected_count, cameras) in expected.items():
        cohort_dir = tmp_path / cohort
        cohort_dir.mkdir()
        result, output = run_export(cohort_dir, cohort)
        assert result.returncode == 0, result.stderr
        for side in ("before", "after"):
            args = json.loads((output / f"{side}.args.json").read_text())
            patterns = args[args.index("-v") + 1].split(",")
            assert len(patterns) == expected_count
            assert all(any(f"/{camera}_*" in pattern for camera in cameras)
                       for pattern in patterns)
            assert all(("/2nd round/" in pattern) == (side == "after")
                       for pattern in patterns)
            assert args[args.index("--num-trainings") + 1] == (
                "2" if side == "before" else "1"
            )
            assert args[args.index("--hm-sync-bucket") + 1] == (
                "5" if side == "before" else "1"
            )
            assert args[args.index("--fontFamily") + 1] == "Arial"
            assert args[args.index("--fs") + 1] == "20"
        before_args = json.loads((output / "before.args.json").read_text())
        before_patterns = before_args[before_args.index("-v") + 1].split(",")
        if cohort == "film-slide":
            assert not any("2025-07-31/c31_*" in pattern for pattern in before_patterns)
        elif cohort == "mock-slide":
            assert any("2025-07-31/c32_*" in pattern for pattern in before_patterns)
        else:
            assert any("2025-08-0[56789]/c31_*" in pattern for pattern in before_patterns)
            assert any("2025-08-1[0123]/c61_*" in pattern for pattern in before_patterns)


def test_unknown_cohort_is_rejected(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "unknown", "export"],
        capture_output=True,
        text=True,
        env={**os.environ, "FILM_PAIR_DIR": str(tmp_path)},
    )
    assert result.returncode == 2
    assert "Unknown cohort" in result.stderr
