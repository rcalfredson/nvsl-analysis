import json
from types import SimpleNamespace

import pytest

from scripts import recalculate_training_reward_tables as batch


SUMMARY = """rewards per minute:
means with 95% confidence intervals:
  n = 10  (in "()" below if different)
  t1, exp fly: 6.2 ±1.5 (9)
  t2, exp fly: 0.0 ±0.0
  t3, exp fly: nan ±nan (0)

rewards per distance traveled [m⁻¹], exp fly:
interval: one frame after first synchronized reward through training end
means with 95% confidence intervals:
  n = 10  (in "()" below if different)
  t1, exp fly: 100.00 ±10.00
  t2, exp fly: 100.00 ±10.00
  t3, exp fly: 100.00 ±10.00

rewards per distance traveled [m⁻¹], exp-minus-yoked:
interval: one frame after first synchronized reward through training end
means with 95% confidence intervals:
  n = 10  (in "()" below if different)
  t1, exp-minus-yoked: -2.50 ±1.00 (8)
  t2, exp-minus-yoked: 0.00 ±0.00
  t3, exp-minus-yoked: nan ±nan (0)

writing imgs/example.png...
"""


def test_manifest_and_chamber_commands():
    manifest = batch.load_manifest(batch.DEFAULT_MANIFEST)
    cohorts = manifest["cohorts"]
    assert len(cohorts) == 53
    assert cohorts[20]["metadata"][1] == "upd2/upd3 genetic manipulation experiments"
    assert all("Yang  Chen" not in v for c in cohorts for v in c["videos"])
    large = batch.analysis_command(cohorts[0], "python")
    htl = batch.analysis_command(cohorts[2], "python")
    assert large[large.index("-f") + 1] == "0-1"
    assert large[large.index("--rCC") + 1] == "15"
    assert htl[htl.index("-f") + 1] == "0-9"
    assert htl[htl.index("--rmCC") + 1] == "5"
    assert htl[htl.index("-v") + 1] == ",".join(cohorts[2]["videos"])


def test_parse_preserves_ci_n_zero_and_missing_and_selects_paired_rpd():
    result = batch.parse_summaries(SUMMARY)
    assert result["rpm"] == {"n": 10, "trainings": {
        "1": "6.2 ±1.5 (9)", "2": "0.0 ±0.0", "3": "nan ±nan (0)",
    }}
    assert result["rpd"]["trainings"]["1"] == "-2.50 ±1.00 (8)"


@pytest.mark.parametrize("text", [
    SUMMARY.replace("  t2, exp fly: 0.0 ±0.0\n", ""),
    SUMMARY + SUMMARY,
    SUMMARY.split(batch.HEADERS["rpd"])[0],
    SUMMARY.replace("n = 10", "n = 10, 20"),
    SUMMARY.replace("(9)", "(11)"),
    SUMMARY.replace("  t2, exp-minus-yoked:", "  t1, exp-minus-yoked:"),
])
def test_rejects_incomplete_ambiguous_or_invalid_summaries(text):
    with pytest.raises(ValueError):
        batch.parse_summaries(text)


def test_batch_failure_resume_running_files_and_html(tmp_path, monkeypatch):
    manifest = batch.load_manifest(batch.DEFAULT_MANIFEST)
    manifest["cohorts"] = manifest["cohorts"][:2]
    manifest["cohorts"][0]["metadata"][2] = 'A>B & "C"'
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    args = SimpleNamespace(manifest=path, python="python", out=tmp_path / "out", resume=False, dry_run=False)
    monkeypatch.setattr(batch, "source_digest", lambda: "source-v1")
    monkeypatch.setattr(batch, "validate_video_patterns", lambda c: None)
    calls = []

    def run(command, *, cwd, stdout, stderr):
        calls.append(command)
        stdout.write(SUMMARY)
        return SimpleNamespace(returncode=1 if len(calls) == 2 else 0)

    monkeypatch.setattr(batch.subprocess, "run", run)
    with pytest.raises(ValueError, match="analysis exited 1"):
        batch.run_batch(args)
    assert "cohort_01" in (args.out / "rpm_running.tsv").read_text()
    assert "cohort_02" not in (args.out / "rpd_running.tsv").read_text()
    assert not (args.out / "rpm_table_html.txt").exists()
    assert (args.out / "logs/cohort_02.log").exists()
    with pytest.raises(ValueError, match="use --resume"):
        batch.run_batch(args)

    args.resume = True
    batch.run_batch(args)
    assert len(calls) == 3  # The successful first cohort is not rerun.
    assert calls[1] == calls[2]
    rpm_html = (args.out / "rpm_table_html.txt").read_text()
    rpd_html = (args.out / "rpd_table_html.txt").read_text()
    assert rpm_html.count("<tr>") == rpd_html.count("<tr>") == 3
    assert 'A&gt;B &amp; &quot;C&quot;' in rpm_html
    assert "6.2 ±1.5 (9)" in rpm_html
    assert "-2.50 ±1.00 (8)" in rpd_html
    assert "100.00" not in rpd_html
    assert "cohort_02" in (args.out / "rpd_running.tsv").read_text()
    monkeypatch.setattr(batch, "source_digest", lambda: "source-v2")
    with pytest.raises(ValueError, match="source changed"):
        batch.run_batch(args)


def test_dry_run_does_not_write_or_execute(tmp_path, monkeypatch, capsys):
    args = SimpleNamespace(manifest=batch.DEFAULT_MANIFEST, python="python", out=tmp_path / "out", resume=False, dry_run=True)
    monkeypatch.setattr(batch.subprocess, "run", lambda *a, **k: pytest.fail("dry run executed analysis"))
    batch.run_batch(args)
    text = capsys.readouterr().out
    assert text.count("-u ") == 53
    assert "[cohort_53]" in text
    assert not args.out.exists()


def test_video_pattern_check_accepts_analysis_avi_files_and_rejects_missing_patterns(tmp_path):
    cohort = {"id": "example", "videos": [str(tmp_path / "c1_*")]}
    (tmp_path / "c1__2025-05-30__19-22-58.trx").touch()
    with pytest.raises(ValueError, match=r"matched no \.avi"):
        batch.validate_video_patterns(cohort)
    (tmp_path / "c1__2025-05-30__19-22-58.avi").touch()
    batch.validate_video_patterns(cohort)
    cohort["videos"].append(str(tmp_path / "c2_*"))
    (tmp_path / "c2_video.AVI").touch()
    batch.validate_video_patterns(cohort)
    cohort["videos"].append(str(tmp_path / "c3_*"))
    with pytest.raises(ValueError, match="c3_"):
        batch.validate_video_patterns(cohort)


def test_resume_empty_checkpoint_after_validator_fix(tmp_path, monkeypatch):
    manifest = batch.load_manifest(batch.DEFAULT_MANIFEST)
    manifest["cohorts"] = manifest["cohorts"][:1]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    (tmp_path / "results.json").write_text(json.dumps({"signature": "old-validator", "results": {}}))
    args = SimpleNamespace(manifest=path, python="python", out=tmp_path, resume=True, dry_run=False)
    monkeypatch.setattr(batch, "validate_video_patterns", lambda c: None)

    def run(command, *, cwd, stdout, stderr):
        stdout.write(SUMMARY)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, "run", run)
    batch.run_batch(args)
    assert (tmp_path / "rpm_table_html.txt").exists()
