import csv
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from src.analysis.tracking_gap_review import compare_annotations
from src.exporting.tracking_gap_review_worksheet import WORKSHEET_JS, render_worksheet


def episode():
    return dict(review_id="abc", video_path="/videos/example.avi", fly=1, role="yok",
                circle="control", training=2, sync_buckets="4;5", duration_s=2.,
                source="gap_review", strata="state_change", selected_gap_ids="[2]",
                review_start_frame=10, review_stop_frame_exclusive=20, calculated_entry_frames="[12]")


def annotation():
    return dict(review_id="abc", review_status="reviewed", observed_entry_frames="[12]",
                confidence="high", reviewer="AB", notes='Visible entry, then exit.\n"Clear view"')


class Inspect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.scripts, self.in_data, self.payload = [], 0, False, ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr" and "data-review-id" in attrs:
            self.rows.append(attrs["data-review-id"])
        if tag == "script":
            self.scripts += 1
            self.in_data = attrs.get("id") == "review-data"

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False

    def handle_data(self, data):
        if self.in_data:
            self.payload += data


def test_table_contains_background_and_preserves_existing_notes_safely():
    row = annotation()
    row["notes"] += "</textarea><script>alert('x')</script>"
    rendered = render_worksheet([episode()], [row], bundle=Path("/bundle"), python="python", script="review.py")
    parser = Inspect()
    parser.feed(rendered)
    assert parser.rows == ["abc"]
    assert parser.scripts == 2
    assert json.loads(parser.payload)["annotations"][0] == row
    assert "example.avi" in rendered and "[10, 20)" in rendered
    assert "Inclusive last frame: 19" in rendered
    assert "Fly ID" in rendered and "Role" in rendered
    assert "&lt;/textarea&gt;&lt;script&gt;" in rendered
    assert "Calculated entry count" not in rendered


def test_identity_mismatch_or_duplicates_fail():
    with pytest.raises(ValueError, match="match episode IDs"):
        render_worksheet([episode()], [], bundle="", python="", script="")
    with pytest.raises(ValueError, match="match episode IDs"):
        render_worksheet([episode()], [annotation(), annotation()], bundle="", python="", script="")


def node_result(rows, *, selected_episode=None):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is optional; needed only to execute worksheet export tests")
    program = """
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0,'utf8'));
const scope = {};
vm.runInNewContext(input.source,scope);
process.stdout.write(JSON.stringify({
  errors:input.rows.map(r=>scope.validationError(r,input.episode)),
  csv:scope.annotationsCSV(input.rows),
  case_number:scope.caseNumber(input.episode,0)
}));
"""
    result = subprocess.run([node, "-e", program], input=json.dumps(dict(
        source=WORKSHEET_JS, rows=rows, episode=selected_episode or episode())), text=True,
        capture_output=True, check=True)
    return json.loads(result.stdout)


def test_javascript_export_roundtrips_into_python_comparison():
    expected = annotation()
    result = node_result([expected])
    assert result["errors"] == [""]
    exported = list(csv.DictReader(io.StringIO(result["csv"])))
    assert exported == [expected]
    comparison, = compare_annotations([episode()], exported)
    assert comparison["observed_count"] == 1
    assert comparison["observed_minus_calculated"] == 0
    assert comparison["reviewer"] == "AB"


@pytest.mark.parametrize("status,frames,confidence", [
    ("reviewed", "", "high"), ("reviewed", "[20]", "high"),
    ("reviewed", "[12,12]", "high"), ("reviewed", "[true]", "high"),
    ("reviewed", "[12]", ""), ("pending", "[]", ""),
])
def test_export_validation_rejects_incomplete_or_out_of_bounds_rows(status, frames, confidence):
    row = dict(annotation(), review_status=status, observed_entry_frames=frames, confidence=confidence)
    assert node_result([row])["errors"][0]


def test_explicit_zero_and_pending_are_valid_but_different():
    zero = dict(annotation(), observed_entry_frames="[]")
    pending = dict(annotation(), review_status="pending", observed_entry_frames="", confidence="")
    assert node_result([zero])["errors"] == [""]
    assert node_result([pending])["errors"] == [""]


def test_worksheet_validation_uses_global_case_number_with_legacy_fallback():
    assert node_result([annotation()], selected_episode=dict(episode(), case_number=53))["case_number"] == 53
    assert node_result([annotation()])["case_number"] == 1
