import json
from zipfile import ZipFile

import pytest

from src.analysis.tracking_gap_review_import import (
    match_observations, parse_observed_frames, read_ods_observations,
)
from src.analysis.tracking_gap_review import compare_annotations, exclude_previous_windows
from src.analysis.tracking_gap_review_import import reconcile, select_observations


def episode():
    return dict(review_id="a", video_path="/data/video.avi", fly="1", role="exp",
                circle="reward", source="gap_review", review_start_frame="10",
                review_stop_frame_exclusive="20", calculated_entry_frames="[12]")


def observation():
    return dict(case=1, video_filename="video.avi", fly="1", role="exp", circle="reward",
                scored_frames="[10, 20)", interval_code="G", observed_entry_text="13", notes="")


def test_ods_repeated_cells_and_rich_text(tmp_path):
    path=tmp_path/"audit.ods"
    xml='''<root xmlns:t="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
      xmlns:x="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
      <t:table t:name="Sheet1"><t:table-row>
      <t:table-cell><x:p>1</x:p></t:table-cell>
      <t:table-cell t:number-columns-repeated="7"/>
      <t:table-cell><x:p>fly<x:s x:c="2"/><x:span>enters</x:span></x:p></t:table-cell>
      <t:table-cell t:number-columns-repeated="10000"/>
      </t:table-row></t:table></root>'''
    with ZipFile(path,"w") as archive:
        archive.writestr("content.xml",xml)
    row,=read_ods_observations(path)
    assert row["case"]==1 and row["notes"]=="fly  enters"
    with pytest.raises(ValueError,match="not found"):
        read_ods_observations(path,"wrong")


@pytest.mark.parametrize("text,expected",[("none",[]),("[]",[]),("",None),("unknown",None),("12, 14",[12,14])])
def test_missing_observation_is_not_zero(text,expected):
    assert parse_observed_frames(text)==expected


def test_identity_validation_and_confidence_remain_explicit():
    ep=episode()
    rows,annotations=match_observations([observation()],[ep])
    assert annotations[0]["confidence"]==annotations[0]["reviewer"]==""
    with pytest.raises(ValueError,match="confidence"):
        compare_annotations([ep],annotations)
    compared=compare_annotations([ep],annotations,require_confidence=False)
    assert compared[0]["observed_minus_calculated"]==0
    report,=reconcile(rows,[ep],compared,[])
    assert report["category"]=="matching_count_small_timing_offset"
    assert report["interval_type"]=="gap_in_context_only"
    wrong=dict(observation(),circle="control")
    with pytest.raises(ValueError,match="identity"):
        match_observations([wrong],[ep])


def test_other_circle_and_count_disagreement_are_flagged():
    row=dict(observation(),observed_entry_text="none",notes="enters control circle")
    rows,annotations=match_observations([row],[episode()])
    compared=compare_annotations([episode()],annotations,require_confidence=False)
    report,=reconcile(rows,[episode()],compared,[])
    assert report["category"]=="count_disagreement_requires_recheck"
    assert "other circle" in report["annotation_flags"]


def test_exclude_prior_windows_uses_half_open_bounds_and_circle():
    old=episode()
    touching=dict(old,review_start_frame=20,review_stop_frame_exclusive=30)
    overlapping=dict(old,review_start_frame=19,review_stop_frame_exclusive=30)
    other=dict(overlapping,circle="control")
    assert exclude_previous_windows([touching,overlapping,other],[old])==[touching,other]


def test_new_context_only_code_can_be_imported():
    ep=dict(episode(),case_number=53,interval_type="gap_in_context_only")
    row=dict(observation(),case=53,interval_code="C")
    _,annotations=match_observations([row],[ep])
    assert json.loads(annotations[0]["observed_entry_frames"])==[13]


def test_cumulative_selection_and_explicit_partial_import():
    episodes = [dict(episode(), case_number=n, review_id=str(n)) for n in (53, 54, 55)]
    rows = [dict(observation(), case=n) for n in (1, 53, 54)]
    with pytest.raises(ValueError, match="missing selected cases.*55"):
        select_observations(rows, episodes)
    selected, eps, coverage = select_observations(rows, episodes, allow_partial=True)
    assert [r["case"] for r in selected] == [53, 54]
    assert [e["case_number"] for e in eps] == [53, 54]
    assert coverage == dict(included_cases=[53, 54], missing_cases=[55],
                            excluded_spreadsheet_cases=[1], excluded_bundle_cases=[])
    selected, eps, coverage = select_observations(rows, episodes, case_start=54, case_end=54)
    assert [r["case"] for r in selected] == [54]
    assert coverage["excluded_spreadsheet_cases"] == [1, 53]
    assert coverage["excluded_bundle_cases"] == [53, 55]


def test_selection_rejects_duplicates_even_outside_range():
    episodes = [dict(episode(), case_number=n) for n in (1, 1, 53)]
    with pytest.raises(ValueError, match="duplicate global"):
        select_observations([dict(observation(), case=53)], episodes, case_start=53)
    with pytest.raises(ValueError, match="exactly once"):
        match_observations([observation(), observation()], [episode()])


@pytest.mark.parametrize("start,end", [(0, 3), (1, 0), (4, 3)])
def test_invalid_selection_bounds(start, end):
    with pytest.raises(ValueError, match="Case"):
        select_observations([observation()], [dict(episode(), case_number=1)],
                            case_start=start, case_end=end)


def test_partial_import_does_not_relax_identity_validation():
    row = dict(observation(), case=53, circle="control")
    eps = [dict(episode(), case_number=53), dict(episode(), case_number=54)]
    selected, included, _ = select_observations([row], eps, allow_partial=True)
    with pytest.raises(ValueError, match="identity"):
        match_observations(selected, included)


def test_unmatched_events_take_precedence_over_small_offsets():
    ep = dict(episode(), calculated_entry_frames="[12, 18]")
    rows, annotations = match_observations([dict(observation(), observed_entry_text="11, 15")], [ep])
    comparisons = compare_annotations([ep], annotations, require_confidence=False)
    result, = reconcile(rows, [ep], comparisons, [])
    assert result["observed_minus_calculated"] == 0
    assert result["calculated_minus_observed_frame_offsets"] == "[1]"
    assert result["unmatched_calculated_frames"] == "[18]"
    assert result["category"] == "matching_count_temporal_mismatch"


def test_circle_specific_zero_note_is_flagged_without_changing_entries():
    row = dict(observation(), notes="no reward circle interactions during scored frames")
    rows, annotations = match_observations([row], [episode()])
    result, = reconcile(rows, [episode()], compare_annotations(
        [episode()], annotations, require_confidence=False), [])
    assert "nonempty" in result["annotation_flags"]
    assert result["observed_count"] == 1


def test_new_bundle_missing_count_is_authoritative_over_filtered_candidates():
    ep = dict(episode(), scored_missing_frames="3")
    rows, annotations = match_observations([observation()], [ep])
    result, = reconcile(rows, [ep], compare_annotations([ep], annotations, require_confidence=False), [])
    assert result["scored_missing_frames"] == 3
    assert result["interval_type"] == "gap_in_scored_interval"
