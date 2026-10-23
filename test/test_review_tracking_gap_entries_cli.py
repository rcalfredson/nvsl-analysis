import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import review_tracking_gap_entries as cli


def test_build_refuses_to_overwrite_manual_annotations(tmp_path):
    annotations=tmp_path/"annotations.csv"
    annotations.write_text("existing manual work")
    with pytest.raises(ValueError, match="existing annotations"):
        cli.build(SimpleNamespace(output_dir=tmp_path))
    assert annotations.read_text()=="existing manual work"


def bundle(tmp_path):
    episode=dict(review_id="a",video_path="v",fly=1,role="exp",circle="reward",source="gap_review",
                 review_start_frame=10,review_stop_frame_exclusive=20,calculated_entry_frames="[12]")
    cli.write_csv(tmp_path/"episodes.csv",[episode])
    cli.write_csv(tmp_path/"annotations.csv",[dict(review_id="a",review_status="pending",observed_entry_frames="",
                                                confidence="",reviewer="",notes="")])
    manifest=dict(episodes_sha256=hashlib.sha256((tmp_path/"episodes.csv").read_bytes()).hexdigest())
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    return episode


def test_pending_summary_does_not_claim_zero_observed_errors(tmp_path):
    bundle(tmp_path)
    cli.summarize(SimpleNamespace(bundle=tmp_path,annotations=None,tolerance_frames=2))
    row,=cli.read_csv(tmp_path/"review_summary.csv")
    assert row["reviewed"]=="0" and row["pending"]=="1"
    assert row["observed_entries"]==row["net_observed_minus_calculated"]==""
    assert (tmp_path/"review_summary.html").exists()


def test_edited_episode_file_fails_hash_check(tmp_path):
    bundle(tmp_path)
    with (tmp_path/"episodes.csv").open("a") as handle:
        handle.write("changed\n")
    with pytest.raises(ValueError,match="episodes.csv changed"):
        cli.load_episodes(tmp_path)


def test_changed_source_fails_before_video_review(tmp_path):
    source=tmp_path/"raw.trx"
    source.write_bytes(b"abc")
    stat=source.stat()
    (tmp_path/"manifest.json").write_text(json.dumps(dict(source_files=[dict(
        path=str(source),size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)])))
    source.write_bytes(b"abcd")
    with pytest.raises(ValueError,match="Source file changed"):
        cli.verify_sources(tmp_path)


def test_renderer_does_not_draw_a_position_during_missing_tracking(monkeypatch):
    import cv2
    calls=[]
    actual=cv2.drawMarker
    def marker(*args,**kwargs):
        calls.append(args[1])
        return actual(*args,**kwargs)
    monkeypatch.setattr(cv2,"drawMarker",marker)
    context=dict(pair=1,role="exp",floor=((10,10),(90,90)),x=np.array([np.nan,50.]),y=np.array([np.nan,50.]),
                 border_px=1.,geometries=[dict(training=2,reward=(50,50,10),control=(50,50,30))])
    episode=dict(training=2,circle="reward",review_start_frame=0,review_stop_frame_exclusive=2,
                 calculated_entry_frames="[]",review_id="a")
    raw=np.zeros((100,100,3),np.uint8)
    missing=cli.render_frame(raw,0,context,episode)
    assert calls==[]
    assert missing.shape[0]>raw.shape[0]
    cli.render_frame(raw,1,context,episode)
    assert calls==[(50,50)]
    assert not raw.any()


def test_calculated_entry_cues_are_opt_in(monkeypatch):
    import cv2
    labels=[]
    actual=cv2.putText
    def text(image,label,*args,**kwargs):
        labels.append(label)
        return actual(image,label,*args,**kwargs)
    monkeypatch.setattr(cv2,"putText",text)
    context=dict(pair=1,role="exp",floor=((10,10),(90,90)),x=np.array([50.]),y=np.array([50.]),
                 border_px=1.,geometries=[dict(training=2,reward=(50,50,10),control=(50,50,30))])
    episode=dict(training=2,circle="reward",review_start_frame=0,review_stop_frame_exclusive=1,
                 calculated_entry_frames="[0]",review_id="a")
    raw=np.zeros((100,100,3),np.uint8)
    cli.render_frame(raw,0,context,episode)
    assert "CALCULATED ENTRY" not in labels
    cli.render_frame(raw,0,context,episode,show_calculated=True)
    assert "CALCULATED ENTRY" in labels


def test_worksheet_preserves_annotation_file_and_refuses_accidental_overwrite(tmp_path):
    bundle(tmp_path)
    before=(tmp_path/"annotations.csv").read_bytes()
    args=SimpleNamespace(bundle=tmp_path,output=None,overwrite=False)
    cli.worksheet(args)
    assert (tmp_path/"worksheet.html").exists()
    assert (tmp_path/"annotations.csv").read_bytes()==before
    with pytest.raises(ValueError,match="already exists"):
        cli.worksheet(args)
