import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.heatmap_pairing import (
    eligibility_report, paired_keys, prepare_paired_heatmaps,
)
from src.analysis.video_analysis import VideoAnalysis


def options(**kwargs):
    defaults = dict(hm_sync_bucket=5, syncBucketLenMin=10,
                    hm_sync_bucket_head_minutes=None, hm_sync_bucket_tail_minutes=5,
                    hm_periods=("training",), hm_pair_export=None, hm_pair_manifest=None)
    return SimpleNamespace(**(defaults | kwargs))


def video(name, *, start=10, stop=600, bad=False, valid=True, walking=True):
    return SimpleNamespace(
        fn=name, f=0,
        trns=[SimpleNamespace(start=0, stop=stop)],
        trx=[SimpleNamespace(bad=lambda: bad, nan=np.full(700, not valid))],
        heatmap=[[((np.ones((2, 2)) if walking else np.zeros((2, 2))), 50, None)]],
        _min2f=lambda minutes: int(minutes * 10),
        _syncBucket=lambda trn, df: (start, 6, []),
    )


class _MutableTrajectory:
    def __init__(self, *, bad=False):
        self._bad = bad
        self.nan = np.zeros(700, dtype=bool)
        self.pctInC_SB = {"slideConc": [[42.0]]}

    def bad(self, value=None):
        if value is not None:
            self._bad = value
        return self._bad


def video_with_nan_reward_pi(*, bad=False):
    va = video("recording.avi")
    va.trx = [_MutableTrajectory(bad=bad)]
    va.flies = (0,)
    va.noyc = True
    va.rewardPIPre = [np.nan]
    va.rewardPI = [[np.nan]]
    va.rewardPiPst = [[np.nan, np.nan]]
    va.rewardPiPstNoSync = [[np.nan, np.nan]]
    va.buckets = np.array([np.nan])
    return va


def manifest(tmp_path, rows):
    path = tmp_path / "pairs.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair_id", "before_video", "before_fly", "after_video", "after_fly"])
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("kwargs,reason,occupancy", [
    ({}, "", True),
    ({"start": None}, "missing_sync", False),
    ({"stop": 500}, "incomplete_bucket", False),
    ({"bad": True}, "rejected_trajectory", False),
    ({"valid": False}, "no_valid_bucket_frames", False),
    ({"walking": False}, "empty_heatmap_window", True),
])
def test_eligibility(kwargs, reason, occupancy):
    row = eligibility_report([video("a.avi", **kwargs)], options(), 0)["records"][0]
    assert row["reason"] == reason
    assert row["occupancy_ok"] == occupancy


def test_one_valid_frame_passes_without_reward_count_threshold():
    va = video("a.avi", valid=False)
    va.trx[0].nan[450] = False
    assert eligibility_report([va], options(), 0)["records"][0]["reason"] == ""


def test_valid_slide_circle_bucket_remains_eligible_when_reward_pi_is_nan():
    va = video_with_nan_reward_pi()
    assert np.isfinite(va.trx[0].pctInC_SB["slideConc"][0][0])

    VideoAnalysis.rewardPiCombined(va)
    record = eligibility_report([va], options(hm_sync_bucket=1), 0)["records"][0]

    assert record["occupancy_ok"] is True
    assert record["heatmap_ok"] is True
    assert record["reason"] == ""


def test_genuinely_bad_trajectory_remains_ineligible_after_reward_pi_output():
    va = video_with_nan_reward_pi(bad=True)

    VideoAnalysis.rewardPiCombined(va)
    record = eligibility_report([va], options(hm_sync_bucket=1), 0)["records"][0]

    assert record["occupancy_ok"] is False
    assert record["reason"] == "rejected_trajectory"


def test_same_pairs_both_sides_and_audited_missing_recording(tmp_path):
    before = eligibility_report([video("a"), video("b")], options(), 0)
    after = eligibility_report([video("aa"), video("bb", walking=False)], options(), 0)
    path = manifest(tmp_path, [("1", "a", 0, "aa", 0), ("2", "b", 0, "bb", 0),
                               ("3", "c", 0, "cc", 0)])
    left, audit = paired_keys(path, before, after, "before")
    right, _ = paired_keys(path, before, after, "after")
    assert len(left) == len(right) == 1
    assert [r["included"] for r in audit] == [True, False, False]
    assert audit[1]["after_reason"] == "empty_heatmap_window"
    assert audit[2]["before_reason"] == "missing_recording"


def test_duplicate_fly_mapping_rejected(tmp_path):
    report = eligibility_report([video("a")], options(), 0)
    path = manifest(tmp_path, [("1", "a", 0, "aa", 0), ("2", "a", 0, "bb", 0)])
    with pytest.raises(ValueError, match="reused"):
        paired_keys(path, report, report, "before")


def test_export_then_filter_and_detect_stale_report(tmp_path):
    before_vas = [video("a"), video("b")]
    after_vas = [video("aa"), video("bb", valid=False)]
    before_path, after_path = tmp_path / "before.json", tmp_path / "after.json"
    for vas, path in ((before_vas, before_path), (after_vas, after_path)):
        _, export_only = prepare_paired_heatmaps(vas, options(hm_pair_export=path), [0])
        assert export_only
    pairs = manifest(tmp_path, [("1", "a", 0, "aa", 0), ("2", "b", 0, "bb", 0)])
    opts = options(hm_pair_manifest=pairs, hm_pair_side="before",
                   hm_pair_before_report=before_path, hm_pair_after_report=after_path,
                   hm_pair_audit=tmp_path / "audit.csv")
    selected, export_only = prepare_paired_heatmaps(before_vas, opts, [0])
    assert selected == before_vas[:1]
    assert not export_only
    assert "no_valid_bucket_frames" in opts.hm_pair_audit.read_text()
    before_vas[0].heatmap[0][0][0][0, 0] = 5
    with pytest.raises(ValueError, match="differs"):
        prepare_paired_heatmaps(before_vas, opts, [0])


def test_requires_one_training_and_training_only(tmp_path):
    opts = options(hm_pair_export=tmp_path / "report.json")
    with pytest.raises(ValueError, match="one training"):
        prepare_paired_heatmaps([], opts, [0, 1])
    opts.hm_periods = ("training", "post")
    with pytest.raises(ValueError, match="training-only"):
        prepare_paired_heatmaps([], opts, [0])


def test_unpaired_path_unchanged():
    vas = [video("a")]
    assert prepare_paired_heatmaps(vas, options(), [0]) == (vas, False)


def test_plot_integration_pairs_on_experimental_fly_and_renders_both_fly_roles(tmp_path):
    # Load the plotting function without executing analyze.py's CLI/main program.
    import ast
    from pathlib import Path
    import cv2
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from src.utils import util
    from src.utils.common import CT
    from src.plotting.heatmap_style import apply_heatmap_text_layout

    before = [video("a"), video("b")]
    after = [video("aa"), video("bb", walking=False)]
    before_path, after_path = tmp_path / "before.json", tmp_path / "after.json"
    for vas, path in ((before, before_path), (after, after_path)):
        prepare_paired_heatmaps(vas, options(hm_pair_export=path), [0])
    pairs = manifest(tmp_path, [("1", "a", 0, "aa", 0), ("2", "b", 0, "bb", 0)])
    opts = options(hm_pair_manifest=pairs, hm_pair_side="before",
                   hm_pair_before_report=before_path, hm_pair_after_report=after_path,
                   hm_pair_audit=tmp_path / "audit.csv", hm="I", bg=None,
                   fontSize=12, num_trainings="1", hm_vmin=None, hm_vmax=None,
                   imageFormat="png")
    for va in before:
        va.gidx, va.ct, va.flies, va.noyc = 0, CT.large, [0, 1], False
        va.heatmap.append([(np.full((2, 2), 2.0), 50, None)])
        va.heatmapOOB = False
        va.trns[0].sname = lambda: "t1"
        va.trns[0].circles = lambda f: []
    source = ast.parse(Path("analyze.py").read_text())
    fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "plotHeatmaps")
    writes = []
    namespace = dict(
        opts=opts, util=util, np=np, mpl=mpl, plt=plt, cv2=cv2, CT=CT,
        OP_LIN="I", P=False, F2T=False, HEATMAP_DIV=2,
        HEATMAPS_IMG_FILE="heatmaps%s.png", pch=lambda a, b: b, pcap=lambda s: s,
        _resolve_heatmap_training_indices=lambda trns, selector: [0],
        _parse_hm_bounds_arg=lambda value, name: (None, None, None),
        prepare_paired_heatmaps=prepare_paired_heatmaps,
        apply_heatmap_text_layout=apply_heatmap_text_layout,
        writeImage=lambda *args, **kwargs: writes.append(args),
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "analyze.py", "exec"), namespace)
    try:
        namespace["plotHeatmaps"](before)
        assert len(writes) == 2
        fig = plt.gcf()
        assert len(fig.axes) == 3  # experimental, yoked control, and colorbar
        assert any(ax.get_title(loc="right") == "n=1" for ax in fig.axes)
        assert "False" in opts.hm_pair_audit.read_text()
    finally:
        plt.close("all")


def test_early_skipped_record_does_not_require_heatmap_or_training():
    from src.analysis.heatmap_pairing import skipped_eligibility_record

    rejected = SimpleNamespace(trx=[SimpleNamespace(bad=lambda: True)])
    row = skipped_eligibility_record(rejected, "before.avi", 0)
    assert row["reason"] == "rejected_trajectory"
    assert row["occupancy_ok"] is row["heatmap_ok"] is False
    assert row["heatmap_digest"] is None
    assert skipped_eligibility_record(SimpleNamespace(), "other.avi", 0)["reason"] == "analysis_skipped"
    assert skipped_eligibility_record(SimpleNamespace(isyc=True), "yoked.avi", 1)["reason"] == "not_experimental_fly"


def test_skipped_before_fly_is_matched_and_audited_as_rejected(tmp_path):
    from src.analysis.heatmap_pairing import skipped_eligibility_record

    skipped = skipped_eligibility_record(
        SimpleNamespace(trx=[SimpleNamespace(bad=lambda: True)]), "b", 0
    )
    opts = options(_hm_pair_skipped_records=[skipped])
    before = eligibility_report([video("a")], opts, 0)
    after = eligibility_report([video("aa"), video("bb")], options(), 0)
    path = manifest(tmp_path, [("1", "a", 0, "aa", 0), ("2", "b", 0, "bb", 0)])
    keep, audit = paired_keys(path, before, after, "after")
    assert len(keep) == 1
    assert audit[1]["before_reason"] == "rejected_trajectory"
    assert audit[1]["included"] is False


def test_analysis_loop_exports_all_skipped_flies(tmp_path):
    """Execute the actual collection/early-report block from analyze.py."""
    import ast
    from pathlib import Path
    from src.analysis.heatmap_pairing import skipped_eligibility_record
    from src.utils.parsers import parse_training_selector

    source = ast.parse(Path("analyze.py").read_text())
    analyze_fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
    start = next(i for i, n in enumerate(analyze_fn.body)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "(vas, va)")
    stop = next(i for i in range(start, len(analyze_fn.body))
                if isinstance(analyze_fn.body[i], ast.If)
                and ast.unparse(analyze_fn.body[i].test) == "vas")
    analyze_fn.body = analyze_fn.body[start:stop]
    opts = options(hm_pair_export=tmp_path / "before.json", num_trainings="2")
    rejected = SimpleNamespace(trx=[SimpleNamespace(bad=lambda: True)], skipped=lambda: True)
    namespace = dict(
        opts=opts, cns=[51], fns=["b.avi"], ng=1, fn2fs={"b.avi": [[0]]},
        VideoAnalysis=lambda *args: rejected, skipped_eligibility_record=skipped_eligibility_record,
        prepare_paired_heatmaps=prepare_paired_heatmaps, parse_training_selector=parse_training_selector,
    )
    exec(compile(ast.Module(body=[analyze_fn], type_ignores=[]), "analyze.py", "exec"), namespace)
    namespace["analyze"]()
    report = json.loads(opts.hm_pair_export.read_text())
    assert report["window"]["training"] == 2
    assert len(report["records"]) == 1
    assert report["records"][0]["reason"] == "rejected_trajectory"
    # Running again resets the per-run collection rather than accumulating rows.
    namespace["analyze"]()
    assert len(json.loads(opts.hm_pair_export.read_text())["records"]) == 1


def test_old_reports_require_regeneration(tmp_path):
    report = eligibility_report([video("a")], options(), 0)
    report["version"] = 1
    with pytest.raises(ValueError, match="regenerate both reports"):
        paired_keys(tmp_path / "unused.csv", report, report, "before")


def test_all_skipped_render_writes_rejection_audit(tmp_path):
    from src.analysis.heatmap_pairing import skipped_eligibility_record

    skipped = skipped_eligibility_record(
        SimpleNamespace(trx=[SimpleNamespace(bad=lambda: True)]), "b", 0
    )
    opts = options(_hm_pair_skipped_records=[skipped])
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps(eligibility_report([], opts, 0)))
    after.write_text(json.dumps(eligibility_report([video("bb")], options(), 0)))
    opts.hm_pair_manifest = manifest(tmp_path, [("1", "b", 0, "bb", 0)])
    opts.hm_pair_side = "before"
    opts.hm_pair_before_report, opts.hm_pair_after_report = before, after
    opts.hm_pair_audit = tmp_path / "audit.csv"
    with pytest.raises(ValueError, match="No eligible heatmap pairs"):
        prepare_paired_heatmaps([], opts, [0])
    assert "rejected_trajectory" in opts.hm_pair_audit.read_text()
