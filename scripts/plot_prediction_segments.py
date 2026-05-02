#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nvsl-analysis")

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.video_analysis import VideoAnalysis
from src.plotting.event_chain_plotter import EventChainPlotter
import src.utils.util as util

class FlyDetector:
    pass  # required for unpickle()


@dataclass
class SegmentRow:
    row: dict[str, str]
    input_index: int
    fly_key: tuple[str, str]
    fly_idx: int
    training_idx: int
    anchor_reward_frame: int
    end_reward_frame: int
    duration_frames: float


class PlotOptions(SimpleNamespace):
    def __getattr__(self, name: str) -> Any:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot selected between-reward trajectory segments exported by "
            "flygen-ml export_prediction_segments."
        )
    )
    p.add_argument("--segments-csv", required=True, help="Input exported segment CSV.")
    p.add_argument(
        "--output-dir",
        default="imgs/ml_prediction_errors",
        help="Directory for per-fly segment images.",
    )
    p.add_argument("--max-flies", type=int, default=None)
    p.add_argument("--max-segments-per-fly", type=int, default=None)
    p.add_argument(
        "--selection",
        choices=("first", "longest", "random"),
        default="first",
        help="How to choose segments within each fly group.",
    )
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument(
        "--include-training-end",
        action="store_true",
        help="Include segments censored by training end.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--image-format",
        "--imgFormat",
        dest="image_format",
        default="png",
        help="Output image format.",
    )
    p.add_argument("--pad", type=int, default=5, help="Frames of plotting context.")
    p.add_argument("--zoom", action="store_true", help="Zoom around reward circle.")
    p.add_argument("--zoom-radius-mm", type=float, default=None)
    p.add_argument(
        "--video-path-column",
        default=None,
        help=(
            "Optional CSV column containing the source .avi path. If omitted, "
            "the script derives it from data_path/trx_path."
        ),
    )
    return p.parse_args()


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "na"}


def _as_bool(value: Any) -> bool:
    if _is_blank(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _as_int(value: Any, field: str) -> int:
    if _is_blank(value):
        raise ValueError(f"missing {field}")
    return int(float(str(value).strip()))


def _as_float(value: Any, default: float = math.nan) -> float:
    if _is_blank(value):
        return default
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def _sanitize(value: Any, *, max_len: int = 80) -> str:
    text = str(value or "na").strip()
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    text = text.strip("._")
    return (text or "na")[:max_len]


def _format_float(value: Any, digits: int = 3) -> str:
    x = _as_float(value)
    if not math.isfinite(x):
        return "na"
    return f"{x:.{digits}f}"


def _fly_idx(row: dict[str, str]) -> int:
    for key in ("experimental_fly_idx", "fly_idx", "fly_id"):
        value = row.get(key)
        if not _is_blank(value):
            try:
                return int(float(str(value).strip()))
            except ValueError:
                pass
    return 0


def _training_idx(row: dict[str, str]) -> int:
    value = row.get("training_idx")
    if _is_blank(value):
        return 0
    return int(float(str(value).strip()))


def _fly_key(row: dict[str, str]) -> tuple[str, str]:
    return (str(row.get("fly_id") or ""), str(row.get("sample_key") or ""))


def _read_segments(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _filter_rows(
    rows: list[dict[str, str]], *, include_training_end: bool
) -> tuple[list[SegmentRow], Counter]:
    counts: Counter = Counter()
    kept: list[SegmentRow] = []
    for i, row in enumerate(rows):
        if not include_training_end and _as_bool(row.get("terminated_by_training_end")):
            counts["terminal_training_end"] += 1
            continue
        if _is_blank(row.get("end_reward_frame")):
            counts["missing_end_reward_frame"] += 1
            continue
        try:
            seg = SegmentRow(
                row=row,
                input_index=i,
                fly_key=_fly_key(row),
                fly_idx=_fly_idx(row),
                training_idx=_training_idx(row),
                anchor_reward_frame=_as_int(
                    row.get("anchor_reward_frame"), "anchor_reward_frame"
                ),
                end_reward_frame=_as_int(row.get("end_reward_frame"), "end_reward_frame"),
                duration_frames=_as_float(row.get("duration_frames")),
            )
        except ValueError as exc:
            counts[f"invalid_row:{exc}"] += 1
            continue
        kept.append(seg)
    return kept, counts


def _select_segments(
    segments: list[SegmentRow],
    *,
    selection: str,
    max_segments: int | None,
    rng: random.Random,
) -> list[SegmentRow]:
    selected = list(segments)
    if selection == "longest":
        selected.sort(
            key=lambda s: (
                -s.duration_frames if math.isfinite(s.duration_frames) else math.inf,
                s.input_index,
            )
        )
    elif selection == "random":
        rng.shuffle(selected)
    else:
        selected.sort(key=lambda s: s.input_index)
    if max_segments is not None:
        selected = selected[: max(0, int(max_segments))]
    return selected


def _default_plot_options(image_format: str) -> PlotOptions:
    return PlotOptions(
        timeit=False,
        allowYC=True,
        allowMismatch=True,
        annotate=False,
        contact_geometry="horizontal",
        turn_prob_by_dist=False,
        outside_circle_radii=False,
        cTurnAnlyz=False,
        rTurnAnlyz=False,
        wall=False,
        boundary=False,
        agarose=False,
        turn=False,
        chooseOrientations=False,
        jaabaOut=False,
        matFile=False,
        play=False,
        circle=False,
        plotTrx=False,
        plotThm=False,
        plotThmNorm=False,
        rdp=False,
        showTrackIssues=False,
        showRewardMismatch=False,
        showPlots=False,
        move=False,
        ol=False,
        green=False,
        hm=False,
        plotAll=False,
        skip=0,
        skipPI=False,
        minVis=0,
        numBuckets=None,
        piBucketLenMin=None,
        syncBucketLenMin=10,
        postBucketLenMin=3,
        rpiPostBucketLenMin=3,
        piTh=10,
        adbTh=5,
        numRewardsCompare=100,
        radiusMult=1.3,
        radiusMultCC=None,
        radiusCC=None,
        delayCheckMult=None,
        pre_explor_grid_sz=1.0,
        imageFormat=image_format,
        event_chain_reward_circle_color="grey",
        wall_cache=True,
        wall_cache_dir=None,
        wall_cache_refresh=False,
        wall_orientation="all",
        wall_debug=False,
        bnd_ct_plots=False,
        bnd_ct_plot_mode=None,
        bnd_ct_plot_start_fm=None,
    )


def _candidate_video_paths(row: dict[str, str], video_col: str | None) -> list[str]:
    candidates: list[str] = []
    if video_col and row.get(video_col):
        candidates.append(row[video_col])
    for key in ("video_path", "avi_path", "data_path", "trx_path"):
        value = row.get(key)
        if not value:
            continue
        candidates.append(value)
        root, ext = os.path.splitext(value)
        if ext.lower() in {".data", ".trx"}:
            candidates.append(root + ".avi")
    out: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        path = os.path.expanduser(str(path).strip())
        if path and path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _resolve_video_path(row: dict[str, str], video_col: str | None) -> str:
    candidates = _candidate_video_paths(row, video_col)
    for path in candidates:
        if path.lower().endswith(".avi") and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "could not resolve source .avi from row paths; tried: "
        + ", ".join(candidates)
    )


def _prefill_file_cache(video_path: str, row: dict[str, str]) -> None:
    data_path = str(row.get("data_path") or "").strip()
    trx_path = str(row.get("trx_path") or "").strip()
    if not data_path or not trx_path:
        return
    if not (os.path.exists(data_path) and os.path.exists(trx_path)):
        return
    VideoAnalysis.fileCache[video_path] = [
        util.unpickle(data_path),
        util.unpickle(trx_path),
    ]


def _load_plot_context(
    row: dict[str, str],
    *,
    fly_idx: int,
    image_format: str,
    video_col: str | None,
) -> tuple[VideoAnalysis, EventChainPlotter]:
    video_path = _resolve_video_path(row, video_col)
    _prefill_file_cache(video_path, row)

    opts = _default_plot_options(image_format)
    va = VideoAnalysis.__new__(VideoAnalysis)
    va.opts = opts
    va.gidx = 0
    va.f = fly_idx
    va._skipped = True
    va._loadData(video_path)
    if getattr(va, "isyc", False):
        raise RuntimeError(f"fly {fly_idx} is a yoked-control-only selection")
    va.flies = (0,) if va.noyc else (0, 1)
    va.setOptionDefaults()
    va._initTrx()
    va._initSlots()
    va._readNoteFile(video_path)
    va._skipped = bool(va.trx[0].bad())
    if va._skipped:
        raise RuntimeError(f"trajectory for fly {fly_idx} is marked bad")
    return va, EventChainPlotter(trj=va.trx[0], va=va, image_format=image_format)


def _resolve_training_idx(va: VideoAnalysis, raw_idx: int, reward_frame: int) -> int:
    candidates = []
    for idx in (raw_idx, raw_idx - 1):
        if idx not in candidates and 0 <= idx < len(va.trns):
            candidates.append(idx)
    for idx in candidates:
        trn = va.trns[idx]
        if int(trn.start) <= int(reward_frame) < int(trn.stop):
            return idx
    if candidates:
        return candidates[0]
    raise IndexError(
        f"training_idx={raw_idx} is out of range for {len(va.trns)} training(s)"
    )


def _metadata_label(row: dict[str, str]) -> str:
    parts = [
        f"actual={row.get('prediction_actual_label', 'na')}",
        f"pred={row.get('prediction_predicted_label', 'na')}",
        f"prob={_format_float(row.get('prediction_probability'))}",
        f"margin={_format_float(row.get('prediction_decision_margin'))}",
        f"genotype={row.get('genotype', 'na')}",
        f"cohort={row.get('cohort', 'na')}",
        f"n_segments={row.get('prediction_n_segments', 'na')}",
        f"evidence={row.get('prediction_evidence_bin', 'na')}",
    ]
    return " | " + " | ".join(parts)


def _fly_dir_name(seg: SegmentRow) -> str:
    row = seg.row
    bits = [
        "fly",
        _sanitize(row.get("fly_id") or seg.fly_idx),
        "sample",
        _sanitize(row.get("sample_key") or "na", max_len=48),
        "actual",
        _sanitize(row.get("prediction_actual_label")),
        "pred",
        _sanitize(row.get("prediction_predicted_label")),
        "p",
        _sanitize(_format_float(row.get("prediction_probability"))),
    ]
    return "_".join(bits)


def _segment_file_name(
    seg: SegmentRow, image_format: str, *, training_idx: int | None = None
) -> str:
    if training_idx is None:
        training_idx = seg.training_idx
    row = seg.row
    stem = "__".join(
        [
            f"seg{_sanitize(row.get('segment_id') or seg.input_index)}",
            f"trn{training_idx + 1}",
            f"rw{seg.anchor_reward_frame}-{seg.end_reward_frame}",
            f"actual-{_sanitize(row.get('prediction_actual_label'))}",
            f"pred-{_sanitize(row.get('prediction_predicted_label'))}",
            f"p{_sanitize(_format_float(row.get('prediction_probability')))}",
            f"margin{_sanitize(_format_float(row.get('prediction_decision_margin')))}",
        ]
    )
    return f"{stem}.{image_format.lstrip('.')}"


def main() -> None:
    args = parse_args()
    rows = _read_segments(args.segments_csv)
    segments, skip_counts = _filter_rows(
        rows, include_training_end=bool(args.include_training_end)
    )

    grouped: dict[tuple[str, str], list[SegmentRow]] = defaultdict(list)
    for seg in segments:
        grouped[seg.fly_key].append(seg)

    fly_groups = sorted(
        grouped.items(), key=lambda item: min(seg.input_index for seg in item[1])
    )
    if args.max_flies is not None:
        fly_groups = fly_groups[: max(0, int(args.max_flies))]

    rng = random.Random(args.random_seed)
    selected_by_fly = [
        (
            key,
            _select_segments(
                segs,
                selection=args.selection,
                max_segments=args.max_segments_per_fly,
                rng=rng,
            ),
        )
        for key, segs in fly_groups
    ]

    if args.dry_run:
        print(f"[prediction_segments] rows read: {len(rows)}")
        print(f"[prediction_segments] valid plottable rows: {len(segments)}")
        print(f"[prediction_segments] fly groups selected: {len(selected_by_fly)}")
        for _, segs in selected_by_fly:
            if not segs:
                continue
            first = segs[0]
            print(
                "[prediction_segments] "
                f"{_fly_dir_name(first)}: selected {len(segs)} / "
                f"{len(grouped[first.fly_key])} segment(s)"
            )
        print_summary(len(grouped), len(rows), 0, skip_counts)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    context_cache: dict[tuple[str, str, int], tuple[VideoAnalysis, EventChainPlotter]] = {}
    plotted = 0
    failed = 0
    image_format = args.image_format.lstrip(".")

    for _, segs in selected_by_fly:
        if not segs:
            continue
        fly_dir = os.path.join(args.output_dir, _fly_dir_name(segs[0]))
        os.makedirs(fly_dir, exist_ok=True)
        for seg in segs:
            row = seg.row
            cache_key = (
                str(row.get("data_path") or ""),
                str(row.get("trx_path") or ""),
                int(seg.fly_idx),
            )
            try:
                if cache_key not in context_cache:
                    context_cache[cache_key] = _load_plot_context(
                        row,
                        fly_idx=seg.fly_idx,
                        image_format=image_format,
                        video_col=args.video_path_column,
                    )
                va, plotter = context_cache[cache_key]
                trn_index = _resolve_training_idx(
                    va, seg.training_idx, seg.anchor_reward_frame
                )
                out_path = os.path.join(
                    fly_dir,
                    _segment_file_name(seg, image_format, training_idx=trn_index),
                )
                annotation = (
                    f"segment_id = {row.get('segment_id', 'na')}\n"
                    f"frames = {seg.anchor_reward_frame}->{seg.end_reward_frame}\n"
                    f"duration = {row.get('duration_frames', 'na')} frames\n"
                    f"qc_flags = {row.get('qc_flags', '') or 'none'}"
                )
                plotter.plot_between_reward_interval(
                    trn_index=trn_index,
                    start_reward=seg.anchor_reward_frame,
                    end_reward=seg.end_reward_frame,
                    pad=max(0, int(args.pad)),
                    zoom=bool(args.zoom),
                    zoom_radius_mm=args.zoom_radius_mm,
                    image_format=image_format,
                    role_idx=0,
                    out_path=out_path,
                    title_suffix=_metadata_label(row),
                    annotation_text=annotation,
                )
                plt.close("all")
                plotted += 1
            except Exception as exc:
                failed += 1
                print(
                    "[prediction_segments] WARNING: failed to plot "
                    f"row {seg.input_index} segment_id={row.get('segment_id', 'na')}: "
                    f"{type(exc).__name__}: {exc}"
                )

    print_summary(len(grouped), len(rows), plotted, skip_counts)
    if failed:
        print(f"[prediction_segments] failed while plotting: {failed}")
    print(f"[prediction_segments] output dir: {args.output_dir}")


def print_summary(
    flies_considered: int,
    rows_read: int,
    rows_plotted: int,
    skip_counts: Counter,
) -> None:
    print("[prediction_segments] summary")
    print(f"  flies considered: {flies_considered}")
    print(f"  segment rows read: {rows_read}")
    print(f"  segment rows plotted: {rows_plotted}")
    print(
        "  skipped terminal training-end segments: "
        f"{skip_counts.get('terminal_training_end', 0)}"
    )
    print(
        "  skipped missing end_reward_frame: "
        f"{skip_counts.get('missing_end_reward_frame', 0)}"
    )
    other = sum(
        count
        for key, count in skip_counts.items()
        if key not in {"terminal_training_end", "missing_end_reward_frame"}
    )
    if other:
        print(f"  skipped other invalid rows: {other}")


if __name__ == "__main__":
    main()
