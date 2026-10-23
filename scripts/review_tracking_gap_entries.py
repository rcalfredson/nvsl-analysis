#!/usr/bin/env python3
"""Build, play, and summarize a manual circle-entry review of tracking gaps."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_tracking_gap_endpoints import PILOT, gap_rows, load_context
from src.utils.tracking_gap_review_io import load_episodes, read_csv, verify_sources, write_csv
from src.analysis.tracking_gap_review import (
    add_observed_controls, compare_annotations, exclude_previous_windows, finalize_episodes, gap_candidates,
    merge_review_windows, reconstructed_entries, sample_candidates,
)


def context_key(row):
    return row["video_path"], int(row["fly"]), row["role"]


def write_worksheet(bundle, episodes, annotations, output):
    from src.exporting.tracking_gap_review_worksheet import render_worksheet
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_worksheet(episodes, annotations, bundle=bundle,
                                      python=sys.executable, script=Path(__file__).resolve()))


def worksheet(args):
    bundle = args.bundle.resolve()
    output = args.output or bundle / "worksheet.html"
    if output.exists() and not args.overwrite:
        raise ValueError("Worksheet already exists; choose --output or explicitly use --overwrite")
    write_worksheet(bundle, load_episodes(bundle), read_csv(bundle / "annotations.csv"), output)
    print(f"Wrote notes worksheet to {output}; source annotations.csv was not changed.")


def build(args):
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new/empty output directory; existing annotations will not be overwritten")
    case_metadata = {}
    if args.cases_csv:
        selection = read_csv(args.cases_csv)
        cases = [(row["video_path"], int(row["fly"]), row["role"]) for row in selection]
        if not cases or len(set(cases)) != len(cases):
            raise ValueError("Cases CSV must contain distinct video_path, fly, role selections")
        case_metadata = {(str(Path(row["video_path"]).resolve()), int(row["fly"]), row["role"]):row for row in selection}
    elif args.pilot:
        cases = PILOT
    else:
        if args.pair is None or args.role is None:
            raise ValueError("--video requires --pair and --role")
        cases = [(args.video, args.pair, args.role)]
    contexts = [load_context(*case) for case in cases]
    for context in contexts:
        key = context["video_path"], context["pair"], context["role"]
        context["case_selection_metadata"] = case_metadata.get(key, {})
        context["cohort"] = case_metadata.get(key, {}).get("cohort", "")
    candidates, events = [], {}
    for context in contexts:
        candidates += gap_candidates(context, gap_rows(context), padding_seconds=args.padding_seconds,
                                     near_mm=args.near_mm, long_seconds=args.long_seconds)
        key = context["video_path"], context["pair"], context["role"]
        events[key] = reconstructed_entries(context)
    original_candidate_count = len(candidates)
    previous = [row for path in args.exclude_bundle for row in load_episodes(path)]
    candidates = exclude_previous_windows(candidates, previous)
    selected = sample_candidates(candidates, args.per_stratum, args.seed)
    merged = merge_review_windows(selected)
    merged = add_observed_controls(merged, contexts, args.observed_controls, args.seed+1, excluded=previous)
    episodes = finalize_episodes(merged, contexts, events)
    for i, episode in enumerate(episodes, args.case_offset+1):
        episode["case_number"] = i
    if not episodes:
        raise ValueError("No review intervals available")
    annotations = [dict(review_id=ep["review_id"], review_status="pending", observed_entry_frames="",
                        confidence="", reviewer="", notes="") for ep in episodes]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "candidates.csv", candidates, fields=list(candidates[0]) if candidates else ["video_path", "gap_id"])
    write_csv(out / "sampled_candidates.csv", selected, fields=list(selected[0]) if selected else ["video_path", "gap_id"])
    write_csv(out / "episodes.csv", episodes)
    write_csv(out / "annotations.csv", annotations)
    event_rows = [dict(video_path=c["video_path"], fly=c["pair"], role=c["role"], training=training,
                       circle=circle, entry_frame=int(frame))
                  for c in contexts for (training, circle), frames in events[c["video_path"], c["pair"], c["role"]].items()
                  for frame in frames]
    write_csv(out / "calculated_entries.csv", event_rows,
              fields=["video_path", "fly", "role", "training", "circle", "entry_frame"])
    manifest = dict(
        seed=args.seed, per_stratum=args.per_stratum, observed_controls=args.observed_controls,
        padding_seconds=args.padding_seconds, near_boundary_mm=args.near_mm, long_gap_seconds=args.long_seconds,
        training=2, event_source="offline production-entry-rule reconstruction, not independent ground truth",
        case_offset=args.case_offset, candidate_count_before_exclusion=original_candidate_count,
        cases_csv=str(args.cases_csv.resolve()) if args.cases_csv else None,
        cases_csv_sha256=hashlib.sha256(args.cases_csv.read_bytes()).hexdigest() if args.cases_csv else None,
        excluded_bundles=[dict(path=str(path.resolve()), episodes_sha256=hashlib.sha256((path/"episodes.csv").read_bytes()).hexdigest()) for path in args.exclude_bundle],
        options="large2, rCC=15 mm, 10-minute actual-reward synchronized buckets",
        frame_convention="zero-based, scored windows [start, stop); score every entry in the entire window",
        source_files=[], contexts=[],
        code_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__).resolve(), ROOT/"src/analysis/tracking_gap_review.py",
            ROOT/"scripts/audit_tracking_gap_endpoints.py", ROOT/"src/utils/tracking_gap_review_io.py",
            ROOT/"src/analysis/trajectory.py",
            ROOT/"src/analysis/training.py", ROOT/"src/analysis/video_analysis.py")},
    )
    for context in contexts:
        manifest["contexts"].append({k:v for k,v in context.items() if k not in ("x", "y", "first_frame")})
        for suffix in (".avi", ".data", ".trx"):
            path = Path(context["video_path"]).with_suffix(suffix)
            stat = path.stat()
            manifest["source_files"].append(dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    manifest["episodes_sha256"] = hashlib.sha256((out/"episodes.csv").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    counts = Counter((r["video_path"], r["role"], r["circle"], r["stratum"]) for r in candidates)
    sampled_counts = Counter((r["video_path"], r["role"], r["circle"], r["stratum"]) for r in selected)
    write_csv(out / "sampling.csv", [dict(video_path=k[0], role=k[1], circle=k[2], stratum=k[3],
                                         population=n, sampled=sampled_counts[k]) for k,n in sorted(counts.items())],
              fields=["video_path", "role", "circle", "stratum", "population", "sampled"])
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Circle-entry review</title>',
             '<style>body{font-family:system-ui;margin:2rem;line-height:1.5}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.5rem}pre{white-space:pre-wrap}</style>',
             '<h1>Tracking-gap circle-entry review</h1><p><a href="worksheet.html">Open the editable notes worksheet</a></p>',
             '<p>Edit annotations.csv. Use pending, reviewed, or unscorable. For reviewed intervals, enter a JSON list of absolute entry frame numbers (explicit [] for zero), confidence high/medium/low, reviewer, and notes. Score every entry in the specified half-open interval for the named circle, including the padded context. Do not infer unobserved visits from interpolation.</p>',
             '<p>The sampling is stratified for diagnosis, not a representative estimate of overall error. Review windows that overlap for the same fly/circle are merged. Different circle rows are distinct counting tasks. Near-boundary and long-gap thresholds are triage choices.</p>',
             '<p>Reviewer controls: space play/pause, a/d one frame, b/f ten frames, n/p next/previous case, digits then g jump to an absolute frame, q quit. The viewer does not write annotations. Pink cross = observed target position; missing tracking is labeled and no interpolated position is drawn.</p>',
             '<table><tr><th>Case</th><th>Video / fly / role</th><th>Circle</th><th>Scored frames [start,stop)</th><th>SB</th><th>Source / strata</th><th>Calculated entries</th><th>Review ID / command</th></tr>']
    for ep in episodes:
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), "play", "--bundle", str(out), "--review-id", ep["review_id"]])
        cells = [ep["case_number"], f"{Path(ep['video_path']).name} / fly {ep['fly']} / {ep['role']}", ep["circle"],
                 f"[{ep['review_start_frame']}, {ep['review_stop_frame_exclusive']})",
                 ep["sync_buckets"], ep["source"]+" / "+ep["strata"], ep["calculated_entry_frames"], ep["review_id"]+"\n"+command]
        parts.append('<tr>'+''.join('<td><pre>'+html.escape(str(c))+'</pre></td>' for c in cells)+'</tr>')
    parts.append('</table></html>')
    (out / "index.html").write_text("\n".join(parts))
    write_worksheet(out, episodes, annotations, out / "worksheet.html")
    write_notes_table(out, episodes)
    print(f"Built {len(candidates)} gap/circle candidates; sampled {len(selected)}; exported {len(episodes)} nonoverlapping review tasks to {out}")


def write_notes_table(out, episodes):
    headers = ["Case", "Video filename", "Fly ID", "Role", "Circle", "Scored frames [start, stop)",
               "Interval type (G=gap, C=context-only gap, T=tracked)", "Observed entry frames", "Notes"]
    rows = []
    for ep in episodes:
        code = {"gap_in_scored_interval":"G", "gap_in_context_only":"C", "fully_tracked_comparison":"T"}[ep["interval_type"]]
        rows.append([ep["case_number"], Path(ep["video_path"]).name, ep["fly"], ep["role"], ep["circle"],
                     f"[{ep['review_start_frame']}, {ep['review_stop_frame_exclusive']})", code, "", ""])
    with (out/"review_table.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerows([headers, *rows])
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Manual audit notes</title>',
             '<style>body{font:14px system-ui;margin:1.5rem}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.5rem;text-align:left;vertical-align:top}</style>',
             '<p>G: missing frames in the scored interval. C: gap-related case with the gap outside scored frames. T: fully tracked comparison. Score the named circle only, within [start,stop).</p>',
             '<table border="1" cellpadding="8"><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>']
    for row in rows:
        parts.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>')
    parts.append('</tbody></table></html>')
    (out/"wiki_table.html").write_text('\n'.join(parts))


def render_frame(raw_frame, frame, context, episode, show_calculated=False):
    import cv2
    image = raw_frame.copy()
    geometry = next(g for g in context["geometries"] if g["training"] == int(episode["training"]))
    for circle in ("reward", "control"):
        cx, cy, radius = geometry[circle]
        color = (0, 255, 255) if circle == episode["circle"] else (160, 160, 160)
        cv2.circle(image, (round(cx), round(cy)), round(radius), color, 1)
        cv2.circle(image, (round(cx), round(cy)), round(radius+context["border_px"]), color, 1)
    x, y = context["x"][frame], context["y"][frame]
    observed = bool(np.isfinite([x, y]).all())
    if observed:
        cv2.drawMarker(image, (round(x), round(y)), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
    (left, top), (right, bottom) = context["floor"]
    height, width = image.shape[:2]
    image = image[max(0,int(top)-25):min(height,int(bottom)+26), max(0,int(left)-25):min(width,int(right)+26)]
    image = cv2.copyMakeBorder(image, 112, 10, 10, 10, cv2.BORDER_CONSTANT)
    scored = int(episode["review_start_frame"]) <= frame < int(episode["review_stop_frame_exclusive"])
    entries = json.loads(episode["calculated_entry_frames"])
    lines = [f"fly {context['pair']} {context['role']} | {episode['circle']} | frame {frame}",
             f"score [{episode['review_start_frame']}, {episode['review_stop_frame_exclusive']}) | {'SCORED' if scored else 'CONTEXT ONLY'}",
             "TRACKING OBSERVED" if observed else "TRACKING MISSING - no position inferred",
             "CALCULATED ENTRY" if show_calculated and frame in entries else
             (f"case {episode['case_number']} | " if episode.get("case_number") else "review ")+episode["review_id"]]
    for i,line in enumerate(lines):
        cv2.putText(image, line, (8, 22+25*i), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (255,255,255) if observed or i!=2 else (0,120,255), 1, cv2.LINE_AA)
    return image


def play(args):
    import cv2
    bundle = args.bundle.resolve()
    verify_sources(bundle)
    episodes = load_episodes(bundle)
    ids = [r["review_id"] for r in episodes]
    if args.review_id and args.review_id not in ids:
        raise ValueError("Unknown review ID")
    index = ids.index(args.review_id) if args.review_id else 0
    cache, cap, active = {}, None, None
    try:
        while True:
            ep = episodes[index]
            key = context_key(ep)
            if key not in cache:
                cache[key] = load_context(*key)
            context = cache[key]
            if active != index:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(context["video_path"])
                if not cap.isOpened():
                    raise ValueError("Could not open recording")
                lower = max(0, int(ep["review_start_frame"])-int(context["fps"]*args.context_seconds))
                upper = min(len(context["x"])-1, int(ep["review_stop_frame_exclusive"])+int(context["fps"]*args.context_seconds))
                frame, paused, digits, active = lower, True, "", index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, raw = cap.read()
            if not ok:
                raise ValueError(f"Could not decode frame {frame}")
            cv2.imshow("Circle-entry review", render_frame(raw, frame, context, ep, args.show_calculated))
            keycode = cv2.waitKeyEx(0 if paused else max(1, round(1000/context["fps"])))
            if keycode == ord("q"):
                break
            if keycode == ord(" "):
                paused = not paused
            elif keycode in (ord("n"), ord("p")):
                index = (index+(1 if keycode==ord("n") else -1)) % len(episodes)
            elif keycode in (ord("a"), ord("d"), ord("b"), ord("f"), 2424832, 2555904, 65361, 65363):
                delta = -1 if keycode in (ord("a"),2424832,65361) else 1 if keycode in (ord("d"),2555904,65363) else -10 if keycode==ord("b") else 10
                frame = min(upper, max(lower, frame+delta))
                paused = True
            elif ord("0") <= keycode <= ord("9"):
                digits += chr(keycode)
                paused = True
            elif keycode == ord("g") and digits:
                frame = min(upper, max(lower, int(digits)))
                digits, paused = "", True
            elif not paused:
                frame = min(upper, frame+1)
                if frame == upper:
                    paused = True
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def summarize(args):
    bundle = args.bundle.resolve()
    episodes = load_episodes(bundle)
    annotations = read_csv(args.annotations or bundle/"annotations.csv")
    comparisons = compare_annotations(episodes, annotations, args.tolerance_frames,
                                      require_confidence=not getattr(args, "allow_missing_confidence", False))
    write_csv(bundle / "review_comparison.csv", comparisons)
    groups = {}
    for row in comparisons:
        groups.setdefault((row["video_path"], row["fly"], row["role"], row["circle"], row["source"]), []).append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        reviewed = [r for r in rows if r["review_status"] == "reviewed"]
        summary.append(dict(video_path=key[0], fly=key[1], role=key[2], circle=key[3], source=key[4],
                            intervals=len(rows), reviewed=len(reviewed),
                            unscorable=sum(r["review_status"]=="unscorable" for r in rows),
                            pending=sum(r["review_status"]=="pending" for r in rows),
                            reviewed_frames=sum(r["reviewed_frames"] for r in reviewed),
                            calculated_entries_reviewed=sum(r["calculated_count"] for r in reviewed),
                            observed_entries=sum(r["observed_count"] for r in reviewed) if reviewed else None,
                            net_observed_minus_calculated=sum(r["observed_minus_calculated"] for r in reviewed) if reviewed else None,
                            low_confidence_intervals=sum(r["confidence"]=="low" for r in reviewed),
                            unspecified_confidence_intervals=sum(not r["confidence"] for r in reviewed)))
    write_csv(bundle / "review_summary.csv", summary)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Manual review summary</title>',
             '<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.5rem}</style>',
             '<h1>Manual circle-entry review summary</h1>',
             '<p>Only reviewed intervals contribute observed counts. Blank values mean no scored observations. Pending and unscorable intervals are not error-free intervals. These diagnostic samples do not support a population error estimate or an automatic PI/SLI correction.</p>',
             '<p>Count disagreement and temporal disagreement are separate. An unmatched event can reflect a timing offset rather than an extra or missed visit. Inspect review_comparison.csv and the video before interpreting individual mismatches.</p>',
             '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in summary[0])+'</tr>']
    for row in summary:
        parts.append('<tr>'+''.join('<td>'+html.escape(str(v) if v is not None else '—')+'</td>' for v in row.values())+'</tr>')
    parts.append('</table><p><a href="index.html">Review cases</a> · <a href="review_comparison.csv">Individual comparisons</a></p></html>')
    (bundle / "review_summary.html").write_text('\n'.join(parts))
    print(f"Reviewed {sum(r['review_status']=='reviewed' for r in comparisons)}/{len(comparisons)} intervals. No PI/SLI correction or population error estimate is inferred from partial reviews.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    group = b.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--video", type=Path)
    group.add_argument("--cases-csv", type=Path, help="List of video_path, fly, role; optional cohort and selection metadata")
    b.add_argument("--pair", type=int, choices=(0,1))
    b.add_argument("--role", choices=("exp","yok"))
    b.add_argument("--output-dir", type=Path, default=ROOT/"results/tracking_gap_entry_review/pilot")
    b.add_argument("--per-stratum", type=int, default=2)
    b.add_argument("--observed-controls", type=int, default=1)
    b.add_argument("--seed", type=int, default=20261019)
    b.add_argument("--padding-seconds", type=float, default=1.)
    b.add_argument("--near-mm", type=float, default=1.)
    b.add_argument("--long-seconds", type=float, default=2.)
    b.add_argument("--case-offset", type=int, default=0, help="Continue display case numbers after an earlier batch")
    b.add_argument("--exclude-bundle", type=Path, action="append", default=[], help="Exclude all scored windows in an earlier bundle")
    b.set_defaults(action=build)
    p = sub.add_parser("play")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--review-id")
    p.add_argument("--context-seconds", type=float, default=2.)
    p.add_argument("--show-calculated", action="store_true", help="Show calculated-entry cues; hidden by default during manual scoring")
    p.set_defaults(action=play)
    s = sub.add_parser("summarize")
    s.add_argument("--bundle", type=Path, required=True)
    s.add_argument("--annotations", type=Path)
    s.add_argument("--tolerance-frames", type=int, default=2)
    s.add_argument("--allow-missing-confidence", action="store_true",
                   help="Accept imported legacy reviews with blank confidence; report blanks explicitly")
    s.set_defaults(action=summarize)
    w = sub.add_parser("worksheet", help="Generate an editable HTML notes table for an existing bundle")
    w.add_argument("--bundle", type=Path, required=True)
    w.add_argument("--output", type=Path)
    w.add_argument("--overwrite", action="store_true")
    w.set_defaults(action=worksheet)
    args = parser.parse_args()
    if args.command=="build" and (args.per_stratum<1 or args.observed_controls<0 or
                                  not all(np.isfinite(v) and v>=0 for v in (args.padding_seconds,args.near_mm,args.long_seconds))):
        parser.error("Sampling counts and thresholds must be finite and nonnegative; per-stratum must be positive")
    if args.command=="build" and args.pilot and (args.pair is not None or args.role is not None):
        parser.error("--pair/--role apply only with --video")
    if args.command=="build" and args.cases_csv and (args.pair is not None or args.role is not None):
        parser.error("--pair/--role must be supplied in the cases CSV")
    if args.command=="build" and args.case_offset<0:
        parser.error("Case offset must be nonnegative")
    if args.command=="summarize" and args.tolerance_frames<0:
        parser.error("Tolerance must be nonnegative")
    if args.command=="play" and (not np.isfinite(args.context_seconds) or args.context_seconds<0):
        parser.error("Context seconds must be finite and nonnegative")
    args.action(args)


if __name__ == "__main__":
    main()
