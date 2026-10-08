#!/usr/bin/env python3
"""Inspect walking in the Figure 1p/1q open-loop videos, without full analysis.

Reads video lists from manuscript_figure_panels.ipynb and tracking sidecars.
Uses the existing Trajectory interpolation and >2 mm/s walking definition.
Percentages use all frames in [startTrain, startPost), including interpolated
frames; raw tracking loss is reported separately. No preference is calculated.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import glob
import io
import json
from pathlib import Path
import re
import shlex
import sys
from types import SimpleNamespace

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.agarose_preinvestigation import load_pickle, write_csv
from src.analysis.trajectory import Trajectory
from src.exporting.cross_experiment_correlation import recording_identity, subject_key
from src.utils.common import CT


def notebook_video_patterns(notebook):
    """Extract only the two open-loop -v arguments, without executing cells."""
    cells = {c.get("id"): "".join(c["source"]) for c in json.loads(Path(notebook).read_text())["cells"]}
    # Recipe cells contain adjacent Python string literals. AST parsing avoids
    # executing notebook setup or any analysis commands.
    import ast
    result = {}
    for panel, cell_id in (("1p", "figure-1p-export"), ("1q", "figure-1q-exports")):
        candidates = []
        for node in ast.walk(ast.parse(cells[cell_id])):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                candidates.append(node.value)
            elif isinstance(node, ast.JoinedStr):
                candidates.append("".join(n.value for n in node.values if isinstance(n, ast.Constant)))
        commands = list({s for s in candidates if 'python analyze.py -v ' in s and '/openloop/' in s})
        if len(commands) != 1:
            raise ValueError(f"Expected one open-loop command for {panel}")
        args = shlex.split(commands[0])
        if args[args.index('-f') + 1] != '0-19':
            raise ValueError(f"Unexpected fly selection for {panel}")
        result[panel] = args[args.index('-v') + 1].split(',')
    return result


def period_masks(frame_nums, nframes):
    """Use half-open training windows and the same LED command mask as posPrefOL."""
    session = np.zeros(nframes, bool)
    starts, stops = frame_nums['startTrain'], frame_nums['startPost']
    if not starts or len(starts) != len(stops):
        raise ValueError("Missing or inconsistent training windows")
    for start, stop in zip(starts, stops):
        if not 0 <= start < stop <= nframes:
            raise ValueError(f"Invalid training window {start}:{stop}")
        session[start:stop] = True
    events = np.zeros(nframes, int)
    def during_session(values):
        indices = np.asarray(values, dtype=int)
        if np.any((indices < 0) | (indices >= nframes)):
            raise ValueError("LED command outside trajectory")
        return indices[session[indices]]
    for key, values in frame_nums.items():
        if re.fullmatch(r'v[1-9]\d*(\.\d+)?', key):
            np.add.at(events, during_session(values), 1)
    np.add.at(events, during_session(frame_nums['v0']), -1)
    # Reset between trainings, matching posPrefOL's per-training event mask.
    for start, stop in zip(starts, stops):
        if stop < nframes:
            events[stop] = -int(events[start:stop].sum())
    led = np.cumsum(events)
    if np.any((led < 0) | (led > 1)):
        raise ValueError("Invalid LED on/off command sequence")
    return {'session': session, 'led_on': session & (led == 1), 'led_off': session & (led == 0)}


def walking_trajectory(x, y, fps, chamber):
    # Only initialize what the existing trajectory preprocessing needs; omit
    # rewards, geometry, images, and all other full-analysis work.
    trj = Trajectory.__new__(Trajectory)
    trj.x, trj.y = np.array(x, dtype=float), np.array(y, dtype=float)
    trj.theta = None
    trj.va = SimpleNamespace(fps=fps, ct=chamber, openLoop=True, on=np.array([], int))
    with redirect_stdout(io.StringIO()):
        if trj._isEmpty():
            return None
        trj._setLostFrameThreshold()
        trj._interpolate()
        trj._calcDistances()
        trj._calcSpeeds()
        trj._setWalking()
    return trj


def inspect_video(panel, video):
    proto = load_pickle(video.with_suffix('.data'))['protocol']
    if proto.get('pt') != 'openLoop' or proto.get('alt', True):
        raise ValueError(f"Expected on/off open-loop protocol: {video}")
    raw = load_pickle(video.with_suffix('.trx'))
    if len(raw['x']) != 20:
        raise ValueError(f"Expected 20 HTL flies: {video}")
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Cannot read frame rate: {video}")
    date, camera = recording_identity(video)
    rows = []
    for fly in range(20):
        frames = proto['frameNums'][fly]
        if not frames:
            raise ValueError(f"Missing protocol for {video}, fly {fly}")
        masks = period_masks(frames, len(raw['x'][fly]))
        trj = walking_trajectory(raw['x'][fly], raw['y'][fly], fps, CT.htl)
        row = dict(panel=panel, recording_date=date, camera_id=camera,
                   fly_id=fly, subject_key=subject_key(date, camera, fly),
                   open_loop_video=str(video), fps=fps, walking_threshold_mm_s=2,
                   status='ok' if trj is not None else 'no_trajectory')
        missing = ~np.isfinite(raw['x'][fly]) | ~np.isfinite(raw['y'][fly])
        for period, mask in masks.items():
            n = int(mask.sum())
            nw = int(np.count_nonzero(trj.walking & mask)) if trj is not None else None
            row[f'{period}_frames'] = n
            row[f'{period}_walking_frames'] = nw
            row[f'{period}_walking_pct'] = 100 * nw / n if nw is not None and n else float('nan')
            row[f'{period}_raw_missing_pct'] = 100 * np.count_nonzero(missing & mask) / n if n else float('nan')
        rows.append(row)
    return rows


def run(notebook, out_dir):
    rows = []
    for panel, patterns in notebook_video_patterns(notebook).items():
        videos = []
        for pattern in patterns:
            matches = sorted(Path(p) for p in glob.glob(pattern) if p.endswith('.avi'))
            if not matches:
                raise FileNotFoundError(f"No videos for {pattern}")
            videos.extend(matches)
        for video in dict.fromkeys(videos):
            print(f"{panel}: {video.name}", flush=True)
            rows.extend(inspect_video(panel, video))
    rows.sort(key=lambda r: (r['panel'], not np.isfinite(r['session_walking_pct']),
                             r['session_walking_pct'] if np.isfinite(r['session_walking_pct']) else 0))
    out_dir = Path(out_dir)
    write_csv(out_dir / 'walking_per_fly.csv', rows)
    summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, panel in zip(axes, ('1p', '1q')):
        selected = [r for r in rows if r['panel'] == panel]
        valid = [r for r in selected if np.isfinite(r['session_walking_pct'])]
        for period, label, color in [('session', 'Full open-loop session', 'black'),
                                     ('led_on', 'LED on', '#e69f00'), ('led_off', 'LED off', '#0072b2')]:
            vals = np.array([r[f'{period}_walking_pct'] for r in selected])
            vals = vals[np.isfinite(vals)]
            summaries.append(dict(panel=panel, period=period, n_selected=len(selected), n_valid=len(vals),
                                  mean_pct=float(np.mean(vals)), median_pct=float(np.median(vals)),
                                  min_pct=float(np.min(vals)), max_pct=float(np.max(vals)),
                                  n_below_10_pct=int(np.count_nonzero(vals < 10))))
            ax.scatter(np.arange(1, len(valid)+1), [r[f'{period}_walking_pct'] for r in valid],
                       s=14, alpha=.75, color=color, label=label)
        ax.set(title=f"Figure {panel}: {len(valid)}/{len(selected)} flies with trajectories",
               xlabel='Fly rank by session walking percentage', ylim=(0, 100))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Walking (% of period frames)')
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / 'walking_percentages.png', dpi=180)
    fig.savefig(out_dir / 'walking_percentages.pdf')
    plt.close(fig)
    write_csv(out_dir / 'walking_summary.csv', summaries)
    for row in summaries:
        print(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=ROOT / 'notebooks/manuscript_figure_panels.ipynb')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'exports/openloop_walking')
    args = parser.parse_args()
    run(args.notebook, args.out_dir)
