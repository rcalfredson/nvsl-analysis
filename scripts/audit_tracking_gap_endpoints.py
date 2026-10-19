#!/usr/bin/env python3
"""Pilot raw tracking-gap endpoints using production large2 geometry.

No interpolation, reward counting, or eligibility filtering.
The three pilot cases are selected based on the flies in the flat
large chamber with defined SLI that have highest lost-frame percentages.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PILOT = [
    ("/media/Synology4/Yang Chen/2025-07-04/c52__2025-07-04__13-09-27.avi", 1, "yok"),
    (
        "/media/Synology4/Yang Chen/2025-07-20/night/c32__2025-07-20__21-08-49.avi",
        1,
        "exp",
    ),
    ("/media/Synology4/Yang Chen/2025-07-27/c52__2025-07-27__12-52-40.avi", 1, "exp"),
]


# 1. Enumerate runs in the original x-NaN mask, matching Trajectory._interpolate.
def missing_runs(x):
    missing = np.isnan(np.asarray(x, dtype=float))
    changes = np.diff(np.r_[False, missing, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def geometry_at_frame(geometries, frame):
    if frame is None:
        return None
    for geometry in geometries:
        if geometry["active_start_frame"] <= frame < geometry["active_stop_frame"]:
            return geometry
    return None


# 2. Measure an observed endpoint. No coordinates are manufactured for missing data.
def endpoint_fields(frame, x, y, floor, px_per_mm, geometries, border_px):
    fields = dict(
        frame=frame,
        x_px=None,
        y_px=None,
        status="no_observed_endpoint",
        nearest_wall=None,
        nearest_wall_signed_mm=None,
        outside_floor=None,
        training=None,
        phase=None,
        reward_boundary_signed_mm=None,
        reward_outer_border_signed_mm=None,
        control_boundary_signed_mm=None,
        control_outer_border_signed_mm=None,
    )
    if frame is None:
        return fields
    fields.update(x_px=float(x[frame]), y_px=float(y[frame]))
    if not np.isfinite([x[frame], y[frame]]).all():
        fields["status"] = "invalid_observed_xy"
        return fields
    if px_per_mm <= 0 or not np.isfinite(px_per_mm):
        raise ValueError("Pixel calibration must be finite and positive")
    (left, top), (right, bottom) = floor
    if left >= right or top >= bottom:
        raise ValueError("Invalid floor rectangle")
    distances = dict(
        left=x[frame] - left,
        right=right - x[frame],
        top=y[frame] - top,
        bottom=bottom - y[frame],
    )
    wall = min(distances, key=distances.get)
    fields.update(
        status="observed",
        nearest_wall=wall,
        nearest_wall_signed_mm=float(distances[wall] / px_per_mm),
        outside_floor=bool(distances[wall] < 0),
    )
    geometry = geometry_at_frame(geometries, frame)
    if geometry is None:
        fields["phase"] = "outside_circle_analysis_ranges"
        return fields
    fields["training"] = geometry["training"]
    fields["phase"] = (
        "pre"
        if frame < geometry["training_start_frame"]
        else "training" if frame < geometry["training_stop_frame"] else "post"
    )
    for name in ("reward", "control"):
        cx, cy, radius = geometry[name]
        signed = float(np.hypot(x[frame] - cx, y[frame] - cy) - radius)
        fields[name + "_boundary_signed_mm"] = signed / px_per_mm
        fields[name + "_outer_border_signed_mm"] = (signed - border_px) / px_per_mm
    return fields


def gap_rows(context):
    x, y = context["x"], context["y"]
    if len(x) != len(y) or not len(x):
        raise ValueError("Raw x/y arrays must have the same nonzero length")
    rows = []
    for gap_id, (start, stop) in enumerate(missing_runs(x)):
        start, stop = int(start), int(stop)
        before = start - 1 if start > 0 else None
        after = stop if stop < len(x) else None
        row = dict(
            video_path=context["video_path"],
            player_pair_id=context["pair"],
            role=context["role"],
            physical_tracking_id=context["physical_id"],
            gap_id=gap_id,
            gap_start_frame=start,
            gap_stop_frame_exclusive=stop,
            last_missing_frame=stop - 1,
            lost_frames=stop - start,
            missing_duration_s=(stop - start) / context["fps"],
            gap_start_time_s=start / context["fps"],
            recovery_time_s=stop / context["fps"] if after is not None else None,
            initial_gap=before is None,
            trailing_gap=after is None,
            counted_in_analyzer_loss_fraction=before is not None,
            observation_bracket_s=(
                (after - before) / context["fps"]
                if before is not None and after is not None
                else None
            ),
        )
        for prefix, frame in (("before", before), ("after", after)):
            measured = endpoint_fields(
                frame,
                x,
                y,
                context["floor"],
                context["px_per_mm"],
                context["geometries"],
                context["border_px"],
            )
            row.update({prefix + "_" + key: value for key, value in measured.items()})
        rows.append(row)
    return rows


# 3. Reuse the production metadata path, without invoking the analysis constructor.
def load_context(video, pair, role):
    from src.analysis.agarose_preinvestigation import load_pickle
    from src.analysis.video_analysis import VideoAnalysis
    from src.utils.common import CT
    from src.utils.constants import BORDER_WIDTH

    video = Path(video).resolve()
    if pair not in (0, 1) or role not in ("exp", "yok"):
        raise ValueError("The pilot accepts large2 pairs 0/1 and exp/yok roles")
    data = load_pickle(video.with_suffix(".data"))
    raw = load_pickle(video.with_suffix(".trx"))
    if len(raw["x"]) != 4:
        raise ValueError("This pilot supports four-trajectory large2 recordings only")
    va = VideoAnalysis.__new__(VideoAnalysis)
    va.f = pair
    va.opts = SimpleNamespace(
        skipFT=False,
        move=False,
        radiusCC=15.0,
        radiusMultCC=None,
        annotate=False,
        rdp=False,
        allowMismatch=True,
    )
    va.fileCache = {str(video): [data, raw]}
    log = io.StringIO()
    try:
        with redirect_stdout(log):
            va._loadData(str(video))
        if va.ct is not CT.large2 or va.nef != 2:
            raise ValueError("Expected production large2 geometry with two fly pairs")
        if not np.isfinite(va.fps) or va.fps <= 0:
            raise ValueError("Invalid video frame rate")
        physical = (
            pair if role == "exp" else (va.ef2yc[pair] if va.ef2yc else pair + va.nef)
        )
        floor = tuple(
            tuple(float(v) for v in corner) for corner in va.ct.floor(va.xf, f=physical)
        )
        scale = float(va.ct.pxPerMmFloor() * va.xf.fctr)
        geometries = []
        for training in va.trns:
            circles = training.circles(0 if role == "exp" else 1)
            if len(circles) != 2:
                raise ValueError(
                    "Expected one reward circle and one concentric control circle"
                )
            geometries.append(
                dict(
                    training=int(training.n),
                    active_start_frame=int(
                        va.startPre if training.n == 1 else training.start
                    ),
                    active_stop_frame=int(training.postStop),
                    training_start_frame=int(training.start),
                    training_stop_frame=int(training.stop),
                    reward=tuple(float(v) for v in circles[0]),
                    control=tuple(float(v) for v in circles[1]),
                )
            )
        context = dict(
            video_path=str(video),
            pair=pair,
            role=role,
            physical_id=int(physical),
            fps=float(va.fps),
            px_per_mm=scale,
            floor=floor,
            border_px=float(BORDER_WIDTH),
            geometries=geometries,
            x=np.asarray(raw["x"][physical], dtype=float),
            y=np.asarray(raw["y"][physical], dtype=float),
            loader_log=log.getvalue(),
            first_frame=va.frame.copy(),
        )
        if len(context["x"]) != va.nf:
            raise ValueError("Trajectory length differs from production frame count")
        return context
    finally:
        if hasattr(va, "cap"):
            va.cap.release()


# 5. Export an audit table and a few endpoints to check manually in the player.
def geometry_rows(context):
    rows = []
    (left, top), (right, bottom) = context["floor"]
    for geometry in context["geometries"]:
        for name in ("reward", "control"):
            cx, cy, radius = geometry[name]
            clearance = min(cx - left, right - cx, cy - top, bottom - cy) - radius
            rows.append(
                dict(
                    video_path=context["video_path"],
                    player_pair_id=context["pair"],
                    role=context["role"],
                    physical_tracking_id=context["physical_id"],
                    training=geometry["training"],
                    circle=name,
                    active_start_frame=geometry["active_start_frame"],
                    active_stop_frame=geometry["active_stop_frame"],
                    training_start_frame=geometry["training_start_frame"],
                    training_stop_frame=geometry["training_stop_frame"],
                    center_x_px=cx,
                    center_y_px=cy,
                    radius_px=radius,
                    border_px=context["border_px"],
                    px_per_mm=context["px_per_mm"],
                    fps=context["fps"],
                    floor_left_px=left,
                    floor_top_px=top,
                    floor_right_px=right,
                    floor_bottom_px=bottom,
                    circle_wall_clearance_mm=clearance / context["px_per_mm"],
                    outer_border_wall_clearance_mm=(clearance - context["border_px"])
                    / context["px_per_mm"],
                )
            )
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to export to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_geometry(contexts, rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, axes = plt.subplots(
        1, len(contexts), figsize=(6 * len(contexts), 6), squeeze=False
    )
    for ax, context in zip(axes[0], contexts):
        ax.imshow(context["first_frame"][..., ::-1])
        (left, top), (right, bottom) = context["floor"]
        ax.add_patch(
            Rectangle(
                (left, top), right - left, bottom - top, fill=False, ec="cyan", lw=1.5
            )
        )
        geometry = next(g for g in context["geometries"] if g["training"] == 2)
        for name, color in (("reward", "lime"), ("control", "yellow")):
            cx, cy, radius = geometry[name]
            ax.add_patch(
                Circle((cx, cy), radius, fill=False, ec=color, lw=1.5, label=name)
            )
        for prefix, color, marker in (
            ("before", "#e95e8a", "o"),
            ("after", "#41b6ff", "+"),
        ):
            selected = [
                r
                for r in rows
                if r["video_path"] == context["video_path"]
                and r["role"] == context["role"]
                and r["player_pair_id"] == context["pair"]
                and r[prefix + "_training"] == 2
                and r[prefix + "_phase"] == "training"
                and r[prefix + "_status"] == "observed"
            ]
            ax.scatter(
                [r[prefix + "_x_px"] for r in selected],
                [r[prefix + "_y_px"] for r in selected],
                s=14,
                alpha=0.55,
                color=color,
                marker=marker,
                label=f"{prefix} T2 gap ({len(selected)})",
            )
        ax.set(
            xlim=(left - 25, right + 25),
            ylim=(bottom + 25, top - 25),
            xlabel="Frame x (px)",
            ylabel="Frame y (px)",
            title=Path(context["video_path"]).name
            + f"\nfly {context['pair']}, {context['role']}",
        )
        ax.legend(fontsize=7)
    fig.suptitle(
        "Geometry sanity check: T2 gap endpoints over recording frame 0 (background only)"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument(
        "--pilot",
        action="store_true",
        help="Run the three manually reviewed trajectories",
    )
    choice.add_argument("--video", type=Path)
    parser.add_argument("--pair", type=int, choices=(0, 1))
    parser.add_argument("--role", choices=("exp", "yok"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp/frame_loss_spatial_audit/pilot_results",
    )
    args = parser.parse_args()
    if args.pilot and (args.pair is not None or args.role is not None):
        parser.error("--pair/--role apply only with --video")
    if not args.pilot and (args.pair is None or args.role is None):
        parser.error("--video requires --pair and --role")
    cases = PILOT if args.pilot else [(args.video, args.pair, args.role)]
    contexts = [load_context(*case) for case in cases]
    rows = [r for context in contexts for r in gap_rows(context)]
    geometries = [r for context in contexts for r in geometry_rows(context)]
    checks, recordings = [], []
    for context in contexts:
        selected = [
            r
            for r in rows
            if r["video_path"] == context["video_path"] and r["role"] == context["role"]
        ]
        internal = [
            r for r in selected if not r["initial_gap"] and not r["trailing_gap"]
        ]
        chosen = {}

        def choose(candidates, reason):
            for row in candidates:
                if row["gap_id"] not in chosen:
                    chosen[row["gap_id"]] = dict(
                        row, check_reason=reason, review_note=""
                    )
                else:
                    chosen[row["gap_id"]]["check_reason"] += "; " + reason

        choose(
            sorted(internal, key=lambda r: r["lost_frames"], reverse=True)[:3],
            "longest internal gaps",
        )
        t2 = [
            r
            for r in internal
            if any(
                r[p + "_training"] == 2 and r[p + "_phase"] == "training"
                for p in ("before", "after")
            )
        ]
        choose(
            sorted(t2, key=lambda r: r["lost_frames"], reverse=True)[:2],
            "longest gaps with T2 endpoint",
        )

        def boundary_distance(row):
            values = [
                abs(row[p + "_" + c + "_boundary_signed_mm"])
                for p in ("before", "after")
                for c in ("reward", "control")
                if row[p + "_training"] == 2
                and row[p + "_phase"] == "training"
                and row[p + "_" + c + "_boundary_signed_mm"] is not None
            ]
            return min(values, default=float("inf"))

        choose(
            sorted(t2, key=boundary_distance)[:2],
            "T2 endpoints closest to a circle boundary",
        )
        choose(
            [r for r in selected if r["initial_gap"] or r["trailing_gap"]],
            "recording edge case",
        )
        checks += list(chosen.values())
        initial = sum(r["lost_frames"] for r in selected if r["initial_gap"])
        lost = sum(
            r["lost_frames"] for r in selected if r["counted_in_analyzer_loss_fraction"]
        )
        denominator = len(context["x"]) - initial
        recordings.append(
            dict(
                video_path=context["video_path"],
                player_pair_id=context["pair"],
                role=context["role"],
                physical_tracking_id=context["physical_id"],
                fps=context["fps"],
                total_frames=len(context["x"]),
                gap_count=len(selected),
                initial_missing_frames=initial,
                counted_lost_frames=lost,
                loss_pct=100 * lost / denominator if denominator else None,
                before_outside_floor=sum(
                    r["before_outside_floor"] is True for r in selected
                ),
                after_outside_floor=sum(
                    r["after_outside_floor"] is True for r in selected
                ),
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "gaps.csv", rows)
    write_csv(args.output_dir / "geometry.csv", geometries)
    write_csv(args.output_dir / "manual_checks.csv", checks)
    write_csv(args.output_dir / "recordings.csv", recordings)
    plot_geometry(contexts, rows, args.output_dir / "geometry_check.png")
    manifest = dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frame_convention="zero-based; missing runs [start, stop); before=start-1; after=stop",
        timing="nominal seconds from frame/fps; missing duration differs from observation bracket by one frame",
        options=dict(
            chamber="large2", control_radius_mm=15, all_recorded_trainings=True
        ),
        limits="Gap endpoints do not locate the fly during missing frames. No SLI or circle-entry reliability inference is made.",
        recordings=[],
    )
    for context in contexts:
        record = {
            k: v for k, v in context.items() if k not in ("x", "y", "first_frame")
        }
        record["source_files"] = []
        for suffix in (".avi", ".data", ".trx"):
            path = Path(context["video_path"]).with_suffix(suffix)
            stat = path.stat()
            record["source_files"].append(
                dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            )
        manifest["recordings"].append(record)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Exported {len(rows)} gaps, {len(geometries)} circle geometries, and {len(checks)} manual checks to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
