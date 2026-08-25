#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.training import Training
from src.utils.constants import LGC2
from src.plotting.agarose_virtual_control_summary import (
    load_chamber_placement_values,
    plot_agarose_virtual_control_summary,
)
from src.plotting.palettes import ACCENT_BLUE, ACCENT_ORANGE
from src.utils import util
from src.utils.common import CT, Xformer


# Old rt-trx .data files reference __main__.FlyDetector. The plotting command
# needs only the protocol dictionary, but this compatibility shell allows the
# existing project unpickler to load those files without importing analyze.py.
class FlyDetector:
    pass


def _average_video_frames(video_path, frame_index=0, n_frames=30):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open background video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    frames = []
    for _ in range(max(1, int(n_frames))):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(np.asarray(frame, dtype=np.float32))
    cap.release()
    if not frames:
        raise ValueError(f"could not read frames from background video: {video_path}")
    return np.mean(frames, axis=0).clip(0, 255).astype(np.uint8)


def _large_chamber_geometry_from_protocol(
    video_path,
    *,
    frame,
    protocol_index=0,
    training_index_1based=2,
    inner_offset_mm=0.0,
    outer_delta_mm=0.5,
):
    data_path = Path(video_path).with_suffix(".data")
    payload = util.unpickle(str(data_path))
    if not isinstance(payload, dict) or "protocol" not in payload:
        raise ValueError(f"could not read protocol metadata from {data_path}")
    info = payload["protocol"].get("info", [])
    if protocol_index < 0 or protocol_index >= len(info):
        raise IndexError(
            f"protocol index {protocol_index} is out of range for {len(info)} chambers"
        )
    training_idx = int(training_index_1based) - 1
    cpos = info[protocol_index]["cPos"]
    radii = info[protocol_index]["r"]
    reward_cx, reward_cy = (float(v) for v in cpos[training_idx])
    reward_r = float(radii[training_idx])

    # Use the analysis coordinate-transform path itself. In particular, Xformer
    # accounts for source-frame resizing and CT.arenaWells supplies the exact
    # nominal 4 mm well radius and transformed centers used by the metric.
    template_match = payload["protocol"].get("tm")
    if not isinstance(template_match, dict):
        raise ValueError(f"protocol metadata in {data_path} has no template match")
    trx_path = Path(video_path).with_suffix(".trx")
    trx_payload = util.unpickle(str(trx_path))
    if not isinstance(trx_payload, dict) or "x" not in trx_payload:
        raise ValueError(f"could not read trajectory metadata from {trx_path}")
    chamber_type = CT.get(len(trx_payload["x"]), LGC2)
    if chamber_type not in (CT.large, CT.large2):
        raise ValueError(
            f"background video resolves to {chamber_type}, not a large chamber"
        )
    xf = Xformer(template_match, chamber_type, frame, fy=False)
    floor_tl, floor_br = list(chamber_type.floor(xf, f=protocol_index))
    arena_cx, arena_cy = np.mean((floor_tl, floor_br), axis=0)
    nominal_agarose_radius_px, physical_centers = chamber_type.arenaWells(
        xf, protocol_index
    )
    px_per_mm = chamber_type.pxPerMmFloor() * xf.fctr
    inner_radius_px = (
        float(nominal_agarose_radius_px) + float(inner_offset_mm) * px_per_mm
    )
    outer_radius_px = (
        float(nominal_agarose_radius_px) + float(outer_delta_mm) * px_per_mm
    )
    return {
        "reward_circle": (reward_cx, reward_cy, reward_r),
        "chamber_type": chamber_type,
        "arena_center": (arena_cx, arena_cy),
        "physical_centers": physical_centers,
        "nominal_agarose_radius_px": nominal_agarose_radius_px,
        "inner_radius_px": inner_radius_px,
        "outer_radius_px": outer_radius_px,
        "px_per_mm": px_per_mm,
        "floor_bounds": (
            float(floor_tl[0]),
            float(floor_tl[1]),
            float(floor_br[0]),
            float(floor_br[1]),
        ),
    }


def _rotated_centers(centers, arena_center, rotation_deg):
    centers = np.asarray(centers, dtype=float)
    arena_center = np.asarray(arena_center, dtype=float)
    theta = np.deg2rad(float(rotation_deg))
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )
    return (arena_center + (centers - arena_center) @ rotation.T)


def _farthest_center_indices(
    centers, reward_center, *, tie_tolerance_px
):
    centers = np.asarray(centers, dtype=float)
    reward_center = np.asarray(reward_center, dtype=float)
    distances = np.linalg.norm(centers - reward_center, axis=1)
    return tuple(
        int(idx)
        for idx in np.flatnonzero(
            distances >= float(np.max(distances)) - float(tie_tolerance_px)
        )
    )


def save_geometry_annotation(
    video_path,
    *,
    out_path,
    rotation_deg=45.0,
    frame_index=0,
    average_frames=30,
    protocol_index=0,
    training_index_1based=2,
    inner_offset_mm=0.0,
    outer_delta_mm=0.5,
    farthest_from_reward_only=False,
    dpi=220,
):
    frame = _average_video_frames(video_path, frame_index, average_frames)
    geometry = _large_chamber_geometry_from_protocol(
        video_path,
        frame=frame,
        protocol_index=protocol_index,
        training_index_1based=training_index_1based,
        inner_offset_mm=inner_offset_mm,
        outer_delta_mm=outer_delta_mm,
    )
    physical_centers = np.asarray(geometry["physical_centers"], dtype=float)
    physical_indices = virtual_indices = None
    if farthest_from_reward_only:
        reward_center = geometry["reward_circle"][:2]
        tie_tolerance_px = 0.25 * geometry["px_per_mm"]
        physical_indices = _farthest_center_indices(
            physical_centers,
            reward_center,
            tie_tolerance_px=tie_tolerance_px,
        )
        virtual_centers = _rotated_centers(
            physical_centers,
            geometry["arena_center"],
            rotation_deg,
        )
        virtual_indices = _farthest_center_indices(
            virtual_centers,
            reward_center,
            tie_tolerance_px=tie_tolerance_px,
        )
    Training.annotateAgaroseVirtualControlGeometry(
        frame,
        reward_circle=geometry["reward_circle"],
        physical_centers=physical_centers,
        arena_center=geometry["arena_center"],
        inner_radius_px=geometry["inner_radius_px"],
        outer_radius_px=geometry["outer_radius_px"],
        rotation_deg=rotation_deg,
        physical_indices=physical_indices,
        virtual_indices=virtual_indices,
    )
    x0, y0, x1, y1 = geometry["floor_bounds"]
    pad = 8
    height, width = frame.shape[:2]
    x0 = max(0, int(round(x0)) - pad)
    y0 = max(0, int(round(y0)) - pad)
    x1 = min(width, int(round(x1)) + pad)
    y1 = min(height, int(round(y1)) + pad)
    crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(crop)
    ax.axis("off")
    ax.set_title(
        "Farthest-from-reward dual-circle subset"
        if farthest_from_reward_only
        else "Dual-circle placement geometry"
    )
    position_suffix = " retained" if farthest_from_reward_only else ""
    ax.legend(
        handles=[
            Line2D([0], [0], color="white", lw=2.5, label="Reward circle"),
            Line2D([0], [0], color=ACCENT_ORANGE, lw=3, label=f"Physical positions{position_suffix}"),
            Line2D([0], [0], color=ACCENT_BLUE, lw=3, label=f"Virtual positions{position_suffix} ({rotation_deg:g}°)"),
            Line2D([0], [0], color="0.35", lw=1.8, linestyle="--", label="Outer approach circle"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _bundle_variant_metadata(bundle_path):
    with np.load(bundle_path, allow_pickle=True) as bundle:
        return {
            "farthest_only": bool(
                np.asarray(
                    bundle.get("agarose_farthest_from_reward_only", False)
                ).item()
            ),
            "wall_facing_only": bool(
                np.asarray(
                    bundle.get("agarose_wall_facing_entry_only", False)
                ).item()
            ),
            "wall_reference": str(
                np.asarray(
                    bundle.get("agarose_wall_facing_reference", "arena")
                ).item()
            ),
            "inner_offset_mm": float(
                np.asarray(
                    bundle.get("agarose_inner_radius_offset_mm", 0.0)
                ).item()
            ),
            "outer_offset_mm": float(
                np.asarray(
                    bundle.get("agarose_outer_radius_offset_mm", 0.5)
                ).item()
            ),
            "exclude_reward_arc": bool(
                np.asarray(
                    bundle.get(
                        "agarose_exclude_reward_facing_arc_entries", False
                    )
                ).item()
            ),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Plot physical-versus-virtual agarose control results and geometry."
    )
    parser.add_argument("--agarose-bundle", required=True)
    parser.add_argument("--flat-bundle", required=True)
    parser.add_argument("--mode", choices=("exp", "ctrl", "exp_minus_ctrl"), default="exp")
    parser.add_argument("--training-index", type=int, default=2)
    parser.add_argument("--sync-bucket-index", type=int, default=-1)
    parser.add_argument("--sync-bucket-start-index", type=int)
    parser.add_argument("--sync-bucket-end-index", type=int)
    parser.add_argument(
        "--out", default="imgs/agarose_virtual_control_summary.png"
    )
    parser.add_argument("--title")
    parser.add_argument("--background-video")
    parser.add_argument(
        "--geometry-out", default="imgs/agarose_virtual_control_geometry.png"
    )
    parser.add_argument("--background-frame", type=int, default=0)
    parser.add_argument("--background-average-frames", type=int, default=30)
    parser.add_argument("--background-protocol-index", type=int, default=0)
    parser.add_argument("--virtual-rotation-deg", type=float, default=45.0)
    args = parser.parse_args()

    agarose_variant = _bundle_variant_metadata(args.agarose_bundle)
    flat_variant = _bundle_variant_metadata(args.flat_bundle)
    if agarose_variant != flat_variant:
        raise ValueError(
            "agarose and flat bundles use different dual-circle metric variants"
        )
    farthest_only = agarose_variant["farthest_only"]

    common = dict(
        mode=args.mode,
        training_index_1based=args.training_index,
        bucket_index=args.sync_bucket_index,
        bucket_start_index_1based=args.sync_bucket_start_index,
        bucket_end_index_1based=args.sync_bucket_end_index,
    )
    agarose = load_chamber_placement_values(
        args.agarose_bundle, label="Agarose large", **common
    )
    flat = load_chamber_placement_values(
        args.flat_bundle, label="Flat large", **common
    )
    result = plot_agarose_virtual_control_summary(
        agarose,
        flat,
        out_path=args.out,
        title=(
            args.title
            if args.title is not None
            else (
                "Farthest-from-reward dual-circle comparison"
                if farthest_only
                else None
            )
        ),
    )
    print(
        f"[agarose-virtual-control] wrote {result['out_path']} "
        f"(interaction Δ={result['interaction_difference']:.6g}, "
        f"p={result['interaction_p_value']:.6g})"
    )
    if args.background_video:
        geometry_out = save_geometry_annotation(
            args.background_video,
            out_path=args.geometry_out,
            rotation_deg=args.virtual_rotation_deg,
            frame_index=args.background_frame,
            average_frames=args.background_average_frames,
            protocol_index=args.background_protocol_index,
            training_index_1based=args.training_index,
            inner_offset_mm=agarose_variant["inner_offset_mm"],
            outer_delta_mm=agarose_variant["outer_offset_mm"],
            farthest_from_reward_only=farthest_only,
        )
        print(f"[agarose-virtual-control] wrote {geometry_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
