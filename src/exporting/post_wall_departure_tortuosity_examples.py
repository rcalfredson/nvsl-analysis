from __future__ import annotations

import csv
import os

import numpy as np

from src.analysis.between_reward_filters import (
    min_between_reward_sync_bucket_trajectories,
)
from src.analysis.sync_bucket_presence_filters import (
    exp_target_sync_bucket_eligibility_mask,
)
from src.exporting.com_sli_bundle import _safe_group_label
from src.exporting.post_wall_departure_tortuosity_sli_bundle import (
    collect_post_wall_departure_tortuosity,
)
from src.plotting.event_chain_plotter import EventChainPlotter


MANIFEST_FIELDS = [
    "group",
    "role",
    "rank",
    "eligible_episodes",
    "video",
    "trajectory_index",
    "training",
    "segment_start",
    "segment_stop",
    "last_wall_contact_frame",
    "departure_frame",
    "path_mm",
    "direct_distance_to_reward_circle_mm",
    "tortuosity",
    "image",
]


def export_post_wall_departure_tortuosity_examples(vas, opts, gls, out_dir):
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if not vas_ok:
        print("[post-wall-departure-tortuosity-examples] no non-skipped videos")
        return
    records, details, _windows = collect_post_wall_departure_tortuosity(
        vas_ok, opts
    )
    threshold = max(1, min_between_reward_sync_bucket_trajectories(opts))
    target_eligible = exp_target_sync_bucket_eligibility_mask(vas_ok, opts)
    role_mode = str(
        getattr(opts, "post_wall_departure_tortuosity_examples_role", "exp")
        or "exp"
    ).lower()
    allowed_roles = {0} if role_mode == "exp" else {1}
    if role_mode == "both":
        allowed_roles = {0, 1}
    eligible_units = {
        (vi, role_idx)
        for vi, video_roles in enumerate(records)
        for role_idx, values in enumerate(video_roles)
        if len(values) >= threshold
        and role_idx in allowed_roles
        and (role_idx != 0 or bool(target_eligible[vi]))
    }
    candidates = [
        detail
        for detail in details
        if (int(detail["video_index"]), int(detail["role_idx"])) in eligible_units
    ]
    candidates.sort(key=lambda row: float(row["tortuosity"]), reverse=True)

    limit = max(
        1,
        int(
            getattr(opts, "post_wall_departure_tortuosity_examples_num", 12)
            or 12
        ),
    )
    max_per_fly = max(
        0,
        int(
            getattr(
                opts,
                "post_wall_departure_tortuosity_examples_max_per_fly",
                1,
            )
            or 0
        ),
    )
    selected = []
    fly_counts: dict[tuple[int, int], int] = {}
    for detail in candidates:
        unit = (int(detail["video_index"]), int(detail["role_idx"]))
        if max_per_fly and fly_counts.get(unit, 0) >= max_per_fly:
            continue
        selected.append(detail)
        fly_counts[unit] = fly_counts.get(unit, 0) + 1
        if len(selected) >= limit:
            break

    os.makedirs(out_dir, exist_ok=True)
    group_label = _safe_group_label(opts, gls)
    image_format = str(getattr(opts, "imageFormat", "png") or "png").lower()
    zoom_radius_mm = getattr(
        opts,
        "post_wall_departure_tortuosity_examples_zoom_radius_mm",
        None,
    )
    manifest_rows = []
    for rank, detail in enumerate(selected, start=1):
        va = detail["va"]
        role_idx = int(detail["role_idx"])
        role_label = "exp" if role_idx == 0 else "ctrl"
        video_base = os.path.splitext(os.path.basename(str(va.fn)))[0]
        filename = (
            f"rank{rank:02d}__{video_base}"
            f"__fly{int(detail['trajectory_index'])}"
            f"__T{int(detail['training_idx']) + 1}"
            f"__frames{int(detail['segment_start'])}-{int(detail['segment_stop'])}"
            f".{image_format}"
        )
        out_path = os.path.join(out_dir, role_label, filename)
        annotation = (
            f"rank: {rank}/{len(candidates)}\n"
            f"tortuosity: {float(detail['tortuosity']):.3f}\n"
            f"post-wall path: {float(detail['path_mm']):.3f} mm\n"
            f"direct distance: {float(detail['direct_mm']):.3f} mm\n"
            f"last wall frame: {int(detail['last_wall_contact_frame'])}\n"
            f"departure frame: {int(detail['departure_frame'])}"
        )
        zoom = zoom_radius_mm is not None and float(zoom_radius_mm) > 0
        EventChainPlotter(
            trj=detail["traj"],
            va=va,
            image_format=image_format,
        ).plot_between_reward_interval(
            trn_index=int(detail["training_idx"]),
            start_reward=int(detail["segment_start"]),
            end_reward=int(detail["segment_stop"]),
            role_idx=role_idx,
            pad=0,
            zoom=zoom,
            zoom_radius_mm=float(zoom_radius_mm) if zoom else None,
            out_path=out_path,
            title_suffix=f"| {group_label} | {role_label}",
            annotation_text=annotation,
            annotation_location="figure-right",
            annotation_wrap_width=30,
            title_wrap_width=90,
            highlight_start_frame=int(detail["departure_frame"]),
            highlight_stop_frame=int(detail["metric_stop"]),
            highlight_exclude_nonwalking=bool(detail["exclude_nonwalk"]),
            highlight_label="Path after final wall departure",
            comparison_line_start_xy=detail["departure_xy"],
            comparison_line_stop_xy=detail["reward_edge_xy"],
            comparison_line_label="Direct distance to reward circle",
        )
        manifest_rows.append(
            {
                "group": group_label,
                "role": role_label,
                "rank": rank,
                "eligible_episodes": len(candidates),
                "video": str(va.fn),
                "trajectory_index": int(detail["trajectory_index"]),
                "training": int(detail["training_idx"]) + 1,
                "segment_start": int(detail["segment_start"]),
                "segment_stop": int(detail["segment_stop"]),
                "last_wall_contact_frame": int(
                    detail["last_wall_contact_frame"]
                ),
                "departure_frame": int(detail["departure_frame"]),
                "path_mm": float(detail["path_mm"]),
                "direct_distance_to_reward_circle_mm": float(
                    detail["direct_mm"]
                ),
                "tortuosity": float(detail["tortuosity"]),
                "image": out_path,
            }
        )

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(
        "[post-wall-departure-tortuosity-examples] wrote "
        f"{len(manifest_rows)} images and {manifest_path}"
    )
