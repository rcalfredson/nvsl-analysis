from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Wedge


def make_agarose_dual_circle_debug_rows(vas) -> list[dict]:
    rows: list[dict] = []
    for va in vas:
        payload = getattr(va, "agarose_dual_circle_debug_rows", None)
        if not payload:
            continue
        rows.extend(dict(row) for row in payload)
    return rows


def save_agarose_dual_circle_debug_table(vas, out_path: str | Path) -> None:
    rows = make_agarose_dual_circle_debug_rows(vas)
    if not rows:
        print("[agarose-debug] no dual-circle debug rows to write")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[agarose-debug] wrote dual-circle debug CSV: {out_path}")


def _rotate_centers(centers, arena_center, angle_deg):
    centers = np.asarray(centers, dtype=float)
    theta = math.radians(float(angle_deg))
    rotation = np.asarray(
        (
            (math.cos(theta), -math.sin(theta)),
            (math.sin(theta), math.cos(theta)),
        ),
        dtype=float,
    )
    return (centers - arena_center) @ rotation.T + arena_center


def _read_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


def _debug_image_candidates(
    vas,
    training_index=None,
    sync_bucket_start_index=None,
    sync_bucket_end_index=None,
):
    candidates = []
    for va in vas:
        for row in getattr(va, "agarose_dual_circle_debug_rows", ()) or ():
            if row.get("fly_role") != "exp":
                continue
            if not bool(row.get("included_by_active_filters", True)):
                continue
            if training_index is not None:
                if row.get("sync_training_idx_1based") != int(training_index):
                    continue
            bucket = row.get("sync_bucket_idx_1based")
            if sync_bucket_start_index is not None:
                if bucket == "" or int(bucket) < int(sync_bucket_start_index):
                    continue
            if sync_bucket_end_index is not None:
                if bucket == "" or int(bucket) > int(sync_bucket_end_index):
                    continue
            candidates.append((va, row))

    # Round-robin across geometry and outcome so a small gallery does not become
    # all physical episodes (the analysis generates those rows first).
    groups = {}
    for candidate in candidates:
        row = candidate[1]
        key = (row.get("geometry"), bool(row.get("avoids_inner")))
        groups.setdefault(key, []).append(candidate)
    ordered = []
    keys = sorted(groups, key=lambda key: (str(key[0]), key[1]))
    while keys:
        next_keys = []
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop(0))
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return ordered


def _episode_site_index(row):
    prefix = "well" if row.get("geometry") == "physical_agarose" else "virtual_site"
    for label in str(row.get("start_well_labels", "")).split("|"):
        if label.startswith(prefix):
            try:
                return int(label[len(prefix):]) - 1
            except ValueError:
                pass
    return None


def _render_agarose_dual_circle_debug_image(va, row, out_path):
    entry = int(row["episode_start_frame"])
    frame = _read_video_frame(row["video"], entry)
    if frame is None:
        print(
            f"[agarose-debug] could not read frame {entry} "
            f"from {row['video']}; skipping"
        )
        return False

    trj_idx = int(row["trx_idx"])
    abs_fly = int(row["absolute_fly"])
    trj = va.trx[trj_idx]
    floor_tl, floor_br = tuple(va.ct.floor(va.xf, f=abs_fly))
    floor_tl = np.asarray(floor_tl, dtype=float)
    floor_br = np.asarray(floor_br, dtype=float)
    arena_center = 0.5 * (floor_tl + floor_br)
    inner_radius, physical_centers = va.ct.arenaWells(va.xf, abs_fly)
    physical_centers = np.asarray(physical_centers, dtype=float)
    virtual_centers = _rotate_centers(
        physical_centers, arena_center, float(row.get("center_rotation_deg", 45.0))
    )
    center_shift_px = float(
        row.get("agarose_center_outward_shift_mm", 0.0)
    ) * (va.ct.pxPerMmFloor() * va.xf.fctr)
    if center_shift_px:
        def _shift_outward(centers):
            radial = centers - arena_center
            norms = np.linalg.norm(radial, axis=1)
            return centers + center_shift_px * radial / norms[:, np.newaxis]

        physical_centers = _shift_outward(physical_centers)
        virtual_centers = _shift_outward(virtual_centers)
    active_centers = (
        physical_centers
        if row.get("geometry") == "physical_agarose"
        else virtual_centers
    )
    site_idx = _episode_site_index(row)
    if site_idx is None or not 0 <= site_idx < len(active_centers):
        return False
    site_center = active_centers[site_idx]
    outer_radius = float(inner_radius) + float(row["agarose_outer_delta_mm"]) * (
        va.ct.pxPerMmFloor() * va.xf.fctr
    )

    x = np.asarray(np.ma.filled(trj.x, np.nan), dtype=float)
    y = np.asarray(np.ma.filled(trj.y, np.nan), dtype=float)
    context_start = max(0, int(row["debug_context_start_frame"]))
    context_stop = min(len(x), int(row["debug_context_stop_frame"]))
    episode_stop = min(len(x), int(row["episode_stop_frame"]))
    entry_point = np.asarray((x[entry], y[entry]), dtype=float)
    if not np.all(np.isfinite(entry_point)):
        return False

    figure = Figure(figsize=(7.0, 7.0), dpi=140)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.91)
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_xlim(float(floor_tl[0]), float(floor_br[0]))
    ax.set_ylim(float(floor_br[1]), float(floor_tl[1]))
    ax.set_aspect("equal")

    geometry_styles = (
        (physical_centers, "#2b8cbe", "physical"),
        (virtual_centers, "#e67e22", "virtual"),
    )
    active_name = "physical" if row.get("geometry") == "physical_agarose" else "virtual"
    for centers, color, name in geometry_styles:
        alpha = 0.34 if name == active_name else 0.16
        for center in centers:
            ax.add_patch(
                Circle(
                    center,
                    inner_radius,
                    fill=False,
                    color=color,
                    lw=0.65,
                    alpha=alpha,
                )
            )
            ax.add_patch(
                Circle(
                    center,
                    outer_radius,
                    fill=False,
                    color=color,
                    lw=0.65,
                    ls="--",
                    alpha=alpha,
                )
            )

    training_value = (
        row.get("sync_training_idx_1based")
        or row.get("training_pre_idx_1based")
        or (1 if row.get("global_pre_last10m") else None)
    )
    reward_center = None
    if training_value != "" and training_value is not None:
        training_idx = int(training_value) - 1
        if 0 <= training_idx < len(va.trns):
            reward_circles = va.trns[training_idx].circles(trj_idx)
            if reward_circles:
                reward_x, reward_y, reward_radius = reward_circles[0]
                reward_center = np.asarray((reward_x, reward_y), dtype=float)
                ax.add_patch(
                    Circle(
                        (reward_x, reward_y),
                        reward_radius,
                        fill=False,
                        color="#31a354",
                        lw=0.85,
                        alpha=0.55,
                    )
                )
                ax.text(
                    reward_x,
                    reward_y - reward_radius - 3,
                    "reward",
                    color="#238b45",
                    fontsize=7.5,
                    ha="center",
                )

    reference_name = str(row.get("wall_facing_reference", "arena"))
    reference_center = (
        reward_center
        if reference_name == "reward" and reward_center is not None
        else arena_center
    )
    outward = site_center - reference_center
    outward_unit = outward / np.linalg.norm(outward)
    outward_angle = math.degrees(math.atan2(outward_unit[1], outward_unit[0]))
    active_color = "#2b8cbe" if active_name == "physical" else "#e67e22"
    ax.add_patch(
        Wedge(
            site_center,
            outer_radius,
            outward_angle - 90,
            outward_angle + 90,
            color=active_color,
            alpha=0.09,
            lw=0,
        )
    )
    perpendicular = np.asarray((-outward_unit[1], outward_unit[0]))
    divider = np.vstack(
        (
            site_center - outer_radius * perpendicular,
            site_center + outer_radius * perpendicular,
        )
    )
    ax.plot(divider[:, 0], divider[:, 1], color=active_color, lw=0.7, alpha=0.5)

    ax.plot(
        x[context_start:context_stop],
        y[context_start:context_stop],
        color="#6a51a3",
        lw=1.15,
        alpha=0.78,
        label="trajectory context",
    )
    ax.plot(
        x[entry:episode_stop],
        y[entry:episode_stop],
        color="#54278f",
        lw=2.0,
        alpha=0.95,
        label="outer-circle episode",
    )
    ax.scatter(
        *entry_point,
        s=30,
        color="#ffd92f",
        edgecolor="black",
        lw=0.6,
        zorder=5,
    )
    ax.scatter(
        *site_center,
        s=16,
        color=active_color,
        edgecolor="white",
        lw=0.5,
        zorder=5,
    )

    ax.annotate(
        "",
        xy=entry_point,
        xytext=site_center,
        arrowprops=dict(arrowstyle="->", color="#d7301f", lw=1.8),
        zorder=6,
    )
    vector_midpoint = 0.5 * (entry_point + site_center)
    ax.text(
        vector_midpoint[0],
        vector_midpoint[1],
        r"  $p-c$",
        color="#d7301f",
        fontsize=9,
        weight="bold",
    )
    outward_tip = site_center + 0.8 * outer_radius * outward_unit
    ax.annotate(
        "",
        xy=outward_tip,
        xytext=site_center,
        arrowprops=dict(arrowstyle="->", color="#238b45", lw=1.0, ls="--"),
    )
    ax.text(
        outward_tip[0],
        outward_tip[1],
        r"  outward direction "
        + (r"$(c-r)$" if reference_name == "reward" else r"$(c-a)$"),
        color="#238b45",
        fontsize=7.5,
    )

    training = row.get("sync_training_idx_1based") or "pre"
    bucket = row.get("sync_bucket_idx_1based") or "–"
    outcome = "AVOID" if row.get("avoids_inner") else "CONTACT"
    alignment = float(row.get("entry_wall_alignment", np.nan))
    ax.set_title(
        f"{active_name.capitalize()} site {site_idx + 1} · {outcome} · "
        f"wall-facing={bool(row.get('wall_facing_entry'))}\n"
        f"alignment $\\cos\\theta$ = {alignment:.3f} · "
        f"training {training}, bucket {bucket} · entry frame {entry}",
        fontsize=10,
    )
    ax.legend(loc="lower right", framealpha=0.75, fontsize=7.5)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path)
    figure.clear()
    return True


def save_agarose_dual_circle_debug_images(
    vas,
    out_dir: str | Path,
    *,
    max_images=12,
    training_index=None,
    sync_bucket_start_index=None,
    sync_bucket_end_index=None,
) -> list[Path]:
    """Write a balanced gallery of real episodes used by the active filters."""
    max_images = max(0, int(max_images))
    out_dir = Path(out_dir)
    written = []
    for va, row in _debug_image_candidates(
        vas,
        training_index=training_index,
        sync_bucket_start_index=sync_bucket_start_index,
        sync_bucket_end_index=sync_bucket_end_index,
    ):
        if len(written) >= max_images:
            break
        geometry = "physical" if row["geometry"] == "physical_agarose" else "virtual"
        outcome = "avoid" if row.get("avoids_inner") else "contact"
        filename = (
            f"{len(written) + 1:02d}_{geometry}_{outcome}_fly{row['absolute_fly']}"
            f"_frame{row['episode_start_frame']}.png"
        )
        out_path = out_dir / filename
        if _render_agarose_dual_circle_debug_image(va, row, out_path):
            written.append(out_path)
    print(f"[agarose-debug] wrote {len(written)} annotated episode image(s): {out_dir}")
    return written
