"""Geometry-only audit for reward-centered agarose control sites.

This module deliberately does not inspect trajectories or avoidance outcomes.
It enumerates sitewise rotations around the applicable reward center and reports
which candidates satisfy physical-agarose separation and wall-matching rules.
"""

from __future__ import annotations

import csv
from functools import lru_cache
import itertools
from pathlib import Path

import numpy as np


DEFAULT_WALL_TIERS_MM = {
    "strict": (0.5, 1.0),
    "moderate": (1.0, 2.0),
    "relaxed": (2.0, 3.0),
}


@lru_cache(maxsize=None)
def _leggauss(n):
    return np.polynomial.legendre.leggauss(int(n))


def rotate_about(point, origin, angle_deg):
    """Rotate a 2-D point around an origin."""
    point = np.asarray(point, dtype=float)
    origin = np.asarray(origin, dtype=float)
    theta = np.deg2rad(float(angle_deg))
    rotation = np.asarray(
        ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))),
        dtype=float,
    )
    return origin + rotation @ (point - origin)


def signed_wall_clearances_mm(center, floor_bounds, outer_radius_px, px_per_mm):
    """Return nearest-to-farthest signed outer-circle wall clearances."""
    x, y = (float(v) for v in center)
    x0, y0, x1, y1 = (float(v) for v in floor_bounds)
    center_wall_distances = np.asarray((x - x0, x1 - x, y - y0, y1 - y))
    return np.sort((center_wall_distances - float(outer_radius_px)) / px_per_mm)


def circle_outside_rectangle_fraction(center, radius, floor_bounds, quadrature_n=96):
    """Return the fraction of a circle lying outside an axis-aligned rectangle."""
    cx, cy = (float(v) for v in center)
    radius = float(radius)
    x0, y0, x1, y1 = (float(v) for v in floor_bounds)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("floor_bounds must define a nonempty rectangle")
    left = max(cx - radius, x0)
    right = min(cx + radius, x1)
    if right <= left:
        return 1.0

    # Integrate the circle/rectangle intersection as vertical strips. Fixed
    # Gauss-Legendre quadrature is deterministic and accurately handles both
    # single-wall caps and corner truncation without double-counting them.
    nodes, weights = _leggauss(int(quadrature_n))
    xs = 0.5 * (right - left) * nodes + 0.5 * (right + left)
    half_heights = np.sqrt(np.maximum(0.0, radius**2 - (xs - cx) ** 2))
    lower = np.maximum(cy - half_heights, y0)
    upper = np.minimum(cy + half_heights, y1)
    heights = np.maximum(0.0, upper - lower)
    inside_area = 0.5 * (right - left) * float(np.dot(weights, heights))
    circle_area = np.pi * radius**2
    return float(np.clip(1.0 - inside_area / circle_area, 0.0, 1.0))


def _reward_ring_bisector_candidates(physical_center, reward_center, physical_centers):
    """Return exact intersections with bisectors of pairs that are nearest there."""
    physical_center = np.asarray(physical_center, dtype=float)
    reward_center = np.asarray(reward_center, dtype=float)
    physical_centers = np.asarray(physical_centers, dtype=float)
    reward_radius = float(np.linalg.norm(physical_center - reward_center))
    if reward_radius <= 0:
        return []
    raw = []
    for first_idx, second_idx in itertools.combinations(range(len(physical_centers)), 2):
        first = physical_centers[first_idx]
        second = physical_centers[second_idx]
        normal = second - first
        normal_sq = float(np.dot(normal, normal))
        if normal_sq <= 0:
            continue
        constant = 0.5 * (float(np.dot(second, second)) - float(np.dot(first, first)))
        offset = (constant - float(np.dot(normal, reward_center))) / normal_sq
        foot = reward_center + offset * normal
        foot_distance_sq = float(np.dot(foot - reward_center, foot - reward_center))
        remaining_sq = reward_radius**2 - foot_distance_sq
        tolerance = 1e-9 * max(1.0, reward_radius**2)
        if remaining_sq < -tolerance:
            continue
        tangent = np.asarray((-normal[1], normal[0]), dtype=float) / np.sqrt(normal_sq)
        displacement = np.sqrt(max(0.0, remaining_sq))
        intersections = [foot] if displacement <= 1e-9 else [
            foot + displacement * tangent,
            foot - displacement * tangent,
        ]
        for candidate in intersections:
            distances = np.linalg.norm(physical_centers - candidate, axis=1)
            pair_distance = 0.5 * (
                float(distances[first_idx]) + float(distances[second_idx])
            )
            other_indices = [
                idx for idx in range(len(physical_centers))
                if idx not in (first_idx, second_idx)
            ]
            # The bisected pair must truly flank the candidate: neither of the
            # other agarose sites may be appreciably closer than the pair.
            if any(
                distances[idx] < pair_distance - 1e-7
                for idx in other_indices
            ):
                continue
            source_vector = physical_center - reward_center
            candidate_vector = candidate - reward_center
            angle = np.degrees(
                np.arctan2(candidate_vector[1], candidate_vector[0])
                - np.arctan2(source_vector[1], source_vector[0])
            ) % 360.0
            raw.append(
                {
                    "point": candidate,
                    "angle_deg": float(angle),
                    "neighbor_pair": (first_idx + 1, second_idx + 1),
                }
            )

    # Symmetric layouts can produce the same point from multiple bisectors.
    # Merge those representations so such a point is not oversampled later.
    merged = []
    for item in raw:
        match = next(
            (
                existing
                for existing in merged
                if np.linalg.norm(existing["point"] - item["point"]) <= 1e-7
            ),
            None,
        )
        if match is None:
            merged.append(item)
        else:
            match["neighbor_pair"] = tuple(
                sorted(set(match["neighbor_pair"]) | set(item["neighbor_pair"]))
            )
    return merged


def audit_site_candidates(
    *,
    physical_centers,
    physical_site_index,
    reward_center,
    floor_bounds,
    nominal_radius_px,
    outer_radius_px,
    px_per_mm,
    angle_step_deg=5.0,
    agarose_buffer_mm=1.0,
    max_outside_area_fraction=0.25,
    candidate_method="analytical",
    wall_tiers_mm=None,
):
    """Enumerate and classify reward-centered candidates for one physical site."""
    physical_centers = np.asarray(physical_centers, dtype=float)
    reward_center = np.asarray(reward_center, dtype=float)
    physical_center = physical_centers[int(physical_site_index)]
    floor_bounds = tuple(float(v) for v in floor_bounds)
    px_per_mm = float(px_per_mm)
    angle_step_deg = float(angle_step_deg)
    agarose_buffer_mm = float(agarose_buffer_mm)
    max_outside_area_fraction = float(max_outside_area_fraction)
    candidate_method = str(candidate_method)
    tiers = DEFAULT_WALL_TIERS_MM if wall_tiers_mm is None else wall_tiers_mm
    if physical_centers.ndim != 2 or physical_centers.shape[1:] != (2,):
        raise ValueError("physical_centers must have shape (n_sites, 2)")
    if reward_center.shape != (2,) or not np.all(np.isfinite(reward_center)):
        raise ValueError("reward_center must be a finite 2-D point")
    if not np.isfinite(angle_step_deg) or angle_step_deg <= 0 or angle_step_deg > 360:
        raise ValueError("angle_step_deg must be in (0, 360]")
    if not np.isfinite(agarose_buffer_mm) or agarose_buffer_mm < 0:
        raise ValueError("agarose_buffer_mm must be finite and nonnegative")
    if (
        not np.isfinite(max_outside_area_fraction)
        or max_outside_area_fraction < 0
        or max_outside_area_fraction > 1
    ):
        raise ValueError("max_outside_area_fraction must be in [0, 1]")
    if px_per_mm <= 0 or not np.isfinite(px_per_mm):
        raise ValueError("px_per_mm must be finite and positive")
    if candidate_method not in {"analytical", "grid"}:
        raise ValueError("candidate_method must be 'analytical' or 'grid'")

    physical_reward_distance_mm = float(
        np.linalg.norm(physical_center - reward_center) / px_per_mm
    )
    physical_wall = signed_wall_clearances_mm(
        physical_center, floor_bounds, outer_radius_px, px_per_mm
    )
    x0, y0, x1, y1 = floor_bounds
    rows = []
    if candidate_method == "analytical":
        candidate_specs = _reward_ring_bisector_candidates(
            physical_center, reward_center, physical_centers
        )
    else:
        candidate_specs = [
            {
                "point": rotate_about(physical_center, reward_center, angle_deg),
                "angle_deg": float(angle_deg),
                "neighbor_pair": (),
            }
            for angle_deg in np.arange(0.0, 360.0, angle_step_deg)
        ]
    for candidate_spec in candidate_specs:
        candidate = np.asarray(candidate_spec["point"], dtype=float)
        angle_deg = float(candidate_spec["angle_deg"])
        candidate_wall = signed_wall_clearances_mm(
            candidate, floor_bounds, outer_radius_px, px_per_mm
        )
        center_inside = bool(
            x0 <= candidate[0] <= x1 and y0 <= candidate[1] <= y1
        )
        outside_area_fraction = circle_outside_rectangle_fraction(
            candidate, outer_radius_px, floor_bounds
        )
        surface_gaps_mm = (
            np.linalg.norm(physical_centers - candidate, axis=1)
            - float(outer_radius_px)
            - float(nominal_radius_px)
        ) / px_per_mm
        min_agarose_gap_mm = float(np.min(surface_gaps_mm))
        nearest_order = np.argsort(surface_gaps_mm, kind="stable")
        nearest_idx = int(nearest_order[0])
        second_idx = int(nearest_order[1]) if len(nearest_order) > 1 else nearest_idx
        passes_buffer = bool(min_agarose_gap_mm >= agarose_buffer_mm - 1e-9)
        reward_distance_mm = float(
            np.linalg.norm(candidate - reward_center) / px_per_mm
        )
        row = {
            "physical_site": int(physical_site_index) + 1,
            "candidate_method": candidate_method,
            "angle_deg": float(angle_deg),
            "balanced_neighbor_sites": "|".join(
                str(site) for site in candidate_spec["neighbor_pair"]
            ),
            "candidate_x_px": float(candidate[0]),
            "candidate_y_px": float(candidate[1]),
            "reward_distance_mm": reward_distance_mm,
            "reward_distance_error_mm": reward_distance_mm
            - physical_reward_distance_mm,
            "min_physical_agarose_gap_mm": min_agarose_gap_mm,
            "nearest_agarose_site": nearest_idx + 1,
            "second_nearest_agarose_site": second_idx + 1,
            "nearest_agarose_gap_mm": float(surface_gaps_mm[nearest_idx]),
            "second_nearest_agarose_gap_mm": float(surface_gaps_mm[second_idx]),
            "nearest_two_gap_imbalance_mm": float(
                abs(surface_gaps_mm[second_idx] - surface_gaps_mm[nearest_idx])
            ),
            "center_inside_floor": center_inside,
            "outside_floor_area_fraction": outside_area_fraction,
            "passes_outside_area_limit": bool(
                outside_area_fraction <= max_outside_area_fraction + 1e-9
            ),
            "passes_agarose_buffer": passes_buffer,
            "physical_nearest_wall_clearance_mm": float(physical_wall[0]),
            "candidate_nearest_wall_clearance_mm": float(candidate_wall[0]),
            "nearest_wall_difference_mm": float(
                abs(candidate_wall[0] - physical_wall[0])
            ),
            "physical_second_wall_clearance_mm": float(physical_wall[1]),
            "candidate_second_wall_clearance_mm": float(candidate_wall[1]),
            "second_wall_difference_mm": float(
                abs(candidate_wall[1] - physical_wall[1])
            ),
            "passes_hard_geometry": bool(center_inside and passes_buffer),
        }
        row["passes_primary_geometry"] = bool(
            row["passes_hard_geometry"] and row["passes_outside_area_limit"]
        )
        for tier, (nearest_tol, second_tol) in tiers.items():
            row[f"passes_{tier}_nearest_only"] = bool(
                row["passes_hard_geometry"]
                and row["nearest_wall_difference_mm"] <= float(nearest_tol) + 1e-9
            )
            row[f"passes_{tier}"] = bool(
                row[f"passes_{tier}_nearest_only"]
                and row["second_wall_difference_mm"] <= float(second_tol) + 1e-9
            )
        rows.append(row)
    return rows


def _count_complete_draws(candidate_rows_by_site, outer_radius_px, cap=10_000):
    """Count mutually nonoverlapping complete draws, stopping at ``cap``."""
    candidates = [
        [np.asarray((row["candidate_x_px"], row["candidate_y_px"]), dtype=float)
         for row in rows]
        for rows in candidate_rows_by_site
    ]
    if not candidates or any(not site for site in candidates):
        return 0, False
    candidates.sort(key=len)
    minimum_distance = 2.0 * float(outer_radius_px)
    count = 0

    def visit(site_idx, selected):
        nonlocal count
        if count >= cap:
            return
        if site_idx == len(candidates):
            count += 1
            return
        for candidate in candidates[site_idx]:
            if all(np.linalg.norm(candidate - other) >= minimum_distance - 1e-9
                   for other in selected):
                next_selected = selected + [candidate]
                if any(
                    not any(
                        all(
                            np.linalg.norm(future_candidate - other)
                            >= minimum_distance - 1e-9
                            for other in next_selected
                        )
                        for future_candidate in future_site
                    )
                    for future_site in candidates[site_idx + 1 :]
                ):
                    continue
                visit(site_idx + 1, next_selected)
                if count >= cap:
                    return

    visit(0, [])
    return count, count >= cap


def select_maximin_complete_draw(candidate_rows_by_site, outer_radius_px, seed=101):
    """Select the nonoverlapping draw maximizing worst, then total agarose gap."""
    candidates = [list(rows) for rows in candidate_rows_by_site]
    if not candidates or any(not rows for rows in candidates):
        return None
    site_order = sorted(range(len(candidates)), key=lambda idx: len(candidates[idx]))
    minimum_distance = 2.0 * float(outer_radius_px)
    rng = np.random.default_rng(int(seed))
    tie_breakers = {
        id(row): float(rng.random()) for rows in candidates for row in rows
    }
    best = None
    best_score = None

    def visit(order_idx, selected):
        nonlocal best, best_score
        if order_idx == len(site_order):
            gaps = [
                float(row["min_physical_agarose_gap_mm"])
                for _site_idx, row, _point in selected
            ]
            outside = [
                float(row["outside_floor_area_fraction"])
                for _site_idx, row, _point in selected
            ]
            score = (
                min(gaps),
                sum(gaps),
                -sum(outside),
                sum(tie_breakers[id(row)] for _site_idx, row, _point in selected),
            )
            if best_score is None or score > best_score:
                best_score = score
                best = list(selected)
            return
        site_idx = site_order[order_idx]
        for row in candidates[site_idx]:
            point = np.asarray(
                (float(row["candidate_x_px"]), float(row["candidate_y_px"])),
                dtype=float,
            )
            if all(
                np.linalg.norm(point - other_point) >= minimum_distance - 1e-9
                for _other_site, _other_row, other_point in selected
            ):
                visit(order_idx + 1, selected + [(site_idx, row, point)])

    visit(0, [])
    if best is None:
        return None
    return {site_idx: row for site_idx, row, _point in best}


def build_geometry_audit_rows(
    vas,
    *,
    angle_step_deg=5.0,
    agarose_buffer_mm=1.0,
    max_outside_area_fraction=0.25,
    candidate_method="analytical",
    inner_radius_offset_mm=0.0,
    outer_delta_mm=0.5,
    wall_tiers_mm=None,
):
    """Build detailed candidate rows and per-site/tier summaries from analyses."""
    # Keep the pure geometry helpers usable in lightweight environments that do
    # not have the video-analysis dependencies (notably OpenCV) installed.
    from src.utils.common import CT

    tiers = DEFAULT_WALL_TIERS_MM if wall_tiers_mm is None else wall_tiers_mm
    inner_radius_offset_mm = float(inner_radius_offset_mm)
    outer_delta_mm = float(outer_delta_mm)
    if not np.isfinite(inner_radius_offset_mm) or inner_radius_offset_mm < 0:
        raise ValueError("inner_radius_offset_mm must be finite and nonnegative")
    if not np.isfinite(outer_delta_mm) or outer_delta_mm <= inner_radius_offset_mm:
        raise ValueError(
            "outer_delta_mm must be finite and greater than inner_radius_offset_mm"
        )
    candidate_rows = []
    summary_rows = []
    for video_index, va in enumerate(vas):
        if getattr(va, "ct", None) not in (CT.large, CT.large2):
            continue
        video = str(getattr(va, "fn", f"va_{video_index}"))
        px_per_mm = float(va.ct.pxPerMmFloor() * va.xf.fctr)
        for fly_index in getattr(va, "flies", (0,)):
            transform_fly = va.trxf[fly_index]
            radius_from_wells, physical_centers = va.ct.arenaWells(
                va.xf, transform_fly
            )
            nominal_radius_px = float(radius_from_wells)
            outer_radius_px = nominal_radius_px + float(outer_delta_mm) * px_per_mm
            floor_tl, floor_br = list(va.ct.floor(va.xf, f=transform_fly))
            floor_bounds = (
                float(floor_tl[0]), float(floor_tl[1]),
                float(floor_br[0]), float(floor_br[1]),
            )
            for training_idx, training in enumerate(va.trns):
                reward_circles = training.circles(fly_index)
                if not reward_circles:
                    continue
                reward_center = tuple(float(v) for v in reward_circles[0][:2])
                reward_radius_px = float(reward_circles[0][2])
                site_rows = []
                for site_idx in range(len(physical_centers)):
                    rows = audit_site_candidates(
                        physical_centers=physical_centers,
                        physical_site_index=site_idx,
                        reward_center=reward_center,
                        floor_bounds=floor_bounds,
                        nominal_radius_px=nominal_radius_px,
                        outer_radius_px=outer_radius_px,
                        px_per_mm=px_per_mm,
                        angle_step_deg=angle_step_deg,
                        agarose_buffer_mm=agarose_buffer_mm,
                        max_outside_area_fraction=max_outside_area_fraction,
                        candidate_method=candidate_method,
                        wall_tiers_mm=tiers,
                    )
                    common = {
                        "video": video,
                        "video_index": int(video_index),
                        "va_fly": getattr(va, "f", None),
                        "trajectory_fly": int(fly_index),
                        "role": "exp" if fly_index == 0 else "ctrl",
                        "training": int(training_idx) + 1,
                        "reward_x_px": reward_center[0],
                        "reward_y_px": reward_center[1],
                        "reward_radius_px": reward_radius_px,
                        "physical_x_px": float(physical_centers[site_idx][0]),
                        "physical_y_px": float(physical_centers[site_idx][1]),
                        "floor_x0_px": floor_bounds[0],
                        "floor_y0_px": floor_bounds[1],
                        "floor_x1_px": floor_bounds[2],
                        "floor_y1_px": floor_bounds[3],
                        "nominal_agarose_radius_px": nominal_radius_px,
                        "outer_radius_px": outer_radius_px,
                        "px_per_mm": px_per_mm,
                        "angle_step_deg": float(angle_step_deg),
                        "agarose_buffer_mm": float(agarose_buffer_mm),
                        "max_outside_area_fraction": float(
                            max_outside_area_fraction
                        ),
                        "candidate_method": str(candidate_method),
                        "inner_radius_offset_mm": float(inner_radius_offset_mm),
                        "outer_delta_mm": float(outer_delta_mm),
                    }
                    rows = [{**common, **row} for row in rows]
                    candidate_rows.extend(rows)
                    site_rows.append(rows)

                hard_by_site = [
                    [row for row in rows if row["passes_hard_geometry"]]
                    for rows in site_rows
                ]
                hard_complete_count, hard_count_capped = _count_complete_draws(
                    hard_by_site, outer_radius_px
                )
                primary_by_site = [
                    [row for row in rows if row["passes_primary_geometry"]]
                    for rows in site_rows
                ]
                primary_complete_count, primary_count_capped = _count_complete_draws(
                    primary_by_site, outer_radius_px
                )
                for tier, tolerances in tiers.items():
                    nearest_only_by_site = [
                        [
                            row
                            for row in rows
                            if row[f"passes_{tier}_nearest_only"]
                        ]
                        for rows in site_rows
                    ]
                    eligible_by_site = [
                        [row for row in rows if row[f"passes_{tier}"]]
                        for rows in site_rows
                    ]
                    nearest_complete_count, nearest_count_capped = (
                        _count_complete_draws(nearest_only_by_site, outer_radius_px)
                    )
                    complete_count, count_capped = _count_complete_draws(
                        eligible_by_site, outer_radius_px
                    )
                    for site_idx, eligible in enumerate(eligible_by_site):
                        hard_eligible = [
                            row for row in site_rows[site_idx]
                            if row["passes_hard_geometry"]
                        ]
                        primary_eligible = primary_by_site[site_idx]
                        summary_rows.append(
                            {
                                "video": video,
                                "video_index": int(video_index),
                                "va_fly": getattr(va, "f", None),
                                "trajectory_fly": int(fly_index),
                                "role": "exp" if fly_index == 0 else "ctrl",
                                "training": int(training_idx) + 1,
                                "physical_site": int(site_idx) + 1,
                                "tier": tier,
                                "nearest_wall_tolerance_mm": float(tolerances[0]),
                                "second_wall_tolerance_mm": float(tolerances[1]),
                                "hard_geometry_candidate_count": len(hard_eligible),
                                "hard_geometry_30deg_sector_count": len(
                                    {
                                        int(row["angle_deg"] // 30.0)
                                        for row in hard_eligible
                                    }
                                ),
                                "hard_geometry_complete_draw_exists": bool(
                                    hard_complete_count
                                ),
                                "hard_geometry_complete_draw_count": int(
                                    hard_complete_count
                                ),
                                "hard_geometry_complete_draw_count_capped": bool(
                                    hard_count_capped
                                ),
                                "primary_candidate_count": len(primary_eligible),
                                "primary_30deg_sector_count": len(
                                    {
                                        int(row["angle_deg"] // 30.0)
                                        for row in primary_eligible
                                    }
                                ),
                                "primary_complete_draw_exists": bool(
                                    primary_complete_count
                                ),
                                "primary_complete_draw_count": int(
                                    primary_complete_count
                                ),
                                "primary_complete_draw_count_capped": bool(
                                    primary_count_capped
                                ),
                                "nearest_only_candidate_count": len(
                                    nearest_only_by_site[site_idx]
                                ),
                                "nearest_only_30deg_sector_count": len(
                                    {
                                        int(row["angle_deg"] // 30.0)
                                        for row in nearest_only_by_site[site_idx]
                                    }
                                ),
                                "nearest_only_complete_draw_exists": bool(
                                    nearest_complete_count
                                ),
                                "nearest_only_complete_draw_count": int(
                                    nearest_complete_count
                                ),
                                "nearest_only_complete_draw_count_capped": bool(
                                    nearest_count_capped
                                ),
                                "eligible_candidate_count": len(eligible),
                                "eligible_30deg_sector_count": len(
                                    {int(row["angle_deg"] // 30.0) for row in eligible}
                                ),
                                "complete_draw_exists": bool(complete_count),
                                "complete_draw_count": int(complete_count),
                                "complete_draw_count_capped": bool(count_capped),
                                "angle_step_deg": float(angle_step_deg),
                                "candidate_method": str(candidate_method),
                                "agarose_buffer_mm": float(agarose_buffer_mm),
                                "max_outside_area_fraction": float(
                                    max_outside_area_fraction
                                ),
                                "inner_radius_offset_mm": float(inner_radius_offset_mm),
                                "outer_delta_mm": float(outer_delta_mm),
                            }
                        )
    return candidate_rows, summary_rows


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"agarose geometry audit produced no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_geometry_audit_csv(vas, opts, output_path):
    """Export detailed candidates and a sibling per-tier summary CSV."""
    candidate_rows, summary_rows = build_geometry_audit_rows(
        vas,
        angle_step_deg=float(
            getattr(opts, "agarose_reward_audit_angle_step_deg", 5.0)
        ),
        agarose_buffer_mm=float(
            getattr(opts, "agarose_reward_audit_buffer_mm", 1.0)
        ),
        max_outside_area_fraction=float(
            getattr(opts, "agarose_reward_audit_max_outside_area_frac", 0.25)
        ),
        candidate_method=str(
            getattr(opts, "agarose_reward_audit_candidate_method", "analytical")
        ),
        inner_radius_offset_mm=float(
            getattr(opts, "agarose_inner_radius_offset_mm", 0.0)
        ),
        outer_delta_mm=float(getattr(opts, "agarose_outer_delta_mm", 0.5)),
    )
    output_path = Path(output_path)
    summary_path = output_path.with_name(
        f"{output_path.stem}_summary{output_path.suffix or '.csv'}"
    )
    _write_csv(output_path, candidate_rows)
    _write_csv(summary_path, summary_rows)
    configurations = {
        (row["video_index"], row["va_fly"], row["trajectory_fly"], row["training"])
        for row in summary_rows
    }
    print(
        f"[agarose reward geometry audit] wrote {len(candidate_rows)} candidates "
        f"for {len(configurations)} fly/training configurations to {output_path}"
    )
    print(f"[agarose reward geometry audit] wrote summary to {summary_path}")
    return output_path, summary_path
