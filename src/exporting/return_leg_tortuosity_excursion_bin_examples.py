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
from src.exporting.return_leg_tortuosity_excursion_bin_sli_bundle import (
    _collect_records,
    _binning_mode,
    _combined_exclusion_mask,
    _effective_windowing,
    _parse_edges,
    _parse_pairs,
    _resolved_bins,
    _selected_training_indices,
    _top_fraction,
)
from src.plotting.between_reward_segment_metrics import (
    dist_traveled_mm_masked,
    max_radial_distance_mm_masked,
    net_displacement_mm_masked,
)
from src.plotting.event_chain_plotter import EventChainPlotter
from src.utils import util


def _group_label(record, opts, gls) -> str:
    explicit = getattr(opts, "export_group_label", None)
    if explicit:
        return str(explicit)
    try:
        group_idx = int(record["va"].gidx)
    except Exception:
        group_idx = 0
    if gls and 0 <= group_idx < len(gls) and gls[group_idx]:
        return str(gls[group_idx])
    return _safe_group_label(opts, gls)


def _bin_index(value: float, lower: np.ndarray, upper: np.ndarray) -> int | None:
    matches = np.flatnonzero((lower <= float(value)) & (float(value) < upper))
    return int(matches[0]) if matches.size else None


def _last_wall_frame(record) -> int | None:
    wc = record.get("wall_mask")
    if wc is None:
        return None
    fi = int(record["window_start"])
    s = int(record["segment_start"])
    e = int(record["segment_stop"])
    start = max(0, min(s - fi, len(wc)))
    stop = max(0, min(e - fi, len(wc)))
    idx = np.flatnonzero(np.asarray(wc[start:stop], dtype=bool))
    if not idx.size:
        return None
    return int(fi + start + idx[-1])


def _metric_components(record) -> tuple[float, float, float]:
    exclude_wall_frames = bool(record.get("exclude_wall_frames", False))
    path_mask = _combined_exclusion_mask(
        record["nonwalk_mask"],
        record["wall_mask"],
        exclude_nonwalk=bool(record["exclude_nonwalk"]),
        exclude_wall_frames=exclude_wall_frames,
    )
    common = dict(
        traj=record["traj"],
        s=int(record["metric_start"]),
        e=int(record["segment_stop"]),
        fi=int(record["window_start"]),
        px_per_mm=float(record["px_per_mm"]),
        min_keep_frames=int(record["min_walk_frames"]),
    )
    path_mm = dist_traveled_mm_masked(
        **common,
        nonwalk_mask=path_mask,
        exclude_nonwalk=path_mask is not None,
    )
    displacement_mm = net_displacement_mm_masked(
        **common,
        nonwalk_mask=path_mask,
        exclude_nonwalk=path_mask is not None,
    )
    max_radius_mm = max_radial_distance_mm_masked(
        **common,
        nonwalk_mask=record["nonwalk_mask"],
        exclude_nonwalk=bool(record["exclude_nonwalk"]),
        center_xy=record["reward_center_xy"],
    )
    return float(path_mm), float(displacement_mm), float(max_radius_mm)


def _format_bin(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g} mm"


MANIFEST_FIELDS = [
    "group",
    "role",
    "bin_index",
    "bin_lower_mm",
    "bin_upper_mm",
    "rank",
    "eligible_episodes_in_group_bin",
    "video",
    "trajectory_index",
    "training",
    "segment_start",
    "segment_stop",
    "global_max_frame",
    "return_start_frame",
    "last_wall_frame",
    "episode_bin_distance_mm",
    "episode_bin_distance_semantics",
    "return_path_mm",
    "return_displacement_mm",
    "return_max_radius_mm",
    "tortuosity",
    "metric_mode",
    "return_start_mode",
    "exclude_wall_contact_frames",
    "image",
]


def export_return_leg_tortuosity_excursion_bin_examples(
    vas,
    opts,
    gls,
    out_dir: str,
) -> None:
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if not vas_ok:
        print("[return-leg-tortuosity-examples] no non-skipped videos")
        return

    if _binning_mode(opts) != "absolute_distance":
        raise ValueError(
            "Ranked return-leg tortuosity trajectory examples currently require "
            "absolute-distance binning."
        )

    pair_spec = _parse_pairs(opts)
    legacy_distances = pair_spec[2] if pair_spec is not None else _parse_edges(opts)[1]
    selected_trainings = _selected_training_indices(vas_ok, opts)
    skip_first, keep_first = _effective_windowing(opts)
    details: list[dict] = []
    records, _windows = _collect_records(
        vas_ok,
        opts,
        selected_trainings=selected_trainings,
        skip_first=skip_first,
        keep_first=keep_first,
        legacy_distances=legacy_distances,
        episode_callback=details.append,
    )
    lower, upper, _edges, _pair_mode, _legacy = _resolved_bins(opts, records)

    role_mode = str(
        getattr(
            opts,
            "return_leg_tortuosity_excursion_bin_examples_role",
            "exp",
        )
        or "exp"
    ).lower()
    allowed_roles = {0} if role_mode == "exp" else {1}
    if role_mode == "both":
        allowed_roles = {0, 1}

    target_eligible = exp_target_sync_bucket_eligibility_mask(vas_ok, opts)
    min_segments = min_between_reward_sync_bucket_trajectories(opts)
    top_fraction = _top_fraction(opts)
    per_bin = max(
        1,
        int(
            getattr(
                opts,
                "return_leg_tortuosity_excursion_bin_examples_per_bin",
                6,
            )
            or 6
        ),
    )
    max_per_fly = max(
        0,
        int(
            getattr(
                opts,
                "return_leg_tortuosity_excursion_bin_examples_max_per_fly",
                0,
            )
            or 0
        ),
    )

    indexed: dict[tuple[int, int, int], list[dict]] = {}
    for record in details:
        role_idx = int(record["role_idx"])
        if role_idx not in allowed_roles:
            continue
        vi = int(record["video_index"])
        if role_idx == 0 and not bool(target_eligible[vi]):
            continue
        bin_idx = _bin_index(record["radial_mm"], lower, upper)
        if bin_idx is None:
            continue
        record["bin_idx"] = bin_idx
        indexed.setdefault((vi, role_idx, bin_idx), []).append(record)

    eligible: list[dict] = []
    for unit_records in indexed.values():
        if len(unit_records) < min_segments:
            continue
        n_selected = max(1, int(np.ceil(top_fraction * len(unit_records))))
        ranked = sorted(
            unit_records,
            key=lambda item: float(item["tortuosity"]),
            reverse=True,
        )
        eligible.extend(ranked[:n_selected])

    grouped: dict[tuple[str, int, int], list[dict]] = {}
    for record in eligible:
        key = (
            _group_label(record, opts, gls),
            int(record["role_idx"]),
            int(record["bin_idx"]),
        )
        grouped.setdefault(key, []).append(record)

    os.makedirs(out_dir, exist_ok=True)
    manifest_rows: list[dict] = []
    image_format = str(getattr(opts, "imageFormat", "png") or "png").lower()
    zoom_override = getattr(
        opts,
        "return_leg_tortuosity_excursion_bin_examples_zoom_radius_mm",
        None,
    )

    for (group_label, role_idx, bin_idx), candidates in sorted(grouped.items()):
        ranked = sorted(
            candidates,
            key=lambda item: float(item["tortuosity"]),
            reverse=True,
        )
        selected = []
        fly_counts: dict[int, int] = {}
        for record in ranked:
            vi = int(record["video_index"])
            if max_per_fly and fly_counts.get(vi, 0) >= max_per_fly:
                continue
            selected.append(record)
            fly_counts[vi] = fly_counts.get(vi, 0) + 1
            if len(selected) >= per_bin:
                break

        lo = float(lower[bin_idx])
        hi = float(upper[bin_idx])
        group_slug = util.slugify(group_label) or "group"
        role_label = "exp" if role_idx == 0 else "ctrl"
        bin_slug = f"bin{bin_idx + 1}_{lo:g}-{hi:g}mm"
        bin_dir = os.path.join(out_dir, group_slug, role_label, bin_slug)
        os.makedirs(bin_dir, exist_ok=True)

        for rank, record in enumerate(selected, start=1):
            va = record["va"]
            traj = record["traj"]
            video_base = os.path.splitext(os.path.basename(str(va.fn)))[0]
            path_mm, displacement_mm, max_radius_mm = _metric_components(record)
            last_wall_frame = _last_wall_frame(record)
            mode = str(record["metric_mode"])
            distance_semantics = (
                "distance past reward-circle perimeter"
                if record["legacy_distances"]
                else "distance from reward-circle center"
            )
            denominator = (
                max_radius_mm
                if mode == "path_over_max_radius"
                else displacement_mm
            )
            annotation = (
                f"rank: {rank}/{len(candidates)}\n"
                f"tortuosity: {float(record['tortuosity']):.3f}\n"
                f"max-distance bin: {_format_bin(lo, hi)}\n"
                f"episode bin distance: {float(record['radial_mm']):.3f} mm\n"
                f"return path: {path_mm:.3f} mm\n"
                f"net displacement: {displacement_mm:.3f} mm\n"
                f"return max radius: {max_radius_mm:.3f} mm\n"
                f"metric denominator: {denominator:.3f} mm\n"
                f"return start: {int(record['metric_start'])}\n"
                f"wall-contact path frames excluded: "
                f"{bool(record.get('exclude_wall_frames', False))}\n"
                f"last wall frame: "
                f"{'none' if last_wall_frame is None else last_wall_frame}"
            )
            filename = (
                f"rank{rank:02d}__{video_base}__fly{int(record['trajectory_index'])}"
                f"__T{int(record['training_idx']) + 1}"
                f"__frames{int(record['segment_start'])}-{int(record['segment_stop'])}"
                f".{image_format}"
            )
            out_path = os.path.join(bin_dir, filename)
            if zoom_override is None:
                zoom_radius_mm = hi + 1.0 if np.isfinite(hi) else None
                zoom = zoom_radius_mm is not None
            else:
                zoom_radius_mm = float(zoom_override)
                zoom = zoom_radius_mm > 0
                if not zoom:
                    zoom_radius_mm = None

            EventChainPlotter(
                trj=traj,
                va=va,
                image_format=image_format,
            ).plot_between_reward_interval(
                trn_index=int(record["training_idx"]),
                start_reward=int(record["segment_start"]),
                end_reward=int(record["segment_stop"]),
                role_idx=role_idx,
                pad=0,
                zoom=zoom,
                zoom_radius_mm=zoom_radius_mm,
                out_path=out_path,
                title_suffix=(
                    f"| {group_label} | {role_label} | "
                    f"{record['return_start_mode']}"
                ),
                annotation_text=annotation,
                annotation_location="figure-right",
                annotation_wrap_width=30,
                title_wrap_width=90,
                highlight_start_frame=int(record["metric_start"]),
                highlight_stop_frame=int(record["segment_stop"]),
                highlight_exclude_nonwalking=bool(record["exclude_nonwalk"]),
                highlight_excluded_frame_mask=(
                    record["wall_mask"]
                    if record.get("exclude_wall_frames", False)
                    else None
                ),
                highlight_excluded_frame_mask_start=int(record["window_start"]),
                highlight_label="Measured return leg",
            )

            manifest_rows.append(
                {
                    "group": group_label,
                    "role": role_label,
                    "bin_index": bin_idx,
                    "bin_lower_mm": lo,
                    "bin_upper_mm": hi,
                    "rank": rank,
                    "eligible_episodes_in_group_bin": len(candidates),
                    "video": str(va.fn),
                    "trajectory_index": int(record["trajectory_index"]),
                    "training": int(record["training_idx"]) + 1,
                    "segment_start": int(record["segment_start"]),
                    "segment_stop": int(record["segment_stop"]),
                    "global_max_frame": int(record["global_max_frame"]),
                    "return_start_frame": int(record["metric_start"]),
                    "last_wall_frame": (
                        "" if last_wall_frame is None else int(last_wall_frame)
                    ),
                    "episode_bin_distance_mm": float(record["radial_mm"]),
                    "episode_bin_distance_semantics": distance_semantics,
                    "return_path_mm": path_mm,
                    "return_displacement_mm": displacement_mm,
                    "return_max_radius_mm": max_radius_mm,
                    "tortuosity": float(record["tortuosity"]),
                    "metric_mode": mode,
                    "return_start_mode": str(record["return_start_mode"]),
                    "exclude_wall_contact_frames": bool(
                        record.get("exclude_wall_frames", False)
                    ),
                    "image": out_path,
                }
            )

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        if manifest_rows:
            writer.writerows(manifest_rows)
    print(
        "[return-leg-tortuosity-examples] wrote "
        f"{len(manifest_rows)} images and {manifest_path}"
    )
