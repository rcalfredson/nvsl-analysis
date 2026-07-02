from __future__ import annotations

import csv
import os

import numpy as np

from src.analysis.between_reward_filters import (
    min_between_reward_sync_bucket_trajectories,
)
from src.analysis.sync_bucket_presence_filters import (
    exp_target_sync_bucket_eligibility_mask,
    exp_target_sync_bucket_filter_payload,
    mask_by_exp_target_sync_bucket_filter,
)
from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)
from src.exporting.wall_contact_episode_filter import (
    episode_overlaps_wall_contact,
    wall_contact_regions_for_trj,
)
from src.utils.parsers import parse_distances, parse_training_selector


def _dbg(opts, msg):
    if opts is not None and bool(
        getattr(opts, "return_prob_outer_radius_debug", False)
    ):
        print(msg)


def _dbg_if(debug: bool, msg: str):
    if bool(debug):
        print(msg)


def _selected_training_indices(vas, opts) -> list[int]:
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if not vas_ok:
        return []

    n_trn = len(getattr(vas_ok[0], "trns", []) or [])
    raw = getattr(opts, "return_prob_outer_radius_trainings", None)
    if not raw:
        return list(range(n_trn))

    selected = []
    for idx1 in parse_training_selector(raw):
        idx0 = int(idx1) - 1
        if 0 <= idx0 < n_trn:
            selected.append(idx0)

    selected = sorted(set(selected))
    if selected:
        return selected

    print(
        "[export] WARNING: --return-prob-outer-radius-trainings selected no valid "
        "trainings; falling back to all trainings."
    )
    return list(range(n_trn))


def _outer_radii_mm(opts) -> tuple[np.ndarray, bool]:
    raw = getattr(opts, "return_prob_outer_radius_outer_radii_mm", None)
    legacy = False
    if raw is None or not str(raw).strip():
        raw = getattr(opts, "return_prob_outer_radius_outer_deltas_mm", None)
        legacy = raw is not None and str(raw).strip()
    if raw is None or not str(raw).strip():
        raise ValueError(
            "--return-prob-outer-radius-outer-radii-mm is required when exporting "
            "return-prob-outer-radius bundles."
        )
    vals = np.asarray(parse_distances(str(raw)), dtype=float)
    if vals.size == 0:
        raise ValueError(
            "--return-prob-outer-radius-outer-radii-mm must contain at least one value."
        )
    return vals, bool(legacy)


def _effective_windowing(opts) -> tuple[int, int, int]:
    raw_skip = getattr(opts, "return_prob_outer_radius_skip_first_sync_buckets", None)
    raw_keep = getattr(opts, "return_prob_outer_radius_keep_first_sync_buckets", None)
    skip_first = (
        getattr(opts, "skip_first_sync_buckets", 0) if raw_skip is None else raw_skip
    )
    keep_first = (
        getattr(opts, "keep_first_sync_buckets", 0) if raw_keep is None else raw_keep
    )
    last_sync = int(
        getattr(opts, "return_prob_outer_radius_last_sync_buckets", 0) or 0
    )
    return max(0, int(skip_first or 0)), max(0, int(keep_first or 0)), max(
        0, last_sync
    )


def _selected_windows_for_va(
    va,
    selected_trainings: list[int],
    *,
    skip_first: int,
    keep_first: int,
    last_sync_buckets: int,
):
    windows = []
    sync_ranges = getattr(va, "sync_bucket_ranges", None)

    for t_idx in selected_trainings:
        if t_idx >= len(getattr(va, "trns", [])):
            continue
        trn = va.trns[t_idx]
        if trn is None or not trn.isCircle():
            continue

        if sync_ranges and t_idx < len(sync_ranges) and sync_ranges[t_idx]:
            rr = list(sync_ranges[t_idx])
            rr = rr[max(0, int(skip_first)) :]
            if keep_first > 0:
                rr = rr[: int(keep_first)]
            if last_sync_buckets > 0:
                rr = rr[-int(last_sync_buckets) :]
            if not rr:
                continue
            windows.append(
                {
                    "training_idx": int(t_idx),
                    "training_name": trn.name(),
                    "start": int(rr[0][0]),
                    "stop": int(rr[-1][1]),
                    "bucket_ranges": [(int(a), int(b)) for (a, b) in rr],
                }
            )
        else:
            windows.append(
                {
                    "training_idx": int(t_idx),
                    "training_name": trn.name(),
                    "start": int(trn.start),
                    "stop": int(trn.stop),
                    "bucket_ranges": [(int(trn.start), int(trn.stop))],
                }
            )
    return windows


def _frame_in_windows(frame: int, windows) -> bool:
    for win in windows:
        if int(win["start"]) <= frame < int(win["stop"]):
            return True
    return False


_DEBUG_EPISODE_FIELDS = [
    "video_index",
    "video_id",
    "fly_idx",
    "fly_role",
    "training_idx",
    "training_name",
    "radius_idx",
    "requested_outer_mm",
    "radius_mode",
    "window_start",
    "window_stop",
    "episode_idx",
    "anchor_reward",
    "reward_entry",
    "start",
    "stop",
    "event_frame",
    "returns",
    "end_reason",
    "wall_overlap",
    "included_in_metric",
    "reward_cx_px",
    "reward_cy_px",
    "reward_radius_px",
    "reward_radius_mm",
    "outer_radius_px",
    "outer_radius_mm",
    "px_per_mm",
    "reward_delta_mm",
    "border_width_mm",
    "start_x_px",
    "start_y_px",
    "event_x_px",
    "event_y_px",
    "start_distance_px",
    "event_distance_px",
    "start_distance_mm",
    "event_distance_mm",
]


def _wall_contact_regions_if_available(trj):
    try:
        return trj.boundary_event_stats["wall"]["all"]["edge"][
            "boundary_contact_regions"
        ]
    except (KeyError, TypeError, AttributeError):
        return None


def _return_prob_episode_geometry(
    trj,
    trn,
    ep: dict,
    *,
    requested_outer_mm: float,
    legacy_outer_radii: bool,
    reward_radius_mm: float | None,
    reward_delta_mm: float | None,
    border_width_mm: float,
) -> dict:
    nan_diag = {
        "reward_cx_px": float("nan"),
        "reward_cy_px": float("nan"),
        "reward_radius_px": float("nan"),
        "reward_radius_mm": float("nan"),
        "outer_radius_px": float("nan"),
        "outer_radius_mm": float("nan"),
        "px_per_mm": float("nan"),
        "reward_delta_mm": (
            float("nan") if reward_delta_mm is None else float(reward_delta_mm)
        ),
        "border_width_mm": float(border_width_mm),
        "start_x_px": float("nan"),
        "start_y_px": float("nan"),
        "event_x_px": float("nan"),
        "event_y_px": float("nan"),
        "start_distance_px": float("nan"),
        "event_distance_px": float("nan"),
        "start_distance_mm": float("nan"),
        "event_distance_mm": float("nan"),
    }

    if trn is None or not trn.isCircle():
        return nan_diag
    try:
        cs = trn.circles(getattr(trj, "f", 0))
    except Exception:
        cs = []
    if not cs:
        return nan_diag
    cx, cy, r_px = cs[0]
    try:
        cx = float(cx)
        cy = float(cy)
        r_px = float(r_px)
    except (TypeError, ValueError):
        return nan_diag
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(r_px)):
        return nan_diag

    try:
        fctr = float(getattr(trj.va.xf, "fctr", 1.0) or 1.0)
        px_per_mm = float(trj.va.ct.pxPerMmFloor()) * fctr
    except Exception:
        px_per_mm = float("nan")
    if not (np.isfinite(px_per_mm) and px_per_mm > 0.0):
        return {
            **nan_diag,
            "reward_cx_px": cx,
            "reward_cy_px": cy,
            "reward_radius_px": r_px,
        }

    base_reward_radius_mm = r_px / px_per_mm
    resolved_reward_radius_mm = (
        float(reward_radius_mm)
        if reward_radius_mm is not None
        else base_reward_radius_mm + float(reward_delta_mm or 0.0)
    )
    resolved_outer_radius_mm = (
        resolved_reward_radius_mm + float(requested_outer_mm)
        if legacy_outer_radii
        else float(requested_outer_mm)
    )
    reward_r_px = resolved_reward_radius_mm * px_per_mm
    outer_r_px = resolved_outer_radius_mm * px_per_mm

    out = {
        **nan_diag,
        "reward_cx_px": cx,
        "reward_cy_px": cy,
        "reward_radius_px": r_px,
        "reward_radius_mm": resolved_reward_radius_mm,
        "outer_radius_px": outer_r_px,
        "outer_radius_mm": resolved_outer_radius_mm,
        "px_per_mm": px_per_mm,
    }

    x = getattr(trj, "x", None)
    y = getattr(trj, "y", None)
    if x is None or y is None:
        return out
    event_frame = int(ep.get("event_frame", int(ep.get("stop", 0)) - 1))
    for prefix, frame in (
        ("start", int(ep.get("start", 0))),
        ("event", event_frame),
    ):
        if 0 <= frame < len(x) and 0 <= frame < len(y):
            px = float(x[frame])
            py = float(y[frame])
            if np.isfinite(px) and np.isfinite(py):
                dist_px = float(np.hypot(px - cx, py - cy))
                out[f"{prefix}_x_px"] = px
                out[f"{prefix}_y_px"] = py
                out[f"{prefix}_distance_px"] = dist_px
                out[f"{prefix}_distance_mm"] = dist_px / px_per_mm
    return out


def _write_return_prob_outer_radius_debug_episodes_csv(
    vas,
    *,
    out_csv: str,
    outer_radii_mm: np.ndarray,
    legacy_outer_radii: bool,
    reward_radius_mm: float | None,
    reward_delta_mm: float | None,
    border_width_mm: float,
    selected_trainings: list[int],
    skip_first: int,
    keep_first: int,
    last_sync_buckets: int,
    exclude_wall_contact: bool = False,
) -> int:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    n_rows = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DEBUG_EPISODE_FIELDS)
        writer.writeheader()

        for vi, va in enumerate(vas):
            vid = getattr(va, "fn", f"va_{vi}")
            va_fly_idx = getattr(va, "f", None)
            video_id = f"{vid}::f{va_fly_idx if va_fly_idx is not None else -1}"
            windows = _selected_windows_for_va(
                va,
                selected_trainings,
                skip_first=skip_first,
                keep_first=keep_first,
                last_sync_buckets=last_sync_buckets,
            )
            if not windows:
                continue
            windows_by_training = {int(win["training_idx"]): [win] for win in windows}

            for radius_idx, outer_radius_mm in enumerate(outer_radii_mm):
                for fly_idx, trj in enumerate(getattr(va, "trx", [])):
                    if getattr(trj, "_bad", False):
                        continue
                    if fly_idx > 1:
                        continue
                    ctrl = bool(fly_idx == 1)
                    if ctrl and getattr(va, "noyc", False):
                        continue

                    fly_role = "ctrl" if ctrl else "exp"
                    wall_regions = _wall_contact_regions_if_available(trj)
                    for t_idx in selected_trainings:
                        if t_idx >= len(getattr(va, "trns", [])):
                            continue
                        trn = va.trns[t_idx]
                        if trn is None or not trn.isCircle():
                            continue
                        if t_idx not in windows_by_training:
                            continue

                        episodes = trj.reward_return_probability_episodes_for_training(
                            trn=trn,
                            outer_radius_mm=(
                                None if legacy_outer_radii else float(outer_radius_mm)
                            ),
                            outer_delta_mm=(
                                float(outer_radius_mm) if legacy_outer_radii else None
                            ),
                            reward_radius_mm=reward_radius_mm,
                            reward_delta_mm=reward_delta_mm,
                            border_width_mm=float(border_width_mm),
                            ctrl=ctrl,
                            debug=False,
                        )
                        if not episodes:
                            continue

                        for ep_idx, ep in enumerate(episodes):
                            event_t = int(ep.get("event_frame", int(ep["stop"]) - 1))
                            matched_windows = [
                                win
                                for win in windows_by_training[t_idx]
                                if _frame_in_windows(event_t, [win])
                            ]
                            if not matched_windows:
                                continue
                            wall_overlap = episode_overlaps_wall_contact(
                                ep, wall_regions
                            )
                            included = not (bool(exclude_wall_contact) and wall_overlap)
                            for win in matched_windows:
                                row = {
                                    "video_index": int(vi),
                                    "video_id": video_id,
                                    "fly_idx": (
                                        int(va_fly_idx)
                                        if va_fly_idx is not None
                                        else int(fly_idx)
                                    ),
                                    "fly_role": fly_role,
                                    "training_idx": int(t_idx),
                                    "training_name": trn.name(),
                                    "radius_idx": int(radius_idx),
                                    "requested_outer_mm": float(outer_radius_mm),
                                    "radius_mode": (
                                        "delta" if legacy_outer_radii else "radius"
                                    ),
                                    "window_start": int(win["start"]),
                                    "window_stop": int(win["stop"]),
                                    "episode_idx": int(ep_idx),
                                    "event_frame": int(event_t),
                                    "wall_overlap": bool(wall_overlap),
                                    "included_in_metric": bool(included),
                                }
                                row.update(
                                    _return_prob_episode_geometry(
                                        trj,
                                        trn,
                                        ep,
                                        requested_outer_mm=float(outer_radius_mm),
                                        legacy_outer_radii=legacy_outer_radii,
                                        reward_radius_mm=reward_radius_mm,
                                        reward_delta_mm=reward_delta_mm,
                                        border_width_mm=border_width_mm,
                                    )
                                )
                                for key in _DEBUG_EPISODE_FIELDS:
                                    if key not in row and key in ep:
                                        row[key] = ep[key]
                                writer.writerow(row)
                                n_rows += 1
    return n_rows


def _compute_return_prob_curves(
    vas,
    *,
    outer_radii_mm: np.ndarray,
    legacy_outer_radii: bool,
    reward_radius_mm: float | None,
    reward_delta_mm: float | None,
    border_width_mm: float,
    selected_trainings: list[int],
    skip_first: int,
    keep_first: int,
    last_sync_buckets: int,
    debug: bool,
    min_trajectories: int = 0,
    exclude_wall_contact: bool = False,
):
    n_videos = len(vas)
    n_radii = int(outer_radii_mm.size)
    ratio_exp = np.full((n_videos, n_radii), np.nan, dtype=float)
    ratio_ctrl = np.full((n_videos, n_radii), np.nan, dtype=float)
    ret_exp = np.zeros((n_videos, n_radii), dtype=int)
    ret_ctrl = np.zeros((n_videos, n_radii), dtype=int)
    total_exp = np.zeros((n_videos, n_radii), dtype=int)
    total_ctrl = np.zeros((n_videos, n_radii), dtype=int)
    windows_meta = []
    warned_missing_wall_contact = [False]

    for vi, va in enumerate(vas):
        vid = getattr(va, "fn", f"va_{vi}")
        windows = _selected_windows_for_va(
            va,
            selected_trainings,
            skip_first=skip_first,
            keep_first=keep_first,
            last_sync_buckets=last_sync_buckets,
        )
        windows_meta.append(windows)
        if not windows:
            _dbg_if(
                debug,
                f"[return-prob-outer-radius] {vid}: no eligible windows after selection",
            )
            continue

        windows_by_training = {
            int(win["training_idx"]): [win] for win in windows
        }

        for ri, outer_radius_mm in enumerate(outer_radii_mm):
            for fly_idx, trj in enumerate(getattr(va, "trx", [])):
                if getattr(trj, "_bad", False):
                    continue

                returns = 0
                total = 0
                ctrl = bool(fly_idx == 1)
                wall_regions = wall_contact_regions_for_trj(
                    trj,
                    enabled=bool(exclude_wall_contact),
                    warned_missing=warned_missing_wall_contact,
                    log_tag="return-prob-outer-radius",
                )

                for t_idx in selected_trainings:
                    if t_idx >= len(getattr(va, "trns", [])):
                        continue
                    trn = va.trns[t_idx]
                    if trn is None or not trn.isCircle():
                        continue
                    if t_idx not in windows_by_training:
                        continue
                    if ctrl and getattr(va, "noyc", False):
                        continue

                    episodes = trj.reward_return_probability_episodes_for_training(
                        trn=trn,
                        outer_radius_mm=(
                            None if legacy_outer_radii else float(outer_radius_mm)
                        ),
                        outer_delta_mm=(
                            float(outer_radius_mm) if legacy_outer_radii else None
                        ),
                        reward_radius_mm=reward_radius_mm,
                        reward_delta_mm=reward_delta_mm,
                        border_width_mm=float(border_width_mm),
                        ctrl=ctrl,
                        debug=False,
                    )
                    if not episodes:
                        continue

                    for ep in episodes:
                        if episode_overlaps_wall_contact(ep, wall_regions):
                            continue
                        event_t = int(ep["stop"]) - 1
                        if not _frame_in_windows(event_t, windows_by_training[t_idx]):
                            continue
                        total += 1
                        if bool(ep.get("returns", False)):
                            returns += 1

                ratio = np.nan if total <= 0 else (float(returns) / float(total))
                if min_trajectories > 0 and total < int(min_trajectories):
                    ratio = np.nan
                if fly_idx == 0:
                    ret_exp[vi, ri] = int(returns)
                    total_exp[vi, ri] = int(total)
                    ratio_exp[vi, ri] = ratio
                elif fly_idx == 1:
                    ret_ctrl[vi, ri] = int(returns)
                    total_ctrl[vi, ri] = int(total)
                    ratio_ctrl[vi, ri] = ratio

            _dbg_if(
                debug,
                (
                    f"[return-prob-outer-radius] {vid}: outer_radius={float(outer_radius_mm):g} "
                    f"exp={ret_exp[vi, ri]}/{total_exp[vi, ri]}"
                    + (
                        ""
                        if getattr(va, "noyc", False)
                        else f" ctrl={ret_ctrl[vi, ri]}/{total_ctrl[vi, ri]}"
                    )
                ),
            )

    return ratio_exp, ratio_ctrl, ret_exp, ret_ctrl, total_exp, total_ctrl, windows_meta


def export_return_prob_outer_radius_sli_bundle(vas, opts, gls, out_fn):
    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return

    outer_radii_mm, legacy_outer_radii = _outer_radii_mm(opts)
    selected_trainings = _selected_training_indices(vas_ok, opts)
    skip_first, keep_first, last_sync_buckets = _effective_windowing(opts)
    reward_radius_mm = getattr(opts, "return_prob_reward_radius_mm", None)
    reward_delta_mm = getattr(opts, "return_prob_reward_delta_mm", None)
    if reward_radius_mm is not None:
        reward_radius_mm = float(reward_radius_mm)
    elif reward_delta_mm is None:
        reward_delta_mm = 0.0
    if reward_delta_mm is not None:
        reward_delta_mm = float(reward_delta_mm)
    border_width_mm = float(
        getattr(opts, "return_prob_border_width_mm", 0.1) or 0.1
    )
    debug = bool(getattr(opts, "return_prob_outer_radius_debug", False))
    exclude_wall_contact = bool(
        getattr(opts, "return_prob_outer_radius_exclude_wall_contact", False)
    )
    min_trajectories = min_between_reward_sync_bucket_trajectories(opts)

    group_label = _safe_group_label(opts, gls)
    _dbg(opts, f"[return-prob-outer-radius] out={out_fn}")
    _dbg(
        opts,
        (
            f"[return-prob-outer-radius] group={group_label!r} "
            f"trainings={[t + 1 for t in selected_trainings]} "
            f"skip_first={skip_first} keep_first={keep_first} "
            f"last_sync_buckets={last_sync_buckets} "
            f"outer_radii_mm={outer_radii_mm.tolist()}"
            f"{' (legacy deltas)' if legacy_outer_radii else ''}"
        ),
    )

    (
        ratio_exp,
        ratio_ctrl,
        ret_exp,
        ret_ctrl,
        total_exp,
        total_ctrl,
        windows_meta,
    ) = _compute_return_prob_curves(
        vas_ok,
        outer_radii_mm=outer_radii_mm,
        legacy_outer_radii=legacy_outer_radii,
        reward_radius_mm=reward_radius_mm,
        reward_delta_mm=reward_delta_mm,
        border_width_mm=border_width_mm,
        selected_trainings=selected_trainings,
        skip_first=skip_first,
        keep_first=keep_first,
        last_sync_buckets=last_sync_buckets,
        min_trajectories=min_trajectories,
        debug=debug,
        exclude_wall_contact=exclude_wall_contact,
    )

    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as e:
        print(
            "[export] WARNING: failed to compute SLI for return-prob-outer-radius "
            f"bundle: {e}"
        )
        sli = np.full((len(vas_ok),), np.nan, dtype=float)
        sli_ts = np.full((len(vas_ok), 0, 0), np.nan, dtype=float)

    target_sync_bucket_eligible = exp_target_sync_bucket_eligibility_mask(vas_ok, opts)
    ratio_exp = mask_by_exp_target_sync_bucket_filter(ratio_exp, target_sync_bucket_eligible)
    sli = mask_by_exp_target_sync_bucket_filter(sli, target_sync_bucket_eligible)
    sli_ts = mask_by_exp_target_sync_bucket_filter(sli_ts, target_sync_bucket_eligible)

    try:
        training_names = np.array(
            [vas_ok[0].trns[t_idx].name() for t_idx in selected_trainings],
            dtype=object,
        )
    except Exception:
        training_names = np.array([], dtype=object)

    try:
        video_fns = np.array([getattr(va, "fn", "") for va in vas_ok], dtype=object)
        fly_ids = np.array([int(getattr(va, "f", -1)) for va in vas_ok], dtype=int)
        video_ids = np.array(
            [f"{fn}::f{f}" for fn, f in zip(video_fns, fly_ids)],
            dtype=object,
        )
    except Exception:
        video_fns = np.array([""] * len(vas_ok), dtype=object)
        fly_ids = np.array([-1] * len(vas_ok), dtype=int)
        video_ids = np.array([f"va_{i}" for i in range(len(vas_ok))], dtype=object)

    window_strings = []
    for vi, wins in enumerate(windows_meta):
        if not wins:
            window_strings.append("")
            continue
        parts = []
        for win in wins:
            bucket_ranges = ",".join(
                f"[{int(a)},{int(b)})" for (a, b) in win["bucket_ranges"]
            )
            parts.append(
                f"T{int(win['training_idx']) + 1}:{win['training_name']}:{bucket_ranges}"
            )
        window_strings.append(" | ".join(parts))
        _dbg(
            opts,
            f"[return-prob-outer-radius] {video_ids[vi]} windows={window_strings[-1]}",
        )

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(
        out_fn,
        sli=np.asarray(sli, dtype=float),
        sli_ts=np.asarray(sli_ts, dtype=float),
        fraction_within_radius_outer_radius_ratio_exp=np.asarray(
            ratio_exp, dtype=float
        ),
        fraction_within_radius_outer_radius_ratio_ctrl=np.asarray(
            ratio_ctrl, dtype=float
        ),
        fraction_within_radius_outer_radius_return_exp=np.asarray(
            ret_exp, dtype=int
        ),
        fraction_within_radius_outer_radius_return_ctrl=np.asarray(
            ret_ctrl, dtype=int
        ),
        fraction_within_radius_outer_radius_total_exp=np.asarray(
            total_exp, dtype=int
        ),
        fraction_within_radius_outer_radius_total_ctrl=np.asarray(
            total_ctrl, dtype=int
        ),
        btw_rwd_sync_bucket_min_trajectories=np.array(
            min_trajectories, dtype=int
        ),
        **exp_target_sync_bucket_filter_payload(
            vas_ok,
            opts,
            prefix="exp_target_sync_bucket_filter",
        ),
        fraction_within_radius_outer_radius_outer_radii_mm=np.asarray(
            outer_radii_mm, dtype=float
        ),
        fraction_within_radius_outer_radius_outer_deltas_mm=np.asarray(
            outer_radii_mm if legacy_outer_radii else [], dtype=float
        ),
        fraction_within_radius_outer_radius_trainings=np.asarray(
            np.array(selected_trainings, dtype=int) + 1, dtype=int
        ),
        fraction_within_radius_outer_radius_skip_first_sync_buckets=np.array(
            skip_first, dtype=int
        ),
        fraction_within_radius_outer_radius_keep_first_sync_buckets=np.array(
            keep_first, dtype=int
        ),
        fraction_within_radius_outer_radius_last_sync_buckets=np.array(
            last_sync_buckets, dtype=int
        ),
        fraction_within_radius_outer_radius_reward_radius_mm=np.array(
            np.nan if reward_radius_mm is None else reward_radius_mm, dtype=float
        ),
        fraction_within_radius_outer_radius_reward_delta_mm=np.array(
            np.nan if reward_delta_mm is None else reward_delta_mm, dtype=float
        ),
        fraction_within_radius_outer_radius_border_width_mm=np.array(
            border_width_mm, dtype=float
        ),
        fraction_within_radius_outer_radius_exclude_wall_contact=np.array(
            exclude_wall_contact, dtype=bool
        ),
        fraction_within_radius_outer_radius_window_summary=np.asarray(
            window_strings, dtype=object
        ),
        group_label=np.array(group_label, dtype=object),
        training_names=training_names,
        video_fns=video_fns,
        fly_ids=fly_ids,
        video_ids=video_ids,
        sli_training_idx=np.array(getattr(opts, "best_worst_trn", 1) - 1, dtype=int),
        sli_use_training_mean=np.array(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
    )
    debug_episodes_csv = getattr(
        opts, "return_prob_outer_radius_debug_episodes_csv", None
    )
    if debug_episodes_csv:
        n_debug_rows = _write_return_prob_outer_radius_debug_episodes_csv(
            vas_ok,
            out_csv=str(debug_episodes_csv),
            outer_radii_mm=outer_radii_mm,
            legacy_outer_radii=legacy_outer_radii,
            reward_radius_mm=reward_radius_mm,
            reward_delta_mm=reward_delta_mm,
            border_width_mm=border_width_mm,
            selected_trainings=selected_trainings,
            skip_first=skip_first,
            keep_first=keep_first,
            last_sync_buckets=last_sync_buckets,
            exclude_wall_contact=exclude_wall_contact,
        )
        print(
            "[export] Wrote return-prob-outer-radius debug episodes CSV: "
            f"{debug_episodes_csv} (rows={n_debug_rows})"
        )
    print(
        "[export] Wrote return-prob-outer-radius+SLI bundle: "
        f"{out_fn} (n={len(vas_ok)})"
    )
