from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from src.analysis.episode_filters import (
    EPISODE_TYPE_INNER_EXIT_REENTRY,
    episode_filter_accounting_payload,
    min_episode_count_for_type,
)
from src.exporting.com_sli_bundle import _safe_group_label
from src.exporting.turnback_excursion_bin_sli_bundle import (
    _frame_in_windows,
    _selected_windows_for_va,
)
from src.exporting.wall_contact_episode_filter import (
    episode_overlaps_wall_contact,
    wall_contact_regions_for_trj,
)
from src.utils.parsers import parse_training_selector
from src.utils.util import meanConfInt

Y_LABEL = "Home-vector heading alignment at re-entry"
BASE_TITLE = "Turnback home-vector heading alignment"


def _selected_training_indices(vas, opts) -> list[int]:
    """
    Resolve selected trainings for this metric.

    Defaults to Training 2, matching the common analysis window requested for
    turnback home-vector alignment. Uses this metric's own CLI namespace rather
    than the turnback-excursion-bin namespace.
    """
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if not vas_ok:
        return []

    n_trn = len(getattr(vas_ok[0], "trns", []) or [])
    raw = getattr(opts, "turnback_home_vector_alignment_trainings", None)
    if raw is None or not str(raw).strip():
        raw = "2"

    selected = []
    for idx1 in parse_training_selector(str(raw)):
        idx0 = int(idx1) - 1
        if 0 <= idx0 < n_trn:
            selected.append(idx0)

    selected = sorted(set(selected))
    if selected:
        return selected

    print(
        "[export] WARNING: --turnback-home-vector-alignment-trainings selected no "
        "valid trainings; falling back to all trainings."
    )
    return list(range(n_trn))


def _effective_windowing(opts) -> tuple[int, int, int]:
    """
    Resolve sync-bucket windowing for this metric.

    Defaults to skip first sync bucket and keep the next four, i.e. sync buckets
    2 through 5 for Training 2.
    """
    raw_skip = getattr(
        opts, "turnback_home_vector_alignment_skip_first_sync_buckets", None
    )
    raw_keep = getattr(
        opts, "turnback_home_vector_alignment_keep_first_sync_buckets", None
    )
    raw_last = getattr(opts, "turnback_home_vector_alignment_last_sync_buckets", None)

    skip_first = 1 if raw_skip is None else raw_skip
    keep_first = 4 if raw_keep is None else raw_keep
    last_sync = 0 if raw_last is None else raw_last

    return (
        max(0, int(skip_first or 0)),
        max(0, int(keep_first or 0)),
        max(0, int(last_sync or 0)),
    )


def _sli_subset_label(opts) -> str | None:
    sli_group = getattr(opts, "turnback_home_vector_alignment_sli_group", "all")
    if sli_group == "all":
        return None

    if sli_group == "top":
        frac = getattr(opts, "top_sli_fraction", None)
        if frac is None:
            frac = 0.25
        return f"top {float(frac) * 100:g}% SLI flies"

    if sli_group == "bottom":
        frac = getattr(opts, "bottom_sli_fraction", None)
        if frac is None:
            frac = 0.25
        return f"bottom {float(frac) * 100:g}% SLI flies"

    return None


def _sli_selection_meta(opts) -> dict:
    return {
        "sli_group": getattr(opts, "turnback_home_vector_alignment_sli_group", "all"),
        "top_sli_fraction": getattr(opts, "top_sli_fraction", None),
        "bottom_sli_fraction": getattr(opts, "bottom_sli_fraction", None),
        "sli_selection": {
            "training": getattr(opts, "best_worst_trn", None),
            "use_training_mean": bool(getattr(opts, "sli_use_training_mean", False)),
            "skip_first_sync_buckets": getattr(
                opts, "sli_select_skip_first_sync_buckets", None
            ),
            "keep_first_sync_buckets": getattr(
                opts, "sli_select_keep_first_sync_buckets", None
            ),
            "bucket": getattr(opts, "sli_select_bucket", None),
        },
    }


def _panel_label(
    selected_trainings, skip_first: int, keep_first: int, last_sync_buckets: int
) -> str:
    trn_txt = (
        "Training " + ",".join(str(int(i) + 1) for i in selected_trainings)
        if selected_trainings
        else "Selected trainings"
    )

    if last_sync_buckets:
        sb_txt = f"last {int(last_sync_buckets)} sync buckets"
    elif keep_first:
        first = int(skip_first) + 1
        last = int(skip_first) + int(keep_first)
        sb_txt = f"sync buckets {first}-{last}"
    elif skip_first:
        sb_txt = f"sync buckets after first {int(skip_first)} skipped"
    else:
        sb_txt = "all sync buckets"

    return f"{trn_txt}, {sb_txt}"


def _unit_id_for_va_fly(va, vi: int, fly_idx: int) -> str:
    video_id = os.path.splitext(
        os.path.basename(str(getattr(va, "fn", f"video_{vi}")))
    )[0]
    if getattr(va, "f", None) is not None:
        return f"{video_id}:fly{int(getattr(va, 'f')) + int(getattr(va, 'nef', 0)) * int(fly_idx)}"
    return f"{video_id}:fly{int(fly_idx)}"


def _safe_finite_xy_at(trj, frame_idx: int) -> tuple[float, float] | None:
    if frame_idx < 0:
        return None
    if frame_idx >= len(getattr(trj, "x", [])) or frame_idx >= len(
        getattr(trj, "y", [])
    ):
        return None

    x = float(trj.x[frame_idx])
    y = float(trj.y[frame_idx])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def _nearest_finite_xy(
    trj,
    target_idx: int,
    *,
    lo: int,
    hi: int,
    prefer: str,
) -> tuple[int, float, float] | None:
    """
    Find a finite x/y sample inside [lo, hi], preferring a sample near target_idx.

    prefer="before" searches target, target-1, ... then target+1, ...
    prefer="after" searches target, target+1, ... then target-1, ...

    The asymmetric preference keeps the heading window close to the intended
    pre/post re-entry samples while still surviving short windows and sparse NaNs.
    """
    lo = max(0, int(lo))
    hi = min(int(hi), len(getattr(trj, "x", [])) - 1, len(getattr(trj, "y", [])) - 1)
    if hi < lo:
        return None

    target_idx = min(max(int(target_idx), lo), hi)

    if prefer == "before":
        order = list(range(target_idx, lo - 1, -1)) + list(
            range(target_idx + 1, hi + 1)
        )
    elif prefer == "after":
        order = list(range(target_idx, hi + 1)) + list(
            range(target_idx - 1, lo - 1, -1)
        )
    else:
        raise ValueError(f"unknown finite-sample preference: {prefer!r}")

    seen = set()
    for idx in order:
        if idx in seen:
            continue
        seen.add(idx)
        xy = _safe_finite_xy_at(trj, idx)
        if xy is None:
            continue
        x, y = xy
        return int(idx), float(x), float(y)

    return None


def heading_vector_at_reentry(
    trj,
    event_frame: int,
    *,
    window_radius_frames: int = 2,
    training_start: int | None = None,
    training_stop: int | None = None,
) -> tuple[float, float] | None:
    """
    Estimate heading from local displacement around the re-entry event.

    We intentionally use x/y displacement rather than Trajectory.theta here.
    The default uses net displacement across a small symmetric frame window:
    position_after - position_before.

    NaNs, training bounds, and short windows are handled by choosing the nearest
    finite sample on each side of the requested window. If no nonzero finite
    displacement can be recovered, return None.
    """
    radius = max(0, int(window_radius_frames or 0))
    event_frame = int(event_frame)

    lo_bound = 0 if training_start is None else int(training_start)
    hi_bound = len(getattr(trj, "x", [])) - 1
    hi_bound = min(hi_bound, len(getattr(trj, "y", [])) - 1)
    if training_stop is not None:
        hi_bound = min(hi_bound, int(training_stop) - 1)

    before_target = event_frame - radius
    after_target = event_frame + radius

    before = _nearest_finite_xy(
        trj, before_target, lo=lo_bound, hi=event_frame, prefer="before"
    )
    after = _nearest_finite_xy(
        trj, after_target, lo=event_frame, hi=hi_bound, prefer="after"
    )

    # Fallback for radius=0 or one-sided short windows: use the nearest distinct
    # finite samples around the event. This is deliberately a fallback, not the
    # main estimator.
    if before is None or after is None or before[0] == after[0]:
        before = _nearest_finite_xy(
            trj,
            event_frame - 1,
            lo=lo_bound,
            hi=max(lo_bound, event_frame - 1),
            prefer="before",
        )
        after = _nearest_finite_xy(
            trj,
            event_frame + 1,
            lo=min(hi_bound, event_frame + 1),
            hi=hi_bound,
            prefer="after",
        )

    if before is None or after is None or before[0] == after[0]:
        return None

    _bi, bx, by = before
    _ai, ax, ay = after
    hx = float(ax - bx)
    hy = float(ay - by)
    if not (np.isfinite(hx) and np.isfinite(hy)):
        return None
    if np.hypot(hx, hy) <= 0.0:
        return None
    return hx, hy


def home_vector_from_reentry_to_center(
    *,
    cx: float,
    cy: float,
    radius_px: float,
    x: float,
    y: float,
) -> tuple[float, float] | None:
    """
    Return perimeter_point -> reward_center for the re-entry radial line.

    If re-entry is already on the perimeter, the perimeter point is effectively
    the observed re-entry position. If the point is slightly inside/outside due
    to thresholding, projecting to the perimeter keeps the home vector radial.
    """
    rx = float(x) - float(cx)
    ry = float(y) - float(cy)
    d = float(np.hypot(rx, ry))
    if not np.isfinite(d) or d <= 0.0:
        return None

    perimeter_x = float(cx) + float(radius_px) * rx / d
    perimeter_y = float(cy) + float(radius_px) * ry / d

    hx = float(cx) - perimeter_x
    hy = float(cy) - perimeter_y
    if not (np.isfinite(hx) and np.isfinite(hy)):
        return None
    if np.hypot(hx, hy) <= 0.0:
        return None
    return hx, hy


def cosine_alignment(heading_vec, home_vec) -> float:
    hx, hy = map(float, heading_vec)
    vx, vy = map(float, home_vec)

    hnorm = float(np.hypot(hx, hy))
    vnorm = float(np.hypot(vx, vy))
    if hnorm <= 0.0 or vnorm <= 0.0:
        return float("nan")

    val = (hx * vx + hy * vy) / (hnorm * vnorm)
    if not np.isfinite(val):
        return float("nan")

    # Retain the raw cosine scale [-1, 1], but clip tiny floating point spillover.
    return float(np.clip(val, -1.0, 1.0))


def episode_home_vector_alignment(
    trj, trn, ep: dict, *, window_radius_frames: int = 2
) -> float:
    event_frame = int(ep["stop"]) - 1

    event_xy = _safe_finite_xy_at(trj, event_frame)
    if event_xy is None:
        return float("nan")
    x, y = event_xy

    try:
        cx, cy, reward_radius_px = trn.circles(getattr(trj, "f", 0))[0]
    except Exception:
        return float("nan")

    inner_radius_px = ep.get("inner_radius_px", np.nan)
    if not np.isfinite(inner_radius_px):
        inner_radius_px = ep.get("effective_inner_radius_mm", np.nan)
        px_per_mm = ep.get("px_per_mm", np.nan)
        if np.isfinite(inner_radius_px) and np.isfinite(px_per_mm):
            inner_radius_px = float(inner_radius_px) * float(px_per_mm)

    if not np.isfinite(inner_radius_px):
        inner_radius_px = float(reward_radius_px)

    heading_vec = heading_vector_at_reentry(
        trj,
        event_frame,
        window_radius_frames=window_radius_frames,
        training_start=int(getattr(trn, "start", 0)),
        training_stop=int(getattr(trn, "stop", len(getattr(trj, "x", [])))),
    )
    if heading_vec is None:
        return float("nan")

    home_vec = home_vector_from_reentry_to_center(
        cx=float(cx),
        cy=float(cy),
        radius_px=float(inner_radius_px),
        x=float(x),
        y=float(y),
    )

    if home_vec is None:
        return float("nan")

    return cosine_alignment(heading_vec, home_vec)


def _mean_ci(
    vals: np.ndarray, *, ci_conf: float = 0.95
) -> tuple[float, float, float, int]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m, lo, hi, n = meanConfInt(vals, conf=float(ci_conf))
    return float(m), float(lo), float(hi), int(n)


def _iter_successful_selected_episode_values_for_fly(
    va,
    trj,
    fly_idx: int,
    *,
    selected_trainings: list[int],
    skip_first: int,
    keep_first: int,
    last_sync_buckets: int,
    exclude_wall_contact: bool,
    window_radius_frames: int,
    inner_radius_mm: float | None,
    inner_delta_mm: float | None,
    outer_radius_mm: float | None,
    outer_delta_mm: float | None,
    border_width_mm: float,
    radius_offset_px: float,
):
    windows = _selected_windows_for_va(
        va,
        selected_trainings,
        skip_first=skip_first,
        keep_first=keep_first,
        last_sync_buckets=last_sync_buckets,
    )
    if not windows:
        return

    windows_by_training: dict[int, list[dict]] = {}
    for win in windows:
        windows_by_training.setdefault(int(win["training_idx"]), []).append(win)

    wall_regions = wall_contact_regions_for_trj(
        trj,
        enabled=bool(exclude_wall_contact),
        warned_missing=[False],
        log_tag="turnback-home-vector-alignment",
    )

    for t_idx in selected_trainings:
        if t_idx >= len(getattr(va, "trns", [])):
            continue
        trn = va.trns[t_idx]
        if trn is None or not trn.isCircle():
            continue
        if t_idx not in windows_by_training:
            continue

        episodes = trj.reward_turnback_dual_circle_episodes_for_training(
            trn=trn,
            inner_radius_mm=inner_radius_mm,
            inner_delta_mm=inner_delta_mm,
            outer_radius_mm=outer_radius_mm,
            outer_delta_mm=outer_delta_mm,
            border_width_mm=float(border_width_mm),
            debug=False,
            radius_offset_px=float(radius_offset_px),
        )
        if not episodes:
            continue

        for ep in episodes:
            if not bool(ep.get("turns_back", False)):
                continue
            if ep.get("end_reason", "reenter_inner") != "reenter_inner":
                continue
            if episode_overlaps_wall_contact(ep, wall_regions):
                continue

            event_t = int(ep["stop"]) - 1
            if not _frame_in_windows(event_t, windows_by_training[t_idx]):
                continue

            val = episode_home_vector_alignment(
                trj, trn, ep, window_radius_frames=window_radius_frames
            )
            if np.isfinite(val):
                yield float(val)


def _collect_per_fly_values(
    vas,
    opts,
    *,
    selected_trainings: list[int],
    skip_first,
    keep_first,
    last_sync_buckets: int,
):
    min_episodes = min_episode_count_for_type(opts, EPISODE_TYPE_INNER_EXIT_REENTRY)

    inner_radius_mm = getattr(
        opts, "turnback_home_vector_alignment_inner_radius_mm", None
    )
    inner_delta_mm = getattr(
        opts, "turnback_home_vector_alignment_inner_delta_mm", None
    )
    outer_radius_mm = getattr(
        opts, "turnback_home_vector_alignment_outer_radius_mm", None
    )
    outer_delta_mm = getattr(
        opts, "turnback_home_vector_alignment_outer_delta_mm", None
    )

    # Default to the regular turnback geometry, but only successful inner re-entry
    # episodes contribute values.
    if inner_radius_mm is None:
        inner_radius_mm = getattr(opts, "turnback_inner_radius_mm", None)
    if inner_delta_mm is None:
        inner_delta_mm = getattr(opts, "turnback_inner_delta_mm", None)
    if outer_radius_mm is None:
        outer_radius_mm = getattr(opts, "turnback_outer_radius_mm", None)
    if outer_delta_mm is None:
        outer_delta_mm = getattr(opts, "turnback_outer_delta_mm", None)

    inner_radius_mm = None if inner_radius_mm is None else float(inner_radius_mm)
    inner_delta_mm = 0.0 if inner_delta_mm is None else float(inner_delta_mm)
    outer_radius_mm = None if outer_radius_mm is None else float(outer_radius_mm)
    outer_delta_mm = 2.0 if outer_delta_mm is None else float(outer_delta_mm)

    border_width_mm = float(
        getattr(
            opts,
            "turnback_home_vector_alignment_border_width_mm",
            getattr(opts, "turnback_border_width_mm", 0.1),
        )
        or 0.1
    )
    radius_offset_px = float(
        getattr(
            opts,
            "turnback_home_vector_alignment_inner_radius_offset_px",
            getattr(opts, "turnback_inner_radius_offset_px", 0.0),
        )
        or 0.0
    )
    exclude_wall_contact = bool(
        getattr(opts, "turnback_home_vector_alignment_exclude_wall_contact", False)
    )
    window_radius_frames = max(
        0,
        int(
            getattr(opts, "turnback_home_vector_alignment_window_radius_frames", 2) or 0
        ),
    )

    per_unit_ids = []
    per_unit_values = []
    per_unit_episode_counts = []

    for vi, va in enumerate(vas):
        if getattr(va, "_skipped", False):
            continue

        for fly_idx, trj in enumerate(getattr(va, "trx", [])):
            if fly_idx > 1:
                continue
            if fly_idx == 1 and getattr(va, "noyc", False):
                continue
            if fly_idx != 0:
                # The currently targeted three-group plot is one scalar per experimental fly.
                # Keeping this explicit avoids accidentally mixing yoked controls.
                continue
            if getattr(trj, "_bad", False):
                continue

            vals = np.asarray(
                list(
                    _iter_successful_selected_episode_values_for_fly(
                        va,
                        trj,
                        fly_idx,
                        selected_trainings=selected_trainings,
                        skip_first=skip_first,
                        keep_first=keep_first,
                        last_sync_buckets=last_sync_buckets,
                        exclude_wall_contact=exclude_wall_contact,
                        window_radius_frames=window_radius_frames,
                        inner_radius_mm=inner_radius_mm,
                        inner_delta_mm=inner_delta_mm,
                        outer_radius_mm=outer_radius_mm,
                        outer_delta_mm=outer_delta_mm,
                        border_width_mm=border_width_mm,
                        radius_offset_px=radius_offset_px,
                    )
                ),
                dtype=float,
            )
            vals = vals[np.isfinite(vals)]
            n_ep = int(vals.size)
            if n_ep < int(min_episodes):
                continue

            per_unit_ids.append(_unit_id_for_va_fly(va, vi, fly_idx))
            per_unit_values.append(float(np.mean(vals)))
            per_unit_episode_counts.append(n_ep)

    return (
        np.asarray(per_unit_ids, dtype=object),
        np.asarray(per_unit_values, dtype=float),
        np.asarray(per_unit_episode_counts, dtype=int),
        {
            "inner_radius_mm": (
                np.nan if inner_radius_mm is None else float(inner_radius_mm)
            ),
            "inner_delta_mm": (
                np.nan if inner_delta_mm is None else float(inner_delta_mm)
            ),
            "outer_radius_mm": (
                np.nan if outer_radius_mm is None else float(outer_radius_mm)
            ),
            "outer_delta_mm": (
                np.nan if outer_delta_mm is None else float(outer_delta_mm)
            ),
            "border_width_mm": float(border_width_mm),
            "radius_offset_px": float(radius_offset_px),
            "exclude_wall_contact": bool(exclude_wall_contact),
            "window_radius_frames": int(window_radius_frames),
            "min_turnback_episodes": int(min_episodes),
        },
    )


def export_turnback_home_vector_alignment_sli_bundle(vas, opts, gls, out_fn):
    """
    Export one scalar per experimental fly:
    mean cos(heading_at_reentry - home_vector) over successful re-entry episodes.

    Output intentionally matches the scalar-bar NPZ schema consumed by
    scripts/plot_overlay_training_metric_scalar_bars.py.
    """
    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return

    selected_trainings = _selected_training_indices(vas_ok, opts)
    skip_first, keep_first, last_sync_buckets = _effective_windowing(opts)

    ids, values, episode_counts, metric_meta = _collect_per_fly_values(
        vas_ok,
        opts,
        selected_trainings=selected_trainings,
        skip_first=skip_first,
        keep_first=keep_first,
        last_sync_buckets=last_sync_buckets,
    )

    mean, ci_lo, ci_hi, n_units = _mean_ci(values, ci_conf=0.95)

    meta = {
        "log_tag": "turnback-home-vector-alignment",
        "y_label": Y_LABEL,
        "base_title": BASE_TITLE,
        "pool_trainings": True,
        "subset_label": _sli_subset_label(opts),
        "ci": True,
        "ci_conf": 0.95,
        "skip_first_sync_buckets": int(skip_first),
        "keep_first_sync_buckets": int(keep_first),
        "last_sync_buckets": int(last_sync_buckets),
        "training_selection": {
            "trainings_user": getattr(
                opts, "turnback_home_vector_alignment_trainings", None
            ),
            "trainings_effective": [int(i) + 1 for i in selected_trainings],
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "group_label": _safe_group_label(opts, gls),
        "metric": "turnback_home_vector_alignment",
        "heading_convention": (
            "displacement-derived heading from local x/y trajectory window around "
            "the re-entry event; does not use Trajectory.theta"
        ),
        **_sli_selection_meta(opts),
        **metric_meta,
    }

    panel_label = _panel_label(
        selected_trainings,
        skip_first, keep_first, last_sync_buckets
    )
    subset_label = _sli_subset_label(opts)
    if subset_label:
        panel_label = f"{panel_label}; {subset_label}"

    payload = {
        "panel_labels": np.asarray([panel_label], dtype=object),
        "per_unit_values_panel": np.asarray([values], dtype=object),
        "per_unit_ids_panel": np.asarray([ids], dtype=object),
        "mean": np.asarray([mean], dtype=float),
        "ci_lo": np.asarray([ci_lo], dtype=float),
        "ci_hi": np.asarray([ci_hi], dtype=float),
        "n_units_panel": np.asarray([n_units], dtype=int),
        "meta_json": json.dumps(meta, sort_keys=True),
        # Extra diagnostics, ignored by the scalar-bar plotter but useful for audits.
        "turnback_home_vector_alignment_episode_counts": episode_counts,
        "turnback_home_vector_alignment_trainings": np.asarray(
            selected_trainings, dtype=int
        ),
        "turnback_home_vector_alignment_skip_first_sync_buckets": np.array(
            skip_first, dtype=int
        ),
        "turnback_home_vector_alignment_keep_first_sync_buckets": np.array(
            keep_first, dtype=int
        ),
        "turnback_home_vector_alignment_last_sync_buckets": np.array(
            last_sync_buckets, dtype=int
        ),
        "turnback_home_vector_alignment_window_radius_frames": np.array(
            int(metric_meta["window_radius_frames"]), dtype=int
        ),
        "min_turnback_episodes": np.array(
            int(metric_meta["min_turnback_episodes"]), dtype=int
        ),
        **episode_filter_accounting_payload(
            "episode_filter_turnback_home_vector_alignment",
            episode_counts,
            int(metric_meta["min_turnback_episodes"]),
            observed=np.ones_like(episode_counts, dtype=bool),
        ),
    }

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(out_fn, **payload)

    print(
        f"[export] Wrote turnback home-vector alignment scalar bundle: {out_fn} "
        f"(n={int(n_units)})"
    )
