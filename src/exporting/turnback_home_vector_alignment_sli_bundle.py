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
HEADING_ESTIMATOR_MEAN = "mean"
HEADING_ESTIMATOR_REENTRY_MEAN = "reentry_mean"
HEADING_ESTIMATOR_ENDPOINT = "endpoint"
HEADING_ESTIMATOR_CHOICES = (
    HEADING_ESTIMATOR_MEAN,
    HEADING_ESTIMATOR_REENTRY_MEAN,
    HEADING_ESTIMATOR_ENDPOINT,
)


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


def _finite_xy_frame_samples(trj, *, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return frame indices and finite x/y samples from inclusive range [lo, hi].
    """
    n = min(len(getattr(trj, "x", [])), len(getattr(trj, "y", [])))
    if n <= 0:
        return np.zeros((0,), dtype=int), np.zeros((0, 2), dtype=float)

    lo = max(0, int(lo))
    hi = min(int(hi), n - 1)
    if hi < lo:
        return np.zeros((0,), dtype=int), np.zeros((0, 2), dtype=float)

    frames = np.arange(lo, hi + 1, dtype=int)
    xs = np.asarray(trj.x[lo : hi + 1], dtype=float)
    ys = np.asarray(trj.y[lo : hi + 1], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(mask):
        return np.zeros((0,), dtype=int), np.zeros((0, 2), dtype=float)
    return frames[mask], np.column_stack([xs[mask], ys[mask]]).astype(
        float, copy=False
    )


def _count_interpolated_heading_frames(trj, frames: np.ndarray) -> int:
    nan_mask = getattr(trj, "nan", None)
    if nan_mask is None:
        return 0

    try:
        arr = np.asarray(nan_mask, dtype=bool)
    except Exception:
        return 0

    if arr.ndim != 1 or arr.size == 0:
        return 0

    idx = np.asarray(frames, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < arr.size)]
    if idx.size == 0:
        return 0
    return int(np.count_nonzero(arr[idx]))


def _within_interpolated_heading_budget(
    trj,
    frames: np.ndarray,
    max_interpolated_frames: int | None,
) -> bool:
    if max_interpolated_frames is None:
        return True
    return _count_interpolated_heading_frames(trj, frames) <= int(
        max_interpolated_frames
    )


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


def _endpoint_heading_vector_at_reentry(
    trj,
    event_frame: int,
    *,
    window_radius_frames: int = 2,
    max_interpolated_frames: int | None = None,
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
    if not _within_interpolated_heading_budget(
        trj, np.asarray([_bi, _ai], dtype=int), max_interpolated_frames
    ):
        return None

    hx = float(ax - bx)
    hy = float(ay - by)
    if not (np.isfinite(hx) and np.isfinite(hy)):
        return None
    if np.hypot(hx, hy) <= 0.0:
        return None
    return hx, hy


def _mean_heading_vector_at_reentry(
    trj,
    event_frame: int,
    *,
    window_radius_frames: int = 2,
    anchor: str = "border",
    max_interpolated_frames: int | None = None,
    training_start: int | None = None,
    training_stop: int | None = None,
) -> tuple[float, float] | None:
    """
    Estimate heading from the displacement between mean pre/post positions.

    anchor="border" uses frames symmetrically around the crossing boundary: for
    radius 2, before=(event-2,event-1) and after=(event,event+1). The legacy
    anchor="reentry" excludes the event frame and uses frames on either side of
    it: before=(event-2,event-1), after=(event+1,event+2).
    """
    radius = max(0, int(window_radius_frames or 0))
    if radius <= 0:
        return _endpoint_heading_vector_at_reentry(
            trj,
            event_frame,
            window_radius_frames=radius,
            max_interpolated_frames=max_interpolated_frames,
            training_start=training_start,
            training_stop=training_stop,
        )

    event_frame = int(event_frame)
    lo_bound = 0 if training_start is None else int(training_start)
    hi_bound = len(getattr(trj, "x", [])) - 1
    hi_bound = min(hi_bound, len(getattr(trj, "y", [])) - 1)
    if training_stop is not None:
        hi_bound = min(hi_bound, int(training_stop) - 1)

    before_frames, before = _finite_xy_frame_samples(
        trj,
        lo=max(lo_bound, event_frame - radius),
        hi=min(hi_bound, event_frame - 1),
    )
    if anchor == "border":
        after_lo = event_frame
        after_hi = event_frame + radius - 1
    elif anchor == "reentry":
        after_lo = event_frame + 1
        after_hi = event_frame + radius
    else:
        raise ValueError(f"unknown heading anchor: {anchor!r}")
    after_frames, after = _finite_xy_frame_samples(
        trj,
        lo=max(lo_bound, after_lo),
        hi=min(hi_bound, after_hi),
    )

    if before.size == 0 or after.size == 0:
        return _endpoint_heading_vector_at_reentry(
            trj,
            event_frame,
            window_radius_frames=radius,
            max_interpolated_frames=max_interpolated_frames,
            training_start=training_start,
            training_stop=training_stop,
        )

    sampled_frames = np.concatenate([before_frames, after_frames])
    if not _within_interpolated_heading_budget(
        trj, sampled_frames, max_interpolated_frames
    ):
        return None

    before_xy = np.mean(before, axis=0)
    after_xy = np.mean(after, axis=0)
    hx = float(after_xy[0] - before_xy[0])
    hy = float(after_xy[1] - before_xy[1])
    if not (np.isfinite(hx) and np.isfinite(hy)):
        return None
    if np.hypot(hx, hy) <= 0.0:
        return None
    return hx, hy


def heading_vector_at_reentry(
    trj,
    event_frame: int,
    *,
    window_radius_frames: int = 2,
    estimator: str = HEADING_ESTIMATOR_MEAN,
    max_interpolated_frames: int | None = None,
    training_start: int | None = None,
    training_stop: int | None = None,
) -> tuple[float, float] | None:
    """
    Estimate heading from local displacement around the re-entry event.

    We intentionally use x/y displacement rather than Trajectory.theta here.
    The default estimator uses mean pre/post positions anchored symmetrically
    around the border crossing. Use estimator="reentry_mean" for the earlier
    event-centered mean, or estimator="endpoint" for the original single-sample
    endpoint displacement across the requested window.
    """
    estimator = str(estimator or HEADING_ESTIMATOR_MEAN)
    if estimator == HEADING_ESTIMATOR_MEAN:
        return _mean_heading_vector_at_reentry(
            trj,
            event_frame,
            window_radius_frames=window_radius_frames,
            anchor="border",
            max_interpolated_frames=max_interpolated_frames,
            training_start=training_start,
            training_stop=training_stop,
        )
    if estimator == HEADING_ESTIMATOR_REENTRY_MEAN:
        return _mean_heading_vector_at_reentry(
            trj,
            event_frame,
            window_radius_frames=window_radius_frames,
            anchor="reentry",
            max_interpolated_frames=max_interpolated_frames,
            training_start=training_start,
            training_stop=training_stop,
        )
    if estimator == HEADING_ESTIMATOR_ENDPOINT:
        return _endpoint_heading_vector_at_reentry(
            trj,
            event_frame,
            window_radius_frames=window_radius_frames,
            max_interpolated_frames=max_interpolated_frames,
            training_start=training_start,
            training_stop=training_stop,
        )
    raise ValueError(f"unknown heading estimator: {estimator!r}")


def home_vector_from_reentry_to_center(
    *,
    cx: float,
    cy: float,
    x: float,
    y: float,
) -> tuple[float, float] | None:
    """
    Return the vector from the fly's re-entry position to the reward center.

    For the concentric turnback regions used here, this has the same direction
    as projecting the re-entry point to the inner-circle perimeter first. Since
    cosine alignment depends only on direction, the direct vector is sufficient.
    """
    hx = float(cx) - float(x)
    hy = float(cy) - float(y)
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
    trj,
    trn,
    ep: dict,
    *,
    window_radius_frames: int = 2,
    heading_estimator: str = HEADING_ESTIMATOR_MEAN,
    max_interpolated_heading_frames: int | None = None,
) -> float:
    event_frame = int(ep["stop"]) - 1

    event_xy = _safe_finite_xy_at(trj, event_frame)
    if event_xy is None:
        return float("nan")
    x, y = event_xy

    try:
        cx, cy, _reward_radius_px = trn.circles(getattr(trj, "f", 0))[0]
    except Exception:
        return float("nan")

    heading_vec = heading_vector_at_reentry(
        trj,
        event_frame,
        window_radius_frames=window_radius_frames,
        estimator=heading_estimator,
        max_interpolated_frames=max_interpolated_heading_frames,
        training_start=int(getattr(trn, "start", 0)),
        training_stop=int(getattr(trn, "stop", len(getattr(trj, "x", [])))),
    )
    if heading_vec is None:
        return float("nan")

    home_vec = home_vector_from_reentry_to_center(
        cx=float(cx),
        cy=float(cy),
        x=float(x),
        y=float(y),
    )

    if home_vec is None:
        return float("nan")

    return cosine_alignment(heading_vec, home_vec)


def _episode_within_windows(ep: dict, windows) -> bool:
    """
    Return True if the entire episode lies inside a selected analysis window.

    This intentionally checks the full selected window span, not individual sync
    buckets, so episodes may cross from one included bucket to another.
    """
    start = int(ep["start"])
    stop = int(ep["stop"])
    for win in windows:
        if int(win["start"]) <= start and stop <= int(win["stop"]):
            return True
    return False


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
    heading_estimator: str,
    max_interpolated_heading_frames: int | None,
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

            if not _episode_within_windows(ep, windows_by_training[t_idx]):
                continue

            event_t = int(ep["stop"]) - 1
            if not _frame_in_windows(event_t, windows_by_training[t_idx]):
                continue

            val = episode_home_vector_alignment(
                trj,
                trn,
                ep,
                window_radius_frames=window_radius_frames,
                heading_estimator=heading_estimator,
                max_interpolated_heading_frames=max_interpolated_heading_frames,
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
    heading_estimator = str(
        getattr(
            opts,
            "turnback_home_vector_alignment_heading_estimator",
            HEADING_ESTIMATOR_MEAN,
        )
        or HEADING_ESTIMATOR_MEAN
    )
    if heading_estimator not in HEADING_ESTIMATOR_CHOICES:
        raise ValueError(
            "turnback_home_vector_alignment_heading_estimator must be one of "
            f"{HEADING_ESTIMATOR_CHOICES!r}; got {heading_estimator!r}"
        )
    raw_max_interp = getattr(
        opts,
        "turnback_home_vector_alignment_max_interpolated_heading_frames",
        0,
    )
    if raw_max_interp is None:
        raw_max_interp = 0
    max_interpolated_heading_frames = (
        None if int(raw_max_interp) < 0 else max(0, int(raw_max_interp))
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
                        heading_estimator=heading_estimator,
                        max_interpolated_heading_frames=max_interpolated_heading_frames,
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
            "heading_estimator": heading_estimator,
            "max_interpolated_heading_frames": (
                -1
                if max_interpolated_heading_frames is None
                else int(max_interpolated_heading_frames)
            ),
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
            "displacement-derived heading from local x/y trajectory samples around "
            "the re-entry event; does not use Trajectory.theta"
        ),
        **_sli_selection_meta(opts),
        **metric_meta,
    }

    panel_label = _panel_label(
        selected_trainings, skip_first, keep_first, last_sync_buckets
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
        "turnback_home_vector_alignment_heading_estimator": np.array(
            str(metric_meta["heading_estimator"]), dtype=object
        ),
        "turnback_home_vector_alignment_max_interpolated_heading_frames": np.array(
            int(metric_meta["max_interpolated_heading_frames"]), dtype=int
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
