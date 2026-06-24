from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
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
HEADING_ESTIMATOR_ONE_POINT = "one_point"
HEADING_ESTIMATOR_REENTRY_MEAN = "reentry_mean"
HEADING_ESTIMATOR_ENDPOINT = "endpoint"
HEADING_ESTIMATOR_CHOICES = (
    HEADING_ESTIMATOR_MEAN,
    HEADING_ESTIMATOR_ONE_POINT,
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


def _resolve_max_interpolated_heading_frames(opts) -> int | None:
    raw = getattr(
        opts,
        "turnback_home_vector_alignment_max_interpolated_heading_frames",
        0,
    )
    if raw is None:
        raw = 0
    raw = int(raw)
    return None if raw < 0 else max(0, raw)


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


def _display_fly_id_for_va_fly(va, fly_idx: int) -> int:
    if getattr(va, "f", None) is not None:
        try:
            return int(getattr(va, "f"))
        except (TypeError, ValueError):
            pass
    return int(fly_idx)


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


def _heading_components_at_reentry(
    trj,
    event_frame: int,
    *,
    window_radius_frames: int = 2,
    estimator: str = HEADING_ESTIMATOR_MEAN,
    max_interpolated_frames: int | None = None,
    training_start: int | None = None,
    training_stop: int | None = None,
) -> dict | None:
    radius = max(0, int(window_radius_frames or 0))
    event_frame = int(event_frame)
    estimator = str(estimator or HEADING_ESTIMATOR_MEAN)
    if estimator == HEADING_ESTIMATOR_ONE_POINT:
        radius = 1

    if estimator == HEADING_ESTIMATOR_ENDPOINT or radius <= 0:
        lo_bound = 0 if training_start is None else int(training_start)
        hi_bound = len(getattr(trj, "x", [])) - 1
        hi_bound = min(hi_bound, len(getattr(trj, "y", [])) - 1)
        if training_stop is not None:
            hi_bound = min(hi_bound, int(training_stop) - 1)

        before = _nearest_finite_xy(
            trj,
            event_frame - radius,
            lo=lo_bound,
            hi=event_frame,
            prefer="before",
        )
        after = _nearest_finite_xy(
            trj,
            event_frame + radius,
            lo=event_frame,
            hi=hi_bound,
            prefer="after",
        )
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
        before_frames = np.asarray([before[0]], dtype=int)
        after_frames = np.asarray([after[0]], dtype=int)
        before_xy = np.asarray([before[1], before[2]], dtype=float)
        after_xy = np.asarray([after[1], after[2]], dtype=float)
    else:
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
        if estimator in (HEADING_ESTIMATOR_MEAN, HEADING_ESTIMATOR_ONE_POINT):
            after_lo = event_frame
            after_hi = event_frame + radius - 1
        elif estimator == HEADING_ESTIMATOR_REENTRY_MEAN:
            after_lo = event_frame + 1
            after_hi = event_frame + radius
        else:
            raise ValueError(f"unknown heading estimator: {estimator!r}")
        after_frames, after = _finite_xy_frame_samples(
            trj,
            lo=max(lo_bound, after_lo),
            hi=min(hi_bound, after_hi),
        )
        if before.size == 0 or after.size == 0:
            return _heading_components_at_reentry(
                trj,
                event_frame,
                window_radius_frames=radius,
                estimator=HEADING_ESTIMATOR_ENDPOINT,
                max_interpolated_frames=max_interpolated_frames,
                training_start=training_start,
                training_stop=training_stop,
            )
        before_xy = np.mean(before, axis=0)
        after_xy = np.mean(after, axis=0)

    sampled_frames = np.concatenate([before_frames, after_frames])
    if not _within_interpolated_heading_budget(
        trj, sampled_frames, max_interpolated_frames
    ):
        return None
    vector = np.asarray(after_xy - before_xy, dtype=float)
    if not np.all(np.isfinite(vector)) or np.hypot(vector[0], vector[1]) <= 0.0:
        return None
    return {
        "estimator": estimator,
        "before_frames": before_frames,
        "after_frames": after_frames,
        "sampled_frames": sampled_frames,
        "before_xy": before_xy,
        "after_xy": after_xy,
        "vector": vector,
        "interpolated_heading_frames": _count_interpolated_heading_frames(
            trj, sampled_frames
        ),
    }


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
    if estimator == HEADING_ESTIMATOR_ONE_POINT:
        return _mean_heading_vector_at_reentry(
            trj,
            event_frame,
            window_radius_frames=1,
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


def _turnback_event_frame(ep: dict) -> int:
    """
    Resolve the outcome frame for turnback-home-vector alignment.

    The dual-circle episode generator currently records successful re-entry
    episodes with stop equal to the first frame inside the inner circle. The
    generic event_frame field and stop - 1 therefore point to the final outside
    frame for this outcome.
    """
    if ep.get("end_reason", "reenter_inner") == "reenter_inner":
        return int(ep["stop"])
    return int(ep.get("event_frame", int(ep["stop"]) - 1))


def episode_home_vector_alignment(
    trj,
    trn,
    ep: dict,
    *,
    window_radius_frames: int = 2,
    heading_estimator: str = HEADING_ESTIMATOR_MEAN,
    max_interpolated_heading_frames: int | None = None,
) -> float:
    event_frame = _turnback_event_frame(ep)

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


def episode_home_vector_alignment_components(
    trj,
    trn,
    ep: dict,
    *,
    window_radius_frames: int = 2,
    max_interpolated_heading_frames: int | None = None,
) -> dict:
    event_frame = _turnback_event_frame(ep)
    event_xy = _safe_finite_xy_at(trj, event_frame)
    if event_xy is None:
        return {}
    x, y = event_xy

    try:
        cx, cy, _reward_radius_px = trn.circles(getattr(trj, "f", 0))[0]
    except Exception:
        return {}
    home_vec = home_vector_from_reentry_to_center(
        cx=float(cx),
        cy=float(cy),
        x=float(x),
        y=float(y),
    )
    if home_vec is None:
        return {}

    out = {}
    for estimator in HEADING_ESTIMATOR_CHOICES:
        comp = _heading_components_at_reentry(
            trj,
            event_frame,
            window_radius_frames=window_radius_frames,
            estimator=estimator,
            max_interpolated_frames=max_interpolated_heading_frames,
            training_start=int(getattr(trn, "start", 0)),
            training_stop=int(getattr(trn, "stop", len(getattr(trj, "x", [])))),
        )
        if comp is None:
            continue
        val = cosine_alignment(comp["vector"], home_vec)
        if not np.isfinite(val):
            continue
        comp = dict(comp)
        comp["alignment"] = float(val)
        out[estimator] = comp
    return out


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

            event_t = _turnback_event_frame(ep)
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
    max_interpolated_heading_frames = _resolve_max_interpolated_heading_frames(opts)

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


_EXAMPLE_FIELDS = [
    "rank",
    "group",
    "video",
    "fly",
    "training",
    "episode_start",
    "event_frame",
    "episode_stop",
    "inner_boundary_px",
    "pre_reentry_frame",
    "pre_reentry_distance_px",
    "pre_reentry_margin_px",
    "event_distance_px",
    "event_margin_px",
    "alignment_endpoint",
    "alignment_one_point",
    "alignment_reentry_mean",
    "alignment_border_mean",
    "delta_border_minus_endpoint",
    "delta_border_minus_one_point",
    "delta_border_minus_reentry_mean",
    "rank_mode",
    "rank_score",
    "random_seed",
    "endpoint_frames",
    "one_point_frames",
    "reentry_mean_frames",
    "border_mean_frames",
    "endpoint_before_xy",
    "endpoint_after_xy",
    "endpoint_vector",
    "one_point_before_xy",
    "one_point_after_xy",
    "one_point_vector",
    "reentry_mean_before_xy",
    "reentry_mean_after_xy",
    "reentry_mean_vector",
    "reentry_mean_before_crosses_inner_boundary",
    "reentry_mean_after_crosses_inner_boundary",
    "reentry_mean_any_side_crosses_inner_boundary",
    "border_mean_before_xy",
    "border_mean_after_xy",
    "border_mean_vector",
    "border_mean_before_crosses_inner_boundary",
    "border_mean_after_crosses_inner_boundary",
    "border_mean_any_side_crosses_inner_boundary",
    "image",
]

_BOUNDARY_CROSSING_ESTIMATORS = (
    HEADING_ESTIMATOR_REENTRY_MEAN,
    HEADING_ESTIMATOR_MEAN,
)

_BOUNDARY_CROSSING_SUMMARY_FIELDS = [
    "group",
    "estimator",
    "side",
    "n_episodes",
    "n_crossing",
    "fraction_crossing",
]


def _frame_list_str(frames) -> str:
    return ";".join(str(int(f)) for f in np.asarray(frames, dtype=int).reshape(-1))


def _xy_list_str(vals) -> str:
    arr = np.asarray(vals, dtype=float).reshape(-1)
    return ";".join(f"{float(v):.6g}" for v in arr)


def _bool_csv(val: bool) -> str:
    return "1" if bool(val) else "0"


def _example_inner_boundary_diagnostics(trj, trn, ep: dict, event_frame: int) -> dict:
    try:
        cx = float(ep.get("reward_cx_px", np.nan))
        cy = float(ep.get("reward_cy_px", np.nan))
        if not (np.isfinite(cx) and np.isfinite(cy)):
            cx, cy, _reward_r_px = trn.circles(getattr(trj, "f", 0))[0]
            cx = float(cx)
            cy = float(cy)
    except Exception:
        cx = cy = np.nan

    inner_radius_px = float(ep.get("inner_radius_px", np.nan))
    inner_border_px = float(ep.get("inner_border_px", 0.0) or 0.0)
    inner_boundary_px = inner_radius_px + max(0.0, inner_border_px)

    def _distance_at(frame: int) -> float:
        xy = _safe_finite_xy_at(trj, int(frame))
        if xy is None or not (np.isfinite(cx) and np.isfinite(cy)):
            return float("nan")
        x, y = xy
        return float(np.hypot(float(x) - cx, float(y) - cy))

    prev_frame = int(event_frame) - 1
    prev_dist = _distance_at(prev_frame)
    event_dist = _distance_at(int(event_frame))
    return {
        "inner_boundary_px": float(inner_boundary_px),
        "pre_reentry_frame": int(prev_frame),
        "pre_reentry_distance_px": float(prev_dist),
        "pre_reentry_margin_px": (
            float(prev_dist - inner_boundary_px)
            if np.isfinite(prev_dist) and np.isfinite(inner_boundary_px)
            else float("nan")
        ),
        "event_distance_px": float(event_dist),
        "event_margin_px": (
            float(event_dist - inner_boundary_px)
            if np.isfinite(event_dist) and np.isfinite(inner_boundary_px)
            else float("nan")
        ),
    }


def _frame_boundary_margins(
    trj,
    frames,
    *,
    cx: float,
    cy: float,
    inner_boundary_px: float,
) -> np.ndarray:
    margins = []
    if not (
        np.isfinite(cx) and np.isfinite(cy) and np.isfinite(inner_boundary_px)
    ):
        return np.asarray([], dtype=float)

    for frame in np.asarray(frames, dtype=int).reshape(-1):
        xy = _safe_finite_xy_at(trj, int(frame))
        if xy is None:
            continue
        x, y = xy
        dist = float(np.hypot(float(x) - float(cx), float(y) - float(cy)))
        if np.isfinite(dist):
            margins.append(dist - float(inner_boundary_px))
    return np.asarray(margins, dtype=float)


def _sample_set_crosses_inner_boundary(
    trj,
    frames,
    *,
    cx: float,
    cy: float,
    inner_boundary_px: float,
) -> bool:
    margins = _frame_boundary_margins(
        trj, frames, cx=cx, cy=cy, inner_boundary_px=inner_boundary_px
    )
    margins = margins[np.isfinite(margins)]
    if margins.size < 2:
        return False
    return bool(np.nanmin(margins) <= 0.0 and np.nanmax(margins) > 0.0)


def _component_boundary_crossing_diagnostics(
    trj,
    trn,
    ep: dict,
    comps: dict,
) -> dict:
    try:
        cx = float(ep.get("reward_cx_px", np.nan))
        cy = float(ep.get("reward_cy_px", np.nan))
        if not (np.isfinite(cx) and np.isfinite(cy)):
            cx, cy, _reward_r_px = trn.circles(getattr(trj, "f", 0))[0]
            cx = float(cx)
            cy = float(cy)
    except Exception:
        cx = cy = np.nan

    inner_radius_px = float(ep.get("inner_radius_px", np.nan))
    inner_border_px = float(ep.get("inner_border_px", 0.0) or 0.0)
    inner_boundary_px = inner_radius_px + max(0.0, inner_border_px)

    out = {}
    name_by_estimator = {
        HEADING_ESTIMATOR_REENTRY_MEAN: "reentry_mean",
        HEADING_ESTIMATOR_MEAN: "border_mean",
    }
    for estimator in _BOUNDARY_CROSSING_ESTIMATORS:
        comp = comps.get(estimator)
        if comp is None:
            continue
        name = name_by_estimator[estimator]
        before = _sample_set_crosses_inner_boundary(
            trj,
            comp["before_frames"],
            cx=cx,
            cy=cy,
            inner_boundary_px=inner_boundary_px,
        )
        after = _sample_set_crosses_inner_boundary(
            trj,
            comp["after_frames"],
            cx=cx,
            cy=cy,
            inner_boundary_px=inner_boundary_px,
        )
        out[f"{name}_before_crosses_inner_boundary"] = before
        out[f"{name}_after_crosses_inner_boundary"] = after
        out[f"{name}_any_side_crosses_inner_boundary"] = bool(before or after)
    return out


def _boundary_crossing_summary_rows(records, *, group_label: str) -> list[dict]:
    name_by_estimator = {
        HEADING_ESTIMATOR_REENTRY_MEAN: "reentry_mean",
        HEADING_ESTIMATOR_MEAN: "border_mean",
    }
    rows = []
    for estimator in _BOUNDARY_CROSSING_ESTIMATORS:
        name = name_by_estimator[estimator]
        for side in ("before", "after", "any_side"):
            key = f"{name}_{side}_crosses_inner_boundary"
            vals = [bool(rec.get(key, False)) for rec in records]
            n = len(vals)
            n_crossing = int(sum(vals))
            rows.append(
                {
                    "group": group_label,
                    "estimator": name,
                    "side": side,
                    "n_episodes": n,
                    "n_crossing": n_crossing,
                    "fraction_crossing": (
                        f"{(n_crossing / n):.6g}" if n else "nan"
                    ),
                }
            )
    return rows


def _example_rank_mode(opts) -> str:
    mode = str(
        getattr(
            opts,
            "turnback_home_vector_alignment_examples_rank_mode",
            "border_minus_endpoint",
        )
        or "border_minus_endpoint"
    )
    valid = {
        "border_minus_endpoint",
        "abs_border_minus_endpoint",
        "border_minus_one_point",
        "abs_border_minus_one_point",
        "border_minus_reentry_mean",
        "abs_border_minus_reentry_mean",
        "random",
    }
    if mode not in valid:
        raise ValueError(
            "turnback_home_vector_alignment_examples_rank_mode must be one of "
            f"{sorted(valid)!r}; got {mode!r}"
        )
    return mode


def _example_random_seed(opts) -> int:
    return int(
        getattr(opts, "turnback_home_vector_alignment_examples_random_seed", 1) or 1
    )


def _example_rank_score(record: dict, rank_mode: str) -> float:
    if rank_mode == "border_minus_endpoint":
        return float(record["delta_border_minus_endpoint"])
    if rank_mode == "abs_border_minus_endpoint":
        return abs(float(record["delta_border_minus_endpoint"]))
    if rank_mode == "border_minus_one_point":
        return float(record["delta_border_minus_one_point"])
    if rank_mode == "abs_border_minus_one_point":
        return abs(float(record["delta_border_minus_one_point"]))
    if rank_mode == "border_minus_reentry_mean":
        return float(record["delta_border_minus_reentry_mean"])
    if rank_mode == "abs_border_minus_reentry_mean":
        return abs(float(record["delta_border_minus_reentry_mean"]))
    if rank_mode == "random":
        return float(record["random_score"])
    raise ValueError(f"unknown ranking mode: {rank_mode!r}")


def _draw_vector(ax, origin, vec, *, color: str, label: str, scale: float) -> None:
    vx, vy = np.asarray(vec, dtype=float)
    norm = float(np.hypot(vx, vy))
    if norm <= 0.0 or not np.isfinite(norm):
        return
    dx = float(vx / norm * scale)
    dy = float(vy / norm * scale)
    ax.annotate(
        "",
        xy=(float(origin[0]) + dx, float(origin[1]) + dy),
        xytext=(float(origin[0]), float(origin[1])),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=2.3,
            mutation_scale=15,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )


def _draw_path_time_arrow(ax, xs, ys, finite_mask) -> None:
    finite_mask = np.asarray(finite_mask, dtype=bool)
    adjacent_starts = np.flatnonzero(finite_mask[:-1] & finite_mask[1:])
    if adjacent_starts.size == 0:
        return

    best = None
    best_dist = -np.inf
    preferred = len(xs) * 0.45
    for i0 in adjacent_starts:
        i0 = int(i0)
        i1 = i0 + 1
        dx = float(xs[i1] - xs[i0])
        dy = float(ys[i1] - ys[i0])
        dist = float(np.hypot(dx, dy))
        if not np.isfinite(dist) or dist <= 0.0:
            continue
        # Prefer a clearly visible adjacent-frame step near the middle of the
        # plotted episode, while never connecting across a missing-frame gap.
        center_penalty = abs((i0 + i1) * 0.5 - preferred) * 0.01
        score = dist - center_penalty
        if score > best_dist:
            best = (i0, i1)
            best_dist = score

    if best is None:
        return

    i0, i1 = best
    ax.annotate(
        "",
        xy=(float(xs[i1]), float(ys[i1])),
        xytext=(float(xs[i0]), float(ys[i0])),
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=2.0,
            mutation_scale=13,
            shrinkA=1,
            shrinkB=1,
        ),
        zorder=4,
    )


def _choose_text_box_location(ax, points: list[tuple[float, float]]):
    finite_points = np.asarray(
        [
            (float(x), float(y))
            for x, y in points
            if np.isfinite(float(x)) and np.isfinite(float(y))
        ],
        dtype=float,
    )
    candidates = [
        {
            "x": 0.02,
            "y": 0.98,
            "ha": "left",
            "va": "top",
            "box": (0.00, 0.52, 0.68, 1.00),
        },
        {
            "x": 0.98,
            "y": 0.98,
            "ha": "right",
            "va": "top",
            "box": (0.48, 1.00, 0.68, 1.00),
        },
        {
            "x": 0.02,
            "y": 0.02,
            "ha": "left",
            "va": "bottom",
            "box": (0.00, 0.52, 0.00, 0.32),
        },
        {
            "x": 0.98,
            "y": 0.02,
            "ha": "right",
            "va": "bottom",
            "box": (0.48, 1.00, 0.00, 0.32),
        },
    ]

    if finite_points.size == 0:
        return candidates[0]

    axes_points = ax.transAxes.inverted().transform(
        ax.transData.transform(finite_points)
    )
    axes_points = axes_points[
        (axes_points[:, 0] >= 0.0)
        & (axes_points[:, 0] <= 1.0)
        & (axes_points[:, 1] >= 0.0)
        & (axes_points[:, 1] <= 1.0)
    ]
    if axes_points.size == 0:
        return candidates[0]

    best = None
    best_score = None
    for cand in candidates:
        x0, x1, y0, y1 = cand["box"]
        in_box = (
            (axes_points[:, 0] >= x0)
            & (axes_points[:, 0] <= x1)
            & (axes_points[:, 1] >= y0)
            & (axes_points[:, 1] <= y1)
        )
        center = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=float)
        min_dist = float(np.min(np.hypot(*(axes_points - center).T)))
        score = (int(np.count_nonzero(in_box)), -min_dist)
        if best_score is None or score < best_score:
            best = cand
            best_score = score
    return best or candidates[0]


def _plot_home_vector_alignment_example(
    record: dict,
    *,
    out_path: str,
    image_format: str,
) -> None:
    trj = record["trj"]
    trn = record["trn"]
    ep = record["episode"]
    components = record["components"]
    event_frame = int(record["event_frame"])
    start = max(0, int(ep["start"]) - 5)
    stop = min(len(getattr(trj, "x", [])), int(ep["stop"]) + 6)

    xs = np.asarray(trj.x[start:stop], dtype=float)
    ys = np.asarray(trj.y[start:stop], dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.plot(xs[finite], ys[finite], color="0.35", lw=1.2, zorder=2)
    ax.scatter(xs[finite], ys[finite], c="0.20", s=12, zorder=3)
    _draw_path_time_arrow(ax, xs, ys, finite)

    try:
        cx, cy, reward_r_px = trn.circles(getattr(trj, "f", 0))[0]
        cx = float(cx)
        cy = float(cy)
        reward_r_px = float(reward_r_px)
    except Exception:
        cx = cy = reward_r_px = np.nan
    inner_radius_px = float(ep.get("inner_radius_px", np.nan))
    outer_radius_px = float(ep.get("outer_radius_px", np.nan))
    inner_border_px = float(ep.get("inner_border_px", 0.0) or 0.0)
    outer_border_px = float(ep.get("outer_border_px", 0.0) or 0.0)
    inner_boundary_px = inner_radius_px + max(0.0, inner_border_px)
    outer_boundary_px = outer_radius_px + max(0.0, outer_border_px)

    if np.isfinite(cx) and np.isfinite(cy):
        for radius, color, label, linestyle in (
            (reward_r_px, "#888888", "reward circle", "--"),
            (
                inner_boundary_px,
                "#4c78a8",
                "inner crossing boundary",
                "-",
            ),
            (outer_boundary_px, "#c7a252", "outer circle", ":"),
        ):
            if np.isfinite(radius) and radius > 0.0:
                ax.add_patch(
                    Circle(
                        (cx, cy),
                        radius,
                        fill=False,
                        color=color,
                        lw=1.2,
                        linestyle=linestyle,
                        zorder=1,
                    )
                )
        if (
            np.isfinite(inner_radius_px)
            and inner_radius_px > 0.0
            and np.isfinite(inner_boundary_px)
            and inner_boundary_px > inner_radius_px
        ):
            ax.add_patch(
                Circle(
                    (cx, cy),
                    inner_radius_px,
                    fill=False,
                    color="#4c78a8",
                    lw=0.9,
                    linestyle="--",
                    alpha=0.35,
                    zorder=1,
                )
            )

    event_xy = _safe_finite_xy_at(trj, event_frame)
    if event_xy is not None:
        ax.scatter(
            [event_xy[0]],
            [event_xy[1]],
            c="#d62728",
            s=45,
            marker="x",
            zorder=7,
            label="re-entry frame",
        )

    colors = {
        HEADING_ESTIMATOR_ENDPOINT: "#7f7f7f",
        HEADING_ESTIMATOR_ONE_POINT: "#17becf",
        HEADING_ESTIMATOR_REENTRY_MEAN: "#9467bd",
        HEADING_ESTIMATOR_MEAN: "#2ca02c",
    }
    labels = {
        HEADING_ESTIMATOR_ENDPOINT: "endpoint",
        HEADING_ESTIMATOR_ONE_POINT: "one-point",
        HEADING_ESTIMATOR_REENTRY_MEAN: "re-entry mean",
        HEADING_ESTIMATOR_MEAN: "border mean",
    }
    if event_xy is not None:
        scale = max(
            float(np.nanmax(xs[finite]) - np.nanmin(xs[finite])),
            float(np.nanmax(ys[finite]) - np.nanmin(ys[finite])),
            1.0,
        ) * 0.18
        _draw_vector(
            ax,
            event_xy,
            np.asarray([cx - event_xy[0], cy - event_xy[1]], dtype=float),
            color="#d62728",
            label="home vector",
            scale=scale,
        )
        for estimator in (
            HEADING_ESTIMATOR_ENDPOINT,
            HEADING_ESTIMATOR_ONE_POINT,
            HEADING_ESTIMATOR_REENTRY_MEAN,
            HEADING_ESTIMATOR_MEAN,
        ):
            comp = components.get(estimator)
            if comp is None:
                continue
            _draw_vector(
                ax,
                event_xy,
                comp["vector"],
                color=colors[estimator],
                label=labels[estimator],
                scale=scale,
            )
            for frame_kind, marker, alpha in (
                ("before_frames", "o", 0.75),
                ("after_frames", "s", 0.75),
            ):
                frames = np.asarray(comp[frame_kind], dtype=int)
                pts = []
                for frame in frames:
                    xy = _safe_finite_xy_at(trj, int(frame))
                    if xy is not None:
                        pts.append(xy)
                if pts:
                    arr = np.asarray(pts, dtype=float)
                    ax.scatter(
                        arr[:, 0],
                        arr[:, 1],
                        s=26,
                        marker=marker,
                        facecolors="none",
                        edgecolors=colors[estimator],
                        alpha=alpha,
                        zorder=5,
                    )

    for estimator in (
        HEADING_ESTIMATOR_ENDPOINT,
        HEADING_ESTIMATOR_ONE_POINT,
        HEADING_ESTIMATOR_REENTRY_MEAN,
        HEADING_ESTIMATOR_MEAN,
    ):
        comp = components.get(estimator)
        if comp is None:
            continue
        before_xy = np.asarray(comp["before_xy"], dtype=float)
        after_xy = np.asarray(comp["after_xy"], dtype=float)
        if np.all(np.isfinite(before_xy)):
            ax.scatter(
                [before_xy[0]],
                [before_xy[1]],
                s=75,
                marker="o",
                facecolors=colors[estimator],
                edgecolors="white",
                linewidths=0.8,
                alpha=0.35,
                zorder=4,
            )
        if np.all(np.isfinite(after_xy)):
            ax.scatter(
                [after_xy[0]],
                [after_xy[1]],
                s=75,
                marker="s",
                facecolors=colors[estimator],
                edgecolors="white",
                linewidths=0.8,
                alpha=0.35,
                zorder=4,
            )

    text_lines = [
        f"rank {int(record['rank'])}",
        f"video: {os.path.basename(str(record['va'].fn))}",
        f"T{int(record['training_idx']) + 1}, frames {int(ep['start'])}-{int(ep['stop'])}",
        f"endpoint: {record['alignment_endpoint']:.3f}",
        f"one-point: {record['alignment_one_point']:.3f}",
        f"re-entry mean: {record['alignment_reentry_mean']:.3f}",
        f"border mean: {record['alignment_border_mean']:.3f}",
        f"border - endpoint: {record['delta_border_minus_endpoint']:.3f}",
        f"border - one-point: {record['delta_border_minus_one_point']:.3f}",
        f"border - re-entry: {record['delta_border_minus_reentry_mean']:.3f}",
    ]

    ax.set_aspect("equal", adjustable="box")
    x_min = float(np.nanmin(xs[finite]))
    x_max = float(np.nanmax(xs[finite]))
    y_min = float(np.nanmin(ys[finite]))
    y_max = float(np.nanmax(ys[finite]))
    if event_xy is not None:
        x_min = min(x_min, float(event_xy[0]))
        x_max = max(x_max, float(event_xy[0]))
        y_min = min(y_min, float(event_xy[1]))
        y_max = max(y_max, float(event_xy[1]))
    for comp in components.values():
        for key in ("before_xy", "after_xy"):
            pt = np.asarray(comp[key], dtype=float)
            if np.all(np.isfinite(pt)):
                x_min = min(x_min, float(pt[0]))
                x_max = max(x_max, float(pt[0]))
                y_min = min(y_min, float(pt[1]))
                y_max = max(y_max, float(pt[1]))
    span = max(x_max - x_min, y_max - y_min, 1.0)
    if np.isfinite(inner_radius_px) and inner_radius_px > 0.0:
        span = max(span, inner_radius_px * 0.35)
    span = max(span, 8.0)
    pad = span * 0.25
    center_x = (x_min + x_max) * 0.5
    center_y = (y_min + y_max) * 0.5
    half = span * 0.5 + pad
    ax.set_xlim(center_x - half, center_x + half)
    ax.set_ylim(center_y + half, center_y - half)

    text_avoid_points = [(float(x), float(y)) for x, y in zip(xs[finite], ys[finite])]
    if event_xy is not None:
        text_avoid_points.append((float(event_xy[0]), float(event_xy[1])))
    for comp in components.values():
        for key in ("before_xy", "after_xy"):
            pt = np.asarray(comp[key], dtype=float)
            if np.all(np.isfinite(pt)):
                text_avoid_points.append((float(pt[0]), float(pt[1])))
        for key in ("before_frames", "after_frames"):
            for frame in np.asarray(comp[key], dtype=int).reshape(-1):
                xy = _safe_finite_xy_at(trj, int(frame))
                if xy is not None:
                    text_avoid_points.append((float(xy[0]), float(xy[1])))
    text_loc = _choose_text_box_location(ax, text_avoid_points)
    ax.text(
        text_loc["x"],
        text_loc["y"],
        "\n".join(text_lines),
        transform=ax.transAxes,
        va=text_loc["va"],
        ha=text_loc["ha"],
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.82),
        zorder=8,
    )

    ax.set_title("Turnback home-vector alignment estimator comparison")
    uniq = {
        "re-entry frame": Line2D(
            [], [], color="#d62728", marker="x", linestyle="None", markersize=8
        ),
        "path time": Line2D([], [], color="black", lw=2),
        "home vector": Line2D([], [], color="#d62728", lw=3),
        "endpoint": Line2D([], [], color=colors[HEADING_ESTIMATOR_ENDPOINT], lw=3),
        "one-point": Line2D(
            [], [], color=colors[HEADING_ESTIMATOR_ONE_POINT], lw=3
        ),
        "re-entry mean": Line2D(
            [], [], color=colors[HEADING_ESTIMATOR_REENTRY_MEAN], lw=3
        ),
        "border mean": Line2D([], [], color=colors[HEADING_ESTIMATOR_MEAN], lw=3),
    }
    ax.legend(
        [uniq[k] for k in uniq],
        list(uniq.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=8,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, format=image_format, bbox_inches="tight", dpi=160)
    plt.close(fig)


def export_turnback_home_vector_alignment_examples(vas, opts, gls, out_dir):
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if not vas_ok:
        print("[turnback-home-vector-examples] no non-skipped videos")
        return

    selected_trainings = _selected_training_indices(vas_ok, opts)
    skip_first, keep_first, last_sync_buckets = _effective_windowing(opts)
    min_episodes = min_episode_count_for_type(opts, EPISODE_TYPE_INNER_EXIT_REENTRY)
    max_interpolated_heading_frames = _resolve_max_interpolated_heading_frames(opts)
    window_radius_frames = max(
        0,
        int(
            getattr(opts, "turnback_home_vector_alignment_window_radius_frames", 2) or 0
        ),
    )

    inner_radius_mm = getattr(
        opts, "turnback_home_vector_alignment_inner_radius_mm", None
    )
    outer_radius_mm = getattr(
        opts, "turnback_home_vector_alignment_outer_radius_mm", None
    )
    inner_delta_mm = getattr(
        opts, "turnback_home_vector_alignment_inner_delta_mm", None
    )
    outer_delta_mm = getattr(
        opts, "turnback_home_vector_alignment_outer_delta_mm", None
    )
    inner_radius_mm = None if inner_radius_mm is None else float(inner_radius_mm)
    outer_radius_mm = None if outer_radius_mm is None else float(outer_radius_mm)
    inner_delta_mm = (
        getattr(opts, "turnback_inner_delta_mm", 0.0)
        if inner_delta_mm is None
        else float(inner_delta_mm)
    )
    outer_delta_mm = (
        getattr(opts, "turnback_outer_delta_mm", 2.0)
        if outer_delta_mm is None
        else float(outer_delta_mm)
    )
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
    rank_mode = _example_rank_mode(opts)
    random_seed = _example_random_seed(opts)

    candidates = []
    for vi, va in enumerate(vas_ok):
        windows = _selected_windows_for_va(
            va,
            selected_trainings,
            skip_first=skip_first,
            keep_first=keep_first,
            last_sync_buckets=last_sync_buckets,
        )
        windows_by_training = {}
        for win in windows:
            windows_by_training.setdefault(int(win["training_idx"]), []).append(win)
        for fly_idx, trj in enumerate(getattr(va, "trx", [])):
            if fly_idx != 0 or getattr(trj, "_bad", False):
                continue
            wall_regions = wall_contact_regions_for_trj(
                trj,
                enabled=exclude_wall_contact,
                warned_missing=[False],
                log_tag="turnback-home-vector-examples",
            )
            per_fly = []
            for t_idx in selected_trainings:
                if t_idx >= len(getattr(va, "trns", [])):
                    continue
                trn = va.trns[t_idx]
                if trn is None or not trn.isCircle() or t_idx not in windows_by_training:
                    continue
                episodes = trj.reward_turnback_dual_circle_episodes_for_training(
                    trn=trn,
                    inner_radius_mm=inner_radius_mm,
                    inner_delta_mm=inner_delta_mm,
                    outer_radius_mm=outer_radius_mm,
                    outer_delta_mm=outer_delta_mm,
                    border_width_mm=border_width_mm,
                    debug=False,
                    radius_offset_px=radius_offset_px,
                )
                for ep in episodes or []:
                    if not bool(ep.get("turns_back", False)):
                        continue
                    if ep.get("end_reason", "reenter_inner") != "reenter_inner":
                        continue
                    if episode_overlaps_wall_contact(ep, wall_regions):
                        continue
                    if not _episode_within_windows(ep, windows_by_training[t_idx]):
                        continue
                    event_frame = _turnback_event_frame(ep)
                    if not _frame_in_windows(event_frame, windows_by_training[t_idx]):
                        continue
                    comps = episode_home_vector_alignment_components(
                        trj,
                        trn,
                        ep,
                        window_radius_frames=window_radius_frames,
                        max_interpolated_heading_frames=max_interpolated_heading_frames,
                    )
                    if not all(k in comps for k in HEADING_ESTIMATOR_CHOICES):
                        continue
                    rec = {
                        "va": va,
                        "trj": trj,
                        "trn": trn,
                        "video_index": vi,
                        "fly_idx": fly_idx,
                        "training_idx": t_idx,
                        "episode": ep,
                        "event_frame": event_frame,
                        "components": comps,
                        "alignment_endpoint": comps[HEADING_ESTIMATOR_ENDPOINT][
                            "alignment"
                        ],
                        "alignment_one_point": comps[HEADING_ESTIMATOR_ONE_POINT][
                            "alignment"
                        ],
                        "alignment_reentry_mean": comps[
                            HEADING_ESTIMATOR_REENTRY_MEAN
                        ]["alignment"],
                        "alignment_border_mean": comps[HEADING_ESTIMATOR_MEAN][
                            "alignment"
                        ],
                        **_example_inner_boundary_diagnostics(
                            trj, trn, ep, event_frame
                        ),
                    }
                    rec.update(
                        _component_boundary_crossing_diagnostics(
                            trj, trn, ep, comps
                        )
                    )
                    rec["delta_border_minus_endpoint"] = (
                        rec["alignment_border_mean"] - rec["alignment_endpoint"]
                    )
                    rec["delta_border_minus_one_point"] = (
                        rec["alignment_border_mean"] - rec["alignment_one_point"]
                    )
                    rec["delta_border_minus_reentry_mean"] = (
                        rec["alignment_border_mean"] - rec["alignment_reentry_mean"]
                    )
                    per_fly.append(rec)
            if len(per_fly) >= int(min_episodes):
                candidates.extend(per_fly)

    if rank_mode == "random":
        rng = np.random.default_rng(random_seed)
        for rec, score in zip(candidates, rng.random(len(candidates))):
            rec["random_score"] = float(score)

    for rec in candidates:
        rec["rank_mode"] = rank_mode
        rec["rank_score"] = _example_rank_score(rec, rank_mode)

    candidates.sort(
        key=lambda row: (
            float(row["rank_score"]),
            float(row["delta_border_minus_endpoint"]),
            float(row["delta_border_minus_one_point"]),
            float(row["delta_border_minus_reentry_mean"]),
        ),
        reverse=True,
    )
    limit = max(
        1,
        int(getattr(opts, "turnback_home_vector_alignment_examples_num", 24) or 24),
    )
    max_per_fly = max(
        0,
        int(
            getattr(opts, "turnback_home_vector_alignment_examples_max_per_fly", 4)
            or 0
        ),
    )
    selected = []
    fly_counts = {}
    for rec in candidates:
        unit = (int(rec["video_index"]), int(rec["fly_idx"]))
        if max_per_fly and fly_counts.get(unit, 0) >= max_per_fly:
            continue
        selected.append(rec)
        fly_counts[unit] = fly_counts.get(unit, 0) + 1
        if len(selected) >= limit:
            break

    os.makedirs(out_dir, exist_ok=True)
    image_format = str(getattr(opts, "imageFormat", "png") or "png").lower()
    group_label = _safe_group_label(opts, gls)
    rows = []
    for rank, rec in enumerate(selected, start=1):
        rec["rank"] = rank
        video_base = os.path.splitext(os.path.basename(str(rec["va"].fn)))[0]
        display_fly = _display_fly_id_for_va_fly(rec["va"], int(rec["fly_idx"]))
        filename = (
            f"rank{rank:02d}__{video_base}__fly{display_fly}"
            f"__T{int(rec['training_idx']) + 1}"
            f"__event{int(rec['event_frame'])}.{image_format}"
        )
        out_path = os.path.join(out_dir, filename)
        _plot_home_vector_alignment_example(
            rec,
            out_path=out_path,
            image_format=image_format,
        )
        ep = rec["episode"]
        comps = rec["components"]
        rows.append(
            {
                "rank": rank,
                "group": group_label,
                "video": str(rec["va"].fn),
                "fly": int(display_fly),
                "training": int(rec["training_idx"]) + 1,
                "episode_start": int(ep["start"]),
                "event_frame": int(rec["event_frame"]),
                "episode_stop": int(ep["stop"]),
                "inner_boundary_px": float(rec["inner_boundary_px"]),
                "pre_reentry_frame": int(rec["pre_reentry_frame"]),
                "pre_reentry_distance_px": float(rec["pre_reentry_distance_px"]),
                "pre_reentry_margin_px": float(rec["pre_reentry_margin_px"]),
                "event_distance_px": float(rec["event_distance_px"]),
                "event_margin_px": float(rec["event_margin_px"]),
                "alignment_endpoint": float(rec["alignment_endpoint"]),
                "alignment_one_point": float(rec["alignment_one_point"]),
                "alignment_reentry_mean": float(rec["alignment_reentry_mean"]),
                "alignment_border_mean": float(rec["alignment_border_mean"]),
                "delta_border_minus_endpoint": float(
                    rec["delta_border_minus_endpoint"]
                ),
                "delta_border_minus_one_point": float(
                    rec["delta_border_minus_one_point"]
                ),
                "delta_border_minus_reentry_mean": float(
                    rec["delta_border_minus_reentry_mean"]
                ),
                "rank_mode": str(rec["rank_mode"]),
                "rank_score": float(rec["rank_score"]),
                "random_seed": int(random_seed) if rank_mode == "random" else "",
                "endpoint_frames": _frame_list_str(
                    comps[HEADING_ESTIMATOR_ENDPOINT]["sampled_frames"]
                ),
                "one_point_frames": _frame_list_str(
                    comps[HEADING_ESTIMATOR_ONE_POINT]["sampled_frames"]
                ),
                "reentry_mean_frames": _frame_list_str(
                    comps[HEADING_ESTIMATOR_REENTRY_MEAN]["sampled_frames"]
                ),
                "border_mean_frames": _frame_list_str(
                    comps[HEADING_ESTIMATOR_MEAN]["sampled_frames"]
                ),
                "endpoint_before_xy": _xy_list_str(
                    comps[HEADING_ESTIMATOR_ENDPOINT]["before_xy"]
                ),
                "endpoint_after_xy": _xy_list_str(
                    comps[HEADING_ESTIMATOR_ENDPOINT]["after_xy"]
                ),
                "endpoint_vector": _xy_list_str(
                    comps[HEADING_ESTIMATOR_ENDPOINT]["vector"]
                ),
                "one_point_before_xy": _xy_list_str(
                    comps[HEADING_ESTIMATOR_ONE_POINT]["before_xy"]
                ),
                "one_point_after_xy": _xy_list_str(
                    comps[HEADING_ESTIMATOR_ONE_POINT]["after_xy"]
                ),
                "one_point_vector": _xy_list_str(
                    comps[HEADING_ESTIMATOR_ONE_POINT]["vector"]
                ),
                "reentry_mean_before_xy": _xy_list_str(
                    comps[HEADING_ESTIMATOR_REENTRY_MEAN]["before_xy"]
                ),
                "reentry_mean_after_xy": _xy_list_str(
                    comps[HEADING_ESTIMATOR_REENTRY_MEAN]["after_xy"]
                ),
                "reentry_mean_vector": _xy_list_str(
                    comps[HEADING_ESTIMATOR_REENTRY_MEAN]["vector"]
                ),
                "reentry_mean_before_crosses_inner_boundary": _bool_csv(
                    rec["reentry_mean_before_crosses_inner_boundary"]
                ),
                "reentry_mean_after_crosses_inner_boundary": _bool_csv(
                    rec["reentry_mean_after_crosses_inner_boundary"]
                ),
                "reentry_mean_any_side_crosses_inner_boundary": _bool_csv(
                    rec["reentry_mean_any_side_crosses_inner_boundary"]
                ),
                "border_mean_before_xy": _xy_list_str(
                    comps[HEADING_ESTIMATOR_MEAN]["before_xy"]
                ),
                "border_mean_after_xy": _xy_list_str(
                    comps[HEADING_ESTIMATOR_MEAN]["after_xy"]
                ),
                "border_mean_vector": _xy_list_str(
                    comps[HEADING_ESTIMATOR_MEAN]["vector"]
                ),
                "border_mean_before_crosses_inner_boundary": _bool_csv(
                    rec["border_mean_before_crosses_inner_boundary"]
                ),
                "border_mean_after_crosses_inner_boundary": _bool_csv(
                    rec["border_mean_after_crosses_inner_boundary"]
                ),
                "border_mean_any_side_crosses_inner_boundary": _bool_csv(
                    rec["border_mean_any_side_crosses_inner_boundary"]
                ),
                "image": out_path,
            }
        )

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_EXAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = os.path.join(out_dir, "boundary_crossing_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=_BOUNDARY_CROSSING_SUMMARY_FIELDS
        )
        writer.writeheader()
        writer.writerows(
            _boundary_crossing_summary_rows(candidates, group_label=group_label)
        )
    print(
        "[turnback-home-vector-examples] wrote "
        f"{len(rows)} images, {manifest_path}, and {summary_path}"
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
