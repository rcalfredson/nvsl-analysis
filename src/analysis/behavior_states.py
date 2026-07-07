from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np
from scipy.signal import savgol_filter


class BehaviorState(IntEnum):
    REST = 0
    TURN = 1
    RUN = 2


@dataclass(frozen=True)
class BehaviorStateConfig:
    angular_small_turn_rad_s: float = 80.0 * np.pi / 180.0
    angular_large_turn_rad_s: float = 120.0 * np.pi / 180.0
    turn_angular_source: str = "theta_or_path"
    turn_path_angular_alignment: str = "vertex"
    turn_path_min_speed_mm_s: float = 2.0
    head_body_small_turn_mm_s: float = 1.0
    head_body_large_turn_mm_s: float = 1.5
    run_low_mm_s: float = 1.0
    run_high_mm_s: float = 3.0
    savgol_order: int = 3
    savgol_window: int = 5
    angular_mask_shift_frames: int = 2
    path_angular_mask_shift_frames: int = 0
    head_body_shift_frames: int = 4
    turn_score_threshold: float = 1.8
    head_vertex_axis_fraction: float = 0.5
    turn_min_segments: int = 2


DEFAULT_BEHAVIOR_STATE_CONFIG = BehaviorStateConfig()


def _reference_savgol_filter(
    values: np.ndarray,
    *,
    order: int = DEFAULT_BEHAVIOR_STATE_CONFIG.savgol_order,
    window: int = DEFAULT_BEHAVIOR_STATE_CONFIG.savgol_window,
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if np.count_nonzero(finite) > window:
        out[finite] = savgol_filter(out[finite], window, order)
    return out


def _shift_left_zero_tail(values: np.ndarray, frames: int) -> np.ndarray:
    out = np.roll(values, -frames)
    if frames > 0:
        out[-frames:] = 0
    return out


def behavior_state_config_from_opts(opts) -> BehaviorStateConfig:
    cfg = DEFAULT_BEHAVIOR_STATE_CONFIG
    updates = {}

    angular_source = getattr(opts, "behavior_state_turn_angular_source", None)
    if angular_source is not None:
        source = str(angular_source).strip().lower().replace("-", "_")
        if source not in {"theta", "path", "theta_or_path"}:
            raise ValueError(f"Unknown behavior-state turn angular source: {source}")
        updates["turn_angular_source"] = source

    path_alignment = getattr(opts, "behavior_state_turn_path_angular_alignment", None)
    if path_alignment is not None:
        alignment = str(path_alignment).strip().lower().replace("-", "_")
        if alignment not in {"segment_end", "vertex"}:
            raise ValueError(
                f"Unknown behavior-state path angular alignment: {alignment}"
            )
        updates["turn_path_angular_alignment"] = alignment

    path_turn_min_speed = getattr(
        opts, "behavior_state_turn_path_min_speed_mm_s", None
    )
    if path_turn_min_speed is not None:
        updates["turn_path_min_speed_mm_s"] = max(0.0, float(path_turn_min_speed))

    path_shift = getattr(opts, "behavior_state_turn_path_angular_shift_frames", None)
    if path_shift is not None:
        updates["path_angular_mask_shift_frames"] = max(0, int(path_shift))

    small_turn_deg = getattr(opts, "behavior_state_turn_angular_small_deg_s", None)
    if small_turn_deg is not None:
        updates["angular_small_turn_rad_s"] = float(small_turn_deg) * np.pi / 180.0

    large_turn_deg = getattr(opts, "behavior_state_turn_angular_large_deg_s", None)
    if large_turn_deg is not None:
        updates["angular_large_turn_rad_s"] = float(large_turn_deg) * np.pi / 180.0

    min_segments = getattr(opts, "behavior_state_turn_min_segments", None)
    if min_segments is not None:
        updates["turn_min_segments"] = max(1, int(min_segments))

    return replace(cfg, **updates)


def schmitt(values: np.ndarray, *, high: float, low: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = (values > high).astype(np.int8)
    out[values < low] = -1

    nonzero = np.flatnonzero(out)
    if nonzero.size == 0:
        return np.zeros_like(out, dtype=np.int8)

    out[0] = out[nonzero[0]]
    for i in np.flatnonzero(out == 0):
        out[i] = out[i - 1]
    out[out == -1] = 0
    return out.astype(np.int8)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def filter_short_turn_segments(
    turn_mask: np.ndarray,
    *,
    min_segments: int = DEFAULT_BEHAVIOR_STATE_CONFIG.turn_min_segments,
) -> np.ndarray:
    mask = np.asarray(turn_mask, dtype=bool)
    minimum = max(1, int(min_segments))
    if minimum <= 1 or mask.size == 0:
        return mask.copy()

    out = mask.copy()
    for start, stop in _true_runs(out):
        if stop - start + 1 < minimum:
            out[start : stop + 1] = False
    return out


def _angular_turn_score(
    angular_speed_rad_s: np.ndarray,
    *,
    config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
    shift_frames: int | None = None,
) -> np.ndarray:
    angular_speed = np.asarray(angular_speed_rad_s, dtype=np.float64)
    finite = np.isfinite(angular_speed)
    abs_speed = np.zeros_like(angular_speed, dtype=np.float64)
    abs_speed[finite] = np.abs(angular_speed[finite])
    score = (abs_speed > config.angular_small_turn_rad_s).astype(float) + 1.5 * (
        abs_speed > config.angular_large_turn_rad_s
    ).astype(float)
    score[~finite] = 0.0
    if shift_frames is None:
        shift_frames = config.angular_mask_shift_frames
    return _shift_left_zero_tail(score, max(0, int(shift_frames)))


def turn_score_components(
    angular_speed_rad_s: np.ndarray,
    head_speed_mm_s: np.ndarray,
    body_speed_mm_s: np.ndarray,
    *,
    config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
    path_angular_speed_rad_s: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    angular_speed = np.asarray(angular_speed_rad_s, dtype=np.float64)
    head_speed = np.asarray(head_speed_mm_s, dtype=np.float64)
    body_speed = np.asarray(body_speed_mm_s, dtype=np.float64)
    n = min(angular_speed.size, head_speed.size, body_speed.size)
    angular_speed = angular_speed[:n]
    head_speed = head_speed[:n]
    body_speed = body_speed[:n]

    theta_angular_score = _angular_turn_score(
        angular_speed,
        config=config,
        shift_frames=config.angular_mask_shift_frames,
    )
    if path_angular_speed_rad_s is None:
        path_angular_speed = np.full(n, np.nan, dtype=np.float64)
    else:
        path_angular_speed = np.asarray(path_angular_speed_rad_s, dtype=np.float64)[:n]
        if path_angular_speed.size < n:
            tmp = np.full(n, np.nan, dtype=np.float64)
            tmp[: path_angular_speed.size] = path_angular_speed
            path_angular_speed = tmp
    path_angular_score = _angular_turn_score(
        path_angular_speed,
        config=config,
        shift_frames=config.path_angular_mask_shift_frames,
    )
    angular_source = str(config.turn_angular_source).strip().lower().replace("-", "_")
    if angular_source == "theta":
        path_angular_score = np.zeros_like(path_angular_score)
        angular_score = theta_angular_score.copy()
    elif angular_source == "path":
        theta_angular_score = np.zeros_like(theta_angular_score)
        angular_score = path_angular_score.copy()
    elif angular_source == "theta_or_path":
        angular_score = np.maximum(theta_angular_score, path_angular_score)
    else:
        raise ValueError(f"Unknown behavior-state turn angular source: {angular_source}")

    head_body_diff = head_speed - body_speed
    shifted_head_body_diff = _shift_left_zero_tail(
        head_body_diff, config.head_body_shift_frames
    )
    head_body_score = 0.5 * (
        shifted_head_body_diff > config.head_body_small_turn_mm_s
    ).astype(float) + (
        shifted_head_body_diff > config.head_body_large_turn_mm_s
    ).astype(float)

    smoothed_angular_score = _reference_savgol_filter(
        angular_score, order=config.savgol_order, window=config.savgol_window
    )
    smoothed_head_body_score = _reference_savgol_filter(
        head_body_score, order=config.savgol_order, window=config.savgol_window
    )
    turn_score = smoothed_angular_score + smoothed_head_body_score

    return {
        "angular_score": angular_score,
        "theta_angular_score": theta_angular_score,
        "path_angular_score": path_angular_score,
        "head_body_score": head_body_score,
        "smoothed_angular_score": smoothed_angular_score,
        "smoothed_head_body_score": smoothed_head_body_score,
        "turn_score": turn_score,
        "raw_turn_mask": turn_score >= config.turn_score_threshold,
        "head_body_diff_mm_s": head_body_diff,
        "shifted_head_body_diff_mm_s": shifted_head_body_diff,
        "path_angular_speed_rad_s": path_angular_speed,
    }


def translational_speed_mm_s(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    fps: float,
    px_per_mm: float,
    dt_s: np.ndarray | None = None,
    config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
):
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    dr_px = np.hypot(np.diff(x), np.diff(y))
    dr_px = np.append(dr_px[0] if dr_px.size else np.nan, dr_px)
    if dt_s is None:
        dt = np.full_like(dr_px, 1.0 / float(fps), dtype=np.float64)
    else:
        dt = np.asarray(dt_s, dtype=np.float64)
    speed = dr_px / dt / float(px_per_mm)
    return _reference_savgol_filter(
        speed, order=config.savgol_order, window=config.savgol_window
    )


def path_angular_speed_rad_s(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    fps: float,
    dt_s: np.ndarray | None = None,
    body_speed_mm_s: np.ndarray | None = None,
    config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
) -> np.ndarray:
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)
    bearing = np.arctan2(dy, dx)
    db = np.zeros_like(bearing, dtype=np.float64)
    if bearing.size > 1:
        db[1:] = np.diff(bearing)
    db[db > np.pi] -= 2 * np.pi
    db[db < -np.pi] += 2 * np.pi

    if dt_s is None:
        dt = np.full_like(db, 1.0 / float(fps), dtype=np.float64)
    else:
        dt = np.asarray(dt_s, dtype=np.float64)
    out = db / dt
    out[~np.isfinite(bearing)] = np.nan
    alignment = str(config.turn_path_angular_alignment).strip().lower().replace("-", "_")
    if alignment == "vertex":
        shifted = np.full_like(out, np.nan, dtype=np.float64)
        if out.size > 1:
            shifted[:-1] = out[1:]
        out = shifted
    elif alignment != "segment_end":
        raise ValueError(f"Unknown behavior-state path angular alignment: {alignment}")
    if body_speed_mm_s is not None:
        body_speed = np.asarray(body_speed_mm_s, dtype=np.float64)
        n = min(out.size, body_speed.size)
        low_speed = np.ones(out.size, dtype=bool)
        low_speed[:n] = (
            ~np.isfinite(body_speed[:n])
            | (body_speed[:n] < config.turn_path_min_speed_mm_s)
        )
        out[low_speed] = np.nan
    return _reference_savgol_filter(
        out, order=config.savgol_order, window=config.savgol_window
    )


def rotational_speed_rad_s(
    heading_deg: np.ndarray,
    *,
    fps: float,
    dt_s: np.ndarray | None = None,
    config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
) -> np.ndarray:
    heading = np.deg2rad(np.asarray(heading_deg, dtype=np.float64))
    da = np.append(0.0, np.diff(heading))
    da[da > np.pi] -= 2 * np.pi
    da[da < -np.pi] += 2 * np.pi
    if dt_s is None:
        dt = np.full_like(da, 1.0 / float(fps), dtype=np.float64)
    else:
        dt = np.asarray(dt_s, dtype=np.float64)
    return _reference_savgol_filter(
        da / dt, order=config.savgol_order, window=config.savgol_window
    )


def head_vertex_xy_px(
    x_px: np.ndarray,
    y_px: np.ndarray,
    major_axis_px: np.ndarray,
    heading_deg: np.ndarray,
    *,
    config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    major = np.asarray(major_axis_px, dtype=np.float64)
    theta = np.deg2rad(np.asarray(heading_deg, dtype=np.float64))
    radius = major * float(config.head_vertex_axis_fraction)
    return x + radius * np.sin(theta), y - radius * np.cos(theta)


def find_turns(
    angular_speed_rad_s: np.ndarray,
    head_speed_mm_s: np.ndarray,
    body_speed_mm_s: np.ndarray,
    *,
    config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
    path_angular_speed_rad_s: np.ndarray | None = None,
) -> np.ndarray:
    components = turn_score_components(
        angular_speed_rad_s,
        head_speed_mm_s,
        body_speed_mm_s,
        config=config,
        path_angular_speed_rad_s=path_angular_speed_rad_s,
    )
    turn_mask = components["raw_turn_mask"]
    return filter_short_turn_segments(
        turn_mask, min_segments=config.turn_min_segments
    )


def classify_behavior_states(
    body_speed_mm_s: np.ndarray,
    angular_speed_rad_s: np.ndarray,
    head_speed_mm_s: np.ndarray,
    *,
    config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
    path_angular_speed_rad_s: np.ndarray | None = None,
) -> np.ndarray:
    states = np.full(len(body_speed_mm_s), BehaviorState.REST, dtype=np.int8)
    turn_mask = find_turns(
        angular_speed_rad_s,
        head_speed_mm_s,
        body_speed_mm_s,
        config=config,
        path_angular_speed_rad_s=path_angular_speed_rad_s,
    )
    states[turn_mask] = BehaviorState.TURN

    speed = _reference_savgol_filter(
        body_speed_mm_s, order=config.savgol_order, window=config.savgol_window
    )
    walking = schmitt(speed, high=config.run_high_mm_s, low=config.run_low_mm_s)
    run_mask = (states != BehaviorState.TURN) & (walking > 0)
    states[run_mask] = BehaviorState.RUN
    return states


def analyze_trajectory_behavior_states(
    trj, *, config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG
) -> np.ndarray | None:
    if trj.bad() or trj.theta is None or trj.h is None or not trj.va:
        return None

    px_per_mm = trj.va.ct.pxPerMmFloor() * getattr(trj.va.xf, "fctr", 1.0)
    dt_s = _trajectory_dt_s(trj)
    body_speed = translational_speed_mm_s(
        trj.x, trj.y, fps=trj.va.fps, px_per_mm=px_per_mm, dt_s=dt_s, config=config
    )
    head_x, head_y = head_vertex_xy_px(trj.x, trj.y, trj.h, trj.theta, config=config)
    head_speed = translational_speed_mm_s(
        head_x, head_y, fps=trj.va.fps, px_per_mm=px_per_mm, dt_s=dt_s, config=config
    )
    angular_speed = rotational_speed_rad_s(
        trj.theta, fps=trj.va.fps, dt_s=dt_s, config=config
    )
    path_angular_speed = path_angular_speed_rad_s(
        trj.x,
        trj.y,
        fps=trj.va.fps,
        dt_s=dt_s,
        body_speed_mm_s=body_speed,
        config=config,
    )
    trj.behavior_body_speed_mm_s = body_speed
    trj.behavior_head_speed_mm_s = head_speed
    trj.behavior_angular_speed_rad_s = angular_speed
    trj.behavior_path_angular_speed_rad_s = path_angular_speed
    trj.behavior_turn_mask = find_turns(
        angular_speed,
        head_speed,
        body_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
    )
    trj.behavior_state = classify_behavior_states(
        body_speed,
        angular_speed,
        head_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
    )
    return trj.behavior_state


def _trajectory_dt_s(trj) -> np.ndarray | None:
    ts = getattr(trj, "ts", None)
    if ts is None:
        return None

    ts = np.asarray(ts, dtype=np.float64)
    if ts.shape != np.asarray(trj.x).shape or ts.size == 0:
        return None

    fallback_dt = 1.0 / float(trj.va.fps)
    dt = np.empty_like(ts, dtype=np.float64)
    dt[0] = fallback_dt
    if ts.size > 1:
        dt[1:] = np.diff(ts)
    bad = ~np.isfinite(dt) | (dt <= 0)
    if np.any(bad):
        dt[bad] = fallback_dt
    return dt
