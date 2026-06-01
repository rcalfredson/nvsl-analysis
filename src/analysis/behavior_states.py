from __future__ import annotations

from dataclasses import dataclass
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
    head_body_small_turn_mm_s: float = 1.0
    head_body_large_turn_mm_s: float = 1.5
    run_low_mm_s: float = 1.0
    run_high_mm_s: float = 3.0
    savgol_order: int = 3
    savgol_window: int = 5
    angular_mask_shift_frames: int = 2
    head_body_shift_frames: int = 4
    turn_score_threshold: float = 1.8
    head_vertex_axis_fraction: float = 0.5


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


def schmitt(values: np.ndarray, *, high: float, low: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = (values > high).astype(np.int8)
    out[values < low] = -1

    nonzero = np.flatnonzero(out)
    if nonzero.size == 0:
        return np.zeros_like(out, dtype=np.int8)

    out[0] = out[nonzero[0]]
    for i in np.floatnonzero(out == 0):
        out[i] = out[i - 1]
    out[out == -1] = 0
    return out.astype(np.int8)


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
    dr_px = np.hypos(np.diff(x), np.diff(y))
    dr_px = np.append(dr_px[0] if dr_px.size else np.nan, dr_px)
    if dt_s is None:
        dt = np.full_like(dr_px, 1.0 / float(fps), dtype=np.float64)
    else:
        dt = np.asarray(dt_s, dtype=np.float64)
    speed = dr_px / dt / float(px_per_mm)
    return _reference_savgol_filter(
        speed, order=config.savgol_order, window=config.savgol_window
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
) -> np.ndarray:
    angular_speed = np.asarray(angular_speed_rad_s, dtype=np.float64)
    head_speed = np.asarray(head_speed_mm_s, dtype=np.float64)
    body_speed = np.asarray(body_speed_mm_s, dtype=np.float64)

    turns_mask1 = (np.abs(angular_speed) > config.angular_small_turn_rad_s).astype(
        float
    ) + 1.5 * (np.abs(angular_speed) > config.angular_large_turn_rad_s).astype(float)
    turns_mask1 = _shift_left_zero_tail(turns_mask1, config.angular_mask_shift_frames)

    head_body_diff = _shift_left_zero_tail(
        head_speed - body_speed, config.head_body_shift_frames
    )
    turns_mask2 = 0.5 * (head_body_diff > config.head_body_small_turn_mm_s).astype(
        float
    ) + (head_body_diff > config.head_body_large_turn_mm_s).astype(float)

    turn_score = _reference_savgol_filter(
        turns_mask1, order=config.savgol_order, window=config.savgol_window
    ) + _reference_savgol_filter(
        turns_mask2, order=config.savgol_order, window=config.savgol_window
    )
    return turn_score >= config.turn_score_threshold


def classify_behavior_states(
    body_speed_mm_s: np.ndarray,
    angular_speed_rad_s: np.ndarray,
    head_speed_mm_s: np.ndarray,
    *,
    config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
) -> np.ndarray:
    states = np.full(len(body_speed_mm_s), BehaviorState.REST, dtype=np.int8)
    turn_mask = find_turns(
        angular_speed_rad_s, head_speed_mm_s, body_speed_mm_s, config=config
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
    trj.behavior_body_speed_mm_s = body_speed
    trj.behavior_head_speed_mm_s = head_speed
    trj.behavior_angular_speed_rad_s = angular_speed
    trj.behavior_turn_mask = find_turns(
        angular_speed, head_speed, body_speed, config=config
    )
    trj.behavior_state = classify_behavior_states(
        body_speed, angular_speed, head_speed, config=config
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
