from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt


BEHAVIOR_STATE_DETECTOR_HABERKERN = "haberkern"
BEHAVIOR_STATE_DETECTOR_SCHMITT_BUTTERWORTH = "schmitt_butterworth"
BEHAVIOR_STATE_DETECTOR_CHOICES = (
    BEHAVIOR_STATE_DETECTOR_HABERKERN,
    BEHAVIOR_STATE_DETECTOR_SCHMITT_BUTTERWORTH,
)


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
    turn_path_min_segment_speed_mm_s: float | None = None
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
    turn_expand_largest_vertex: bool = False
    turn_absorb_sharp_gaps: bool = False
    turn_sharp_gap_max_segments: int = 2
    turn_sharp_gap_min_peak_ratio: float = 1.0
    turn_rescue_pivot_pauses: bool = False
    turn_pivot_min_angle_rad: float = 100.0 * np.pi / 180.0
    turn_pivot_max_short_segments: int = 1
    turn_pivot_short_segment_max_speed_mm_s: float | None = None
    turn_pivot_flank_min_speed_mm_s: float | None = None


DEFAULT_BEHAVIOR_STATE_CONFIG = BehaviorStateConfig()


@dataclass(frozen=True)
class SchmittButterworthConfig:
    """Configuration for the low-frame-rate Schmitt–Butterworth detector."""

    position_savgol_window: int = 3
    position_savgol_order: int = 1
    walking_start_mm_s: float = 2.0
    stopping_start_mm_s: float = 1.0
    stopped_min_duration_s: float = 0.2
    walking_min_duration_s: float = 0.05
    stop_extension_radius_mm: float = 0.5
    stop_extension_window_s: float = 1.0
    butterworth_order: int = 4
    butterworth_cutoff_hz: float = 2.0
    angular_moving_average_frames: int = 1
    turn_peak_threshold_deg_s: float = 120.0
    turn_flank_frames: int = 2
    turn_merge_radius_mm: float = 0.5
    turn_min_angle_deg: float = 20.0


DEFAULT_SCHMITT_BUTTERWORTH_CONFIG = SchmittButterworthConfig()


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


def behavior_state_detector_from_opts(opts) -> str:
    detector = str(
        getattr(opts, "behavior_state_detector", BEHAVIOR_STATE_DETECTOR_HABERKERN)
        or BEHAVIOR_STATE_DETECTOR_HABERKERN
    ).strip().lower().replace("-", "_")
    if detector not in BEHAVIOR_STATE_DETECTOR_CHOICES:
        raise ValueError(f"Unknown behavior-state detector: {detector}")
    return detector


def _schmitt_butterworth_config_from_opts(opts) -> SchmittButterworthConfig:
    cfg = DEFAULT_SCHMITT_BUTTERWORTH_CONFIG
    updates = {}
    option_fields = {
        "behavior_state_sb_position_savgol_window": (
            "position_savgol_window",
            int,
        ),
        "behavior_state_sb_position_savgol_order": (
            "position_savgol_order",
            int,
        ),
        "behavior_state_sb_walking_start_mm_s": ("walking_start_mm_s", float),
        "behavior_state_sb_stopping_start_mm_s": ("stopping_start_mm_s", float),
        "behavior_state_sb_stopped_min_duration_s": (
            "stopped_min_duration_s",
            float,
        ),
        "behavior_state_sb_walking_min_duration_s": (
            "walking_min_duration_s",
            float,
        ),
        "behavior_state_sb_stop_extension_radius_mm": (
            "stop_extension_radius_mm",
            float,
        ),
        "behavior_state_sb_stop_extension_window_s": (
            "stop_extension_window_s",
            float,
        ),
        "behavior_state_sb_butterworth_cutoff_hz": (
            "butterworth_cutoff_hz",
            float,
        ),
        "behavior_state_sb_angular_moving_average_frames": (
            "angular_moving_average_frames",
            int,
        ),
        "behavior_state_sb_turn_peak_deg_s": (
            "turn_peak_threshold_deg_s",
            float,
        ),
        "behavior_state_sb_turn_flank_frames": ("turn_flank_frames", int),
        "behavior_state_sb_turn_merge_radius_mm": (
            "turn_merge_radius_mm",
            float,
        ),
        "behavior_state_sb_turn_min_angle_deg": ("turn_min_angle_deg", float),
    }
    for option, (field, converter) in option_fields.items():
        value = getattr(opts, option, None)
        if value is not None:
            updates[field] = converter(value)

    out = replace(cfg, **updates)
    if out.position_savgol_window < 1 or out.position_savgol_window % 2 == 0:
        raise ValueError("Schmitt–Butterworth Savitzky–Golay window must be odd")
    if not 0 <= out.position_savgol_order < out.position_savgol_window:
        raise ValueError(
            "Schmitt–Butterworth Savitzky–Golay order must be nonnegative and "
            "smaller than its window"
        )
    if out.stopping_start_mm_s > out.walking_start_mm_s:
        raise ValueError("Stopping threshold cannot exceed walking threshold")
    positive_fields = (
        "stopped_min_duration_s",
        "walking_min_duration_s",
        "stop_extension_radius_mm",
        "stop_extension_window_s",
        "butterworth_cutoff_hz",
        "turn_peak_threshold_deg_s",
        "turn_merge_radius_mm",
        "turn_min_angle_deg",
    )
    if any(getattr(out, field) < 0 for field in positive_fields):
        raise ValueError(
            "Schmitt–Butterworth thresholds and durations must be nonnegative"
        )
    if out.angular_moving_average_frames < 1:
        raise ValueError("Angular moving-average window must be at least one frame")
    if out.turn_flank_frames < 2:
        raise ValueError("Turn line fits require at least two frames per flank")
    return out


def behavior_state_config_from_opts(
    opts,
) -> BehaviorStateConfig | SchmittButterworthConfig:
    if (
        behavior_state_detector_from_opts(opts)
        == BEHAVIOR_STATE_DETECTOR_SCHMITT_BUTTERWORTH
    ):
        return _schmitt_butterworth_config_from_opts(opts)

    cfg = DEFAULT_BEHAVIOR_STATE_CONFIG
    updates = {}

    angular_source = getattr(opts, "behavior_state_turn_angular_source", None)
    if angular_source is not None:
        source = str(angular_source).strip().lower().replace("-", "_")
        if source not in {"theta", "path", "theta_or_path", "path_no_head_body"}:
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

    path_turn_min_segment_speed = getattr(
        opts, "behavior_state_turn_path_min_segment_speed_mm_s", None
    )
    if path_turn_min_segment_speed is not None:
        updates["turn_path_min_segment_speed_mm_s"] = max(
            0.0, float(path_turn_min_segment_speed)
        )

    path_shift = getattr(opts, "behavior_state_turn_path_angular_shift_frames", None)
    if path_shift is not None:
        updates["path_angular_mask_shift_frames"] = max(0, int(path_shift))

    small_turn_deg = getattr(opts, "behavior_state_turn_angular_small_deg_s", None)
    if small_turn_deg is not None:
        updates["angular_small_turn_rad_s"] = float(small_turn_deg) * np.pi / 180.0

    large_turn_deg = getattr(opts, "behavior_state_turn_angular_large_deg_s", None)
    if large_turn_deg is not None:
        updates["angular_large_turn_rad_s"] = float(large_turn_deg) * np.pi / 180.0

    turn_score_threshold = getattr(opts, "behavior_state_turn_score_threshold", None)
    if turn_score_threshold is not None:
        updates["turn_score_threshold"] = float(turn_score_threshold)

    min_segments = getattr(opts, "behavior_state_turn_min_segments", None)
    if min_segments is not None:
        updates["turn_min_segments"] = max(1, int(min_segments))

    expand_largest_vertex = getattr(
        opts, "behavior_state_turn_expand_largest_vertex", None
    )
    if expand_largest_vertex is not None:
        updates["turn_expand_largest_vertex"] = bool(expand_largest_vertex)

    absorb_sharp_gaps = getattr(opts, "behavior_state_turn_absorb_sharp_gaps", None)
    if absorb_sharp_gaps is not None:
        updates["turn_absorb_sharp_gaps"] = bool(absorb_sharp_gaps)

    sharp_gap_max_segments = getattr(
        opts, "behavior_state_turn_sharp_gap_max_segments", None
    )
    if sharp_gap_max_segments is not None:
        updates["turn_sharp_gap_max_segments"] = max(1, int(sharp_gap_max_segments))

    sharp_gap_min_peak_ratio = getattr(
        opts, "behavior_state_turn_sharp_gap_min_peak_ratio", None
    )
    if sharp_gap_min_peak_ratio is not None:
        updates["turn_sharp_gap_min_peak_ratio"] = max(
            0.0, float(sharp_gap_min_peak_ratio)
        )

    rescue_pivot_pauses = getattr(
        opts, "behavior_state_turn_rescue_pivot_pauses", None
    )
    if rescue_pivot_pauses is not None:
        updates["turn_rescue_pivot_pauses"] = bool(rescue_pivot_pauses)

    pivot_min_angle_deg = getattr(
        opts, "behavior_state_turn_pivot_min_angle_deg", None
    )
    if pivot_min_angle_deg is not None:
        updates["turn_pivot_min_angle_rad"] = float(pivot_min_angle_deg) * np.pi / 180.0

    pivot_max_short_segments = getattr(
        opts, "behavior_state_turn_pivot_max_short_segments", None
    )
    if pivot_max_short_segments is not None:
        updates["turn_pivot_max_short_segments"] = max(1, int(pivot_max_short_segments))

    pivot_short_max_speed = getattr(
        opts, "behavior_state_turn_pivot_short_max_speed_mm_s", None
    )
    if pivot_short_max_speed is not None:
        updates["turn_pivot_short_segment_max_speed_mm_s"] = max(
            0.0, float(pivot_short_max_speed)
        )

    pivot_flank_min_speed = getattr(
        opts, "behavior_state_turn_pivot_flank_min_speed_mm_s", None
    )
    if pivot_flank_min_speed is not None:
        updates["turn_pivot_flank_min_speed_mm_s"] = max(
            0.0, float(pivot_flank_min_speed)
        )

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


def expand_turns_to_largest_path_vertex(
    turn_mask: np.ndarray,
    path_angular_speed_rad_s: np.ndarray | None,
    *,
    boundary_vertex_min_rad_s: float = (
        DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s
    ),
) -> np.ndarray:
    mask = np.asarray(turn_mask, dtype=bool)
    out = mask.copy()
    if path_angular_speed_rad_s is None or out.size == 0:
        return out

    path_speed = np.asarray(path_angular_speed_rad_s, dtype=np.float64)
    n = min(out.size, path_speed.size)
    if n == 0:
        return out
    out = out[:n].copy()
    original = mask[:n]
    abs_path_speed = np.abs(path_speed[:n])
    boundary_min = max(0.0, float(boundary_vertex_min_rad_s))

    for start, stop in _true_runs(original):
        if stop >= n:
            continue
        local = abs_path_speed[start : stop + 1]
        finite = np.isfinite(local)
        if not np.any(finite):
            continue
        finite_local_indices = np.flatnonzero(finite)
        local_peak = finite_local_indices[
            int(np.nanargmax(local[finite_local_indices]))
        ]
        vertex = start + int(local_peak)

        candidates = []
        incoming_segment = vertex - 1
        if incoming_segment >= 0 and not original[incoming_segment]:
            would_bridge_previous_turn = (
                incoming_segment - 1 >= 0 and original[incoming_segment - 1]
            )
            if not would_bridge_previous_turn:
                candidates.append(incoming_segment)

        outgoing_segment = vertex
        if outgoing_segment < n and not original[outgoing_segment]:
            would_bridge_next_turn = (
                outgoing_segment + 1 < n and original[outgoing_segment + 1]
            )
            if not would_bridge_next_turn:
                candidates.append(outgoing_segment)

        if candidates:
            out[candidates[0]] = True

        if abs_path_speed[start] >= boundary_min:
            incoming_segment = start - 1
            if incoming_segment >= 0 and not original[incoming_segment]:
                would_bridge_previous_turn = (
                    incoming_segment - 1 >= 0 and original[incoming_segment - 1]
                )
                if not would_bridge_previous_turn:
                    out[incoming_segment] = True

        outgoing_vertex = stop + 1
        if outgoing_vertex < n and abs_path_speed[outgoing_vertex] >= boundary_min:
            outgoing_segment = stop + 1
            if outgoing_segment < n and not original[outgoing_segment]:
                would_bridge_next_turn = (
                    outgoing_segment + 1 < n and original[outgoing_segment + 1]
                )
                if not would_bridge_next_turn:
                    out[outgoing_segment] = True

    if mask.size > n:
        padded = mask.copy()
        padded[:n] = out
        return padded
    return out


def _finite_abs_peak(values: np.ndarray) -> float:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan
    return float(np.max(np.abs(finite_values)))


def _dominant_sign(value: float) -> int:
    if not np.isfinite(value) or value == 0:
        return 0
    return 1 if value > 0 else -1


def _split_opposing_path_vertices(
    turn_mask: np.ndarray,
    path_angular_speed_rad_s: np.ndarray,
    *,
    min_segments: int,
) -> np.ndarray:
    out = np.asarray(turn_mask, dtype=bool).copy()
    path_speed = np.asarray(path_angular_speed_rad_s, dtype=np.float64)
    n = min(out.size, path_speed.size)
    if n == 0:
        return out

    minimum = max(1, int(min_segments))
    for start, stop in _true_runs(out[:n]):
        local = path_speed[start : stop + 1]
        finite_local = np.flatnonzero(np.isfinite(local) & (local != 0))
        if finite_local.size < 2:
            continue

        signed_vertices = [(start + int(i), _dominant_sign(local[int(i)])) for i in finite_local]
        split_candidates = []
        for (left_idx, left_sign), (right_idx, right_sign) in zip(
            signed_vertices, signed_vertices[1:]
        ):
            if left_sign == 0 or right_sign == 0 or left_sign == right_sign:
                continue
            lo, hi = sorted((left_idx, right_idx))
            window = np.abs(path_speed[lo : hi + 1])
            finite = np.isfinite(window)
            if not np.any(finite):
                continue
            finite_offsets = np.flatnonzero(finite)
            weakest_offset = finite_offsets[int(np.nanargmin(window[finite_offsets]))]
            split_idx = lo + int(weakest_offset)
            split_segment = max(start, split_idx - 1)
            candidate_segments = [split_segment]
            if split_idx not in candidate_segments:
                candidate_segments.append(split_idx)
            for candidate_segment in candidate_segments:
                left_len = candidate_segment - start
                right_len = stop - candidate_segment
                if left_len >= minimum and right_len >= minimum:
                    split_candidates.append(
                        (float(window[weakest_offset]), candidate_segment)
                    )
                    break

        if not split_candidates:
            continue
        _, split_idx = min(split_candidates, key=lambda item: item[0])
        out[split_idx] = False

    return filter_short_turn_segments(out, min_segments=minimum)


def absorb_sharp_turn_gaps(
    turn_mask: np.ndarray,
    path_angular_speed_rad_s: np.ndarray | None,
    *,
    max_gap_segments: int = DEFAULT_BEHAVIOR_STATE_CONFIG.turn_sharp_gap_max_segments,
    min_peak_ratio: float = DEFAULT_BEHAVIOR_STATE_CONFIG.turn_sharp_gap_min_peak_ratio,
    boundary_vertex_min_rad_s: float = (
        DEFAULT_BEHAVIOR_STATE_CONFIG.angular_large_turn_rad_s
    ),
    min_segments: int = DEFAULT_BEHAVIOR_STATE_CONFIG.turn_min_segments,
    split_opposing: bool = True,
) -> np.ndarray:
    mask = np.asarray(turn_mask, dtype=bool)
    out = mask.copy()
    if path_angular_speed_rad_s is None or out.size == 0:
        return out

    path_speed = np.asarray(path_angular_speed_rad_s, dtype=np.float64)
    n = min(out.size, path_speed.size)
    if n == 0:
        return out
    out = out[:n].copy()
    original = mask[:n]
    max_gap = max(1, int(max_gap_segments))
    ratio = max(0.0, float(min_peak_ratio))
    boundary_min = max(0.0, float(boundary_vertex_min_rad_s))
    runs = _true_runs(original)

    for (left_start, left_stop), (right_start, right_stop) in zip(runs, runs[1:]):
        gap_start = left_stop + 1
        gap_stop = right_start - 1
        gap_len = gap_stop - gap_start + 1
        if gap_len < 1 or gap_len > max_gap:
            continue

        left_boundary = path_speed[left_stop] if left_stop < n else np.nan
        if np.isfinite(left_boundary) and abs(left_boundary) >= boundary_min:
            out[gap_start] = True

        right_boundary = path_speed[right_start] if right_start < n else np.nan
        if np.isfinite(right_boundary) and abs(right_boundary) >= boundary_min:
            out[gap_stop] = True

        gap_peak = _finite_abs_peak(path_speed[gap_start : gap_stop + 1])
        left_peak = _finite_abs_peak(path_speed[left_start : left_stop + 1])
        right_peak = _finite_abs_peak(path_speed[right_start : right_stop + 1])
        adjacent_peaks = [p for p in (left_peak, right_peak) if np.isfinite(p)]
        if not np.isfinite(gap_peak) or not adjacent_peaks:
            continue
        reference_peak = min(adjacent_peaks)
        if gap_peak >= ratio * reference_peak:
            out[gap_start : gap_stop + 1] = True

    if split_opposing:
        out = _split_opposing_path_vertices(
            out, path_speed[:n], min_segments=min_segments
        )

    if mask.size > n:
        padded = mask.copy()
        padded[:n] = out
        return padded
    return out


def raw_segment_speed_mm_s(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    fps: float,
    px_per_mm: float,
    dt_s: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    n = min(x.size, y.size)
    speed = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return speed

    dr_px = np.hypot(np.diff(x[:n]), np.diff(y[:n]))
    if dt_s is None:
        dt = np.full(n - 1, 1.0 / float(fps), dtype=np.float64)
    else:
        dt_arr = np.asarray(dt_s, dtype=np.float64)[:n]
        dt = dt_arr[1:n] if dt_arr.size >= n else np.full(n - 1, np.nan)
    speed[:-1] = dr_px / dt / float(px_per_mm)
    return speed


def pivot_pause_turn_mask(
    x_px: np.ndarray,
    y_px: np.ndarray,
    segment_speed_mm_s: np.ndarray,
    *,
    body_speed_mm_s: np.ndarray | None = None,
    config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
) -> np.ndarray:
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    segment_speed = np.asarray(segment_speed_mm_s, dtype=np.float64)
    n = min(x.size, y.size, segment_speed.size)
    if body_speed_mm_s is not None:
        body_speed = np.asarray(body_speed_mm_s, dtype=np.float64)[:n]
    else:
        body_speed = None
    out = np.zeros(n, dtype=bool)
    if n < 4:
        return out

    min_segment_speed = config.turn_path_min_segment_speed_mm_s
    if min_segment_speed is None:
        min_segment_speed = config.turn_path_min_speed_mm_s
    short_max = (
        config.turn_pivot_short_segment_max_speed_mm_s
        if config.turn_pivot_short_segment_max_speed_mm_s is not None
        else min_segment_speed
    )
    flank_min = (
        config.turn_pivot_flank_min_speed_mm_s
        if config.turn_pivot_flank_min_speed_mm_s is not None
        else min_segment_speed
    )
    short_max = max(0.0, float(short_max))
    flank_min = max(0.0, float(flank_min))
    min_angle = max(0.0, float(config.turn_pivot_min_angle_rad))
    max_short_segments = max(1, int(config.turn_pivot_max_short_segments))

    for short_start in range(1, n - 2):
        for short_len in range(1, max_short_segments + 1):
            incoming = short_start - 1
            outgoing = short_start + short_len
            if outgoing >= n - 1:
                break

            short_speeds = segment_speed[short_start:outgoing]
            speeds = np.r_[segment_speed[incoming], short_speeds, segment_speed[outgoing]]
            if not np.all(np.isfinite(speeds)):
                continue
            if np.any(short_speeds > short_max):
                continue
            if segment_speed[incoming] < flank_min or segment_speed[outgoing] < flank_min:
                continue
            if body_speed is not None:
                body_window = body_speed[incoming : outgoing + 1]
                if (
                    body_window.size != outgoing - incoming + 1
                    or not np.all(np.isfinite(body_window))
                    or np.any(body_window < config.run_low_mm_s)
                ):
                    continue

            incoming_angle = np.arctan2(
                y[incoming + 1] - y[incoming], x[incoming + 1] - x[incoming]
            )
            outgoing_angle = np.arctan2(
                y[outgoing + 1] - y[outgoing], x[outgoing + 1] - x[outgoing]
            )
            turn_angle = outgoing_angle - incoming_angle
            if turn_angle > np.pi:
                turn_angle -= 2 * np.pi
            elif turn_angle < -np.pi:
                turn_angle += 2 * np.pi
            if abs(turn_angle) >= min_angle:
                out[incoming : outgoing + 1] = True

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
    elif angular_source in {"path", "path_no_head_body"}:
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
    if angular_source == "path_no_head_body":
        head_body_score = np.zeros_like(head_body_score)

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
    px_per_mm: float | None = None,
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
    min_segment_speed = config.turn_path_min_segment_speed_mm_s
    if min_segment_speed is None:
        min_segment_speed = config.turn_path_min_speed_mm_s
    if px_per_mm is not None and float(min_segment_speed) > 0:
        segment_speed = np.hypot(dx, dy) / dt / float(px_per_mm)
        n = min(out.size, segment_speed.size)
        unreliable_vertex = np.ones(out.size, dtype=bool)
        if n > 1:
            incoming = segment_speed[: n - 1]
            outgoing = segment_speed[1:n]
            threshold = float(min_segment_speed)
            unreliable = (
                ~np.isfinite(incoming)
                | ~np.isfinite(outgoing)
                | (incoming < threshold)
                | (outgoing < threshold)
            )
            if alignment == "vertex":
                unreliable_vertex[: n - 1] = unreliable
            else:
                unreliable_vertex[1:n] = unreliable
        out[unreliable_vertex] = np.nan
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
    pivot_turn_mask: np.ndarray | None = None,
) -> np.ndarray:
    components = turn_score_components(
        angular_speed_rad_s,
        head_speed_mm_s,
        body_speed_mm_s,
        config=config,
        path_angular_speed_rad_s=path_angular_speed_rad_s,
    )
    turn_mask = components["raw_turn_mask"]
    turn_mask = filter_short_turn_segments(
        turn_mask, min_segments=config.turn_min_segments
    )
    angular_source = str(config.turn_angular_source).strip().lower().replace("-", "_")
    if config.turn_expand_largest_vertex and angular_source != "theta":
        turn_mask = expand_turns_to_largest_path_vertex(
            turn_mask,
            components["path_angular_speed_rad_s"],
            boundary_vertex_min_rad_s=config.angular_large_turn_rad_s,
        )
    if config.turn_absorb_sharp_gaps and angular_source != "theta":
        turn_mask = absorb_sharp_turn_gaps(
            turn_mask,
            components["path_angular_speed_rad_s"],
            max_gap_segments=config.turn_sharp_gap_max_segments,
            min_peak_ratio=config.turn_sharp_gap_min_peak_ratio,
            min_segments=config.turn_min_segments,
            split_opposing=True,
        )
    if config.turn_rescue_pivot_pauses and angular_source != "theta":
        if pivot_turn_mask is not None:
            pivot_mask = np.asarray(pivot_turn_mask, dtype=bool)
            n = min(turn_mask.size, pivot_mask.size)
            turn_mask[:n] |= pivot_mask[:n]
    return turn_mask


def classify_behavior_states(
    body_speed_mm_s: np.ndarray,
    angular_speed_rad_s: np.ndarray,
    head_speed_mm_s: np.ndarray,
    *,
    config: BehaviorStateConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
    path_angular_speed_rad_s: np.ndarray | None = None,
    pivot_turn_mask: np.ndarray | None = None,
) -> np.ndarray:
    states = np.full(len(body_speed_mm_s), BehaviorState.REST, dtype=np.int8)
    turn_mask = find_turns(
        angular_speed_rad_s,
        head_speed_mm_s,
        body_speed_mm_s,
        config=config,
        path_angular_speed_rad_s=path_angular_speed_rad_s,
        pivot_turn_mask=pivot_turn_mask,
    )
    states[turn_mask] = BehaviorState.TURN

    speed = _reference_savgol_filter(
        body_speed_mm_s, order=config.savgol_order, window=config.savgol_window
    )
    walking = schmitt(speed, high=config.run_high_mm_s, low=config.run_low_mm_s)
    run_mask = (states != BehaviorState.TURN) & (walking > 0)
    states[run_mask] = BehaviorState.RUN
    return states


def smooth_schmitt_butterworth_positions(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    config: SchmittButterworthConfig = DEFAULT_SCHMITT_BUTTERWORTH_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the alternative detector's position-only Savitzky–Golay filter."""
    x = np.asarray(x_px, dtype=np.float64).copy()
    y = np.asarray(y_px, dtype=np.float64).copy()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    window = int(config.position_savgol_window)
    if window > 1 and n >= window:
        order = int(config.position_savgol_order)
        x = savgol_filter(x, window, order, mode="interp")
        y = savgol_filter(y, window, order, mode="interp")
    return x, y


def schmitt_walking_mask(
    speed_mm_s: np.ndarray,
    *,
    walking_start_mm_s: float = (
        DEFAULT_SCHMITT_BUTTERWORTH_CONFIG.walking_start_mm_s
    ),
    stopping_start_mm_s: float = (
        DEFAULT_SCHMITT_BUTTERWORTH_CONFIG.stopping_start_mm_s
    ),
) -> np.ndarray:
    """Inclusive Schmitt trigger, conservatively initialized as stopped."""
    speed = np.asarray(speed_mm_s, dtype=np.float64)
    walking = np.zeros(speed.size, dtype=bool)
    state = False
    for i, value in enumerate(speed):
        if np.isfinite(value):
            if value >= walking_start_mm_s:
                state = True
            elif value <= stopping_start_mm_s:
                state = False
        walking[i] = state
    return walking


def extend_schmitt_stops(
    walking_mask: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    fps: float,
    px_per_mm: float,
    dt_s: np.ndarray | None = None,
    config: SchmittButterworthConfig = DEFAULT_SCHMITT_BUTTERWORTH_CONFIG,
) -> np.ndarray:
    """Perform a frozen, one-pass, fixed-endpoint extension of stopped bouts."""
    walking = np.asarray(walking_mask, dtype=bool)
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    n = min(walking.size, x.size, y.size)
    original = walking[:n].copy()
    out = original.copy()
    max_frames = max(0, n - 1)
    dt_values = None if dt_s is None else np.asarray(dt_s, dtype=np.float64)[:n]
    radius_px = float(config.stop_extension_radius_mm) * float(px_per_mm)

    for start, stop in _true_runs(~original):
        elapsed = 0.0
        for offset in range(1, max_frames + 1):
            frame = start - offset
            if frame < 0:
                break
            elapsed += (
                float(dt_values[frame + 1])
                if dt_values is not None
                else 1.0 / float(fps)
            )
            if elapsed > config.stop_extension_window_s + 1e-12:
                break
            if np.hypot(x[frame] - x[start], y[frame] - y[start]) > radius_px:
                break
            out[frame] = False
        elapsed = 0.0
        for offset in range(1, max_frames + 1):
            frame = stop + offset
            if frame >= n:
                break
            elapsed += (
                float(dt_values[frame])
                if dt_values is not None
                else 1.0 / float(fps)
            )
            if elapsed > config.stop_extension_window_s + 1e-12:
                break
            if np.hypot(x[frame] - x[stop], y[frame] - y[stop]) > radius_px:
                break
            out[frame] = False
    return out


def _binary_bouts(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return []
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries - 1, values.size - 1]
    return [
        (int(start), int(stop), bool(values[start]))
        for start, stop in zip(starts, stops)
    ]


def clean_schmitt_bout_durations(
    walking_mask: np.ndarray,
    *,
    fps: float,
    dt_s: np.ndarray | None = None,
    config: SchmittButterworthConfig = DEFAULT_SCHMITT_BUTTERWORTH_CONFIG,
) -> np.ndarray:
    """Merge undersized bouts until stable; every edit removes a boundary."""
    out = np.asarray(walking_mask, dtype=bool).copy()
    dt_values = (
        None
        if dt_s is None
        else np.asarray(dt_s, dtype=np.float64)[: out.size]
    )
    minimum_s = {
        False: float(config.stopped_min_duration_s),
        True: float(config.walking_min_duration_s),
    }

    def duration_s(start: int, stop: int) -> float:
        if dt_values is None:
            return (stop - start + 1) / float(fps)
        return float(np.sum(dt_values[start : stop + 1]))

    while True:
        bouts = _binary_bouts(out)
        if len(bouts) <= 1:
            return out
        selected = None
        for target_state in (False, True):
            selected = next(
                (
                    (start, stop, state)
                    for start, stop, state in bouts
                    if state == target_state
                    and duration_s(start, stop) + 1e-12 < minimum_s[target_state]
                ),
                None,
            )
            if selected is not None:
                break
        if selected is None:
            return out
        start, stop, state = selected
        out[start : stop + 1] = not state


def schmitt_butterworth_path_angular_velocity_deg_s(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    fps: float,
    dt_s: np.ndarray | None = None,
) -> np.ndarray:
    """Path-bearing angular velocity aligned to the spatial vertex frame."""
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    n = min(x.size, y.size)
    out = np.zeros(n, dtype=np.float64)
    if n < 3:
        return out

    dx = np.diff(x[:n])
    dy = np.diff(y[:n])
    valid_segment = np.isfinite(dx) & np.isfinite(dy) & (np.hypot(dx, dy) > 1e-12)
    bearing = np.arctan2(-dy, dx)
    for vertex in range(1, n - 1):
        if not (valid_segment[vertex - 1] and valid_segment[vertex]):
            continue
        delta = np.arctan2(
            np.sin(bearing[vertex] - bearing[vertex - 1]),
            np.cos(bearing[vertex] - bearing[vertex - 1]),
        )
        if dt_s is None:
            dt = 1.0 / float(fps)
        else:
            dt_values = np.asarray(dt_s, dtype=np.float64)
            dt = 0.5 * (dt_values[vertex] + dt_values[vertex + 1])
        if np.isfinite(dt) and dt > 0:
            out[vertex] = np.rad2deg(delta) / dt
    return out


def _centered_moving_average(values: np.ndarray, frames: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    width = max(1, int(frames))
    if width == 1 or values.size == 0:
        return values.copy()
    left = (width - 1) // 2
    right = width - left - 1
    mode = "reflect" if values.size > max(left, right) else "edge"
    padded = np.pad(values, (left, right), mode=mode)
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def filter_schmitt_butterworth_angular_velocity(
    angular_velocity_deg_s: np.ndarray,
    *,
    fps: float,
    config: SchmittButterworthConfig = DEFAULT_SCHMITT_BUTTERWORTH_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-phase Butterworth and optional moving-average signals."""
    raw = np.asarray(angular_velocity_deg_s, dtype=np.float64)
    nyquist = 0.5 * float(fps)
    cutoff = float(config.butterworth_cutoff_hz)
    if not 0 < cutoff < nyquist:
        raise ValueError(
            f"Butterworth cutoff must be between 0 and Nyquist ({nyquist:g} Hz); "
            f"got {cutoff:g} Hz"
        )
    sos = butter(
        int(config.butterworth_order),
        cutoff,
        btype="lowpass",
        fs=float(fps),
        output="sos",
    )
    try:
        filtered = sosfiltfilt(sos, raw)
    except ValueError:
        # Very short trajectories cannot support forward/backward filter padding.
        filtered = raw.copy()
    averaged = _centered_moving_average(
        filtered, int(config.angular_moving_average_frames)
    )
    return filtered, averaged


def strict_turn_peak_indices(
    angular_velocity_deg_s: np.ndarray,
    *,
    threshold_deg_s: float,
    flank_frames: int,
) -> np.ndarray:
    absolute = np.abs(np.asarray(angular_velocity_deg_s, dtype=np.float64))
    if absolute.size < 3:
        return np.array([], dtype=np.int64)
    indices = np.arange(1, absolute.size - 1)
    keep = (
        (absolute[indices] >= threshold_deg_s)
        & (absolute[indices] > absolute[indices - 1])
        & (absolute[indices] > absolute[indices + 1])
        & (indices >= flank_frames)
        & (indices + flank_frames < absolute.size)
    )
    return indices[keep].astype(np.int64)


def _forward_line_angle_rad(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    t = np.arange(x.size, dtype=np.float64)
    vx = np.polyfit(t, x, 1)[0]
    vy = -np.polyfit(t, y, 1)[0]
    if not np.isfinite(vx + vy) or np.hypot(vx, vy) <= 1e-12:
        return np.nan
    return float(np.arctan2(vy, vx))


def signed_turn_angle_deg(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    first_peak: int,
    last_peak: int,
    flank_frames: int,
) -> float:
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    incoming = slice(first_peak - flank_frames, first_peak)
    outgoing = slice(last_peak + 1, last_peak + flank_frames + 1)
    before = _forward_line_angle_rad(x[incoming], y[incoming])
    after = _forward_line_angle_rad(x[outgoing], y[outgoing])
    if not np.isfinite(before + after):
        return np.nan
    return float(
        np.rad2deg(np.arctan2(np.sin(after - before), np.cos(after - before)))
    )


def consolidate_schmitt_butterworth_peaks(
    peaks: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    filtered_angular_velocity_deg_s: np.ndarray,
    *,
    px_per_mm: float,
    config: SchmittButterworthConfig = DEFAULT_SCHMITT_BUTTERWORTH_CONFIG,
) -> list[dict]:
    peak_values = [int(value) for value in np.asarray(peaks, dtype=np.int64)]
    if not peak_values:
        return []
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    angular = np.asarray(filtered_angular_velocity_deg_s, dtype=np.float64)
    groups: list[list[int]] = [[peak_values[0]]]
    radius_px = config.turn_merge_radius_mm * float(px_per_mm)
    for peak in peak_values[1:]:
        previous = groups[-1][-1]
        distance = np.hypot(x[peak] - x[previous], y[peak] - y[previous])
        if np.isfinite(distance) and distance <= radius_px:
            groups[-1].append(peak)
        else:
            groups.append([peak])

    events = []
    flank = int(config.turn_flank_frames)
    for group in groups:
        representative = max(group, key=lambda frame: (abs(angular[frame]), -frame))
        angle = signed_turn_angle_deg(
            x,
            y,
            first_peak=group[0],
            last_peak=group[-1],
            flank_frames=flank,
        )
        events.append(
            {
                "peaks": tuple(group),
                "representative_peak": int(representative),
                "start": int(group[0] - flank),
                "stop": int(group[-1] + flank),
                "signed_angle_deg": angle,
                "peak_angular_velocity_deg_s": float(angular[representative]),
            }
        )
    return events


def wall_contact_mask_for_behavior_states(trj, n: int) -> np.ndarray:
    """Expand the standard compact wall/all/edge regions into a frame mask."""
    try:
        regions = trj.boundary_event_stats["wall"]["all"]["edge"][
            "boundary_contact_regions"
        ]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "Schmitt–Butterworth behavior states require wall-contact regions"
        ) from exc
    mask = np.zeros(int(n), dtype=bool)
    for region in regions:
        if isinstance(region, slice):
            start = 0 if region.start is None else int(region.start)
            stop = n if region.stop is None else int(region.stop)
        elif hasattr(region, "start") and hasattr(region, "stop"):
            start = 0 if region.start is None else int(region.start)
            stop = n if region.stop is None else int(region.stop)
        else:
            start, stop = int(region[0]), int(region[1])
        mask[max(0, start) : min(n, stop)] = True
    return mask


def analyze_trajectory_schmitt_butterworth_states(
    trj,
    *,
    config: SchmittButterworthConfig = DEFAULT_SCHMITT_BUTTERWORTH_CONFIG,
) -> np.ndarray | None:
    if trj.bad() or not trj.va:
        return None
    x = np.asarray(trj.x, dtype=np.float64)
    y = np.asarray(trj.y, dtype=np.float64)
    n = min(x.size, y.size)
    if n == 0:
        return None

    fps = float(trj.va.fps)
    px_per_mm = trj.va.ct.pxPerMmFloor() * getattr(trj.va.xf, "fctr", 1.0)
    dt_s = _trajectory_dt_s(trj)
    x_smooth, y_smooth = smooth_schmitt_butterworth_positions(
        x[:n], y[:n], config=config
    )
    displacement_px = np.hypot(np.diff(x_smooth), np.diff(y_smooth))
    displacement_px = np.append(
        displacement_px[0] if displacement_px.size else 0.0, displacement_px
    )
    if dt_s is None:
        speed_dt = np.full(n, 1.0 / fps, dtype=np.float64)
    else:
        speed_dt = np.asarray(dt_s, dtype=np.float64)[:n]
    speed = displacement_px / speed_dt / float(px_per_mm)
    preliminary_walking = schmitt_walking_mask(
        speed,
        walking_start_mm_s=config.walking_start_mm_s,
        stopping_start_mm_s=config.stopping_start_mm_s,
    )
    extended_walking = extend_schmitt_stops(
        preliminary_walking,
        x_smooth,
        y_smooth,
        fps=fps,
        px_per_mm=px_per_mm,
        dt_s=dt_s,
        config=config,
    )
    walking = clean_schmitt_bout_durations(
        extended_walking, fps=fps, dt_s=dt_s, config=config
    )
    raw_angular = schmitt_butterworth_path_angular_velocity_deg_s(
        x_smooth, y_smooth, fps=fps, dt_s=dt_s
    )
    butter_angular, final_angular = filter_schmitt_butterworth_angular_velocity(
        raw_angular, fps=fps, config=config
    )
    peaks = strict_turn_peak_indices(
        final_angular,
        threshold_deg_s=config.turn_peak_threshold_deg_s,
        flank_frames=config.turn_flank_frames,
    )
    events = consolidate_schmitt_butterworth_peaks(
        peaks,
        x_smooth,
        y_smooth,
        final_angular,
        px_per_mm=px_per_mm,
        config=config,
    )
    wall_contact = wall_contact_mask_for_behavior_states(trj, n)
    turn_mask = np.zeros(n, dtype=bool)
    for event in events:
        start, stop = event["start"], event["stop"]
        reason = None
        if not np.isfinite(event["signed_angle_deg"]):
            reason = "invalid_angle"
        elif abs(event["signed_angle_deg"]) < config.turn_min_angle_deg:
            reason = "angle_below_threshold"
        elif np.any(~walking[start : stop + 1]):
            reason = "stopped"
        elif np.any(wall_contact[start : stop + 1]):
            reason = "wall_contact"
        event["accepted"] = reason is None
        event["rejection_reason"] = reason
        if reason is None:
            turn_mask[start : stop + 1] = True

    states = np.full(n, BehaviorState.REST, dtype=np.int8)
    states[walking] = BehaviorState.RUN
    states[turn_mask] = BehaviorState.TURN

    trj.behavior_detector = BEHAVIOR_STATE_DETECTOR_SCHMITT_BUTTERWORTH
    trj.behavior_body_speed_mm_s = speed
    trj.behavior_path_angular_speed_rad_s = np.deg2rad(final_angular)
    trj.behavior_turn_mask = turn_mask
    trj.behavior_state = states
    trj.behavior_sb_x_smooth_px = x_smooth
    trj.behavior_sb_y_smooth_px = y_smooth
    trj.behavior_sb_preliminary_walking_mask = preliminary_walking
    trj.behavior_sb_extended_walking_mask = extended_walking
    trj.behavior_sb_walking_mask = walking
    trj.behavior_sb_raw_angular_velocity_deg_s = raw_angular
    trj.behavior_sb_butterworth_angular_velocity_deg_s = butter_angular
    trj.behavior_sb_angular_velocity_deg_s = final_angular
    trj.behavior_sb_peak_indices = peaks
    trj.behavior_sb_turn_events = events
    trj.behavior_sb_wall_contact_mask = wall_contact
    return states


def analyze_trajectory_behavior_states(
    trj,
    *,
    config: BehaviorStateConfig
    | SchmittButterworthConfig = DEFAULT_BEHAVIOR_STATE_CONFIG,
) -> np.ndarray | None:
    if isinstance(config, SchmittButterworthConfig):
        return analyze_trajectory_schmitt_butterworth_states(trj, config=config)

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
        px_per_mm=px_per_mm,
        dt_s=dt_s,
        body_speed_mm_s=body_speed,
        config=config,
    )
    segment_speed = raw_segment_speed_mm_s(
        trj.x,
        trj.y,
        fps=trj.va.fps,
        px_per_mm=px_per_mm,
        dt_s=dt_s,
    )
    if config.turn_rescue_pivot_pauses:
        pivot_mask = pivot_pause_turn_mask(
            trj.x,
            trj.y,
            segment_speed,
            body_speed_mm_s=body_speed,
            config=config,
        )
    else:
        pivot_mask = np.zeros_like(segment_speed, dtype=bool)
    trj.behavior_detector = BEHAVIOR_STATE_DETECTOR_HABERKERN
    trj.behavior_body_speed_mm_s = body_speed
    trj.behavior_head_speed_mm_s = head_speed
    trj.behavior_angular_speed_rad_s = angular_speed
    trj.behavior_path_angular_speed_rad_s = path_angular_speed
    trj.behavior_raw_segment_speed_mm_s = segment_speed
    trj.behavior_pivot_turn_mask = pivot_mask
    trj.behavior_turn_mask = find_turns(
        angular_speed,
        head_speed,
        body_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
        pivot_turn_mask=pivot_mask,
    )
    trj.behavior_state = classify_behavior_states(
        body_speed,
        angular_speed,
        head_speed,
        config=config,
        path_angular_speed_rad_s=path_angular_speed,
        pivot_turn_mask=pivot_mask,
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
