import numpy as np


def _angle_delta(a, b):
    d = abs(b - a)
    return min(d, 2 * np.pi - d)


def _speed_frame_for_velocity_angle(angle_idx):
    return angle_idx + 1


def _velocity_angle_segment_speed(trj, angle_idx):
    speed_idx = _speed_frame_for_velocity_angle(angle_idx)
    if speed_idx < 0 or speed_idx >= len(trj.sp):
        return np.nan, speed_idx
    return trj.sp[speed_idx], speed_idx


def add_circle_turn_fields(trj, va, stats, opts):
    """
    Computes turning/near_turning/turning_indices for the supplied single-radius
    stats dict (in-place).  Re-implements the essentials of
    EllipseToBoundaryDistCalculator.find_subset_at_or_below_duration but
    without any wall geometry assumptions.
    """
    if "turning" in stats:  # already done
        return

    contact_regions = stats["boundary_contact_regions"]
    frames = len(stats["boundary_contact"])
    turning_flag = np.zeros(frames, dtype=np.uint8)
    turning_idxs = []
    totals = []
    rejection_reasons = []
    turn_angle_diagnostics = []

    dur_frames = int(round(opts.turn_duration_thresh * va.fps))
    min_angle = np.pi * opts.min_vel_angle_delta / 180
    min_speed = opts.min_turn_speed
    if hasattr(trj, "pxPerMmFloor"):
        min_speed = trj.pxPerMmFloor * min_speed

    for idx, reg in enumerate(contact_regions):
        angle_window_start = max(0, reg.start - 1)
        angle_window_stop = reg.stop
        event_diag = {
            "start_frame": reg.start,
            "end_frame": reg.stop,
            "duration_frames": reg.stop - reg.start,
            "angle_window_start_frame": angle_window_start,
            "angle_window_stop_frame": angle_window_stop,
            "min_turn_speed_px_per_frame": min_speed,
            "speed_gate": "velocity_angle_forward_segment",
            "low_speed_frames": [],
            "low_speed_speed_frames": [],
            "wall_contact_start_frames": [],
            "used_angle_pairs": [],
            "used_speed_frames": [],
            "vel_angle_deltas": [],
            "angle_pair_mode": "boundary_crossing_window",
            "apex_bridge_mode": "",
            "apex_bridge_pair": None,
            "apex_bridge_rejection": "",
            "total_vel_angle_delta": np.nan,
        }

        if reg.stop - reg.start > dur_frames:
            rejection_reasons.append("too_long")
            totals.append(np.nan)
            turn_angle_diagnostics.append(event_diag)
            continue

        vel_deltas = []
        i = angle_window_start
        while i < angle_window_stop:
            speed, speed_idx = _velocity_angle_segment_speed(trj, i)
            if not np.isfinite(speed) or speed < min_speed:
                event_diag["low_speed_frames"].append(i)
                event_diag["low_speed_speed_frames"].append(speed_idx)
                i += 1
                continue
            j = i + 1
            j_speed, j_speed_idx = _velocity_angle_segment_speed(trj, j)
            while j < angle_window_stop and (
                not np.isfinite(j_speed) or j_speed < min_speed
            ):
                event_diag["low_speed_frames"].append(j)
                event_diag["low_speed_speed_frames"].append(j_speed_idx)
                j += 1
                j_speed, j_speed_idx = _velocity_angle_segment_speed(trj, j)
            if j >= reg.stop:
                break
            delta = _angle_delta(trj.velAngles[i], trj.velAngles[j])
            vel_deltas.append(delta)
            event_diag["used_angle_pairs"].append((i, j))
            event_diag["used_speed_frames"].append((speed_idx, j_speed_idx))
            event_diag["vel_angle_deltas"].append(delta)
            i = j

        total = abs(np.sum(vel_deltas))
        event_diag["low_speed_frames"] = sorted(set(event_diag["low_speed_frames"]))
        event_diag["low_speed_speed_frames"] = sorted(
            set(event_diag["low_speed_speed_frames"])
        )
        event_diag["total_vel_angle_delta"] = total
        totals.append(total)
        turn_angle_diagnostics.append(event_diag)

        if total >= min_angle:
            turning_flag[reg.start : reg.stop] = 1
            turning_idxs.append(idx)
            rejection_reasons.append("turn")
        elif not vel_deltas:
            rejection_reasons.append("no_velocity_angle_pairs")
        else:
            rejection_reasons.append("too_little_velocity_angle_change")

    stats["turning"] = turning_flag
    stats["near_turning"] = np.zeros(frames, dtype=np.uint8)  # keep layout
    stats["turning_indices"] = turning_idxs
    stats["total_vel_angle_deltas"] = totals
    stats["frames_to_skip"] = set()  # parity with wall code
    stats["rejection_reasons"] = rejection_reasons
    stats["turn_angle_diagnostics"] = turn_angle_diagnostics
