import numpy as np


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

    dur_frames = int(round(opts.turn_duration_thresh * va.fps))
    min_angle = np.pi * opts.min_vel_angle_delta / 180
    min_speed = opts.min_turn_speed
    if hasattr(trj, "pxPerMmFloor"):
        min_speed = trj.pxPerMmFloor * min_speed

    for idx, reg in enumerate(contact_regions):
        if reg.stop - reg.start > dur_frames:
            rejection_reasons.append("too_long")
            continue

        vel_deltas = []
        i = reg.start
        while i < reg.stop - 1:
            if trj.sp[i] < min_speed:
                i += 1
                continue
            j = i + 1
            while j < reg.stop and trj.sp[j] < min_speed:
                j += 1
            if j >= reg.stop:
                break
            d = abs(trj.velAngles[j] - trj.velAngles[i])
            vel_deltas.append(min(d, 2 * np.pi - d))
            i = j

        total = abs(np.sum(vel_deltas))
        totals.append(total)

        if total >= min_angle:
            turning_flag[reg.start : reg.stop] = 1
            turning_idxs.append(idx)
            rejection_reasons.append("turn")
        else:
            rejection_reasons.append("too_little_velocity_angle_change")

    stats["turning"] = turning_flag
    stats["near_turning"] = np.zeros(frames, dtype=np.uint8)  # keep layout
    stats["turning_indices"] = turning_idxs
    stats["total_vel_angle_deltas"] = totals
    stats["frames_to_skip"] = set()  # parity with wall code
    stats["rejection_reasons"] = rejection_reasons
