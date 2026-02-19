# src/exporting/wall_contacts_per_reward_interval.py
from __future__ import annotations

import os
from typing import Any

import numpy as np


def _get_wall_contact_regions(trj) -> list[Any]:
    try:
        return trj.boundary_event_stats["wall"]["all"]["edge"][
            "boundary_contact_regions"
        ]
    except Exception:
        return []


def _infer_role_idx(trj) -> int:
    if not hasattr(trj, "f"):
        return 0
    try:
        role = int(getattr(trj, "f"))
    except Exception:
        return 0
    return role if role in (0, 1) else 0


def _training_window_after_skip(
    va, trn, *, t_idx: int, skip_first: int
) -> tuple[int, int]:
    """
    Return (start_frame, stop_frame) spanning the *contiguous range* of frames covered
    by the included sync buckets (>=1 buckets), after dropping the first `skip_first`
    sync buckets.

    Fallback: (trn.start, trn.stop).
    """
    ranges = getattr(va, "sync_bucket_ranges", None)
    if not ranges or t_idx >= len(ranges) or not ranges[t_idx]:
        return (int(trn.start), int(trn.stop))

    rr = ranges[t_idx]
    if skip_first < 0:
        skip_first = 0
    if skip_first >= len(rr):
        return (0, 0)

    rr2 = rr[skip_first:]
    start = int(rr2[0][0])
    stop = int(rr2[-1][1])
    if stop <= start:
        return (0, 0)
    return (start, stop)


def _sanitize_reward_frames(rf, *, start: int, stop: int) -> np.ndarray:
    if rf is None:
        return np.zeros(0, dtype=np.int64)
    rf = np.asarray(rf, dtype=float)
    rf = rf[np.isfinite(rf)]
    if rf.size == 0:
        return np.zeros(0, dtype=np.int64)

    rf = rf[(rf >= start) & (rf < stop)]
    if rf.size == 0:
        return np.zeros(0, dtype=np.int64)

    rf = np.unique(rf.astype(np.int64))
    rf.sort()
    return rf


def _count_regions_by_reward_interval_start(regions, rf: np.ndarray) -> np.ndarray:
    """
    Count regions by which inter-reward interval contains region.start.
    Intervals: [rf[i], rf[i+1]) for i=0..n_rewards-2.
    """
    if rf is None or rf.size < 2:
        return np.zeros(0, dtype=np.int32)

    starts = rf[:-1]
    ends = rf[1:]
    K = starts.size

    counts = np.zeros(K, dtype=np.int32)
    if not regions:
        return counts

    for r in regions:
        sf = int(r.start)
        if sf < int(starts[0]) or sf >= int(ends[-1]):
            continue
        i = int(np.searchsorted(starts, sf, side="right") - 1)
        if i < 0 or i >= K:
            continue
        if sf < int(ends[i]):
            counts[i] += 1

    return counts


def build_wall_contacts_per_reward_interval_payload(va) -> dict:
    """
    Build payload: wall-contact event counts per inter-reward interval for one training.
    Must be called before clean_up_boundary_contact_data().
    """
    trn_1based = int(getattr(va.opts, "wall_contacts_trn", 2))
    t_idx = trn_1based - 1
    skip_k = int(getattr(va.opts, "skip_first_sync_buckets", 0) or 0)

    if not hasattr(va, "trns"):
        raise RuntimeError("va.trns not found")
    if t_idx < 0 or t_idx >= len(va.trns):
        raise ValueError(f"--wall-contacts-trn={trn_1based} out of range")

    trn = va.trns[t_idx]

    # scalar fly_id (chamber location id)
    try:
        fly_id_scalar = int(getattr(va, "f"))
    except Exception:
        fly_id_scalar = -1

    video_fn = getattr(va, "fn", None)
    video_basename = os.path.basename(video_fn) if video_fn else ""

    # Cache per-role reward frames and per-role window
    rewards_by_role: dict[int, np.ndarray] = {}
    window_by_role: dict[int, tuple[int, int]] = {}

    role_idx = []
    fly_id = []
    trj_idx = []
    counts_per_interval = []
    mean_per_interval = []
    n_intervals = []
    n_rewards = []

    for i, trj in enumerate(va.trx):
        if hasattr(trj, "bad") and trj.bad():
            continue

        role = _infer_role_idx(trj)

        if role not in window_by_role:
            w_start, w_stop = _training_window_after_skip(
                va, trn, t_idx=t_idx, skip_first=skip_k
            )
            window_by_role[role] = (w_start, w_stop)

        w_start, w_stop = window_by_role[role]
        if w_stop <= w_start:
            rf = np.zeros(0, dtype=np.int64)
        else:
            if role not in rewards_by_role:
                rf_raw = va._getOn(trn, calc=False, f=role)
                rewards_by_role[role] = _sanitize_reward_frames(
                    rf_raw, start=w_start, stop=w_stop
                )

            rf = rewards_by_role[role]
        regions = _get_wall_contact_regions(trj)
        counts = _count_regions_by_reward_interval_start(regions, rf)

        role_idx.append(role)
        fly_id.append(fly_id_scalar)
        trj_idx.append(int(i))

        counts_per_interval.append(counts.astype(np.int32))
        mean_per_interval.append(
            float(np.mean(counts)) if counts.size > 0 else float("nan")
        )
        n_intervals.append(int(counts.size))
        n_rewards.append(int(rf.size))

    return {
        "training_idx": int(trn_1based),
        "video_basename": video_basename,
        "role_idx": np.asarray(role_idx, dtype=np.int32),
        "fly_id": np.asarray(fly_id, dtype=np.int32),
        "trj_idx": np.asarray(trj_idx, dtype=np.int32),
        "counts_per_interval": counts_per_interval,  # ragged
        "mean_per_interval": np.asarray(mean_per_interval, dtype=np.float32),
        "n_intervals": np.asarray(n_intervals, dtype=np.int32),
        "n_rewards": np.asarray(n_rewards, dtype=np.int32),
    }


def save_wall_contacts_per_reward_interval_npz(
    out_path: str, payloads: list[dict]
) -> None:
    """
    Combine per-VA payloads into one NPZ.
    Pads counts_per_interval to max interval count across rows with -1.
    """
    if not payloads:
        raise ValueError("No payloads to save")

    role_all = []
    fly_all = []
    trj_all = []
    mean_all = []
    ni_all = []
    nr_all = []
    vid_all = []
    counts_rows = []

    training_idx = payloads[0].get("training_idx", -1)

    for p in payloads:
        if int(p.get("training_idx", -1)) != int(training_idx):
            raise ValueError("Mixed training_idx in wall contacts payloads")

        role = np.asarray(p["role_idx"], dtype=np.int32)
        fly = np.asarray(p["fly_id"], dtype=np.int32)
        trj = np.asarray(p["trj_idx"], dtype=np.int32)
        mean = np.asarray(p["mean_per_interval"], dtype=np.float32)
        ni = np.asarray(p["n_intervals"], dtype=np.int32)
        nr = np.asarray(p["n_rewards"], dtype=np.int32)
        vid = str(p.get("video_basename", ""))

        c_list = p["counts_per_interval"]
        if len(c_list) != role.shape[0]:
            raise ValueError("counts_per_interval length mismatch with role_idx")

        for i in range(role.shape[0]):
            role_all.append(role[i])
            fly_all.append(fly[i])
            trj_all.append(trj[i])
            mean_all.append(mean[i])
            ni_all.append(ni[i])
            nr_all.append(nr[i])
            vid_all.append(vid)
            counts_rows.append(np.asarray(c_list[i], dtype=np.int32))

    K_max = max((row.size for row in counts_rows), default=0)
    counts_mat = np.full((len(counts_rows), K_max), -1, dtype=np.int32)
    for i, row in enumerate(counts_rows):
        counts_mat[i, : row.size] = row

    np.savez_compressed(
        out_path,
        role_idx=np.asarray(role_all, dtype=np.int32),
        fly_id=np.asarray(fly_all, dtype=np.int32),
        trj_idx=np.asarray(trj_all, dtype=np.int32),
        mean_per_interval=np.asarray(mean_all, dtype=np.float32),
        n_intervals=np.asarray(ni_all, dtype=np.int32),
        n_rewards=np.asarray(nr_all, dtype=np.int32),
        video_basename=np.asarray(vid_all, dtype=object),
        counts_per_interval=counts_mat,
        training_idx=np.int32(training_idx),
    )
