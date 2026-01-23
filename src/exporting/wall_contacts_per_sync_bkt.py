from __future__ import annotations

import os
from bisect import bisect_right
from typing import Any

import numpy as np


def _get_wall_contact_regions(trj) -> list[Any]:
    """
    Return boundary_contact_regions for wall/all/edge, or [] if missing.
    """
    try:
        return trj.boundary_event_stats["wall"]["all"]["edge"][
            "boundary_contact_regions"
        ]
    except Exception:
        return []


def _infer_role_idx(trj) -> int:
    """
    Role id for this trajectory:
      - trj.f is the role id (0=exp, 1=yok)

    We keep a tiny bit of defensive coding in case trj.f is missing or weird.
    """
    if not hasattr(trj, "f"):
        return 0

    try:
        role = int(getattr(trj, "f"))
    except Exception:
        return 0

    # Accept the common 0/1 encoding; fall back to exp on unexpected values
    if role in (0, 1):
        return role
    return 0


def _count_regions_by_bucket_start(regions, bucket_ranges) -> np.ndarray:
    """
    Count regions by which sync bucket contains region.start.

    Counting by start-frame prevents double-counting events that span buckets.
    """
    B = len(bucket_ranges)
    counts = np.zeros(B, dtype=np.int32)
    if B == 0 or not regions:
        return counts

    starts = [int(s) for (s, _) in bucket_ranges]
    ends = [int(e) for (_, e) in bucket_ranges]

    for r in regions:
        sf = int(r.start)
        b = bisect_right(starts, sf) - 1
        if b < 0 or b >= B:
            continue
        if sf < ends[b]:
            counts[b] += 1

    return counts


def build_wall_contacts_per_sync_bkt_payload(va) -> dict:
    """
    Build a lightweight payload for one VideoAnalysis instance.
    Must be called before clean_up_boundary_contact_data().
    """
    trn_1based = int(getattr(va.opts, "wall_contacts_trn", 2))
    t_idx = trn_1based - 1

    if not hasattr(va, "sync_bucket_ranges"):
        raise RuntimeError(
            "va.sync_bucket_ranges not found. Ensure bySyncBucket() ran."
        )
    if t_idx < 0 or t_idx >= len(va.sync_bucket_ranges):
        raise ValueError(f"--wall-contacts-trn={trn_1based} out of range")

    bucket_ranges = va.sync_bucket_ranges[t_idx]
    B = len(bucket_ranges)

    # scalar fly_id (chamber location id)
    try:
        fly_id_scalar = int(getattr(va, "f"))
    except Exception:
        fly_id_scalar = -1

    video_fn = getattr(va, "fn", None)
    video_basename = os.path.basename(video_fn) if video_fn else ""

    role_idx = []
    fly_id = []
    trj_idx = []
    counts_per_bucket = []
    mean_per_bucket = []
    n_buckets = []

    for i, trj in enumerate(va.trx):
        if hasattr(trj, "bad") and trj.bad():
            continue

        regions = _get_wall_contact_regions(trj)
        counts = _count_regions_by_bucket_start(regions, bucket_ranges)

        role_idx.append(_infer_role_idx(trj))
        fly_id.append(fly_id_scalar)
        trj_idx.append(int(i))

        counts_per_bucket.append(counts.astype(np.int32))
        mean_per_bucket.append(float(np.mean(counts)) if B > 0 else float("nan"))
        n_buckets.append(int(B))

    return {
        "training_idx": int(trn_1based),
        "video_basename": video_basename,
        "role_idx": np.asarray(role_idx, dtype=np.int32),
        "fly_id": np.asarray(fly_id, dtype=np.int32),
        "trj_idx": np.asarray(trj_idx, dtype=np.int32),
        "counts_per_bucket": counts_per_bucket,  # keep ragged list for now
        "mean_per_bucket": np.asarray(mean_per_bucket, dtype=np.float32),
        "n_buckets": np.asarray(n_buckets, dtype=np.int32),
    }


def save_wall_contacts_per_sync_bkt_npz(out_path: str, payloads: list[dict]) -> None:
    """
    Combine per-VA payloads into one NPZ.
    Pads counts_per_bucket to max bucket count across payloads with -1.
    """
    if not payloads:
        raise ValueError("No payloads to save")

    # Gather rows
    role_all = []
    fly_all = []
    trj_all = []
    mean_all = []
    nb_all = []
    vid_all = []
    counts_rows = []

    training_idx = payloads[0].get("training_idx", -1)

    for p in payloads:
        if int(p.get("training_idx", -1)) != int(training_idx):
            # keep it strict; prevents mixing T1/T2 accidentally
            raise ValueError("Mixed training_idx in wall contacts payloads")

        role = np.asarray(p["role_idx"], dtype=np.int32)
        fly = np.asarray(p["fly_id"], dtype=np.int32)
        trj = np.asarray(p["trj_idx"], dtype=np.int32)
        mean = np.asarray(p["mean_per_bucket"], dtype=np.float32)
        nb = np.asarray(p["n_buckets"], dtype=np.int32)
        vid = str(p.get("video_basename", ""))

        # counts_per_bucket is ragged list aligned with rows
        c_list = p["counts_per_bucket"]
        if len(c_list) != role.shape[0]:
            raise ValueError("counts_per_bucket length mismatch with role_idx")

        for i in range(role.shape[0]):
            role_all.append(role[i])
            fly_all.append(fly[i])
            trj_all.append(trj[i])
            mean_all.append(mean[i])
            nb_all.append(nb[i])
            vid_all.append(vid)
            counts_rows.append(np.asarray(c_list[i], dtype=np.int32))

    # Pad counts to max B with -1
    B_max = max((row.size for row in counts_rows), default=0)
    counts_mat = np.full((len(counts_rows), B_max), -1, dtype=np.int32)
    for i, row in enumerate(counts_rows):
        counts_mat[i, : row.size] = row

    np.savez_compressed(
        out_path,
        role_idx=np.asarray(role_all, dtype=np.int32),
        fly_id=np.asarray(fly_all, dtype=np.int32),
        trj_idx=np.asarray(trj_all, dtype=np.int32),
        mean_per_bucket=np.asarray(mean_all, dtype=np.float32),
        n_buckets=np.asarray(nb_all, dtype=np.int32),
        video_basename=np.asarray(vid_all, dtype=object),
        counts_per_bucket=counts_mat,
        training_idx=np.int32(training_idx),
    )
