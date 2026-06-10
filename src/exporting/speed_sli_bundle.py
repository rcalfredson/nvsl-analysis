from __future__ import annotations

import numpy as np

from src.analysis.sync_bucket_presence_filters import (
    exp_target_sync_bucket_eligibility_mask,
    exp_target_sync_bucket_filter_payload,
    mask_by_exp_target_sync_bucket_filter,
)
from src.analysis.sli_bundle_utils import normalize_sli_bundle
from src.exporting.bundle_utils import build_metric_plus_sli_bundle, save_sli_bundle


def _canonical_bucket_plan(vas, opts) -> tuple[int, int, float]:
    va0 = vas[0]
    n_trn = len(getattr(va0, "trns", []))
    try:
        df = va0._min2f(float(getattr(opts, "syncBucketLenMin")))
    except Exception:
        df = va0._numRewardsMsg(True, silent=True)

    n_sb = 0
    for va in vas:
        for trn in getattr(va, "trns", []):
            try:
                _fi, n_here, _on = va._syncBucket(trn, df)
            except Exception:
                continue
            if n_here is not None:
                n_sb = max(n_sb, int(n_here))

    try:
        bucket_len_min = float(df) / float(va0.fps) / 60.0
    except Exception:
        bucket_len_min = np.nan

    return int(n_trn), int(n_sb), float(bucket_len_min)


def _wall_valid_mask(traj, start: int, stop: int) -> np.ndarray | None:
    try:
        boundary_contact = traj.boundary_event_stats["wall"]["all"]["opp_edge"][
            "boundary_contact"
        ]
    except Exception:
        return None

    mask = ~np.asarray(boundary_contact[start:stop], dtype=bool)
    return mask


def _mean_speed_mm_s(
    traj, start: int, stop: int, *, exclude_wall: bool
) -> tuple[float, int]:
    if stop <= start:
        return np.nan, 0
    if getattr(traj, "bad", lambda: False)():
        return np.nan, 0

    sp = np.asarray(getattr(traj, "sp", []), dtype=float)
    if sp.size == 0:
        return np.nan, 0

    start = max(0, int(start))
    stop = min(int(stop), sp.size)
    if stop <= start:
        return np.nan, 0

    values = sp[start:stop]
    if exclude_wall:
        valid_wall = _wall_valid_mask(traj, start, stop)
        if valid_wall is not None and valid_wall.shape[0] == values.shape[0]:
            values = values[valid_wall]

    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, 0

    px_per_mm = float(getattr(traj, "pxPerMmFloor", np.nan))
    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
        return np.nan, int(values.size)
    return float(np.mean(values) / px_per_mm), int(values.size)


def _extract_speed_arrays(vas, opts=None) -> dict[str, np.ndarray]:
    n_videos = len(vas)
    n_trn, n_sb, _bucket_len_min = _canonical_bucket_plan(vas, opts)
    speed_exp = np.full((n_videos, n_trn, n_sb), np.nan, dtype=float)
    speed_ctrl = np.full((n_videos, n_trn, n_sb), np.nan, dtype=float)
    speedN_exp = np.zeros((n_videos, n_trn, n_sb), dtype=int)
    speedN_ctrl = np.zeros((n_videos, n_trn, n_sb), dtype=int)

    if n_videos == 0 or n_trn == 0 or n_sb == 0:
        return {
            "speed_exp": speed_exp,
            "speed_ctrl": speed_ctrl,
            "speedN_exp": speedN_exp,
            "speedN_ctrl": speedN_ctrl,
        }

    va0 = vas[0]
    try:
        df = va0._min2f(float(getattr(opts, "syncBucketLenMin")))
    except Exception:
        df = va0._numRewardsMsg(True, silent=True)
    exclude_wall = bool(getattr(opts, "excl_wall_for_spd", False))

    for vi, va in enumerate(vas):
        for ti, trn in enumerate(getattr(va, "trns", [])):
            if ti >= n_trn:
                break
            try:
                fi, n_here, _on = va._syncBucket(trn, df)
            except Exception:
                continue
            if fi is None or n_here is None:
                continue
            nb_eff = min(n_sb, int(n_here))
            for fly_key, fly_idx in (("exp", 0), ("ctrl", 1)):
                if fly_idx >= len(getattr(va, "trx", [])):
                    continue
                target = speed_exp if fly_key == "exp" else speed_ctrl
                target_n = speedN_exp if fly_key == "exp" else speedN_ctrl
                traj = va.trx[fly_idx]
                for b_idx in range(nb_eff):
                    start = int(fi + b_idx * df)
                    stop = int(start + df)
                    if stop > int(trn.stop):
                        continue
                    mean_speed, n_frames = _mean_speed_mm_s(
                        traj, start, stop, exclude_wall=exclude_wall
                    )
                    target[vi, ti, b_idx] = mean_speed
                    target_n[vi, ti, b_idx] = n_frames

    return {
        "speed_exp": speed_exp,
        "speed_ctrl": speed_ctrl,
        "speedN_exp": speedN_exp,
        "speedN_ctrl": speedN_ctrl,
    }


def build_speed_sli_bundle(vas, opts, gls) -> dict:
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    bundle = build_metric_plus_sli_bundle(
        vas,
        opts,
        gls,
        extract_metric_arrays=lambda vas_ok: _extract_speed_arrays(vas_ok, opts),
        bucket_type="speed",
        print_label="speed",
        require_3d=True,
    )

    target_sync_bucket_eligible = exp_target_sync_bucket_eligibility_mask(vas_ok, opts)
    bundle["speed_exp"] = mask_by_exp_target_sync_bucket_filter(
        bundle["speed_exp"], target_sync_bucket_eligible
    )
    bundle["sli"] = mask_by_exp_target_sync_bucket_filter(
        bundle["sli"], target_sync_bucket_eligible
    )
    bundle["sli_ts"] = mask_by_exp_target_sync_bucket_filter(
        bundle["sli_ts"], target_sync_bucket_eligible
    )
    bundle["speed_units"] = np.asarray("mm/s", dtype=object)
    bundle["speed_exclude_wall_contact"] = np.asarray(
        bool(getattr(opts, "excl_wall_for_spd", False)), dtype=bool
    )
    bundle.update(
        exp_target_sync_bucket_filter_payload(
            vas_ok,
            opts,
            prefix="exp_pi_threshold_filter",
        )
    )
    return normalize_sli_bundle(bundle)


def export_speed_sli_bundle(vas, opts, gls, out_fn):
    bundle = build_speed_sli_bundle(vas, opts, gls)
    save_sli_bundle(bundle, out_fn, print_label="speed")
