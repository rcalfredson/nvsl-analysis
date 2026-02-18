from __future__ import annotations

import os
import numpy as np

from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)


def _dbg(opts, msg):
    if opts is not None and bool(getattr(opts, "turnback_sli_debug", False)):
        print(msg)


def _extract_turnback_arrays(vas, opts=None):
    """
    Returns (all float unless noted):
      ratio_exp:  (n_videos, n_trn, n_sb)
      ratio_ctrl: (n_videos, n_trn, n_sb)  (all-NaN if absent)
      total_exp:  (n_videos, n_trn, n_sb)  int (optional but useful)
      total_ctrl: (n_videos, n_trn, n_sb)  int
    """
    n_videos = len(vas)
    va0 = vas[0]

    # Canonical bucket count: max theoretical # of sync buckets from _syncBucket()
    n_trn = len(getattr(va0, "trns", []))
    n_sb = 0
    try:
        df = va0._numRewardsMsg(True)
        for t in va0.trns:
            _fi, n, _on = va0._syncBucket(t, df)
            # _syncBucket returns n buckets for this training (even if no rewards happen later)
            if n is not None:
                n_sb = max(n_sb, int(n))
        _dbg(
            opts,
            f"[turnback-export] canonical plan: n_trn={n_trn}, n_sb={n_sb} (via _syncBucket)",
        )
    except Exception as e:
        # Fallback: infer from any available computed ratio array
        for va in vas:
            d = getattr(va, "reward_turnback_dual_circle_counts", None)
            if isinstance(d, dict) and d.get("ratio", None) is not None:
                arr = np.asarray(d["ratio"])
                if arr.ndim == 3:
                    n_trn = arr.shape[0]
                    n_sb = max(n_sb, arr.shape[2])
        if n_sb == 0:
            # If we truly cannot infer anything, keep empty
            n_trn = int(n_trn or 0)
            n_sb = 0
        _dbg(
            opts,
            f"[turnback-export] WARNING: _syncBucket failed ({type(e).__name__}: {e}); using fallback inference",
        )
        _dbg(opts, f"[turnback-export] fallback plan: n_trn={n_trn}, n_sb={n_sb}")

    ratio_exp = np.full((n_videos, n_trn, n_sb), np.nan, dtype=float)
    ratio_ctrl = np.full((n_videos, n_trn, n_sb), np.nan, dtype=float)

    total_exp = np.zeros((n_videos, n_trn, n_sb), dtype=int)
    total_ctrl = np.zeros((n_videos, n_trn, n_sb), dtype=int)

    for vi, va in enumerate(vas):
        d = getattr(va, "reward_turnback_dual_circle_counts", None)
        vid = getattr(va, "fn", f"va_{vi}")

        if not isinstance(d, dict) or d.get("ratio", None) is None:
            _dbg(
                opts,
                f"[turnback-export] {vid}: MISSING reward_turnback_dual_circle_counts['ratio']",
            )
            continue

        ratio = np.asarray(d["ratio"])
        _dbg(
            opts,
            f"[turnback-export] {vid}: ratio shape={getattr(ratio, 'shape', None)}",
        )

        if ratio.ndim != 3:
            _dbg(
                opts,
                f"[turnback-export] {vid}: SKIP (ratio.ndim={ratio.ndim}, expected 3)",
            )
            continue
        if ratio.shape[0] != n_trn:
            # keep NaNs rather than misalign
            _dbg(
                opts,
                f"[turnback-export] {vid}: SKIP (n_trn mismatch: ratio.shape[0]={ratio.shape[0]} vs {n_trn})",
            )
            continue
        nb_eff = min(n_sb, ratio.shape[2])

        if ratio.shape[2] < n_sb:
            _dbg(
                opts,
                f"[turnback-export] {vid}: PAD (ratio nb={ratio.shape[2]} < canonical n_sb={n_sb})",
            )
        elif ratio.shape[2] > n_sb:
            _dbg(
                opts,
                f"[turnback-export] {vid}: TRUNC (ratio nb={ratio.shape[2]} > canonical n_sb={n_sb})",
            )

        # exp is fly 0
        exp_slice = ratio[:, 0, :nb_eff]
        _dbg(
            opts,
            f"[turnback-export] {vid}: exp finite={np.isfinite(exp_slice).sum()} / {exp_slice.size}",
        )
        ratio_exp[vi, :, :nb_eff] = exp_slice

        # ctrl is fly 1 if present
        if ratio.shape[1] > 1:
            ctrl_slice = ratio[:, 1, :nb_eff]
            _dbg(
                opts,
                f"[turnback-export] {vid}: ctrl finite={np.isfinite(ctrl_slice).sum()} / {ctrl_slice.size}",
            )
            ratio_ctrl[vi, :, :nb_eff] = ctrl_slice

        # counts are optional but usually present
        tot = d.get("total", None)
        if tot is not None:
            tot = np.asarray(tot)
            if tot.shape == ratio.shape:
                total_exp[vi, :, :nb_eff] = tot[:, 0, :nb_eff]
                if tot.shape[1] > 1:
                    total_ctrl[vi, :, :nb_eff] = tot[:, 1, :nb_eff]

    _dbg(
        opts,
        f"[turnback-export] FINAL ratio_exp finite={np.isfinite(ratio_exp).sum()} / {ratio_exp.size}",
    )
    _dbg(
        opts,
        f"[turnback-export] FINAL ratio_ctrl finite={np.isfinite(ratio_ctrl).sum()} / {ratio_ctrl.size}",
    )

    # Helpful “where are the NaNs” summaries:
    if n_trn and n_sb:
        per_bucket = np.isfinite(ratio_exp).sum(axis=(0, 1))  # (n_sb,)
        per_trn = np.isfinite(ratio_exp).sum(axis=(0, 2))  # (n_trn,)
        _dbg(opts, f"[turnback-export] exp finite per training: {per_trn.tolist()}")
        _dbg(
            opts,
            f"[turnback-export] exp finite per bucket (first 10): {per_bucket[:10].tolist()}",
        )
        _dbg(
            opts,
            f"[turnback-export] exp finite per bucket (last 10): {per_bucket[-10:].tolist()}",
        )

    return ratio_exp, ratio_ctrl, total_exp, total_ctrl


def export_turnback_sli_bundle(vas, opts, gls, out_fn):
    """
    Writes an .npz with:
      - sli: (n_videos,)
      - turnback_ratio_exp:  (n_videos, n_trn, n_sb)
      - turnback_ratio_ctrl: (n_videos, n_trn, n_sb)
      - turnback_total_exp / ctrl: (n_videos, n_trn, n_sb) int
      - group_label, bucket_len_min, training_names, video_ids, SLI metadata
    """
    from analyze import bucketLenForType

    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return

    va0 = vas_ok[0]
    group_label = _safe_group_label(opts, gls)

    _dbg(opts, f"[turnback-export] out={out_fn}")
    _dbg(
        opts,
        f"[turnback-export] group_label={group_label!r}  n_videos={len(vas_ok)}  noyc={getattr(va0, 'noyc', None)}",
    )

    ratio_exp, ratio_ctrl, total_exp, total_ctrl = _extract_turnback_arrays(
        vas_ok, opts
    )

    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as e:
        print(f"[export] WARNING: failed to compute SLI for turnback bundle: {e}")
        sli = np.full((len(vas_ok),), np.nan, dtype=float)
        sli_ts = np.full((len(vas_ok), ratio_exp.shape[1], ratio_exp.shape[2]), np.nan)

    _dbg(opts, f"[turnback-export] SLI finite={np.isfinite(sli).sum()} / {sli.size}")
    _dbg(opts, f"[turnback-export] sli_ts shape={getattr(sli_ts, 'shape', None)}")

    # Metadata
    try:
        bl, _blf = bucketLenForType("turnback_dual_circle")
        bucket_len_min = float(bl)
    except Exception:
        bucket_len_min = np.nan

    try:
        training_names = np.array([t.name() for t in va0.trns], dtype=object)
    except Exception:
        training_names = np.array([], dtype=object)

    try:
        video_fns = np.array([getattr(va, "fn", "") for va in vas_ok], dtype=object)
        fly_ids = np.array([int(getattr(va, "f", -1)) for va in vas_ok], dtype=int)
        video_ids = np.array(
            [f"{fn}::f{f}" for fn, f in zip(video_fns, fly_ids)], dtype=object
        )
    except Exception:
        video_fns = np.array([""] * len(vas_ok), dtype=object)
        fly_ids = np.array([-1] * len(vas_ok), dtype=int)
        video_ids = np.array([f"va_{i}" for i in range(len(vas_ok))], dtype=object)

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(
        out_fn,
        sli=sli,
        sli_ts=sli_ts,
        turnback_ratio_exp=ratio_exp,
        turnback_ratio_ctrl=ratio_ctrl,
        turnback_total_exp=total_exp,
        turnback_total_ctrl=total_ctrl,
        group_label=np.array(group_label, dtype=object),
        bucket_len_min=np.array(bucket_len_min, dtype=float),
        training_names=training_names,
        video_fns=video_fns,
        fly_ids=fly_ids,
        video_ids=video_ids,
        sli_training_idx=np.array(getattr(opts, "best_worst_trn", 1) - 1, dtype=int),
        sli_use_training_mean=np.array(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
        turnback_inner_delta_mm=np.array(
            float(getattr(opts, "turnback_inner_delta_mm", 0.0)), dtype=float
        ),
        turnback_inner_radius_offset_px=np.array(
            float(getattr(opts, "turnback_inner_radius_offset_px", 0.0)), dtype=float
        ),
    )
    print(f"[export] Wrote turnback+SLI bundle: {out_fn} (n={len(vas_ok)})")
    _dbg(
        opts,
        f"[turnback-export] saved keys: turnback_ratio_exp/ctrl, turnback_total_exp/ctrl, sli, sli_ts, metadata",
    )
