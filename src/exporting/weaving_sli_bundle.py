from __future__ import annotations

import os
import numpy as np

from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)


def _dbg(opts, msg: str) -> None:
    if opts is not None and bool(getattr(opts, "weaving_sli_debug", False)):
        print(msg)


def _canonical_n_trn_n_sb(vas, opts=None) -> tuple[int, int]:
    """
    Determine canonical (n_trn, n_sb) for exporting arrays.

    Strategy (matches other bundle exporters):
      1) Prefer max theoretical bucket count from _syncBucket for each training.
      2) Fallback: infer from sync_bucket_ranges lengths across VAs.
      3) Final fallback: keep n_sb=0.
    """
    va0 = vas[0]
    n_trn = len(getattr(va0, "trns", []))
    n_sb = 0

    # 1) Canonical via _syncBucket
    try:
        df = va0._numRewardsMsg(True)
        for t in va0.trns:
            _fi, n, _on = va0._syncBucket(t, df)
            if n is not None:
                n_sb = max(n_sb, int(n))
        _dbg(
            opts,
            f"[weaving-export] canonical plan: n_trn={n_trn}, n_sb={n_sb} (via _syncBucket)",
        )
        return n_trn, n_sb
    except Exception as e:
        _dbg(
            opts,
            f"[weaving-export] WARNING: _syncBucket failed ({type(e).__name__}: {e}); trying sync_bucket_ranges fallback",
        )

    # 2) Fallback via sync_bucket_ranges (effective buckets)
    try:
        for va in vas:
            sbr = getattr(va, "sync_bucket_ranges", None)
            if isinstance(sbr, list):
                n_trn = max(n_trn, len(sbr))
                for ranges in sbr:
                    if isinstance(ranges, list):
                        n_sb = max(n_sb, len(ranges))
        _dbg(
            opts,
            f"[weaving-export] fallback plan: n_trn={n_trn}, n_sb={n_sb} (via sync_bucket_ranges)",
        )
    except Exception as e:
        _dbg(
            opts,
            f"[weaving-export] WARNING: sync_bucket_ranges fallback failed ({type(e).__name__}: {e})",
        )

    n_trn = int(n_trn or 0)
    n_sb = int(n_sb or 0)
    return n_trn, n_sb


def _extract_weaving_arrays(vas, opts=None):
    """
    Extract weaving per-exit ratio per sync bucket.

    Source (new Cython accumulation):
        va.weaving_exit_stats_sb[fly_idx][training_idx] = (weaving_counts[B], total_exits[B])

    Returns:
      ratio_exp:  (n_videos, n_trn, n_sb) float
      ratio_ctrl: (n_videos, n_trn, n_sb) float (all-NaN if absent)
      total_exp:  (n_videos, n_trn, n_sb) int
      total_ctrl: (n_videos, n_trn, n_sb) int
      weave_exp:  (n_videos, n_trn, n_sb) int  numerator
      weave_ctrl: (n_videos, n_trn, n_sb) int
    """
    n_videos = len(vas)
    n_trn, n_sb = _canonical_n_trn_n_sb(vas, opts)

    ratio_exp = np.full((n_videos, n_trn, n_sb), np.nan, dtype=float)
    ratio_ctrl = np.full((n_videos, n_trn, n_sb), np.nan, dtype=float)

    total_exp = np.zeros((n_videos, n_trn, n_sb), dtype=int)
    total_ctrl = np.zeros((n_videos, n_trn, n_sb), dtype=int)

    weave_exp = np.zeros((n_videos, n_trn, n_sb), dtype=int)
    weave_ctrl = np.zeros((n_videos, n_trn, n_sb), dtype=int)

    if n_trn == 0 or n_sb == 0:
        _dbg(opts, "[weaving-export] n_trn==0 or n_sb==0; returning empty arrays")
        return ratio_exp, ratio_ctrl, total_exp, total_ctrl, weave_exp, weave_ctrl

    for vi, va in enumerate(vas):
        vid = getattr(va, "fn", f"va_{vi}")
        stats = getattr(va, "weaving_exit_stats_sb", None)

        if not isinstance(stats, list) or len(stats) == 0:
            _dbg(opts, f"[weaving-export] {vid}: MISSING weaving_exit_stats_sb")
            continue

        for fly_idx, (ratio_out, total_out, weave_out) in [
            (0, (ratio_exp, total_exp, weave_exp)),
            (1, (ratio_ctrl, total_ctrl, weave_ctrl)),
        ]:
            if fly_idx >= len(stats):
                continue
            per_trn = stats[fly_idx]
            if not isinstance(per_trn, list) or len(per_trn) == 0:
                continue

            # Fill per training
            for t_idx in range(min(n_trn, len(per_trn))):
                entry = per_trn[t_idx]
                if entry is None:
                    continue
                try:
                    w_list, tot_list = entry
                except Exception:
                    _dbg(
                        opts,
                        f"[weaving-export] {vid}: bad entry at fly={fly_idx} trn={t_idx}: {entry!r}",
                    )
                    continue

                w = np.asarray(w_list, dtype=int)
                tot = np.asarray(tot_list, dtype=int)
                nb_eff = min(n_sb, w.size, tot.size)
                if nb_eff <= 0:
                    continue

                weave_out[vi, t_idx, :nb_eff] = w[:nb_eff]
                total_out[vi, t_idx, :nb_eff] = tot[:nb_eff]

                with np.errstate(divide="ignore", invalid="ignore"):
                    r = w[:nb_eff] / tot[:nb_eff].astype(float)
                r[tot[:nb_eff] <= 0] = np.nan
                ratio_out[vi, t_idx, :nb_eff] = r

        _dbg(
            opts,
            f"[weaving-export] {vid}: exp finite={np.isfinite(ratio_exp[vi]).sum()} / {ratio_exp[vi].size}",
        )

    _dbg(
        opts,
        f"[weaving-export] FINAL ratio_exp finite={np.isfinite(ratio_exp).sum()} / {ratio_exp.size}",
    )
    _dbg(
        opts,
        f"[weaving-export] FINAL ratio_ctrl finite={np.isfinite(ratio_ctrl).sum()} / {ratio_ctrl.size}",
    )

    if n_trn and n_sb:
        per_bucket = np.isfinite(ratio_exp).sum(axis=(0, 1))
        per_trn = np.isfinite(ratio_exp).sum(axis=(0, 2))
        _dbg(opts, f"[weaving-export] exp finite per training: {per_trn.tolist()}")
        _dbg(
            opts,
            f"[weaving-export] exp finite per bucket (first 10): {per_bucket[:10].tolist()}",
        )
        _dbg(
            opts,
            f"[weaving-export] exp finite per bucket (last 10): {per_bucket[-10:].tolist()}",
        )

    return ratio_exp, ratio_ctrl, total_exp, total_ctrl, weave_exp, weave_ctrl


def export_weaving_sli_bundle(vas, opts, gls, out_fn: str) -> None:
    """
    Write an .npz bundle for weaving-per-exit ratio per sync bucket, plus SLI.

    Saved keys:
      - sli: (n_videos,)
      - sli_ts: (n_videos, n_trn, n_sb)
      - weaving_ratio_exp/ctrl: (n_videos, n_trn, n_sb)
      - weaving_total_exp/ctrl: (n_videos, n_trn, n_sb) int
      - weaving_count_exp/ctrl: (n_videos, n_trn, n_sb) int (numerator)
      - metadata: group_label, bucket_len_min, training_names, video_fns, fly_ids, video_ids, SLI settings
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

    _dbg(opts, f"[weaving-export] out={out_fn}")
    _dbg(
        opts,
        f"[weaving-export] group_label={group_label!r}  n_videos={len(vas_ok)}  noyc={getattr(va0, 'noyc', None)}",
    )

    ratio_exp, ratio_ctrl, total_exp, total_ctrl, weave_exp, weave_ctrl = (
        _extract_weaving_arrays(vas_ok, opts)
    )

    # SLI (same pattern as other bundles)
    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as e:
        print(f"[export] WARNING: failed to compute SLI for weaving bundle: {e}")
        sli = np.full((len(vas_ok),), np.nan, dtype=float)
        sli_ts = np.full(
            (len(vas_ok), ratio_exp.shape[1], ratio_exp.shape[2]), np.nan, dtype=float
        )

    _dbg(opts, f"[weaving-export] SLI finite={np.isfinite(sli).sum()} / {sli.size}")
    _dbg(opts, f"[weaving-export] sli_ts shape={getattr(sli_ts, 'shape', None)}")

    # Metadata
    try:
        bl, _blf = bucketLenForType("weaving")
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

    # Save
    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(
        out_fn,
        sli=sli,
        sli_ts=sli_ts,
        weaving_ratio_exp=ratio_exp,
        weaving_ratio_ctrl=ratio_ctrl,
        weaving_total_exp=total_exp,
        weaving_total_ctrl=total_ctrl,
        weaving_count_exp=weave_exp,
        weaving_count_ctrl=weave_ctrl,
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
        sli_select_skip_first_sync_buckets=np.array(
            (
                0
                if getattr(opts, "sli_select_skip_first_sync_buckets", None) is None
                else max(0, int(getattr(opts, "sli_select_skip_first_sync_buckets")))
            ),
            dtype=int,
        ),
        sli_select_keep_first_sync_buckets=np.array(
            (
                0
                if getattr(opts, "sli_select_keep_first_sync_buckets", None) is None
                else max(0, int(getattr(opts, "sli_select_keep_first_sync_buckets")))
            ),
            dtype=int,
        ),
        weaving_definition=np.array(
            "weaving_ratio = (# reward-circle exits classified as weaving AND not a large turn) / (total reward-circle exits) per sync bucket; exp=fly0 ctrl=fly1",
            dtype=object,
        ),
    )
    print(f"[export] Wrote weaving+SLI bundle: {out_fn} (n={len(vas_ok)})")
    _dbg(
        opts,
        "[weaving-export] saved keys: weaving_ratio_*, weaving_total_*, weaving_count_*, sli, sli_ts, metadata",
    )
