from __future__ import annotations

import os
import numpy as np

from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)


def _dbg(opts, msg):
    if opts is not None and bool(
        getattr(opts, "reward_lgturn_pathlen_sli_debug", False)
    ):
        print(msg)


def _canonical_bucket_plan_from_syncbucket(vas, opts=None):
    """
    Infer canonical (n_trn, n_sb, bucket_len_min) using va._syncBucket().

    Returns:
        n_trn (int): number of trainings
        n_sb (int): max theoretical sync buckets across trainings
        bucket_len_min (float): df / fps / 60
    """
    va0 = vas[0]
    n_trn = len(getattr(va0, "trns", []))
    n_sb = 0
    bucket_len_min = np.nan

    df = va0._numRewardsMsg(True)
    try:
        bucket_len_min = float(df) / float(va0.fps) / 60.0
    except Exception:
        bucket_len_min = np.nan

    for t in getattr(va0, "trns", []):
        _fi, n, _on = va0._syncBucket(t, df)
        if n is not None:
            n_sb = max(n_sb, int(n))

    _dbg(
        opts,
        f"[reward-lgturn-pathlen-export] canonical plan: n_trn={n_trn}, n_sb={n_sb}, bl_min={bucket_len_min:g}",
    )
    return int(n_trn), int(n_sb), float(bucket_len_min)


def _extract_reward_lgturn_pathlen_arrays(vas, opts=None):
    """
    Returns:
      mean_exp:   (n_videos, n_trn, n_sb) float
      mean_ctrl:  (n_videos, n_trn, n_sb) float (all-NaN if absent)
      n_exp:      (n_videos, n_trn, n_sb) int   (# rewards-with-turn per bucket)
      n_ctrl:     (n_videos, n_trn, n_sb) int
      n_rewards:  (n_videos, n_trn, n_sb) int   (total rewards per bucket; shared schedule)
      bucket_len_min: float
    """
    n_videos = len(vas)
    va0 = vas[0]

    # Canonical bucket plan
    try:
        n_trn, n_sb, bucket_len_min = _canonical_bucket_plan_from_syncbucket(vas, opts)
    except Exception as e:
        _dbg(
            opts,
            f"[reward-lgturn-pathlen-export] WARNING: failed canonical plan via _syncBucket: {e}",
        )
        n_trn = len(getattr(va0, "trns", []))
        n_sb = 0
        bucket_len_min = np.nan

    mean_exp = np.full((n_videos, n_trn, n_sb), np.nan, dtype=float)
    mean_ctrl = np.full((n_videos, n_trn, n_sb), np.nan, dtype=float)

    n_exp = np.zeros((n_videos, n_trn, n_sb), dtype=int)
    n_ctrl = np.zeros((n_videos, n_trn, n_sb), dtype=int)
    n_rewards = np.zeros((n_videos, n_trn, n_sb), dtype=int)

    for vi, va in enumerate(vas):
        vid = getattr(va, "fn", f"va_{vi}")

        # Ensure the per-VA sync-bucket metric exists
        if (
            not hasattr(va, "syncMeanRewardLgTurnPathLen")
            or not hasattr(va, "syncMeanRewardLgTurnPathLenN")
            or not hasattr(va, "syncNumRewardsPerBucket")
        ):
            try:
                va.bySyncBucketMeanRewardLgTurnPathLen()
            except Exception as e:
                _dbg(
                    opts,
                    f"[reward-lgturn-pathlen-export] {vid}: FAILED to compute bySyncBucketMeanRewardLgTurnPathLen ({type(e).__name__}: {e})",
                )
                continue

        means = getattr(va, "syncMeanRewardLgTurnPathLen", None)
        counts = getattr(va, "syncMeanRewardLgTurnPathLenN", None)
        totals = getattr(va, "syncNumRewardsPerBucket", None)

        if (
            not isinstance(means, list)
            or not isinstance(counts, list)
            or not isinstance(totals, list)
        ):
            _dbg(
                opts,
                f"[reward-lgturn-pathlen-export] {vid}: MISSING or invalid syncMeanRewardLgTurnPathLen/_N/syncNumRewardsPerBucket",
            )
            continue

        if len(means) != n_trn or len(counts) != n_trn or len(totals) != n_trn:
            _dbg(
                opts,
                f"[reward-lgturn-pathlen-export] {vid}: SKIP (n_trn mismatch: means={len(means)} counts={len(counts)} totals={len(totals)} vs {n_trn})",
            )
            continue

        # exp/ctrl means+counts
        for ti in range(n_trn):
            exp_vals = np.asarray(means[ti].get("exp", []), dtype=float)
            exp_n = np.asarray(counts[ti].get("exp", []), dtype=int)

            nb_eff = min(n_sb, exp_vals.size)
            mean_exp[vi, ti, :nb_eff] = exp_vals[:nb_eff]
            if exp_n.size:
                n_exp[vi, ti, : min(n_sb, exp_n.size)] = exp_n[: min(n_sb, exp_n.size)]

            # ctrl optional
            ctrl_vals = np.asarray(means[ti].get("ctrl", []), dtype=float)
            ctrl_n = np.asarray(counts[ti].get("ctrl", []), dtype=int)
            nb_eff_c = min(n_sb, ctrl_vals.size)
            if nb_eff_c > 0:
                mean_ctrl[vi, ti, :nb_eff_c] = ctrl_vals[:nb_eff_c]
            if ctrl_n.size:
                n_ctrl[vi, ti, : min(n_sb, ctrl_n.size)] = ctrl_n[
                    : min(n_sb, ctrl_n.size)
                ]

            # total rewards per bucket (shared schedule)
            tot = (
                np.asarray(totals[ti], dtype=int)
                if totals[ti] is not None
                else np.asarray([], dtype=int)
            )
            nb_eff_t = min(n_sb, tot.size)
            if nb_eff_t > 0:
                n_rewards[vi, ti, :nb_eff_t] = tot[:nb_eff_t]

        _dbg(
            opts,
            f"[reward-lgturn-pathlen-export] {vid}: exp finite={np.isfinite(mean_exp[vi]).sum()} / {mean_exp[vi].size}",
        )

    _dbg(
        opts,
        f"[reward-lgturn-pathlen-export] FINAL exp finite={np.isfinite(mean_exp).sum()} / {mean_exp.size}",
    )
    _dbg(
        opts,
        f"[reward-lgturn-pathlen-export] FINAL ctrl finite={np.isfinite(mean_ctrl).sum()} / {mean_ctrl.size}",
    )

    return mean_exp, mean_ctrl, n_exp, n_ctrl, n_rewards, bucket_len_min


def export_reward_lgturn_pathlen_sli_bundle(vas, opts, gls, out_fn):
    """
    Writes an .npz with:
      - sli: (n_videos,)
      - sli_ts: (n_videos, n_trn, n_sb)
      - reward_lgturn_pathlen_exp:   (n_videos, n_trn, n_sb) mean pathlen (mm)
      - reward_lgturn_pathlen_ctrl:  (n_videos, n_trn, n_sb) mean pathlen (mm) (NaN if absent)
      - reward_lgturn_pathlenN_exp/ctrl: (n_videos, n_trn, n_sb) int  (# rewards-with-turn)
      - reward_lgturn_rewards:       (n_videos, n_trn, n_sb) int  (# rewards per bucket; shared schedule)
      - group_label, bucket_len_min, training_names, video_ids, SLI metadata
    """
    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return

    va0 = vas_ok[0]
    group_label = _safe_group_label(opts, gls)

    _dbg(opts, f"[reward-lgturn-pathlen-export] out={out_fn}")
    _dbg(
        opts,
        f"[reward-lgturn-pathlen-export] group_label={group_label!r} n_videos={len(vas_ok)} noyc={getattr(va0, "noyc", None)}",
    )

    mean_exp, mean_ctrl, n_exp, n_ctrl, n_rewards, bucket_len_min = (
        _extract_reward_lgturn_pathlen_arrays(vas_ok, opts)
    )

    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as e:
        print(
            f"[export] WARNING: failed to compute SLI for reward-lgturn-pathlen bundle: {e}"
        )
        sli = np.full((len(vas_ok),), np.nan, dtype=float)
        sli_ts = np.full((len(vas_ok), mean_exp.shape[1], mean_exp.shape[2]), np.nan)

    # Metadata
    try:
        training_names = np.array([t.name() for t in va0.trns], dtype=object)
    except Exception:
        training_names = np.array([], dtype=object)

    try:
        video_ids = np.array(
            [getattr(va, "fn", f"va_{i}") for i, va in enumerate(vas_ok)], dtype=object
        )
    except Exception:
        video_ids = np.array([f"va_{i}" for i in range(len(vas_ok))], dtype=object)

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(
        out_fn,
        sli=sli,
        sli_ts=sli_ts,
        reward_lgturn_pathlen_exp=mean_exp,
        reward_lgturn_pathlen_ctrl=mean_ctrl,
        reward_lgturn_pathlenN_exp=n_exp,
        reward_lgturn_pathlenN_ctrl=n_ctrl,
        reward_lgturn_rewards=n_rewards,
        group_label=np.array(group_label, dtype=object),
        bucket_len_min=np.array(bucket_len_min, dtype=float),
        training_names=training_names,
        video_ids=video_ids,
        sli_training_idx=np.array(getattr(opts, "best_worst_trn", 1) - 1, dtype=int),
        sli_use_training_mean=np.array(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
    )
    print(
        f"[export] Wrote reward-lgturn-pathlen+SLI bundle: {out_fn} (n={len(vas_ok)})"
    )
