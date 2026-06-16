"""Cumulative-rewards + SLI bundle export entry points.

This metric answers:

    At each sync-bucket endpoint, how many calc rewards has each fly
    accumulated within the current training?

The exported arrays have shape:
    (n_videos, n_trainings, n_sync_buckets)

The cumulative count resets at the start of each training.
"""

from __future__ import annotations

import os
import numpy as np

from src.analysis.sli_bundle_utils import validate_sli_bundle
from src.analysis.sync_bucket_presence_filters import (
    exp_target_sync_bucket_eligibility_mask,
    exp_target_sync_bucket_filter_payload,
    mask_by_exp_target_sync_bucket_filter,
)
from src.exporting.com_sli_bundle import (
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)


def _calc_reward_on_frames(va, trn, fly_idx: int) -> np.ndarray:
    """Return sorted calc reward-on frames for one fly/training."""
    on = va._getOn(trn, calc=True, ctrl=False, f=fly_idx)
    if on is None:
        return np.asarray([], dtype=int)

    on = np.asarray(on)

    if on.size == 0:
        return np.asarray([], dtype=int)

    # Be permissive in case upstream gives a float array with NaNs.
    if np.issubdtype(on.dtype, np.floating):
        on = on[np.isfinite(on)]

    return np.sort(on.astype(int, copy=False))


def _infer_sync_bucket_df(va) -> int:
    """Infer sync-bucket width in frames using the same helper used elsewhere."""
    df = va._numRewardsMsg(True, silent=True)
    if df is None or int(df) <= 0:
        raise ValueError("Could not infer sync-bucket frame width from VideoAnalysis.")
    return int(df)


def _infer_n_trainings_and_buckets(vas_ok, df: int) -> tuple[int, int]:
    """
    Infer rectangular bundle dimensions.

    This mirrors the existing bundle convention of exporting rectangular
    n_videos x n_trainings x n_buckets arrays.
    """
    n_trainings = max(len(getattr(va, "trns", [])) for va in vas_ok)
    n_buckets = 0

    for va in vas_ok:
        for trn in getattr(va, "trns", []):
            try:
                _fi, nb, _ = va._syncBucket(trn, df)
            except Exception:
                nb = 0
            n_buckets = max(n_buckets, int(nb or 0))

    return int(n_trainings), int(n_buckets)


def _cum_rewards_for_training(
    va, trn, fly_idx: int, *, df: int, nb_out: int
) -> np.ndarray:
    """
    Return cumulative calc reward count at each complete sync-bucket endpoint.

    vals[k] is the number of calc reward-on frames in the current training
    up to endpoint fi + (k + 1) * df.

    Incomplete buckets and bad/missing trajectories remain NaN.
    """
    vals = np.full(nb_out, np.nan, dtype=float)

    if len(getattr(va, "trx", [])) <= fly_idx:
        return vals

    if len(va.trx) > 0 and va._bad(fly_idx):
        return vals

    fi, nb, _ = va._syncBucket(trn, df)
    nb = min(int(nb or 0), int(nb_out))

    if nb <= 0 or fi is None:
        return vals

    fi = int(fi)
    df = int(df)

    starts = np.asarray([fi + k * df for k in range(nb)], dtype=int)
    endpoints = starts + df

    # Match the completeness convention used in bySyncBucketCOM.
    complete = np.asarray([(trn.stop - int(s)) >= df for s in starts], dtype=bool)

    on = _calc_reward_on_frames(va, trn, fly_idx)

    # Keep rewards inside the current training window. The count resets per training.
    on = on[(on >= fi) & (on < int(trn.stop))]

    counts = np.searchsorted(on, endpoints, side="right").astype(float)
    vals[:nb] = np.where(complete, counts, np.nan)

    return vals


def _extract_cum_reward_arrays(vas_ok, opts=None):
    """
    Returns
    -------
    cum_rewards_exp : np.ndarray
        Shape (n_videos, n_trainings, n_bucket).
    cum_rewards_ctrl : np.ndarray
        Shape (n_video, n_trainings, n_bucket). All-NaN if no yoked/control fly.
    """
    n_videos = len(vas_ok)
    va0 = vas_ok[0]
    df = _infer_sync_bucket_df(va0)

    n_trainings, n_buckets = _infer_n_trainings_and_buckets(vas_ok, df)

    cum_rewards_exp = np.full((n_videos, n_trainings, n_buckets), np.nan, dtype=float)
    cum_rewards_ctrl = np.full((n_videos, n_trainings, n_buckets), np.nan, dtype=float)

    for vi, va in enumerate(vas_ok):
        trns = getattr(va, "trns", [])

        for (
            ti,
            trn,
        ) in enumerate(trns):
            if ti >= n_trainings:
                break

            cum_rewards_exp[vi, ti, :] = _cum_rewards_for_training(
                va, trn, fly_idx=0, df=df, nb_out=n_buckets
            )

            if len(getattr(va, "trx", [])) > 1:
                cum_rewards_ctrl[vi, ti, :] = _cum_rewards_for_training(
                    va, trn, fly_idx=1, df=df, nb_out=n_buckets
                )
    return cum_rewards_exp, cum_rewards_ctrl


def build_cum_rewards_sli_bundle(vas, opts, gls) -> dict:
    """
    Build an in-memory cumulative-rewards + SLI bundle.

    Stored keys intentionally mirror the generic metric+SLI bundle convention:

      - sli: scalar SLI per video, used for top/bottom learner filtering
      - sli_ts: SLI time series per video/training/bucket
      - cum_rewards_exp: cumulative calc rewards for experimental flies
      - cum_rewards_ctrl: cumulative calc rewards for yoked control flies
      - group_label
      - bucket_len_min
      - training_names
      - video_ids
    """
    from analyze import bucketLenForType

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        raise ValueError(
            "[export] No non-skipped VideoAnalysis instances for cumulative-rewards+SLI bundle."
        )

    va0 = vas_ok[0]
    group_label = _safe_group_label(opts, gls)

    cum_rewards_exp, cum_rewards_ctrl = _extract_cum_reward_arrays(vas_ok, opts)

    # Compute SLI with the exact helper used by COM bundles.
    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as e:
        print(
            f"[export] WARNING: failed to compute SLI for cumulative-rewards bundle: {e}"
        )
        sli = np.full((len(vas_ok),), np.nan, dtype=float)
        sli_ts = np.full(
            (
                len(vas_ok),
                cum_rewards_exp.shape[1],
                cum_rewards_exp.shape[2],
            ),
            np.nan,
            dtype=float,
        )

    # Match COM-bundle behavior: apply exp-target sync-bucket filtering to
    # the plotted experimental metric and SLI fields.
    target_sync_bucket_eligible = exp_target_sync_bucket_eligibility_mask(vas_ok, opts)
    cum_rewards_exp = mask_by_exp_target_sync_bucket_filter(
        cum_rewards_exp,
        target_sync_bucket_eligible,
    )
    sli = mask_by_exp_target_sync_bucket_filter(sli, target_sync_bucket_eligible)
    sli_ts = mask_by_exp_target_sync_bucket_filter(sli_ts, target_sync_bucket_eligible)

    # Keep ctrl unmasked here, matching the COM exporter pattern where the primary
    # experiment-target filter is applied to the experimental metric + SLI.
    # To have include-ctrl overlays to omit the same excluded videos, mask
    # cum_rewards_ctrl too.

    try:
        # Reuse the same sync-bucket length as COM/rpid-style time-series plots.
        bl, _blf = bucketLenForType("commag")
        bucket_len_min = float(bl)
    except Exception:
        bucket_len_min = np.nan

    try:
        training_names = np.array([t.name() for t in va0.trns], dtype=object)
    except Exception:
        training_names = np.array([], dtype=object)

    try:
        video_ids = np.array(
            [getattr(va, "fn", f"va_{i}") for i, va in enumerate(vas_ok)],
            dtype=object,
        )
    except Exception:
        video_ids = np.array([f"va_{i}" for i in range(len(vas_ok))], dtype=object)

    payload = dict(
        sli=sli,
        sli_ts=sli_ts,
        cum_rewards_exp=cum_rewards_exp,
        cum_rewards_ctrl=cum_rewards_ctrl,
        group_label=np.array(group_label, dtype=object),
        bucket_len_min=np.array(bucket_len_min, dtype=float),
        training_names=training_names,
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
        **exp_target_sync_bucket_filter_payload(
            vas_ok,
            opts,
            prefix="exp_target_sync_bucket_filter",
        ),
    )

    if getattr(opts, "cum_rewards_sli_debug", False):
        print(
            f"[export][cum-rewards-sli-debug] group={group_label} "
            f"n_videos={len(vas_ok)} "
            f"n_trains={cum_rewards_exp.shape[1]} "
            f"nb={cum_rewards_exp.shape[2]}"
        )
        for vi, va in enumerate(vas_ok):
            vid = getattr(va, "fn", f"va_{vi}")
            fin_counts = [
                int(np.isfinite(cum_rewards_exp[vi, ti, :]).sum())
                for ti in range(cum_rewards_exp.shape[1])
            ]
            total_last = [
                (
                    float(
                        cum_rewards_exp[
                            vi,
                            ti,
                            np.flatnonzero(np.isfinite(cum_rewards_exp[vi, ti, :]))[-1],
                        ]
                    )
                    if np.isfinite(cum_rewards_exp[vi, ti, :]).any()
                    else np.nan
                )
                for ti in range(cum_rewards_exp.shape[1])
            ]
            sli_i = float(sli[vi]) if vi < len(sli) and np.isfinite(sli[vi]) else np.nan
            print(
                f"[export][cum-rewards-sli-debug] video={vid} "
                f"finite_exp_per_trn={fin_counts} "
                f"last_exp_per_trn={total_last} "
                f"sli={sli_i:g}"
            )

    return payload


def export_cum_rewards_sli_bundle(vas, opts, gls, out_fn):
    bundle = build_cum_rewards_sli_bundle(vas, opts, gls)

    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    validate_sli_bundle(bundle, path=out_fn)

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(out_fn, **bundle)

    print(
        f"[export] Wrote cumulative-rewards+SLI bundle: {out_fn} "
        f"(n={len(bundle['sli'])})"
    )
