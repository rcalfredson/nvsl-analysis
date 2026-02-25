import numpy as np
import os


def _safe_group_label(opts, gls):
    # Priority: explicit CLI label > groupLabels[0] > fallback
    if getattr(opts, "export_group_label", None):
        return opts.export_group_label
    if gls and len(gls) == 1 and gls[0]:
        return gls[0]
    return "group"


def _extract_commag_arrays(vas):
    """
    Returns:
      commag_exp:  (n_videos, n_trains, nb)
      commag_ctrl: (n_videos, n_trains, nb) (all-NaN if absent)
    """
    n_videos = len(vas)
    va0 = vas[0]
    n_trains = len(va0.syncCOMMag)
    # infer nb from first available training exp list
    nb = len(va0.syncCOMMag[0]["exp"]) if n_trains > 0 else 0

    commag_exp = np.full((n_videos, n_trains, nb), np.nan, dtype=float)
    commag_ctrl = np.full((n_videos, n_trains, nb), np.nan, dtype=float)

    for vi, va in enumerate(vas):
        # Defensively handle VAs that were skipped or that have missing fields
        if not hasattr(va, "syncCOMMag") or va.syncCOMMag is None:
            continue
        if len(va.syncCOMMag) != n_trains:
            # Something is inconsistent in the run;
            # keep NaNs rather than silently misalign.
            continue

        for ti in range(n_trains):
            trn_dict = va.syncCOMMag[ti]
            exp_vals = trn_dict.get("exp", None)
            if exp_vals is not None:
                commag_exp[vi, ti, : min(nb, len(exp_vals))] = exp_vals[:nb]
            ctrl_vals = trn_dict.get("ctrl", None)
            if ctrl_vals is not None:
                commag_ctrl[vi, ti, : min(nb, len(ctrl_vals))] = ctrl_vals[:nb]

    return commag_exp, commag_ctrl


def _compute_sli_scalar_and_timeseries_from_rpid(vas, opts):
    """
    Returns:
      sli_scalar: (n_videos,) float   # used for top/bottom filtering
      sli_ts:     (n_videos, n_trains, nb) float  # exp-ctrl per bucket (plotted)
    """
    from analyze import compute_sli_per_fly, trnsForType, typeCalc, vaVarForType

    if len(vas) == 0:
        return np.array([], dtype=float), np.empty((0, 0, 0), dtype=float)

    va0 = vas[0]
    tp, calc = typeCalc("rpid")
    a = np.array([vaVarForType(va, tp, calc) for va in vas])
    trns = trnsForType(va0, "rpid")
    a = a.reshape((len(vas), len(trns), -1))

    # reshape to raw_4: (n_video, n_trains, n_flies, nb)
    n_videos = len(vas)
    n_trains = len(trns)
    n_flies = len(va0.flies)
    nb = a.shape[2] // n_flies
    raw_4 = a.reshape((n_videos, n_trains, n_flies, nb))

    # FULL time-series SLI to plot:
    # (exp - ctrl) at every bucket, every training, every video
    sli_ts = raw_4[:, :, 0, :] - raw_4[:, :, 1, :]

    # Scalar SLI for filtering:
    sli_training_idx = getattr(opts, "best_worst_trn", 1) - 1
    use_training_mean = bool(getattr(opts, "sli_use_training_mean", False))

    # SLI selection windowing (applies ONLY to the scalar used for best/worst + set-op selection)
    raw_sel_skip = getattr(opts, "sli_select_skip_first_sync_buckets", None)
    raw_sel_keep = getattr(opts, "sli_select_keep_first_sync_buckets", None)
    sel_skip_k = 0 if raw_sel_skip is None else max(0, int(raw_sel_skip))
    sel_keep_k = 0 if raw_sel_keep is None else max(0, int(raw_sel_keep))

    sli_scalar = compute_sli_per_fly(
        raw_4,
        sli_training_idx,
        bucket_idx=None,
        average_over_buckets=use_training_mean,
        skip_first_sync_buckets=sel_skip_k,
        keep_first_sync_buckets=sel_keep_k,
    )
    return np.asarray(sli_scalar, dtype=float), np.asarray(sli_ts, dtype=float)


def export_com_sli_bundle(vas, opts, gls, out_fn):
    """
    Writes an .npz with:
      - sli: (n_videos,)
      - commag_exp: (n_videos, n_trains, nb)
      - commag_ctrl: (n_videos, n_trains, nb)
      - group_label: str
      - bucket_len_min: float (if we can infer it)
      - training_names: (n_trains,) strings
    """
    from analyze import bucketLenForType

    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    # Filter to only analyses that actually ran
    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return

    va0 = vas_ok[0]
    group_label = _safe_group_label(opts, gls)

    # Extract COM magnitude arrays
    commag_exp, commag_ctrl = _extract_commag_arrays(vas_ok)

    # Compute SLI
    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as e:
        print(f"[export] WARNING: failed to compute SLI for bundle: {e}")
        sli = np.full((len(vas_ok),), np.nan, dtype=float)
        # infer n_trains/nb from commag arrays after they are computed (or set empty)
        sli_ts = np.full(
            (len(vas_ok), commag_exp.shape[1], commag_exp.shape[2]), np.nan, dtype=float
        )

    if getattr(opts, "com_sli_debug", False):
        print(
            f"[export][com-sli-debug] group={group_label} "
            f"n_videos={len(vas_ok)} n_trains={commag_exp.shape[1]} nb={commag_exp.shape[2]} out={out_fn}"
        )

        for vi, va in enumerate(vas_ok):
            vid = getattr(va, "fn", f"va_{vi}")
            has = hasattr(va, "syncCOMMag") and (va.syncCOMMag is not None)
            ntr = len(va.syncCOMMag) if has else 0

            fin_counts = [
                int(np.isfinite(commag_exp[vi, ti, :]).sum())
                for ti in range(commag_exp.shape[1])
            ]

            sli_i = (
                float(sli[vi]) if (vi < len(sli) and np.isfinite(sli[vi])) else np.nan
            )

            print(
                f"[export][com-sli-debug] video={vid} has_syncCOMMag={has} "
                f"n_trains={ntr} finite_exp_per_trn={fin_counts} sli={sli_i:g}"
            )

    # Metadata
    try:
        # bucketLenForType returns (bl, blf)
        bl, _blf = bucketLenForType("commag")
        bucket_len_min = float(bl)
    except Exception:
        bucket_len_min = np.nan

    try:
        training_names = np.array([t.name() for t in va0.trns], dtype=object)
    except Exception:
        training_names = np.array([], dtype=object)

    # Optional video identifiers (nice for debugging)
    try:
        video_ids = np.array(
            [getattr(va, "fn", f"va_{i}") for i, va in enumerate(vas_ok)]
        )
    except Exception:
        video_ids = np.array([f"va_{i}" for i in range(len(vas_ok))], dtype=object)

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(
        out_fn,
        sli=sli,
        sli_ts=sli_ts,
        commag_exp=commag_exp,
        commag_ctrl=commag_ctrl,
        group_label=np.array(group_label, dtype=object),
        bucket_len_min=np.array(bucket_len_min, dtype=float),
        training_names=training_names,
        video_ids=video_ids,
        sli_training_idx=np.array(getattr(opts, "best_worst_trn", 1) - 1, dtype=int),
        sli_use_training_mean=np.array(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
    )
    print(f"[export] Wrote COM+SLI bundle: {out_fn} (n={len(vas_ok)})")
