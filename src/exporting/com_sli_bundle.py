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


def _compute_sli_from_rpid(vas, opts):
    """
    Computes per-video SLI scalar using the same logic flags as elsewhere:
      - sli_training_idx from --best-worst-trn (default behavior)
      - average over buckets if --sli-use-training-mean
    """
    from analyze import compute_sli_per_fly, trnsForType, typeCalc, vaVarForType

    if len(vas) == 0:
        return np.array([], dtype=float)

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

    sli_training_idx = getattr(opts, "best_worst_trn", 1) - 1
    use_training_mean = bool(getattr(opts, "sli_use_training_mean", False))

    sli = compute_sli_per_fly(
        raw_4, sli_training_idx, bucket_idx=None, average_over_buckets=use_training_mean
    )
    # Ensure it's a flat float array
    return np.asarray(sli, dtype=float)


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

    # Compute SLI
    try:
        sli = _compute_sli_from_rpid(vas_ok, opts)
    except Exception as e:
        print(f"[export] WARNING: failed to compute SLI for bundle: {e}")
        sli = np.full((len(vas_ok),), np.nan, dtype=float)

    # Extract COM magnitude arrays
    commag_exp, commag_ctrl = _extract_commag_arrays(vas_ok)

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
