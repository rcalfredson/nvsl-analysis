# src/exporting/wallpct_sli_bundle.py
import os
import numpy as np

from src.exporting.com_sli_bundle import (
    _safe_group_label,
    _compute_sli_scalar_and_timeseries_from_rpid,
)


def _extract_wallpct_arrays(vas):
    """
    Returns:
      wallpct_exp:  (n_videos, n_trains, nb)  float in [0,1] with NaNs allowed
      wallpct_ctrl: (n_videos, n_trains, nb)  float in [0,1] with NaNs allowed (all-NaN if absent)
    """
    n_videos = len(vas)

    # Find a reference VA that actually has syncWallPct
    ref = None
    for va in vas:
        swp = getattr(va, "syncWallPct", None)
        if swp is not None and isinstance(swp, (list, tuple)):
            ref = va
            break

    if ref is None:
        return (
            np.full((n_videos, 0, 0), np.nan, dtype=float),
            np.full((n_videos, 0, 0), np.nan, dtype=float),
        )

    swp_ref = getattr(ref, "syncWallPct", []) or []
    n_trains = len(swp_ref)

    # Infer nb (sync-bucket count) by scanning for the first non-empty exp list.
    nb = 0

    # 1) scan trainings within the reference VA
    for ti in range(n_trains):
        trn_dict = swp_ref[ti] if isinstance(swp_ref[ti], dict) else {}
        exp_vals = trn_dict.get("exp", None)
        if exp_vals is not None:
            try:
                nb = len(exp_vals)
            except Exception:
                nb = 0
        if nb > 0:
            break

    # 2) if still unknown, scan other videos (defensive against ref video having no full buckets)
    if nb == 0:
        for va in vas:
            swp = getattr(va, "syncWallPct", None)
            if swp is None or len(swp) != n_trains:
                continue
            for ti in range(n_trains):
                trn_dict = swp[ti] if isinstance(swp[ti], dict) else {}
                exp_vals = trn_dict.get("exp", None)
                if exp_vals is None:
                    continue
                try:
                    nb = len(exp_vals)
                except Exception:
                    nb = 0
                if nb > 0:
                    break
            if nb > 0:
                break

    wallpct_exp = np.full((n_videos, n_trains, nb), np.nan, dtype=float)
    wallpct_ctrl = np.full((n_videos, n_trains, nb), np.nan, dtype=float)

    if nb == 0 or n_trains == 0:
        return wallpct_exp, wallpct_ctrl

    for vi, va in enumerate(vas):
        swp = getattr(va, "syncWallPct", None)
        if swp is None or len(swp) != n_trains:
            continue

        for ti in range(n_trains):
            trn_dict = swp[ti] if isinstance(swp[ti], dict) else {}

            exp_vals = trn_dict.get("exp", None)
            if exp_vals is not None:
                exp_vals = np.asarray(exp_vals, dtype=float).reshape(-1)
                wallpct_exp[vi, ti, : min(nb, exp_vals.size)] = exp_vals[:nb]

            ctrl_vals = trn_dict.get("ctrl", None)
            if ctrl_vals is not None:
                ctrl_vals = np.asarray(ctrl_vals, dtype=float).reshape(-1)
                wallpct_ctrl[vi, ti, : min(nb, ctrl_vals.size)] = ctrl_vals[:nb]

    return wallpct_exp, wallpct_ctrl


def export_wallpct_sli_bundle(vas, opts, gls, out_fn):
    """
    Writes an .npz bundle with:
      - sli: (n_videos,)
      - sli_ts: (n_videos, n_trains, nb)   # exp-ctrl per sync bucket, for plotting if desired
      - wallpct_exp: (n_videos, n_trains, nb)   # fraction [0,1]
      - wallpct_ctrl: (n_videos, n_trains, nb)  # fraction [0,1] (all-NaN if absent)
      - group_label: str
      - bucket_len_min: float
      - training_names: (n_trains,) strings
      - video_ids: (n_videos,) strings
      - sli_training_idx, sli_use_training_mean: exporter settings snapshot
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

    # Extract wallpct arrays
    wallpct_exp, wallpct_ctrl = _extract_wallpct_arrays(vas_ok)

    # Compute SLI (scalar + full time-series)
    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as e:
        print(f"[export] WARNING: failed to compute SLI for wallpct bundle: {e}")
        sli = np.full((len(vas_ok),), np.nan, dtype=float)
        sli_ts = np.full(
            (len(vas_ok), wallpct_exp.shape[1], wallpct_exp.shape[2]),
            np.nan,
            dtype=float,
        )

    # Metadata
    try:
        bl, _blf = bucketLenForType(
            "wallpct"
        )  # sync-bucket length; reuse commag mapping
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

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)
    np.savez_compressed(
        out_fn,
        sli=np.asarray(sli, dtype=float),
        sli_ts=np.asarray(sli_ts, dtype=float),
        wallpct_exp=np.asarray(wallpct_exp, dtype=float),
        wallpct_ctrl=np.asarray(wallpct_ctrl, dtype=float),
        group_label=np.array(group_label, dtype=object),
        bucket_len_min=np.array(bucket_len_min, dtype=float),
        training_names=training_names,
        video_ids=video_ids,
        sli_training_idx=np.array(getattr(opts, "best_worst_trn", 1) - 1, dtype=int),
        sli_use_training_mean=np.array(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
    )
    print(f"[export] Wrote wallpct+SLI bundle: {out_fn} (n={len(vas_ok)})")
