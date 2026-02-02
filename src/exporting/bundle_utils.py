# src/exporting/bundle_utils.py
from __future__ import annotations

import os
import numpy as np

from src.exporting.com_sli_bundle import (
    _safe_group_label,
    _compute_sli_scalar_and_timeseries_from_rpid,
)


def save_metric_plus_sli_bundle(
    vas,
    opts,
    gls,
    out_fn,
    *,
    extract_metric_arrays,  # callable: (vas_ok) -> dict[str, np.ndarray]
    bucket_type: str,  # passed to bucketLenForType
    print_label: str,  # for logs
    require_3d: bool = True,
):
    from analyze import bucketLenForType

    if not out_fn.lower().endswith(".npz"):
        out_fn += ".npz"

    vas_ok = [va for va in vas if not getattr(va, "_skipped", False)]
    if len(vas_ok) == 0:
        print(f"[export] No non-skipped VideoAnalysis instances; not writing {out_fn}")
        return

    va0 = vas_ok[0]
    group_label = _safe_group_label(opts, gls)

    # ---- metric extraction happens here, after filtering ----
    metric_arrays = extract_metric_arrays(vas_ok)
    if not isinstance(metric_arrays, dict) or len(metric_arrays) == 0:
        raise ValueError(
            f"{print_label}: extract_metric_arrays returned empty/invalid dict"
        )

    # Infer (n_trn, nb) from the first metric array and sanity-check shapes
    any_arr = next(iter(metric_arrays.values()))
    any_arr = np.asarray(any_arr)

    if require_3d and any_arr.ndim != 3:
        raise ValueError(
            f"{print_label}: expected 3D metric arrays (n_videos,n_trn,nb); got {any_arr.shape}"
        )

    n_videos = len(vas_ok)
    if any_arr.shape[0] != n_videos:
        raise ValueError(
            f"{print_label}: metric first dim {any_arr.shape[0]} != n_videos {n_videos}"
        )

    n_trn = int(any_arr.shape[1]) if any_arr.ndim >= 2 else 0
    nb = int(any_arr.shape[2]) if any_arr.ndim >= 3 else 0

    for k, v in metric_arrays.items():
        v = np.asarray(v)
        if v.shape[0] != n_videos:
            raise ValueError(
                f"{print_label}: metric '{k}' first dim {v.shape[0]} != n_videos {n_videos}"
            )
        if require_3d and v.ndim != 3:
            raise ValueError(f"{print_label}: metric '{k}' expected 3D, got {v.shape}")
        # optional: enforce same (n_trn, nb) across keys
        if require_3d and (v.shape[1] != n_trn or v.shape[2] != nb):
            raise ValueError(
                f"{print_label}: metric '{k}' shape {v.shape} != ({n_videos},{n_trn},{nb})"
            )

    # ---- SLI ----
    try:
        sli, sli_ts = _compute_sli_scalar_and_timeseries_from_rpid(vas_ok, opts)
    except Exception as e:
        print(f"[export] WARNING: failed to compute SLI for {print_label} bundle: {e}")
        sli = np.full((n_videos,), np.nan, dtype=float)
        sli_ts = np.full((n_videos, n_trn, nb), np.nan, dtype=float)

    # ---- metadata ----
    try:
        bl, _blf = bucketLenForType(bucket_type)
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
        video_ids = np.array([f"va_{i}" for i in range(n_videos)], dtype=object)

    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)

    payload = dict(
        sli=np.asarray(sli, dtype=float),
        sli_ts=np.asarray(sli_ts, dtype=float),
        group_label=np.array(group_label, dtype=object),
        bucket_len_min=np.array(bucket_len_min, dtype=float),
        training_names=training_names,
        video_ids=video_ids,
        sli_training_idx=np.array(getattr(opts, "best_worst_trn", 1) - 1, dtype=int),
        sli_use_training_mean=np.array(
            bool(getattr(opts, "sli_use_training_mean", False))
        ),
    )

    payload.update({k: np.asarray(v) for k, v in metric_arrays.items()})

    np.savez_compressed(out_fn, **payload)
    print(f"[export] Wrote {print_label}+SLI bundle: {out_fn} (n={n_videos})")
