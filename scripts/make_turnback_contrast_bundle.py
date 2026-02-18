#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import numpy as np


# ----------------------------
# Helpers
# ----------------------------


def _load_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _as_scalar(x):
    # np.savez stores scalars as 0-d arrays
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return np.ravel(arr)[0].item()
    return arr


def _as_1d_str_array(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=object)
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.array([str(v) for v in arr], dtype=object)


def _require_same_bucket_plan(A: dict, B: dict, *, labelA="A", labelB="B"):
    """
    Defensive checks to avoid subtracting mismatched bucket/training plans.
    """
    blA = float(_as_scalar(A.get("bucket_len_min", np.nan)))
    blB = float(_as_scalar(B.get("bucket_len_min", np.nan)))
    if np.isfinite(blA) and np.isfinite(blB) and abs(blA - blB) > 1e-9:
        raise ValueError(f"bucket_len_min mismatch: {labelA}={blA} vs {labelB}={blB}")

    tnA = A.get("training_names", None)
    tnB = B.get("training_names", None)
    if tnA is not None and tnB is not None:
        a = _as_1d_str_array(tnA)
        b = _as_1d_str_array(tnB)
        if a.size != b.size:
            raise ValueError(
                f"training_names length mismatch: {labelA}={a.size} vs {labelB}={b.size}"
            )
        if a.size and not np.array_equal(a, b):
            # Content mismatch is a strong signal you're comparing different exports.
            # Show a small hint rather than dumping everything.
            mism = np.nonzero(a != b)[0]
            i0 = int(mism[0]) if mism.size else 0
            raise ValueError(
                "training_names content mismatch "
                f"(first mismatch at idx {i0}: {labelA}={a[i0]!r} vs {labelB}={b[i0]!r})"
            )


def _turnback_shape(bundle: dict, *, label="bundle") -> tuple[int, int, int]:
    if "turnback_ratio_exp" not in bundle:
        raise ValueError(f"{label} missing key 'turnback_ratio_exp'")
    x = np.asarray(bundle["turnback_ratio_exp"], dtype=float)
    if x.ndim != 3:
        raise ValueError(
            f"{label} turnback_ratio_exp must be 3D (n_vid,n_trn,n_sb); got shape={x.shape}"
        )
    return int(x.shape[0]), int(x.shape[1]), int(x.shape[2])


def _summarize_group_turnback_mean_exp(bundle: dict) -> np.ndarray:
    """
    Mean of turnback_ratio_exp over videos (nanmean).
    Returns shape (n_trn, n_sb).
    """
    # Validate shape for clearer errors.
    _turnback_shape(bundle, label="bundle")
    x = np.asarray(bundle["turnback_ratio_exp"], dtype=float)  # (n_vid, n_trn, n_sb)
    return np.nanmean(x, axis=0)


def _save_npz(out_path: str, **kwargs):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **kwargs)


# ----------------------------
# Main
# ----------------------------


def main():
    p = argparse.ArgumentParser(
        description=(
            "Create a between-group turnback contrast *mean* bundle.\n"
            "Optionally compute delta-of-deltas between a jitter contrast and a baseline contrast (mean-only)."
        )
    )

    # Baseline (0px) contrast inputs
    p.add_argument("--a", required=True, help="Group A baseline bundle (.npz)")
    p.add_argument("--b", required=True, help="Group B baseline bundle (.npz)")
    p.add_argument("--out", required=True, help="Output contrast bundle (.npz)")

    # Optional jitter (e.g. +0.5px) contrast inputs
    p.add_argument("--a-jitter", default=None, help="Group A jitter bundle (.npz)")
    p.add_argument("--b-jitter", default=None, help="Group B jitter bundle (.npz)")
    p.add_argument(
        "--out-dodelta",
        default=None,
        help="If jitter bundles provided, write delta-of-deltas summary bundle (.npz)",
    )

    # Misc
    p.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="If set, skip strict metadata/shape checks (bucket_len_min, training_names, turnback_ratio_exp shape).",
    )

    args = p.parse_args()

    A0 = _load_npz(args.a)
    B0 = _load_npz(args.b)

    if not args.allow_metadata_mismatch:
        _require_same_bucket_plan(A0, B0, labelA="A0", labelB="B0")
        _, n_trnA, n_sbA = _turnback_shape(A0, label="A0")
        _, n_trnB, n_sbB = _turnback_shape(B0, label="B0")
        if (n_trnA, n_sbA) != (n_trnB, n_sbB):
            raise ValueError(
                f"turnback_ratio_exp shape mismatch (n_trn,n_sb): A0=({n_trnA},{n_sbA}) vs B0=({n_trnB},{n_sbB})"
            )

    meanA0 = _summarize_group_turnback_mean_exp(A0)
    meanB0 = _summarize_group_turnback_mean_exp(B0)
    c0_mean = meanA0 - meanB0

    # "pseudo bundle" payload for plot_com_sli_bundles compatibility
    pseudo_series = c0_mean[None, :, :]  # (1, n_trn, n_sb)

    # SLI placeholders so _load_bundle() doesn't complain
    n_trn, n_sb = c0_mean.shape
    sli = np.array([np.nan], dtype=float)
    sli_ts = np.full((1, n_trn, n_sb), np.nan, dtype=float)

    group_label = np.array(
        f"{_as_scalar(A0.get('group_label','A'))} - {_as_scalar(B0.get('group_label','B'))}",
        dtype=object,
    )
    video_ids = np.array(["__contrast__"], dtype=object)

    # metadata passthrough (best-effort)
    meta = dict(
        bucket_len_min=A0.get("bucket_len_min", np.array(np.nan, dtype=float)),
        training_names=A0.get("training_names", np.array([], dtype=object)),
        group_label_a=A0.get("group_label", np.array("A", dtype=object)),
        group_label_b=B0.get("group_label", np.array("B", dtype=object)),
        source_a=np.array(args.a, dtype=object),
        source_b=np.array(args.b, dtype=object),
        contrast_method=np.array("mean_only", dtype=object),
        contrast_note=np.array(
            "Mean-only contrast bundle for plotting. No CI/SE computed here.",
            dtype=object,
        ),
        turnback_inner_delta_mm=A0.get(
            "turnback_inner_delta_mm", np.array(np.nan, dtype=float)
        ),
        turnback_inner_radius_offset_px_A=A0.get(
            "turnback_inner_radius_offset_px", np.array(np.nan, dtype=float)
        ),
        turnback_inner_radius_offset_px_B=B0.get(
            "turnback_inner_radius_offset_px", np.array(np.nan, dtype=float)
        ),
    )

    _save_npz(
        args.out,
        contrast_mean=c0_mean,
        turnback_ratio_exp=pseudo_series,
        turnback_ratio_ctrl=np.full_like(pseudo_series, np.nan),
        turnback_total_exp=np.zeros_like(pseudo_series, dtype=int),
        turnback_total_ctrl=np.zeros_like(pseudo_series, dtype=int),
        group_label=group_label,
        video_ids=video_ids,
        sli=sli,
        sli_ts=sli_ts,
        sli_training_idx=A0.get("sli_training_idx", np.array(0, dtype=int)),
        sli_use_training_mean=A0.get("sli_use_training_mean", np.array(False)),
        **meta,
    )
    print(f"[contrast] wrote: {args.out}")
    print(f"[contrast] max |mean| = {np.nanmax(np.abs(c0_mean)):.6f}")

    # Optional delta-of-deltas
    if (args.a_jitter is None) ^ (args.b_jitter is None):
        raise ValueError("Provide both --a-jitter and --b-jitter, or neither.")

    if args.a_jitter and args.b_jitter:
        if not args.out_dodelta:
            raise ValueError("If jitter bundles are provided, also set --out-dodelta.")

        Aj = _load_npz(args.a_jitter)
        Bj = _load_npz(args.b_jitter)

        if not args.allow_metadata_mismatch:
            _require_same_bucket_plan(Aj, Bj, labelA="Aj", labelB="Bj")
            _require_same_bucket_plan(A0, Aj, labelA="A0", labelB="Aj")
            _, n_trnAj, n_sbAj = _turnback_shape(Aj, label="Aj")
            _, n_trnBj, n_sbBj = _turnback_shape(Bj, label="Bj")
            if (n_trnAj, n_sbAj) != (n_trnBj, n_sbBj):
                raise ValueError(
                    f"turnback_ratio_exp shape mismatch (n_trn,n_sb): Aj=({n_trnAj},{n_sbAj}) vs Bj=({n_trnBj},{n_sbBj})"
                )
            if (n_trnAj, n_sbAj) != (n_trn, n_sb):
                raise ValueError(
                    f"baseline vs jitter shape mismatch (n_trn,n_sb): baseline=({n_trn},{n_sb}) vs jitter=({n_trnAj},{n_sbAj})"
                )

        meanAj = _summarize_group_turnback_mean_exp(Aj)
        meanBj = _summarize_group_turnback_mean_exp(Bj)
        cj_mean = meanAj - meanBj

        dodelta_mean = cj_mean - c0_mean

        meta2 = dict(
            source_a_jitter=np.array(args.a_jitter, dtype=object),
            source_b_jitter=np.array(args.b_jitter, dtype=object),
        )

        _save_npz(
            args.out_dodelta,
            dodelta_mean=dodelta_mean,
            # keep baseline + jitter contrasts too (often handy)
            contrast0_mean=c0_mean,
            contrastj_mean=cj_mean,
            **meta,
            **meta2,
        )
        print(f"[dodelta] wrote: {args.out_dodelta}")
        print(f"[dodelta] max |mean| = {np.nanmax(np.abs(dodelta_mean)):.6f}")


if __name__ == "__main__":
    main()
