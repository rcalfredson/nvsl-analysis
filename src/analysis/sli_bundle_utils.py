from __future__ import annotations

import numpy as np


def as_scalar(x):
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


def as_str_array(x):
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.array([str(v) for v in arr], dtype=object)


def load_sli_bundle(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    req = [
        "sli",
        "group_label",
        "bucket_len_min",
        "training_names",
        "video_ids",
        "sli_training_idx",
        "sli_use_training_mean",
    ]
    missing = [k for k in req if k not in d.files]
    if missing:
        raise ValueError(f"Bundle {path} is missing keys: {missing}")

    out = {k: d[k] for k in req}
    out["path"] = path
    out["group_label"] = str(as_scalar(out["group_label"]))
    out["bucket_len_min"] = float(as_scalar(out["bucket_len_min"]))
    out["sli_training_idx"] = int(as_scalar(out["sli_training_idx"]))
    out["sli_use_training_mean"] = bool(as_scalar(out["sli_use_training_mean"]))

    prefixes = (
        "commag_",
        "weaving_",
        "wallpct_",
        "turnback_",
        "agarose_",
        "lgturn_",
        "reward_lgturn_",
        "reward_lv_",
        "sli_",
    )
    for k in d.files:
        if k in out:
            continue
        if k.startswith(prefixes) or k in ("sli_ts",):
            out[k] = d[k]
    return out


def bundle_video_ids(bundle: dict) -> np.ndarray | None:
    key = "video_uid" if "video_uid" in bundle else "video_ids"
    if key not in bundle:
        return None
    return as_str_array(bundle[key])


def align_by_video_ids(base_bundle: dict, comp_bundle: dict) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if "video_ids" not in base_bundle or "video_ids" not in comp_bundle:
        return None, None, 0
    base_ids = bundle_video_ids(base_bundle)
    comp_ids = bundle_video_ids(comp_bundle)
    if base_ids is None or comp_ids is None:
        return None, None, 0

    base_map = {vid: i for i, vid in enumerate(base_ids)}
    comp_map = {vid: i for i, vid in enumerate(comp_ids)}
    common = [vid for vid in comp_ids if vid in base_map]
    if not common:
        return None, None, 0

    base_idx = np.array([base_map[vid] for vid in common], dtype=int)
    comp_idx = np.array([comp_map[vid] for vid in common], dtype=int)
    return base_idx, comp_idx, int(len(common))
