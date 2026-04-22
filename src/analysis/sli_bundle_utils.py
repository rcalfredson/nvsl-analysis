from __future__ import annotations

import numpy as np


REQ_BUNDLE_KEYS = [
    "sli",
    "group_label",
    "bucket_len_min",
    "training_names",
    "video_ids",
    "sli_training_idx",
    "sli_use_training_mean",
]

BUNDLE_ARRAY_PREFIXES = (
    "commag_",
    "between_reward_",
    "cum_reward_sli_",
    "weaving_",
    "wallpct_",
    "turnback_",
    "agarose_",
    "lgturn_",
    "reward_lgturn_",
    "reward_lv_",
    "return_prob_",
    "sli_",
)


def as_scalar(x):
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


def as_str_array(x):
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.array([str(v) for v in arr], dtype=object)


def normalize_sli_bundle(bundle: dict, *, path: str | None = None) -> dict:
    missing = [k for k in REQ_BUNDLE_KEYS if k not in bundle]
    if missing:
        where = path or bundle.get("path") or "<unknown>"
        raise ValueError(f"Bundle {where} is missing keys: {missing}")

    out = dict(bundle)
    out["group_label"] = str(as_scalar(out["group_label"]))
    out["bucket_len_min"] = float(as_scalar(out["bucket_len_min"]))
    out["sli_training_idx"] = int(as_scalar(out["sli_training_idx"]))
    out["sli_use_training_mean"] = bool(as_scalar(out["sli_use_training_mean"]))
    out["training_names"] = as_str_array(out["training_names"])
    out["video_ids"] = as_str_array(out["video_ids"])
    out["sli"] = np.asarray(out["sli"], dtype=float).reshape(-1)
    if "sli_ts" in out:
        out["sli_ts"] = np.asarray(out["sli_ts"], dtype=float)
    out["sli_select_skip_first_sync_buckets"] = int(
        as_scalar(out.get("sli_select_skip_first_sync_buckets", 0))
    )
    out["sli_select_keep_first_sync_buckets"] = int(
        as_scalar(out.get("sli_select_keep_first_sync_buckets", 0))
    )
    if path is not None:
        out["path"] = path
    elif "path" in out:
        out["path"] = str(out["path"])
    return out


def load_sli_bundle(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    missing = [k for k in REQ_BUNDLE_KEYS if k not in d.files]
    if missing:
        raise ValueError(f"Bundle {path} is missing keys: {missing}")

    out = {k: d[k] for k in REQ_BUNDLE_KEYS}
    for k in d.files:
        if k in out:
            continue
        if k.startswith(BUNDLE_ARRAY_PREFIXES) or k in ("sli_ts", "fly_ids"):
            out[k] = d[k]
    return normalize_sli_bundle(out, path=path)


def bundle_video_ids(bundle: dict) -> np.ndarray | None:
    key = "video_uid" if "video_uid" in bundle else "video_ids"
    if key not in bundle:
        return None
    return as_str_array(bundle[key])


def bundle_fly_ids(bundle: dict) -> np.ndarray | None:
    if "fly_ids" not in bundle:
        return None
    return np.asarray(bundle["fly_ids"], dtype=int).reshape(-1)


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
