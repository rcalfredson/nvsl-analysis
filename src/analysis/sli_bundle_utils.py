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
    "commagN_",
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


def _bundle_label(bundle: dict, path: str | None = None) -> str:
    return path or bundle.get("path") or "<unknown>"


def validate_sli_bundle(bundle: dict, *, path: str | None = None) -> None:
    where = _bundle_label(
        bundle,
        path,
    )

    sli = np.asarray(bundle["sli"], dtype=float)
    if sli.ndim != 1:
        raise ValueError(f"Bundle {where} has non-1D sli shape {sli.shape}")

    video_ids = as_str_array(bundle["video_ids"])
    if video_ids.shape[0] != sli.shape[0]:
        raise ValueError(
            f"Bundle {where} has len(video_ids)={video_ids.shape[0]} "
            f"but len(sli)={sli.shape[0]}"
        )

    training_names = as_str_array(bundle["training_names"])
    sli_training_idx = int(as_scalar(bundle["sli_training_idx"]))
    if sli_training_idx < 0:
        raise ValueError(
            f"Bundle {where} has negative sli_training_idx={sli_training_idx}"
        )

    skip = int(as_scalar(bundle.get("sli_select_skip_first_sync_buckets", 0)))
    keep = int(as_scalar(bundle.get("sli_select_keep_first_sync_buckets", 0)))
    if skip < 0:
        raise ValueError(
            f"Bundle {where} has negative sli_select_skip_first_sync_buckets={skip}"
        )
    if keep < 0:
        raise ValueError(
            f"Bundle {where} has negative sli_select_keep_first_sync_buckets={keep}"
        )

    if "sli_ts" not in bundle:
        return

    sli_ts = np.asarray(bundle["sli_ts"], dtype=float)
    if sli_ts.ndim != 3:
        raise ValueError(f"Bundle {where} has non-3D sli_ts shape: {sli_ts.shape}")
    if sli_ts.shape[0] != sli.shape[0]:
        raise ValueError(
            f"Bundle {where} has sli_ts.shape[0]={sli_ts.shape[0]} "
            f"but len(sli)={sli.shape[0]}"
        )
    if training_names.shape[0] != sli_ts.shape[1]:
        raise ValueError(
            f"Bundle {where} has len(training_names)={training_names.shape[0]} "
            f"but sli_ts.shape[1]={sli_ts.shape[1]}"
        )
    if sli_ts.shape[1] > 0 and sli_training_idx >= sli_ts.shape[1]:
        raise ValueError(
            f"Bundle {where} has sli_training_idx={sli_training_idx} "
            f"but sli_ts has {sli_ts.shape[1]} trainings"
        )

    n_sync_buckets = sli_ts.shape[2]
    if skip > n_sync_buckets:
        raise ValueError(
            f"Bundle {where} skips {skip} SLI sync buckets, "
            f"but sli_ts has only {n_sync_buckets}"
        )
    if keep > 0 and skip + keep > n_sync_buckets:
        raise ValueError(
            f"Bundle {where} keeps SLI sync-bucket window [{skip}, {skip + keep}) "
            f"beyond sli_ts bucket count {n_sync_buckets}"
        )


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
    validate_sli_bundle(out, path=path)
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
        if k.startswith(BUNDLE_ARRAY_PREFIXES) or k in (
            "sli_ts",
            "fly_ids",
            "btw_rwd_sync_bucket_min_trajectories",
        ):
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


def align_by_video_ids(
    base_bundle: dict, comp_bundle: dict
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
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
