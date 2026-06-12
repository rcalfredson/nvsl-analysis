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
    "episode_filter_",
    "exp_target_sync_bucket_filter_",
    "exp_pi_threshold_filter_",
    "weaving_",
    "wallpct_",
    "turnback_",
    "agarose_",
    "lgturn_",
    "reward_lgturn_",
    "reward_lv_",
    "speed_",
    "fraction_within_radius_",
    "return_prob_",
    "sli_",
)

FRACTION_WITHIN_RADIUS_EXCURSION_BIN_KEYS = (
    "fraction_within_radius_excursion_bin_ratio_exp",
    "fraction_within_radius_excursion_bin_ratio_ctrl",
    "fraction_within_radius_excursion_bin_return_exp",
    "fraction_within_radius_excursion_bin_return_ctrl",
    "fraction_within_radius_excursion_bin_total_exp",
    "fraction_within_radius_excursion_bin_total_ctrl",
    "fraction_within_radius_excursion_bin_edges_mm",
)

FRACTION_WITHIN_RADIUS_EXCURSION_BIN_LEGACY_KEYS = (
    "return_prob_excursion_bin_ratio_exp",
    "return_prob_excursion_bin_ratio_ctrl",
    "return_prob_excursion_bin_return_exp",
    "return_prob_excursion_bin_return_ctrl",
    "return_prob_excursion_bin_total_exp",
    "return_prob_excursion_bin_total_ctrl",
    "return_prob_excursion_bin_edges_mm",
)

FRACTION_WITHIN_RADIUS_LEGACY_KEY_ALIASES = {
    "return_prob_outer_radius_ratio_exp": "fraction_within_radius_outer_radius_ratio_exp",
    "return_prob_outer_radius_ratio_ctrl": "fraction_within_radius_outer_radius_ratio_ctrl",
    "return_prob_outer_radius_return_exp": "fraction_within_radius_outer_radius_return_exp",
    "return_prob_outer_radius_return_ctrl": "fraction_within_radius_outer_radius_return_ctrl",
    "return_prob_outer_radius_total_exp": "fraction_within_radius_outer_radius_total_exp",
    "return_prob_outer_radius_total_ctrl": "fraction_within_radius_outer_radius_total_ctrl",
    "return_prob_outer_radius_outer_radii_mm": "fraction_within_radius_outer_radius_outer_radii_mm",
    "return_prob_outer_radius_outer_deltas_mm": "fraction_within_radius_outer_radius_outer_deltas_mm",
    "return_prob_outer_radius_trainings": "fraction_within_radius_outer_radius_trainings",
    "return_prob_outer_radius_skip_first_sync_buckets": "fraction_within_radius_outer_radius_skip_first_sync_buckets",
    "return_prob_outer_radius_keep_first_sync_buckets": "fraction_within_radius_outer_radius_keep_first_sync_buckets",
    "return_prob_outer_radius_last_sync_buckets": "fraction_within_radius_outer_radius_last_sync_buckets",
    "return_prob_outer_radius_reward_radius_mm": "fraction_within_radius_outer_radius_reward_radius_mm",
    "return_prob_outer_radius_reward_delta_mm": "fraction_within_radius_outer_radius_reward_delta_mm",
    "return_prob_outer_radius_border_width_mm": "fraction_within_radius_outer_radius_border_width_mm",
    "return_prob_outer_radius_exclude_wall_contact": "fraction_within_radius_outer_radius_exclude_wall_contact",
    "return_prob_outer_radius_window_summary": "fraction_within_radius_outer_radius_window_summary",
    "return_prob_excursion_bin_ratio_exp": "fraction_within_radius_excursion_bin_ratio_exp",
    "return_prob_excursion_bin_ratio_ctrl": "fraction_within_radius_excursion_bin_ratio_ctrl",
    "return_prob_excursion_bin_return_exp": "fraction_within_radius_excursion_bin_return_exp",
    "return_prob_excursion_bin_return_ctrl": "fraction_within_radius_excursion_bin_return_ctrl",
    "return_prob_excursion_bin_total_exp": "fraction_within_radius_excursion_bin_total_exp",
    "return_prob_excursion_bin_total_ctrl": "fraction_within_radius_excursion_bin_total_ctrl",
    "return_prob_excursion_bin_edges_mm": "fraction_within_radius_excursion_bin_edges_mm",
    "return_prob_excursion_bin_radii_mm": "fraction_within_radius_excursion_bin_radii_mm",
    "return_prob_excursion_bin_requested_edges_mm": "fraction_within_radius_excursion_bin_requested_edges_mm",
    "return_prob_excursion_bin_open_ended_upper_bin": "fraction_within_radius_excursion_bin_open_ended_upper_bin",
    "return_prob_excursion_bin_trainings": "fraction_within_radius_excursion_bin_trainings",
    "return_prob_excursion_bin_skip_first_sync_buckets": "fraction_within_radius_excursion_bin_skip_first_sync_buckets",
    "return_prob_excursion_bin_keep_first_sync_buckets": "fraction_within_radius_excursion_bin_keep_first_sync_buckets",
    "return_prob_excursion_bin_last_sync_buckets": "fraction_within_radius_excursion_bin_last_sync_buckets",
    "return_prob_excursion_bin_reward_delta_mm": "fraction_within_radius_excursion_bin_reward_delta_mm",
    "return_prob_excursion_bin_reward_radius_mm": "fraction_within_radius_excursion_bin_reward_radius_mm",
    "return_prob_excursion_bin_border_width_mm": "fraction_within_radius_excursion_bin_border_width_mm",
    "return_prob_excursion_bin_exclude_wall_contact": "fraction_within_radius_excursion_bin_exclude_wall_contact",
    "return_prob_excursion_bin_window_summary": "fraction_within_radius_excursion_bin_window_summary",
    "return_prob_excursion_bin_description": "fraction_within_radius_excursion_bin_description",
}

RETURN_PROB_EXCURSION_BIN_KEYS = FRACTION_WITHIN_RADIUS_EXCURSION_BIN_KEYS

RETURN_PROB_EXCURSION_BIN_LEGACY_KEYS = (
    FRACTION_WITHIN_RADIUS_EXCURSION_BIN_LEGACY_KEYS
)

TURNBACK_EXCURSION_BIN_KEYS = (
    "turnback_excursion_bin_ratio_exp",
    "turnback_excursion_bin_ratio_ctrl",
    "turnback_excursion_bin_turn_exp",
    "turnback_excursion_bin_turn_ctrl",
    "turnback_excursion_bin_total_exp",
    "turnback_excursion_bin_total_ctrl",
    "turnback_excursion_bin_edges_mm",
)

BETWEEN_REWARD_MAXDIST_KEYS = (
    "between_reward_maxdist_exp",
    "between_reward_maxdist_ctrl",
    "between_reward_maxdistN_exp",
    "between_reward_maxdistN_ctrl",
)

BETWEEN_REWARD_RETURN_LEG_DIST_KEYS = (
    "between_reward_return_leg_dist_exp",
    "between_reward_return_leg_dist_ctrl",
    "between_reward_return_leg_distN_exp",
    "between_reward_return_leg_distN_ctrl",
)

COMMAG_KEYS = (
    "commag_exp",
    "commag_ctrl",
)

COMMAG_COUNT_KEYS = (
    "commagN_exp",
    "commagN_ctrl",
)

TURNBACK_RATIO_KEYS = (
    "turnback_ratio_exp",
    "turnback_ratio_ctrl",
    "turnback_total_exp",
    "turnback_total_ctrl",
    "turnback_inner_delta_mm",
    "turnback_outer_delta_mm",
)

TURNBACK_RATIO_PRIMARY_KEYS = (
    "turnback_ratio_exp",
    "turnback_ratio_ctrl",
    "turnback_total_exp",
    "turnback_total_ctrl",
)

AGAROSE_RATIO_KEYS = (
    "agarose_ratio_exp",
    "agarose_ratio_ctrl",
    "agarose_total_exp",
    "agarose_total_ctrl",
    "agarose_avoid_exp",
    "agarose_avoid_ctrl",
)

AGAROSE_PRE_KEYS = (
    "agarose_pre_ratio_exp",
    "agarose_pre_ratio_ctrl",
    "agarose_pre_total_exp",
    "agarose_pre_total_ctrl",
    "agarose_pre_avoid_exp",
    "agarose_pre_avoid_ctrl",
)

AGAROSE_TRAINING_PRE_KEYS = (
    "agarose_training_pre_ratio_exp",
    "agarose_training_pre_ratio_ctrl",
    "agarose_training_pre_total_exp",
    "agarose_training_pre_total_ctrl",
    "agarose_training_pre_avoid_exp",
    "agarose_training_pre_avoid_ctrl",
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


def add_fraction_within_radius_key_aliases(bundle: dict) -> dict:
    """Populate permanent fraction-within-radius keys from legacy return-prob keys."""
    out = dict(bundle)
    for legacy_key, new_key in FRACTION_WITHIN_RADIUS_LEGACY_KEY_ALIASES.items():
        if new_key not in out and legacy_key in out:
            out[new_key] = out[legacy_key]
    return out


def _bundle_label(bundle: dict, path: str | None = None) -> str:
    return path or bundle.get("path") or "<unknown>"


def _exp_target_sync_bucket_eligible(bundle: dict, n_videos: int, where: str):
    eligible = np.asarray(
        bundle.get(
            "exp_target_sync_bucket_filter_eligible",
            bundle.get(
                "exp_pi_threshold_filter_eligible",
                np.ones(n_videos, dtype=bool),
            ),
        ),
        dtype=bool,
    ).reshape(-1)
    if eligible.size != n_videos:
        raise ValueError(
            f"Bundle {where} has exp_target_sync_bucket_filter_eligible.shape="
            f"{eligible.shape} but expected {(n_videos,)}"
        )
    return eligible


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


def _validate_return_prob_metric_array(
    bundle: dict,
    key: str,
    *,
    where: str,
    n_videos: int,
    n_bins: int,
) -> np.ndarray:
    arr = np.asarray(bundle[key], dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Bundle {where} has non-2D {key} shape {arr.shape}")
    if arr.shape != (n_videos, n_bins):
        raise ValueError(
            f"Bundle {where} has {key}.shape={arr.shape} "
            f"but expected {(n_videos, n_bins)}"
        )
    return arr


def _validate_return_prob_count_array(
    bundle: dict,
    key: str,
    *,
    where: str,
    n_videos: int,
    n_bins: int,
) -> np.ndarray:
    arr = np.asarray(bundle[key])
    if arr.ndim != 2:
        raise ValueError(f"Bundle {where} has non-2D {key} shape {arr.shape}")
    if arr.shape != (n_videos, n_bins):
        raise ValueError(
            f"Bundle {where} has {key}.shape={arr.shape} "
            f"but expected {(n_videos, n_bins)}"
        )
    if np.any(~np.isfinite(arr.astype(float))):
        raise ValueError(f"Bundle {where} has non-finite values in {key}")
    if np.any(arr < 0):
        raise ValueError(f"Bundle {where} has negative values in {key}")
    if np.any(arr != np.floor(arr)):
        raise ValueError(f"Bundle {where} has non-integer values in {key}")
    return arr.astype(int, copy=False)


def _validate_3d_distance_metric_array(
    bundle: dict,
    key: str,
    *,
    where: str,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    arr = np.asarray(bundle[key], dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Bundle {where} has non-3D {key} shape {arr.shape}")
    if arr.shape != expected_shape:
        raise ValueError(
            f"Bundle {where} has {key}.shape={arr.shape} "
            f"but expected {expected_shape}"
        )
    if np.any(np.isinf(arr)):
        raise ValueError(f"Bundle {where} has infinite values in {key}")
    finite = np.isfinite(arr)
    if np.any(arr[finite] < 0.0):
        raise ValueError(f"Bundle {where} has negative values in {key}")
    return arr


def _validate_3d_distance_count_array(
    bundle: dict,
    key: str,
    *,
    where: str,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    arr = np.asarray(bundle[key])
    if arr.ndim != 3:
        raise ValueError(f"Bundle {where} has non-3D {key} shape {arr.shape}")
    if arr.shape != expected_shape:
        raise ValueError(
            f"Bundle {where} has {key}.shape={arr.shape} "
            f"but expected {expected_shape}"
        )
    arr_float = arr.astype(float)
    if np.any(~np.isfinite(arr_float)):
        raise ValueError(f"Bundle {where} has non-finite values in {key}")
    if np.any(arr_float < 0):
        raise ValueError(f"Bundle {where} has negative values in {key}")
    if np.any(arr_float != np.floor(arr_float)):
        raise ValueError(f"Bundle {where} has non-integer values in {key}")
    return arr.astype(int, copy=False)


def _expected_sync_bucket_metric_shape(
    bundle: dict,
    *,
    where: str,
) -> tuple[int, int, int]:
    sli = np.asarray(bundle["sli"], dtype=float)
    n_videos = int(sli.shape[0])
    if "sli_ts" not in bundle:
        raise ValueError(
            f"Bundle {where} is missing sli_ts needed to validate sync-bucket metrics"
        )
    sli_ts = np.asarray(bundle["sli_ts"], dtype=float)
    if sli_ts.ndim != 3:
        raise ValueError(f"Bundle {where} has non-3D sli_ts shape: {sli_ts.shape}")
    return (n_videos, int(sli_ts.shape[1]), int(sli_ts.shape[2]))


def validate_between_reward_sync_bucket_distance_bundle(
    bundle: dict,
    *,
    metric_name: str,
    value_exp_key: str,
    value_ctrl_key: str,
    count_exp_key: str,
    count_ctrl_key: str,
    path: str | None = None,
) -> None:
    where = _bundle_label(bundle, path)
    keys = (value_exp_key, value_ctrl_key, count_exp_key, count_ctrl_key)
    missing = [k for k in keys if k not in bundle]
    if missing:
        raise ValueError(f"Bundle {where} is missing {metric_name} keys: {missing}")

    expected_shape = _expected_sync_bucket_metric_shape(bundle, where=where)
    value_exp = _validate_3d_distance_metric_array(
        bundle, value_exp_key, where=where, expected_shape=expected_shape
    )
    value_ctrl = _validate_3d_distance_metric_array(
        bundle, value_ctrl_key, where=where, expected_shape=expected_shape
    )
    count_exp = _validate_3d_distance_count_array(
        bundle, count_exp_key, where=where, expected_shape=expected_shape
    )
    count_ctrl = _validate_3d_distance_count_array(
        bundle, count_ctrl_key, where=where, expected_shape=expected_shape
    )
    min_segments = int(as_scalar(bundle.get("btw_rwd_sync_bucket_min_trajectories", 1)))
    if min_segments < 0:
        raise ValueError(
            f"Bundle {where} has negative btw_rwd_sync_bucket_min_trajectories"
        )
    sufficient_count = max(1, min_segments)

    for key, values, counts in (
        (value_exp_key, value_exp, count_exp),
        (value_ctrl_key, value_ctrl, count_ctrl),
    ):
        finite = np.isfinite(values)
        if np.any(finite & (counts < sufficient_count)):
            raise ValueError(
                f"Bundle {where} has finite {key} where count is below "
                f"{sufficient_count}"
            )


def validate_between_reward_maxdist_bundle(
    bundle: dict, *, path: str | None = None
) -> None:
    validate_between_reward_sync_bucket_distance_bundle(
        bundle,
        metric_name="between-reward max-distance",
        value_exp_key="between_reward_maxdist_exp",
        value_ctrl_key="between_reward_maxdist_ctrl",
        count_exp_key="between_reward_maxdistN_exp",
        count_ctrl_key="between_reward_maxdistN_ctrl",
        path=path,
    )


def validate_between_reward_return_leg_dist_bundle(
    bundle: dict, *, path: str | None = None
) -> None:
    validate_between_reward_sync_bucket_distance_bundle(
        bundle,
        metric_name="between-reward return-leg distance",
        value_exp_key="between_reward_return_leg_dist_exp",
        value_ctrl_key="between_reward_return_leg_dist_ctrl",
        count_exp_key="between_reward_return_leg_distN_exp",
        count_ctrl_key="between_reward_return_leg_distN_ctrl",
        path=path,
    )


def validate_commag_bundle(bundle: dict, *, path: str | None = None) -> None:
    where = _bundle_label(bundle, path)
    missing = [k for k in COMMAG_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"Bundle {where} is missing COM-magnitude keys: {missing}")

    expected_shape = _expected_sync_bucket_metric_shape(bundle, where=where)
    value_exp = _validate_3d_distance_metric_array(
        bundle, "commag_exp", where=where, expected_shape=expected_shape
    )
    value_ctrl = _validate_3d_distance_metric_array(
        bundle, "commag_ctrl", where=where, expected_shape=expected_shape
    )

    bucket_len_min = float(as_scalar(bundle.get("bucket_len_min", np.nan)))
    if not np.isfinite(bucket_len_min) or bucket_len_min <= 0.0:
        raise ValueError(f"Bundle {where} has invalid bucket_len_min={bucket_len_min}")

    min_segments = int(as_scalar(bundle.get("btw_rwd_sync_bucket_min_trajectories", 1)))
    if min_segments < 0:
        raise ValueError(
            f"Bundle {where} has negative btw_rwd_sync_bucket_min_trajectories"
        )

    present_count_keys = [k for k in COMMAG_COUNT_KEYS if k in bundle]
    if present_count_keys and set(present_count_keys) != set(COMMAG_COUNT_KEYS):
        missing_counts = [k for k in COMMAG_COUNT_KEYS if k not in bundle]
        raise ValueError(
            f"Bundle {where} has incomplete COM-magnitude count keys: {missing_counts}"
        )
    if not present_count_keys:
        return

    count_exp = _validate_3d_distance_count_array(
        bundle, "commagN_exp", where=where, expected_shape=expected_shape
    )
    count_ctrl = _validate_3d_distance_count_array(
        bundle, "commagN_ctrl", where=where, expected_shape=expected_shape
    )

    sufficient_count = max(1, min_segments)
    for key, values, counts in (
        ("commag_exp", value_exp, count_exp),
        ("commag_ctrl", value_ctrl, count_ctrl),
    ):
        finite = np.isfinite(values)
        if np.any(finite & (counts < sufficient_count)):
            raise ValueError(
                f"Bundle {where} has finite {key} where count is below "
                f"{sufficient_count}"
            )


def _validate_probability_array(
    bundle: dict,
    key: str,
    *,
    where: str,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    arr = np.asarray(bundle[key], dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Bundle {where} has non-3D {key} shape {arr.shape}")
    if arr.shape != expected_shape:
        raise ValueError(
            f"Bundle {where} has {key}.shape={arr.shape} "
            f"but expected {expected_shape}"
        )
    if np.any(np.isinf(arr)):
        raise ValueError(f"Bundle {where} has infinite values in {key}")
    finite = np.isfinite(arr)
    if np.any((arr[finite] < 0.0) | (arr[finite] > 1.0)):
        raise ValueError(f"Bundle {where} has out-of-range probabilities in {key}")
    return arr


def validate_turnback_ratio_bundle(bundle: dict, *, path: str | None = None) -> None:
    where = _bundle_label(bundle, path)
    missing = [k for k in TURNBACK_RATIO_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"Bundle {where} is missing turnback ratio keys: {missing}")

    expected_shape = _expected_sync_bucket_metric_shape(bundle, where=where)
    ratio_exp = _validate_probability_array(
        bundle, "turnback_ratio_exp", where=where, expected_shape=expected_shape
    )
    ratio_ctrl = _validate_probability_array(
        bundle, "turnback_ratio_ctrl", where=where, expected_shape=expected_shape
    )
    total_exp = _validate_3d_distance_count_array(
        bundle, "turnback_total_exp", where=where, expected_shape=expected_shape
    )
    total_ctrl = _validate_3d_distance_count_array(
        bundle, "turnback_total_ctrl", where=where, expected_shape=expected_shape
    )
    min_total = int(as_scalar(bundle.get("min_turnback_episodes", 1)))
    if min_total < 0:
        raise ValueError(f"Bundle {where} has negative min_turnback_episodes")

    target_sync_bucket_eligible = _exp_target_sync_bucket_eligible(
        bundle, expected_shape[0], where
    )

    for key, ratio, total in (
        ("turnback_ratio_exp", ratio_exp, total_exp),
        ("turnback_ratio_ctrl", ratio_ctrl, total_ctrl),
    ):
        reportable = total >= max(1, min_total)
        if key.endswith("_exp"):
            reportable = reportable & target_sync_bucket_eligible[:, None, None]
        if np.any(~np.isfinite(ratio[reportable])):
            raise ValueError(
                f"Bundle {where} has non-finite {key} where total passes "
                f"reporting threshold {min_total}"
            )
        if np.any(np.isfinite(ratio[~reportable])):
            raise ValueError(
                f"Bundle {where} has finite {key} where total is below "
                f"reporting threshold {min_total}"
            )
        implied_turns = ratio[reportable] * total[reportable]
        if np.any(np.abs(implied_turns - np.rint(implied_turns)) > 1e-10):
            raise ValueError(
                f"Bundle {where} has {key} values inconsistent with integer counts"
            )

    inner_delta_mm = float(as_scalar(bundle["turnback_inner_delta_mm"]))
    outer_delta_mm = float(as_scalar(bundle["turnback_outer_delta_mm"]))
    if not np.isfinite(inner_delta_mm):
        raise ValueError(f"Bundle {where} has non-finite turnback_inner_delta_mm")
    if not np.isfinite(outer_delta_mm):
        raise ValueError(f"Bundle {where} has non-finite turnback_outer_delta_mm")
    if inner_delta_mm < 0.0:
        raise ValueError(f"Bundle {where} has negative turnback_inner_delta_mm")
    if outer_delta_mm <= inner_delta_mm:
        raise ValueError(
            f"Bundle {where} has turnback_outer_delta_mm={outer_delta_mm} "
            f"<= turnback_inner_delta_mm={inner_delta_mm}"
        )
    if "turnback_inner_radius_offset_px" in bundle:
        offset_px = float(as_scalar(bundle["turnback_inner_radius_offset_px"]))
        if not np.isfinite(offset_px):
            raise ValueError(
                f"Bundle {where} has non-finite turnback_inner_radius_offset_px"
            )


def _validate_count_array(
    bundle: dict,
    key: str,
    *,
    where: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(bundle[key])
    if arr.shape != expected_shape:
        raise ValueError(
            f"Bundle {where} has {key}.shape={arr.shape} "
            f"but expected {expected_shape}"
        )
    arr_float = arr.astype(float)
    if np.any(~np.isfinite(arr_float)):
        raise ValueError(f"Bundle {where} has non-finite values in {key}")
    if np.any(arr_float < 0):
        raise ValueError(f"Bundle {where} has negative values in {key}")
    if np.any(arr_float != np.floor(arr_float)):
        raise ValueError(f"Bundle {where} has non-integer values in {key}")
    return arr.astype(int, copy=False)


def _validate_ratio_count_semantics(
    bundle: dict,
    *,
    where: str,
    ratio_key: str,
    avoid_key: str,
    total_key: str,
    expected_shape: tuple[int, ...],
    min_total: int,
    reportable_mask: np.ndarray | None = None,
) -> None:
    ratio = np.asarray(bundle[ratio_key], dtype=float)
    if ratio.shape != expected_shape:
        raise ValueError(
            f"Bundle {where} has {ratio_key}.shape={ratio.shape} "
            f"but expected {expected_shape}"
        )
    if np.any(np.isinf(ratio)):
        raise ValueError(f"Bundle {where} has infinite values in {ratio_key}")
    finite = np.isfinite(ratio)
    if np.any((ratio[finite] < 0.0) | (ratio[finite] > 1.0)):
        raise ValueError(
            f"Bundle {where} has out-of-range probabilities in {ratio_key}"
        )

    avoid = _validate_count_array(
        bundle, avoid_key, where=where, expected_shape=expected_shape
    )
    total = _validate_count_array(
        bundle, total_key, where=where, expected_shape=expected_shape
    )
    if np.any(avoid > total):
        raise ValueError(f"Bundle {where} has {avoid_key} values greater than totals")

    reportable = total >= int(min_total)
    if min_total <= 0:
        reportable = total > 0
    else:
        reportable &= total > 0
    if reportable_mask is not None:
        mask = np.asarray(reportable_mask, dtype=bool)
        try:
            mask = np.broadcast_to(mask, expected_shape)
        except ValueError as exc:
            raise ValueError(
                f"Bundle {where} has {ratio_key} reportable_mask.shape="
                f"{mask.shape} but expected broadcastable to {expected_shape}"
            ) from exc
        reportable &= mask

    if np.any(~np.isfinite(ratio[reportable])):
        raise ValueError(
            f"Bundle {where} has non-finite {ratio_key} where total passes "
            f"reporting threshold {min_total}"
        )
    if np.any(np.isfinite(ratio[~reportable])):
        raise ValueError(
            f"Bundle {where} has finite {ratio_key} where total is below "
            f"reporting threshold {min_total}"
        )

    expected = np.full_like(ratio, np.nan, dtype=float)
    np.divide(avoid, total, out=expected, where=reportable)
    both_finite = np.isfinite(ratio) & np.isfinite(expected)
    if np.any(np.abs(ratio[both_finite] - expected[both_finite]) > 1e-10):
        raise ValueError(f"Bundle {where} has inconsistent {ratio_key} values")


def validate_agarose_sli_bundle(bundle: dict, *, path: str | None = None) -> None:
    where = _bundle_label(bundle, path)
    missing = [k for k in AGAROSE_RATIO_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"Bundle {where} is missing agarose ratio keys: {missing}")

    expected_shape = _expected_sync_bucket_metric_shape(bundle, where=where)
    min_total = int(
        as_scalar(
            bundle.get(
                "min_agarose_episodes",
                bundle.get("agarose_dual_circle_min_total", 10),
            )
        )
    )
    if min_total < 0:
        raise ValueError(f"Bundle {where} has negative min_agarose_episodes")

    n_videos = expected_shape[0]
    target_sync_bucket_eligible = _exp_target_sync_bucket_eligible(
        bundle, n_videos, where
    )

    for suffix in ("exp", "ctrl"):
        _validate_ratio_count_semantics(
            bundle,
            where=where,
            ratio_key=f"agarose_ratio_{suffix}",
            avoid_key=f"agarose_avoid_{suffix}",
            total_key=f"agarose_total_{suffix}",
            expected_shape=expected_shape,
            min_total=min_total,
            reportable_mask=(
                target_sync_bucket_eligible[:, None, None]
                if suffix == "exp"
                else None
            ),
        )

    present_pre = [k for k in AGAROSE_PRE_KEYS if k in bundle]
    if present_pre and set(present_pre) != set(AGAROSE_PRE_KEYS):
        missing_pre = [k for k in AGAROSE_PRE_KEYS if k not in bundle]
        raise ValueError(
            f"Bundle {where} has incomplete agarose pre keys: {missing_pre}"
        )
    if present_pre:
        for suffix in ("exp", "ctrl"):
            _validate_ratio_count_semantics(
                bundle,
                where=where,
                ratio_key=f"agarose_pre_ratio_{suffix}",
                avoid_key=f"agarose_pre_avoid_{suffix}",
                total_key=f"agarose_pre_total_{suffix}",
                expected_shape=(n_videos,),
                min_total=min_total,
                reportable_mask=(
                    target_sync_bucket_eligible if suffix == "exp" else None
                ),
            )
        if "agarose_pre_window_min" in bundle:
            window_min = float(as_scalar(bundle["agarose_pre_window_min"]))
            if not np.isfinite(window_min) or window_min < 0.0:
                raise ValueError(f"Bundle {where} has invalid agarose_pre_window_min")

    present_training_pre = [k for k in AGAROSE_TRAINING_PRE_KEYS if k in bundle]
    if present_training_pre and set(present_training_pre) != set(
        AGAROSE_TRAINING_PRE_KEYS
    ):
        missing_training_pre = [k for k in AGAROSE_TRAINING_PRE_KEYS if k not in bundle]
        raise ValueError(
            f"Bundle {where} has incomplete agarose training-pre keys: "
            f"{missing_training_pre}"
        )
    if present_training_pre:
        n_videos, n_trainings, _n_sync_buckets = expected_shape
        for suffix in ("exp", "ctrl"):
            _validate_ratio_count_semantics(
                bundle,
                where=where,
                ratio_key=f"agarose_training_pre_ratio_{suffix}",
                avoid_key=f"agarose_training_pre_avoid_{suffix}",
                total_key=f"agarose_training_pre_total_{suffix}",
                expected_shape=(n_videos, n_trainings),
                min_total=min_total,
                reportable_mask=(
                    target_sync_bucket_eligible[:, None]
                    if suffix == "exp"
                    else None
                ),
            )
        if "agarose_training_pre_window_min" in bundle:
            window_min = np.asarray(
                bundle["agarose_training_pre_window_min"], dtype=float
            )
            if window_min.shape != (n_trainings,):
                raise ValueError(
                    f"Bundle {where} has agarose_training_pre_window_min.shape="
                    f"{window_min.shape} but expected {(n_trainings,)}"
                )
            finite = np.isfinite(window_min)
            if np.any(window_min[finite] < 0.0):
                raise ValueError(
                    f"Bundle {where} has negative agarose_training_pre_window_min"
                )


def validate_fraction_within_radius_excursion_bin_bundle(
    bundle: dict, *, path: str | None = None
) -> None:
    bundle = add_fraction_within_radius_key_aliases(bundle)
    where = _bundle_label(bundle, path)
    missing = [k for k in RETURN_PROB_EXCURSION_BIN_KEYS if k not in bundle]
    if missing:
        raise ValueError(
            f"Bundle {where} is missing fraction-within-radius excursion-bin keys: {missing}"
        )

    sli = np.asarray(bundle["sli"], dtype=float)
    if sli.ndim != 1:
        raise ValueError(f"Bundle {where} has non-1D sli shape {sli.shape}")
    n_videos = int(sli.shape[0])

    edges = np.asarray(bundle["fraction_within_radius_excursion_bin_edges_mm"], dtype=float)
    if edges.ndim != 1:
        raise ValueError(
            f"Bundle {where} has non-1D fraction_within_radius_excursion_bin_edges_mm "
            f"shape {edges.shape}"
        )
    if edges.size < 2:
        raise ValueError(
            f"Bundle {where} has fewer than two fraction-within-radius bin edges"
        )
    if not np.all(np.isfinite(edges)):
        raise ValueError(
            f"Bundle {where} has non-finite resolved fraction-within-radius bin edges"
        )
    if np.any(np.diff(edges) <= 0):
        raise ValueError(
            f"Bundle {where} has non-increasing fraction-within-radius bin edges"
        )
    n_bins = int(edges.size - 1)

    if "fraction_within_radius_excursion_bin_requested_edges_mm" in bundle:
        requested = np.asarray(
            bundle["fraction_within_radius_excursion_bin_requested_edges_mm"], dtype=float
        )
        if requested.ndim != 1 or requested.size != edges.size:
            raise ValueError(
                f"Bundle {where} has fraction_within_radius_excursion_bin_requested_edges_mm "
                f"shape {requested.shape} but expected {(edges.size,)}"
            )
        if not np.all(np.isfinite(requested[:-1])):
            raise ValueError(
                f"Bundle {where} has non-finite interior requested bin edges"
            )
        if np.any(np.diff(requested) <= 0):
            raise ValueError(f"Bundle {where} has non-increasing requested bin edges")

    if "fraction_within_radius_excursion_bin_window_summary" in bundle:
        window_summary = as_str_array(
            bundle["fraction_within_radius_excursion_bin_window_summary"]
        )
        if window_summary.shape[0] != n_videos:
            raise ValueError(
                f"Bundle {where} has len(fraction_within_radius_excursion_bin_window_summary)="
                f"{window_summary.shape[0]} but len(sli)={n_videos}"
            )

    for key in (
        "fraction_within_radius_excursion_bin_skip_first_sync_buckets",
        "fraction_within_radius_excursion_bin_keep_first_sync_buckets",
        "fraction_within_radius_excursion_bin_last_sync_buckets",
    ):
        if key in bundle and int(as_scalar(bundle[key])) < 0:
            raise ValueError(f"Bundle {where} has negative {key}")

    ratio_exp = _validate_return_prob_metric_array(
        bundle,
        "fraction_within_radius_excursion_bin_ratio_exp",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    ratio_ctrl = _validate_return_prob_metric_array(
        bundle,
        "fraction_within_radius_excursion_bin_ratio_ctrl",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    ret_exp = _validate_return_prob_metric_array(
        bundle,
        "fraction_within_radius_excursion_bin_return_exp",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    ret_ctrl = _validate_return_prob_metric_array(
        bundle,
        "fraction_within_radius_excursion_bin_return_ctrl",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    total_exp = _validate_return_prob_count_array(
        bundle,
        "fraction_within_radius_excursion_bin_total_exp",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    total_ctrl = _validate_return_prob_count_array(
        bundle,
        "fraction_within_radius_excursion_bin_total_ctrl",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    min_segments = int(as_scalar(bundle.get("btw_rwd_sync_bucket_min_trajectories", 1)))
    if min_segments < 0:
        raise ValueError(
            f"Bundle {where} has negative btw_rwd_sync_bucket_min_trajectories"
        )
    sufficient_count = max(1, min_segments)

    target_sync_bucket_eligible = _exp_target_sync_bucket_eligible(
        bundle, n_videos, where
    )

    for key, ratio in (
        ("fraction_within_radius_excursion_bin_ratio_exp", ratio_exp),
        ("fraction_within_radius_excursion_bin_ratio_ctrl", ratio_ctrl),
    ):
        finite = np.isfinite(ratio)
        if np.any((ratio[finite] < 0.0) | (ratio[finite] > 1.0)):
            raise ValueError(f"Bundle {where} has out-of-range probabilities in {key}")

    for key, values, total in (
        ("fraction_within_radius_excursion_bin_return_exp", ret_exp, total_exp),
        ("fraction_within_radius_excursion_bin_return_ctrl", ret_ctrl, total_ctrl),
    ):
        finite = np.isfinite(values)
        if np.any(~finite):
            raise ValueError(f"Bundle {where} has non-finite values in {key}")
        if np.any(values[finite] < 0.0):
            raise ValueError(f"Bundle {where} has negative values in {key}")
        if np.any(values[finite] > total[finite] + 1e-12):
            raise ValueError(f"Bundle {where} has {key} values greater than totals")

    for key, ratio, values, total in (
        (
            "fraction_within_radius_excursion_bin_ratio_exp",
            ratio_exp,
            ret_exp,
            total_exp,
        ),
        (
            "fraction_within_radius_excursion_bin_ratio_ctrl",
            ratio_ctrl,
            ret_ctrl,
            total_ctrl,
        ),
    ):
        expected = np.full_like(values, np.nan, dtype=float)
        reportable = total >= sufficient_count
        if key.endswith("_exp"):
            reportable = reportable & target_sync_bucket_eligible[:, None]
        np.divide(values, total, out=expected, where=reportable)
        both_finite = np.isfinite(ratio) & np.isfinite(expected)
        if np.any(~np.isfinite(ratio[reportable])):
            raise ValueError(
                f"Bundle {where} has non-finite {key} where total passes "
                f"reporting threshold {sufficient_count}"
            )
        if np.any(np.isfinite(ratio[~reportable])):
            raise ValueError(
                f"Bundle {where} has finite {key} where total is below "
                f"reporting threshold {sufficient_count}"
            )
        if np.any(np.abs(ratio[both_finite] - expected[both_finite]) > 1e-10):
            raise ValueError(f"Bundle {where} has inconsistent {key} values")


def validate_return_prob_excursion_bin_bundle(
    bundle: dict, *, path: str | None = None
) -> None:
    validate_fraction_within_radius_excursion_bin_bundle(bundle, path=path)


def validate_turnback_excursion_bin_bundle(
    bundle: dict, *, path: str | None = None
) -> None:
    where = _bundle_label(bundle, path)
    missing = [k for k in TURNBACK_EXCURSION_BIN_KEYS if k not in bundle]
    if missing:
        raise ValueError(
            f"Bundle {where} is missing turnback excursion-bin keys: {missing}"
        )

    sli = np.asarray(bundle["sli"], dtype=float)
    if sli.ndim != 1:
        raise ValueError(f"Bundle {where} has non-1D sli shape {sli.shape}")
    n_videos = int(sli.shape[0])

    pair_mode = bool(
        "turnback_excursion_bin_pair_inner_deltas_mm" in bundle
        or "turnback_excursion_bin_pair_outer_deltas_mm" in bundle
    )

    edges = np.asarray(bundle["turnback_excursion_bin_edges_mm"], dtype=float)
    if edges.ndim != 1:
        raise ValueError(
            f"Bundle {where} has non-1D turnback_excursion_bin_edges_mm "
            f"shape {edges.shape}"
        )
    if pair_mode:
        pair_inner = np.asarray(
            bundle.get("turnback_excursion_bin_pair_inner_deltas_mm", []), dtype=float
        )
        pair_outer = np.asarray(
            bundle.get("turnback_excursion_bin_pair_outer_deltas_mm", []), dtype=float
        )
        if pair_inner.ndim != 1 or pair_outer.ndim != 1:
            raise ValueError(f"Bundle {where} has non-1D turnback pair metadata")
        if pair_inner.size == 0 or pair_inner.size != pair_outer.size:
            raise ValueError(
                f"Bundle {where} has mismatched turnback pair metadata shapes "
                f"{pair_inner.shape} and {pair_outer.shape}"
            )
        if not np.all(np.isfinite(pair_inner)) or not np.all(np.isfinite(pair_outer)):
            raise ValueError(f"Bundle {where} has non-finite turnback pair metadata")
        if np.any(pair_inner < 0.0):
            raise ValueError(f"Bundle {where} has negative turnback pair inner deltas")
        if np.any(pair_outer <= pair_inner):
            raise ValueError(
                f"Bundle {where} has turnback pair outer deltas <= inner deltas"
            )
        n_bins = int(pair_inner.size)
    else:
        if edges.size < 2:
            raise ValueError(f"Bundle {where} has fewer than two turnback bin edges")
        if not np.all(np.isfinite(edges)):
            raise ValueError(
                f"Bundle {where} has non-finite resolved turnback bin edges"
            )
        if np.any(np.diff(edges) <= 0):
            raise ValueError(f"Bundle {where} has non-increasing turnback bin edges")
        n_bins = int(edges.size - 1)

    if "turnback_excursion_bin_requested_edges_mm" in bundle:
        requested = np.asarray(
            bundle["turnback_excursion_bin_requested_edges_mm"], dtype=float
        )
        expected_requested_size = 0 if pair_mode else edges.size
        if requested.ndim != 1 or requested.size != expected_requested_size:
            raise ValueError(
                f"Bundle {where} has turnback_excursion_bin_requested_edges_mm "
                f"shape {requested.shape} but expected {(expected_requested_size,)}"
            )
        if requested.size and not np.all(np.isfinite(requested[:-1])):
            raise ValueError(
                f"Bundle {where} has non-finite interior requested turnback bin edges"
            )
        if requested.size and np.any(np.diff(requested) <= 0):
            raise ValueError(
                f"Bundle {where} has non-increasing requested turnback bin edges"
            )

    if "turnback_excursion_bin_open_ended_upper_bin" in bundle:
        open_ended = bool(
            as_scalar(bundle["turnback_excursion_bin_open_ended_upper_bin"])
        )
        if open_ended and pair_mode:
            raise ValueError(
                f"Bundle {where} marks an open-ended turnback bin in pair mode"
            )
        if open_ended and "turnback_excursion_bin_requested_edges_mm" in bundle:
            requested = np.asarray(
                bundle["turnback_excursion_bin_requested_edges_mm"], dtype=float
            )
            if not np.isposinf(float(requested[-1])):
                raise ValueError(
                    f"Bundle {where} marks an open-ended turnback bin but the "
                    "requested last edge is not inf"
                )

    if "turnback_excursion_bin_trainings" in bundle:
        trainings = np.asarray(bundle["turnback_excursion_bin_trainings"], dtype=int)
        if trainings.ndim != 1:
            raise ValueError(
                f"Bundle {where} has non-1D turnback_excursion_bin_trainings "
                f"shape {trainings.shape}"
            )
        if np.any(trainings < 0):
            raise ValueError(
                f"Bundle {where} has negative turnback_excursion_bin_trainings"
            )

    if "turnback_excursion_bin_window_summary" in bundle:
        window_summary = as_str_array(bundle["turnback_excursion_bin_window_summary"])
        if window_summary.shape[0] != n_videos:
            raise ValueError(
                f"Bundle {where} has len(turnback_excursion_bin_window_summary)="
                f"{window_summary.shape[0]} but len(sli)={n_videos}"
            )

    for key in (
        "turnback_excursion_bin_skip_first_sync_buckets",
        "turnback_excursion_bin_keep_first_sync_buckets",
        "turnback_excursion_bin_last_sync_buckets",
    ):
        if key in bundle and int(as_scalar(bundle[key])) < 0:
            raise ValueError(f"Bundle {where} has negative {key}")

    for key in (
        "turnback_excursion_bin_inner_delta_mm",
        "turnback_excursion_bin_border_width_mm",
        "turnback_excursion_bin_inner_radius_offset_px",
    ):
        if key in bundle and not np.isfinite(float(as_scalar(bundle[key]))):
            raise ValueError(f"Bundle {where} has non-finite {key}")
    if (
        "turnback_excursion_bin_inner_delta_mm" in bundle
        and float(as_scalar(bundle["turnback_excursion_bin_inner_delta_mm"])) < 0.0
    ):
        raise ValueError(
            f"Bundle {where} has negative turnback_excursion_bin_inner_delta_mm"
        )
    if (
        "turnback_excursion_bin_border_width_mm" in bundle
        and float(as_scalar(bundle["turnback_excursion_bin_border_width_mm"])) < 0.0
    ):
        raise ValueError(
            f"Bundle {where} has negative turnback_excursion_bin_border_width_mm"
        )

    ratio_exp = _validate_return_prob_metric_array(
        bundle,
        "turnback_excursion_bin_ratio_exp",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    ratio_ctrl = _validate_return_prob_metric_array(
        bundle,
        "turnback_excursion_bin_ratio_ctrl",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    turn_exp = _validate_return_prob_metric_array(
        bundle,
        "turnback_excursion_bin_turn_exp",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    turn_ctrl = _validate_return_prob_metric_array(
        bundle,
        "turnback_excursion_bin_turn_ctrl",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    total_exp = _validate_return_prob_count_array(
        bundle,
        "turnback_excursion_bin_total_exp",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    total_ctrl = _validate_return_prob_count_array(
        bundle,
        "turnback_excursion_bin_total_ctrl",
        where=where,
        n_videos=n_videos,
        n_bins=n_bins,
    )
    min_total = int(as_scalar(bundle.get("min_turnback_episodes", 1)))
    if min_total < 0:
        raise ValueError(f"Bundle {where} has negative min_turnback_episodes")

    for key, ratio in (
        ("turnback_excursion_bin_ratio_exp", ratio_exp),
        ("turnback_excursion_bin_ratio_ctrl", ratio_ctrl),
    ):
        finite = np.isfinite(ratio)
        if np.any((ratio[finite] < 0.0) | (ratio[finite] > 1.0)):
            raise ValueError(f"Bundle {where} has out-of-range probabilities in {key}")

    for key, values, total in (
        ("turnback_excursion_bin_turn_exp", turn_exp, total_exp),
        ("turnback_excursion_bin_turn_ctrl", turn_ctrl, total_ctrl),
    ):
        finite = np.isfinite(values)
        if np.any(~finite):
            raise ValueError(f"Bundle {where} has non-finite values in {key}")
        if np.any(values[finite] < 0.0):
            raise ValueError(f"Bundle {where} has negative values in {key}")
        if np.any(values[finite] > total[finite] + 1e-12):
            raise ValueError(f"Bundle {where} has {key} values greater than totals")

    target_sync_bucket_eligible = _exp_target_sync_bucket_eligible(
        bundle, n_videos, where
    )

    for key, ratio, values, total in (
        ("turnback_excursion_bin_ratio_exp", ratio_exp, turn_exp, total_exp),
        ("turnback_excursion_bin_ratio_ctrl", ratio_ctrl, turn_ctrl, total_ctrl),
    ):
        expected = np.full_like(values, np.nan, dtype=float)
        reportable = total >= max(1, min_total)
        if key.endswith("_exp"):
            reportable = reportable & target_sync_bucket_eligible[:, None]
        np.divide(values, total, out=expected, where=reportable)
        both_finite = np.isfinite(ratio) & np.isfinite(expected)
        if np.any(~np.isfinite(ratio[reportable])):
            raise ValueError(
                f"Bundle {where} has non-finite {key} where total passes "
                f"reporting threshold {min_total}"
            )
        if np.any(np.isfinite(ratio[~reportable])):
            raise ValueError(
                f"Bundle {where} has finite {key} where total is below "
                f"reporting threshold {min_total}"
            )
        if np.any(np.abs(ratio[both_finite] - expected[both_finite]) > 1e-10):
            raise ValueError(f"Bundle {where} has inconsistent {key} values")


def normalize_sli_bundle(bundle: dict, *, path: str | None = None) -> dict:
    missing = [k for k in REQ_BUNDLE_KEYS if k not in bundle]
    if missing:
        where = path or bundle.get("path") or "<unknown>"
        raise ValueError(f"Bundle {where} is missing keys: {missing}")

    out = add_fraction_within_radius_key_aliases(bundle)
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
    if any(k in out for k in RETURN_PROB_EXCURSION_BIN_KEYS):
        validate_fraction_within_radius_excursion_bin_bundle(out, path=path)
    if any(k in out for k in BETWEEN_REWARD_MAXDIST_KEYS):
        validate_between_reward_maxdist_bundle(out, path=path)
    if any(k in out for k in BETWEEN_REWARD_RETURN_LEG_DIST_KEYS):
        validate_between_reward_return_leg_dist_bundle(out, path=path)
    if any(k in out for k in COMMAG_KEYS):
        validate_commag_bundle(out, path=path)
    if any(k in out for k in TURNBACK_RATIO_PRIMARY_KEYS):
        validate_turnback_ratio_bundle(out, path=path)
    if any(k in out for k in AGAROSE_RATIO_KEYS):
        validate_agarose_sli_bundle(out, path=path)
    if any(k in out for k in TURNBACK_EXCURSION_BIN_KEYS):
        validate_turnback_excursion_bin_bundle(out, path=path)
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
            "min_between_reward_trajectories",
            "min_turnback_episodes",
            "min_agarose_episodes",
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
