from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

SCHEMA_VERSION = 1
SLI_REGRESSION_KEYS = (
    "sli",
    "sli_ts",
    "video_ids",
    "training_names",
    "sli_training_idx",
    "sli_use_training_mean",
    "sli_select_skip_first_sync_buckets",
    "sli_select_keep_first_sync_buckets",
    "group_label",
)

RETURN_PROB_EXCURSION_BIN_REGRESSION_KEYS = (
    "group_label",
    "return_prob_excursion_bin_border_width_mm",
    "return_prob_excursion_bin_edges_mm",
    "return_prob_excursion_bin_keep_first_sync_buckets",
    "return_prob_excursion_bin_last_sync_buckets",
    "return_prob_excursion_bin_open_ended_upper_bin",
    "return_prob_excursion_bin_ratio_ctrl",
    "return_prob_excursion_bin_ratio_exp",
    "return_prob_excursion_bin_requested_edges_mm",
    "return_prob_excursion_bin_return_ctrl",
    "return_prob_excursion_bin_return_exp",
    "return_prob_excursion_bin_reward_delta_mm",
    "return_prob_excursion_bin_skip_first_sync_buckets",
    "return_prob_excursion_bin_total_ctrl",
    "return_prob_excursion_bin_total_exp",
    "return_prob_excursion_bin_trainings",
    "return_prob_excursion_bin_window_summary",
    "sli",
    "sli_select_keep_first_sync_buckets",
    "sli_select_skip_first_sync_buckets",
    "sli_training_idx",
    "sli_ts",
    "sli_use_training_mean",
    "training_names",
    "video_ids",
)

TURNBACK_EXCURSION_BIN_REGRESSION_KEYS = (
    "group_label",
    "sli",
    "sli_select_keep_first_sync_buckets",
    "sli_select_skip_first_sync_buckets",
    "sli_training_idx",
    "sli_ts",
    "sli_use_training_mean",
    "training_names",
    "turnback_excursion_bin_border_width_mm",
    "turnback_excursion_bin_edges_mm",
    "turnback_excursion_bin_inner_delta_mm",
    "turnback_excursion_bin_inner_radius_offset_px",
    "turnback_excursion_bin_keep_first_sync_buckets",
    "turnback_excursion_bin_last_sync_buckets",
    "turnback_excursion_bin_open_ended_upper_bin",
    "turnback_excursion_bin_ratio_ctrl",
    "turnback_excursion_bin_ratio_exp",
    "turnback_excursion_bin_requested_edges_mm",
    "turnback_excursion_bin_skip_first_sync_buckets",
    "turnback_excursion_bin_total_ctrl",
    "turnback_excursion_bin_total_exp",
    "turnback_excursion_bin_trainings",
    "turnback_excursion_bin_turn_ctrl",
    "turnback_excursion_bin_turn_exp",
    "turnback_excursion_bin_window_summary",
    "video_ids",
)

BETWEEN_REWARD_DISTANCE_HIST_REGRESSION_KEYS = (
    "bin_edges",
    "ci_hi",
    "ci_lo",
    "counts",
    "mean",
    "meta_json",
    "n_dropped",
    "n_raw",
    "n_units",
    "n_units_panel",
    "n_used",
    "panel_labels",
    "per_unit_ids_panel",
    "per_unit_panel",
)

BETWEEN_REWARD_CONDITIONED_DISTTRAV_REGRESSION_KEYS = (
    "ci_hi_tail",
    "ci_hi_total",
    "ci_lo_tail",
    "ci_lo_total",
    "mean_tail",
    "mean_total",
    "meta",
    "n_units",
    "per_unit_ids",
    "per_unit_tail",
    "per_unit_total",
    "x_centers",
    "x_edges",
)

BETWEEN_REWARD_MAXDIST_SLI_REGRESSION_KEYS = (
    "between_reward_maxdistN_ctrl",
    "between_reward_maxdistN_exp",
    "between_reward_maxdist_ctrl",
    "between_reward_maxdist_exp",
    "btw_rwd_sync_bucket_min_trajectories",
    "bucket_len_min",
    "group_label",
    "sli",
    "sli_select_keep_first_sync_buckets",
    "sli_select_skip_first_sync_buckets",
    "sli_training_idx",
    "sli_ts",
    "sli_use_training_mean",
    "training_names",
    "video_ids",
)

BETWEEN_REWARD_RETURN_LEG_DIST_SLI_REGRESSION_KEYS = (
    "between_reward_return_leg_distN_ctrl",
    "between_reward_return_leg_distN_exp",
    "between_reward_return_leg_dist_ctrl",
    "between_reward_return_leg_dist_exp",
    "btw_rwd_sync_bucket_min_trajectories",
    "bucket_len_min",
    "group_label",
    "sli",
    "sli_select_keep_first_sync_buckets",
    "sli_select_skip_first_sync_buckets",
    "sli_training_idx",
    "sli_ts",
    "sli_use_training_mean",
    "training_names",
    "video_ids",
)

COMMAG_SLI_REGRESSION_KEYS = (
    "btw_rwd_sync_bucket_min_trajectories",
    "bucket_len_min",
    "commagN_ctrl",
    "commagN_exp",
    "commag_ctrl",
    "commag_exp",
    "group_label",
    "sli",
    "sli_select_keep_first_sync_buckets",
    "sli_select_skip_first_sync_buckets",
    "sli_training_idx",
    "sli_ts",
    "sli_use_training_mean",
    "training_names",
    "video_ids",
)

TURNBACK_RATIO_REGRESSION_KEYS = (
    "bucket_len_min",
    "group_label",
    "sli",
    "sli_select_keep_first_sync_buckets",
    "sli_select_skip_first_sync_buckets",
    "sli_training_idx",
    "sli_ts",
    "sli_use_training_mean",
    "training_names",
    "turnback_inner_delta_mm",
    "turnback_inner_radius_offset_px",
    "turnback_outer_delta_mm",
    "turnback_ratio_ctrl",
    "turnback_ratio_exp",
    "turnback_total_ctrl",
    "turnback_total_exp",
    "video_ids",
)

AGAROSE_SLI_REGRESSION_KEYS = (
    "agarose_avoid_ctrl",
    "agarose_avoid_exp",
    "agarose_pre_avoid_ctrl",
    "agarose_pre_avoid_exp",
    "agarose_pre_label",
    "agarose_pre_ratio_ctrl",
    "agarose_pre_ratio_exp",
    "agarose_pre_total_ctrl",
    "agarose_pre_total_exp",
    "agarose_pre_window_min",
    "agarose_ratio_ctrl",
    "agarose_ratio_exp",
    "agarose_total_ctrl",
    "agarose_total_exp",
    "agarose_training_pre_avoid_ctrl",
    "agarose_training_pre_avoid_exp",
    "agarose_training_pre_label",
    "agarose_training_pre_ratio_ctrl",
    "agarose_training_pre_ratio_exp",
    "agarose_training_pre_total_ctrl",
    "agarose_training_pre_total_exp",
    "agarose_training_pre_window_min",
    "bucket_len_min",
    "group_label",
    "sli",
    "sli_training_idx",
    "sli_ts",
    "sli_use_training_mean",
    "training_names",
    "video_ids",
)

FIRST_N_REWARD_DIAGNOSTICS_REGRESSION_KEYS = (
    "video_basename",
    "video_path",
    "va_tag",
    "fly_idx",
    "selected_subset_label",
    "selected_trainings_label",
    "skip_first_sync_buckets",
    "keep_first_sync_buckets",
    "nth_reward_target",
    "has_selected_window",
    "actual_reward_count_in_selected_window",
    "eligible_for_nth_reward_cutoff",
    "sli",
    "cutoff_frame",
    "cutoff_training",
    "cutoff_time_since_selected_window_start_s",
    "cutoff_time_since_cutoff_training_start_s",
    "time_to_first_actual_reward_s",
    "time_to_nth_actual_reward_s",
    "first_n_reward_span_s",
    "actual_reward_count_by_cutoff",
    "control_reward_count_by_cutoff",
    "actual_circle_entry_count_by_cutoff",
    "control_circle_entry_count_by_cutoff",
    "reward_pi_by_cutoff",
    "actual_entry_minus_reward_count_by_cutoff",
    "control_entry_minus_reward_count_by_cutoff",
    "control_to_actual_entry_ratio_by_cutoff",
    "control_to_actual_reward_ratio_by_cutoff",
    "reward_event_type",
    "selected_reward_count_in_selected_window",
    "time_to_first_selected_reward_s",
    "time_to_nth_selected_reward_s",
    "first_n_selected_reward_span_s",
    "selected_reward_rate_to_nth_per_min",
)

REGRESSION_KEY_PRESETS = {
    "between_reward_conditioned_disttrav": (
        BETWEEN_REWARD_CONDITIONED_DISTTRAV_REGRESSION_KEYS
    ),
    "between_reward_distance_hist": BETWEEN_REWARD_DISTANCE_HIST_REGRESSION_KEYS,
    "between_reward_maxdist_sli": BETWEEN_REWARD_MAXDIST_SLI_REGRESSION_KEYS,
    "between_reward_return_leg_dist_sli": (
        BETWEEN_REWARD_RETURN_LEG_DIST_SLI_REGRESSION_KEYS
    ),
    "agarose_sli": AGAROSE_SLI_REGRESSION_KEYS,
    "commag_sli": COMMAG_SLI_REGRESSION_KEYS,
    "first_n_reward_diagnostics": FIRST_N_REWARD_DIAGNOSTICS_REGRESSION_KEYS,
    "return_prob_excursion_bin": RETURN_PROB_EXCURSION_BIN_REGRESSION_KEYS,
    "sli": SLI_REGRESSION_KEYS,
    "turnback_excursion_bin": TURNBACK_EXCURSION_BIN_REGRESSION_KEYS,
    "turnback_ratio": TURNBACK_RATIO_REGRESSION_KEYS,
}


def _json_bytes(payload) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_float_array(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype.kind == "f":
        out = out.astype("<f8", copy=True)
        out[np.isnan(out)] = np.nan
        return out
    if out.dtype.kind == "c":
        out = out.astype("<c16", copy=True)
        out.real[np.isnan(out.real)] = np.nan
        out.imag[np.isnan(out.imag)] = np.nan
        return out
    return out


def _canonical_array_payload(value) -> tuple[dict, bytes]:
    arr = np.asarray(value)
    meta = {"shape": list(arr.shape), "dtype": str(arr.dtype)}

    if arr.dtype.kind in "fciu?":
        canonical = _normalize_float_array(arr)
        meta.update(
            {
                "kind": "numeric",
                "finite_count": (
                    int(np.isfinite(canonical).sum())
                    if canonical.dtype.kind in "fc"
                    else int(canonical.size)
                ),
                "nan_count": (
                    int(np.isnan(canonical).sum())
                    if canonical.dtype.kind in "fc"
                    else 0
                ),
            }
        )
        return meta, np.ascontiguousarray(canonical).tobytes()

    strings = np.asarray(arr, dtype=str)
    meta.update({"kind": "string", "finite_count": int(strings.size), "nan_count": 0})
    return meta, _json_bytes(strings.reshape(-1).tolist())


def digest_array(value) -> dict:
    meta, canonical = _canonical_array_payload(value)
    return {**meta, "sha256": _hash_bytes(canonical)}


def digest_npz(path: str | Path, *, keys: Iterable[str] | None = None) -> dict:
    path = Path(path)
    requested = None if keys is None else tuple(keys)
    with np.load(path, allow_pickle=True) as data:
        available = set(data.files)
        selected = tuple(sorted(available if requested is None else requested))
        missing = [key for key in selected if key not in available]
        if missing:
            raise KeyError(f"{path} is missing digest keys: {missing}")
        arrays = {key: digest_array(data[key]) for key in selected}

    digest_payload = {
        "schema_version": SCHEMA_VERSION,
        "keys": selected,
        "arrays": arrays,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(path),
        "keys": list(selected),
        "arrays": arrays,
        "sha256": _hash_bytes(_json_bytes(digest_payload)),
    }


def _canonical_csv_cell(raw: str):
    text = "" if raw is None else str(raw)
    stripped = text.strip()
    if stripped:
        try:
            value = float(stripped)
        except Exception:
            return {"kind": "string", "value": text}
        if np.isnan(value):
            return {"kind": "nan"}
        if np.isinf(value):
            return {"kind": "inf", "sign": 1 if value > 0 else -1}
        return {"kind": "number", "value": float(value)}
    return {"kind": "string", "value": text}


def digest_csv(path: str | Path, *, keys: Iterable[str] | None = None) -> dict:
    path = Path(path)
    requested = None if keys is None else tuple(keys)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        available = tuple(reader.fieldnames or ())
        selected = tuple(available if requested is None else requested)
        missing = [key for key in selected if key not in available]
        if missing:
            raise KeyError(f"{path} is missing digest columns: {missing}")
        rows = [
            [_canonical_csv_cell(row.get(key, "")) for key in selected]
            for row in reader
        ]

    digest_payload = {
        "schema_version": SCHEMA_VERSION,
        "keys": selected,
        "rows": rows,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(path),
        "artifact_type": "csv",
        "keys": list(selected),
        "row_count": len(rows),
        "sha256": _hash_bytes(_json_bytes(digest_payload)),
    }


def digest_artifact(
    path: str | Path,
    *,
    keys: Iterable[str] | None = None,
    artifact_type: str | None = None,
) -> dict:
    kind = str(artifact_type or Path(path).suffix.lstrip(".") or "npz").lower()
    if kind == "csv":
        return digest_csv(path, keys=keys)
    if kind == "npz":
        return digest_npz(path, keys=keys)
    raise ValueError(f"Unsupported digest artifact type: {artifact_type!r}")


def write_manifest(
    entries: Iterable[dict], out_path: str | Path, *, project_root: str | Path = "."
) -> dict:
    root = Path(project_root)
    bundles = []
    for entry in entries:
        path = Path(entry["path"])
        keys = entry.get("keys")
        artifact_type = entry.get("type") or path.suffix.lstrip(".") or "npz"
        digest = digest_artifact(root / path, keys=keys, artifact_type=artifact_type)
        bundle = {
            "name": entry.get("name", path.stem),
            "path": str(path),
            "keys": digest["keys"],
            "sha256": digest["sha256"],
            **({"arrays": digest["arrays"]} if "arrays" in digest else {}),
            **({"row_count": digest["row_count"]} if "row_count" in digest else {}),
        }
        if entry.get("_include_type", True) or entry.get("type") is not None:
            bundle["type"] = str(artifact_type).lower()
        bundles.append(bundle)

    manifest = {"schema_version": SCHEMA_VERSION, "bundles": bundles}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def refresh_manifest(
    manifest_path: str | Path,
    *,
    out_path: str | Path | None = None,
    project_root: str | Path = ".",
    keys: Iterable[str] | None = None,
) -> dict:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    override_keys = None if keys is None else tuple(keys)
    entries = []
    for bundle in manifest.get("bundles", []):
        entry = {
            "name": bundle.get("name", Path(bundle["path"]).stem),
            "path": bundle["path"],
            "type": bundle.get("type"),
            "_include_type": "type" in bundle,
            "keys": override_keys if override_keys is not None else bundle.get("keys"),
        }
        entries.append(entry)
    return write_manifest(
        entries,
        manifest_path if out_path is None else out_path,
        project_root=project_root,
    )


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def check_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path = ".",
    require_all: bool = True,
) -> list[str]:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version={manifest.get('schema_version')!r}"
        )

    root = Path(project_root)
    failures = []
    for bundle in manifest.get("bundles", []):
        path = root / bundle["path"]
        if not path.exists():
            message = f"{bundle['name']}: missing bundle {bundle['path']}"
            if require_all:
                failures.append(message)
            continue

        observed = digest_artifact(
            path,
            keys=bundle.get("keys"),
            artifact_type=bundle.get("type"),
        )
        if observed["sha256"] != bundle.get("sha256"):
            failures.append(
                f"{bundle['name']}: digest mismatch "
                f"expected={bundle.get('sha256')} observed={observed['sha256']}"
            )
    return failures


def _parse_key_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    return keys or None


def resolve_regression_key_preset(name: str | None) -> tuple[str, ...] | None:
    if name is None:
        return None
    try:
        return REGRESSION_KEY_PRESETS[str(name)]
    except KeyError as exc:
        known = ", ".join(sorted(REGRESSION_KEY_PRESETS))
        raise KeyError(
            f"Unknown regression key preset {name!r}. Known presets: {known}"
        ) from exc


def _resolve_cli_keys(
    raw_keys: str | None,
    key_preset: str | None,
) -> list[str] | tuple[str, ...] | None:
    keys = _parse_key_list(raw_keys)
    preset_keys = resolve_regression_key_preset(key_preset)
    if keys is not None and preset_keys is not None:
        raise ValueError("Use either --keys or --key-preset, not both.")
    return keys if keys is not None else preset_keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create or check canonical NPZ/CSV digests."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("digest", help="Print a digest for one NPZ or CSV artifact.")
    one.add_argument("path")
    one.add_argument("--keys", help="Comma-separated keys to include.")
    one.add_argument(
        "--key-preset",
        choices=sorted(REGRESSION_KEY_PRESETS),
        help="Named regression key set to include.",
    )
    one.add_argument(
        "--type",
        choices=("npz", "csv"),
        default=None,
        help="Artifact type. Defaults to the path suffix.",
    )

    create = sub.add_parser(
        "write-manifest",
        help="Write a manifest for NPZ bundles or CSV artifacts.",
    )
    create.add_argument("--out", required=True)
    create.add_argument(
        "--root",
        default=".",
        help=(
            "Base directory for resolving bundle paths while computing digests; "
            "manifest paths are still stored as provided."
        ),
    )
    create.add_argument(
        "--bundle",
        action="append",
        required=True,
        help=(
            "Bundle to include in the manifest. Use NAME=PATH to give the entry an "
            "explicit manifest name, or just PATH to use the file stem as the name. "
            "May be repeated."
        ),
    )
    create.add_argument(
        "--keys",
        help=(
            "Comma-separated keys or CSV columns to include in each digest. "
            "Applies to all artifacts in this manifest."
        ),
    )
    create.add_argument(
        "--key-preset",
        choices=sorted(REGRESSION_KEY_PRESETS),
        help="Named regression key set to include in each digest.",
    )

    refresh = sub.add_parser(
        "refresh-manifest",
        help=(
            "Rewrite a manifest by reusing its existing bundle names, paths, "
            "artifact types, and keys unless --keys or --key-preset is provided."
        ),
    )
    refresh.add_argument("manifest")
    refresh.add_argument(
        "--out",
        default=None,
        help="Output path. Defaults to rewriting the input manifest in place.",
    )
    refresh.add_argument("--root", default=".")
    refresh.add_argument(
        "--keys",
        help=(
            "Comma-separated keys or CSV columns to include in each digest. "
            "Overrides keys stored in the existing manifest."
        ),
    )
    refresh.add_argument(
        "--key-preset",
        choices=sorted(REGRESSION_KEY_PRESETS),
        help="Named regression key set to include in each digest.",
    )

    check = sub.add_parser("check-manifest", help="Check bundles against a manifest.")
    check.add_argument("manifest")
    check.add_argument("--root", default=".")
    check.add_argument(
        "--allow-missing",
        action="store_true",
        help="Ignore missing bundle files instead of reporting them as failures.",
    )

    args = parser.parse_args(argv)

    if args.command == "digest":
        keys = _resolve_cli_keys(args.keys, args.key_preset)
        print(
            json.dumps(
                digest_artifact(
                    args.path,
                    keys=keys,
                    artifact_type=args.type,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "write-manifest":
        keys = _resolve_cli_keys(args.keys, args.key_preset)
        entries = []
        for raw in args.bundle:
            if "=" in raw:
                name, path = raw.split("=", 1)
            else:
                path = raw
                name = Path(raw).stem
            entries.append({"name": name, "path": path, "keys": keys})
        write_manifest(entries, args.out, project_root=args.root)
        return 0

    if args.command == "refresh-manifest":
        keys = _resolve_cli_keys(args.keys, args.key_preset)
        refresh_manifest(
            args.manifest,
            out_path=args.out,
            project_root=args.root,
            keys=keys,
        )
        return 0

    failures = check_manifest(
        args.manifest, project_root=args.root, require_all=not args.allow_missing
    )

    if failures:
        for failure in failures:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
