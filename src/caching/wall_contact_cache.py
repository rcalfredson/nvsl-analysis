from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_SUBDIR = ".nvsl_cache"


# ----------------------------- small helpers -----------------------------


def _stable_json(obj: Any) -> str:
    """Deterministic JSON for hashing/comparison."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_identity(path: str) -> Dict[str, Any]:
    """Fast-ish identity: size + mtime. Good enough for v1."""
    st = os.stat(path)
    return {"path": path, "size": int(st.st_size), "mtime": float(st.st_mtime)}


def _video_basename_no_ext(video_fn: str) -> str:
    base = os.path.basename(video_fn)
    return os.path.splitext(base)[0]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_replace(tmp_path: str, final_path: str) -> None:
    os.replace(tmp_path, final_path)


def _to_u8_str(s: str) -> np.ndarray:
    return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)


def _from_u8_str(a: np.ndarray) -> str:
    return bytes(a.astype(np.uint8)).decode("utf-8")


# --------------------------- regions serialization ---------------------------


def regions_to_array(regions: Optional[Iterable[slice]]) -> np.ndarray:
    """
    Convert slice regions -> Nx2 int32 array of (start, stop).
    None / empty => shape (0,2).
    """
    if not regions:
        return np.zeros((0, 2), dtype=np.int32)
    out: List[Tuple[int, int]] = []
    for r in regions:
        if r is None:
            continue
        # Treat None as 0/len? but in your code these should be ints already.
        start = 0 if r.start is None else int(r.start)
        stop = 0 if r.stop is None else int(r.stop)
        out.append((start, stop))
    return np.asarray(out, dtype=np.int32)


def array_to_regions(a: np.ndarray) -> List[slice]:
    """
    Convert Nx2 array (start, stop) -> list[slice].
    """
    if a is None:
        return []
    a = np.asarray(a, dtype=np.int32)
    if a.size == 0:
        return []
    return [slice(int(s), int(e)) for s, e in a]


# ------------------------------ payload IO ------------------------------


def extract_wall_payload_from_trj(trj: Any) -> Dict[str, Any]:
    """
    Extract a normalized (pickle-free) representation of the wall subtree
    from trj.boundary_event_stats.

    Output schema (per fly):
      {
        "wall": {
          <combo>: {
            <ref_pt>: {
              "boundary_contact": uint8 (T,),
              "regions": int32 (N,2),
              "start_idxs": int32 (N,)
            }
          }
        }
      }
    """
    bes = getattr(trj, "boundary_event_stats", None) or {}
    wall = bes.get("wall", {}) if isinstance(bes, dict) else {}

    out: Dict[str, Any] = {"wall": {}}

    for combo, combo_d in wall.items():
        if not isinstance(combo_d, dict):
            continue
        out["wall"][combo] = {}
        for ref_pt, stats in combo_d.items():
            if not isinstance(stats, dict):
                continue

            d: Dict[str, Any] = {}
            if "boundary_contact" in stats:
                bc = np.asarray(stats["boundary_contact"])
                # store as uint8 to keep npz compact and stable
                d["boundary_contact"] = bc.astype(np.uint8, copy=False)

            # key your downstream uses:
            if "boundary_contact_regions" in stats:
                d["regions"] = regions_to_array(stats["boundary_contact_regions"])

            if "contact_start_idxs" in stats:
                d["start_idxs"] = np.asarray(
                    stats["contact_start_idxs"], dtype=np.int32
                )

            # Optional: keep space for later fields without breaking schema
            # (These are NOT required for exclusion.)
            for k in ("closest_boundary_indices", "dist_to_boundary"):
                if k in stats:
                    d[k] = np.asarray(stats[k])

            if "bounds" in stats and isinstance(stats["bounds"], dict):
                if "x" in stats["bounds"]:
                    d["bounds_x"] = np.asarray(stats["bounds"]["x"])
                if "y" in stats["bounds"]:
                    d["bounds_y"] = np.asarray(stats["bounds"]["y"])

            out["wall"][combo][ref_pt] = d

    return out


def apply_wall_payload_to_trj(trj: Any, payload_for_fly: Dict[str, Any]) -> None:
    """
    Apply normalized wall payload back onto trj.boundary_event_stats in the
    shape consumers expect, including reconstructing slice regions.
    """
    if (
        not hasattr(trj, "boundary_event_stats")
        or getattr(trj, "boundary_event_stats") is None
    ):
        trj.boundary_event_stats = {}

    bes = trj.boundary_event_stats
    if not isinstance(bes, dict):
        # extremely defensive: overwrite if something odd
        bes = {}
        trj.boundary_event_stats = bes

    bes.setdefault("wall", {})
    wall_dst = bes["wall"]

    wall_src = (payload_for_fly or {}).get("wall", {})
    if not isinstance(wall_src, dict):
        return

    # overwrite wall subtree entirely from cache (safer than merge)
    bes["wall"] = {}
    wall_dst = bes["wall"]

    for combo, combo_d in wall_src.items():
        wall_dst[combo] = {}
        for ref_pt, d in combo_d.items():
            stats: Dict[str, Any] = {}

            if "boundary_contact" in d:
                stats["boundary_contact"] = np.asarray(d["boundary_contact"]).astype(
                    np.uint8, copy=False
                )

            if "regions" in d:
                stats["boundary_contact_regions"] = array_to_regions(d["regions"])

            if "start_idxs" in d:
                stats["contact_start_idxs"] = np.asarray(
                    d["start_idxs"], dtype=np.int32
                )

            for k in ("closest_boundary_indices", "dist_to_boundary"):
                if k in d:
                    stats[k] = np.asarray(d[k])

            if "bounds_x" in d or "bounds_y" in d:
                b: Dict[str, Any] = {}
                if "bounds_x" in d:
                    b["x"] = np.asarray(d["bounds_x"])
                if "bounds_y" in d:
                    b["y"] = np.asarray(d["bounds_y"])
                stats["bounds"] = b

            wall_dst[combo][ref_pt] = stats


# ------------------------------ manifest + paths ------------------------------


@dataclass(frozen=True)
class WallCacheSpec:
    """
    Everything needed to locate and validate a wall cache file for a given run.
    """

    npz_path: str
    param_hash: str
    expected_manifest: Dict[str, Any]


def build_wall_cache_spec(
    *,
    video_fn: str,
    cache_dir: Optional[str],
    ct_name: str,
    wall_orientation: str,
    thresholds_wall: List[float],
    offset_wall: float,
    boundary_contact_binary_path: str,
    boundary_contact_binary_sha256: str,
    input_identities: Dict[str, Any],
) -> WallCacheSpec:
    """
    Create an expected manifest and an output path for the cache file.
    """
    video_base = _video_basename_no_ext(video_fn)

    wall_params = {
        "ct": ct_name,
        "wall_orientation": wall_orientation,
        "thresholds_wall": [float(thresholds_wall[0]), float(thresholds_wall[1])],
        "offset_wall": float(offset_wall),
    }
    param_hash = hashlib.sha1(_stable_json(wall_params).encode("utf-8")).hexdigest()[:8]

    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(video_fn), DEFAULT_CACHE_SUBDIR)
    _ensure_dir(cache_dir)

    npz_name = f"{video_base}__wall__p{param_hash}__v{CACHE_SCHEMA_VERSION}.npz"
    npz_path = os.path.join(cache_dir, npz_name)

    expected_manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "video_basename": video_base,
        "code": {
            "boundary_contact_binary_path": boundary_contact_binary_path,
            "boundary_contact_binary_sha256": boundary_contact_binary_sha256,
        },
        "inputs": input_identities,
        "wall_params": wall_params,
    }

    return WallCacheSpec(
        npz_path=npz_path, param_hash=param_hash, expected_manifest=expected_manifest
    )


def manifest_matches(
    expected: Dict[str, Any], found: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Strict match on schema, code sha, inputs identities, and wall params.
    Returns (ok, reason_if_not).
    """
    if not isinstance(found, dict):
        return False, "manifest not a dict"

    if found.get("schema_version") != expected.get("schema_version"):
        return False, "schema_version mismatch"

    # code identity
    exp_code = expected.get("code", {})
    got_code = found.get("code", {})
    if got_code.get("boundary_contact_binary_sha256") != exp_code.get(
        "boundary_contact_binary_sha256"
    ):
        return False, "boundary_contact binary hash mismatch"

    # wall params
    if found.get("wall_params") != expected.get("wall_params"):
        return False, "wall_params mismatch"

    # inputs
    if found.get("inputs") != expected.get("inputs"):
        return False, "inputs identity mismatch"

    return True, ""


# ------------------------------ save/load .npz ------------------------------


def save_wall_cache_npz(
    npz_path: str,
    manifest: Dict[str, Any],
    *,
    wall_orientations: Optional[List[str]],
    per_fly_payload: Dict[int, Dict[str, Any]],
) -> None:
    """
    Save cache as a single .npz with:
      - manifest_json (uint8 bytes)
      - wall_orientations (string array) [optional]
      - per-fly arrays under systematic keys
    """
    manifest_full = dict(manifest)
    manifest_full["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_full["payload"] = {
        "flies": sorted([int(f) for f in per_fly_payload.keys()]),
    }

    arrays: Dict[str, np.ndarray] = {}
    arrays["manifest_json"] = _to_u8_str(_stable_json(manifest_full))

    if wall_orientations is not None:
        arrays["wall_orientations"] = np.asarray(list(wall_orientations), dtype="U")

    for fly, payload in per_fly_payload.items():
        fly = int(fly)
        wall = payload.get("wall", {})
        for combo, combo_d in wall.items():
            for ref_pt, d in combo_d.items():
                prefix = f"f{fly}__{combo}__{ref_pt}"
                if "boundary_contact" in d:
                    arrays[f"{prefix}__boundary_contact"] = np.asarray(
                        d["boundary_contact"], dtype=np.uint8
                    )
                if "regions" in d:
                    arrays[f"{prefix}__regions"] = np.asarray(
                        d["regions"], dtype=np.int32
                    )
                if "start_idxs" in d:
                    arrays[f"{prefix}__start_idxs"] = np.asarray(
                        d["start_idxs"], dtype=np.int32
                    )

                # optional extras
                for k in (
                    "closest_boundary_indices",
                    "dist_to_boundary",
                    "bounds_x",
                    "bounds_y",
                ):
                    if k in d:
                        arrays[f"{prefix}__{k}"] = np.asarray(d[k])

    tmp_path = npz_path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp_path, **arrays)
    _atomic_replace(tmp_path, npz_path)


def load_wall_cache_npz(
    npz_path: str,
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]], Optional[List[str]]]:
    """
    Load cache from .npz. Returns (manifest, per_fly_payload, wall_orientations)
    """
    with np.load(npz_path, allow_pickle=False) as z:
        manifest = json.loads(_from_u8_str(z["manifest_json"]))

        wall_orientations: Optional[List[str]] = None
        if "wall_orientations" in z:
            wall_orientations = [str(x) for x in z["wall_orientations"]]

        per_fly: Dict[int, Dict[str, Any]] = {}

        # Discover fly/combo/ref_pt by parsing keys
        for key in z.files:
            if not key.startswith("f") or "__" not in key:
                continue
            if key == "manifest_json" or key == "wall_orientations":
                continue

            # key format: f{fly}__{combo}__{ref_pt}__field
            parts = key.split("__")
            if len(parts) < 4:
                continue
            fly_s = parts[0]  # f0
            try:
                fly = int(fly_s[1:])
            except Exception:
                continue
            combo = parts[1]
            ref_pt = parts[2]
            field = "__".join(
                parts[3:]
            )  # supports fields with __ though we don't use that now

            per_fly.setdefault(fly, {"wall": {}})
            per_fly[fly]["wall"].setdefault(combo, {})
            per_fly[fly]["wall"][combo].setdefault(ref_pt, {})
            per_fly[fly]["wall"][combo][ref_pt][field] = z[key]

        return manifest, per_fly, wall_orientations
