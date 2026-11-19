"""Native, opt-in trajectory selection; no dependency on offline-tracking."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def file_identity(path):
    path = Path(path).resolve(strict=True)
    st = path.stat()
    return dict(path=str(path), size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_tracking(video, original_trx, *, variant="original", manifest=None, fallback=False):
    """Resolve by full canonical video path, never camera/day or basename."""
    video = str(Path(video).resolve(strict=True))
    selected, resolved = original_trx, "original"
    if variant != "original":
        if not manifest:
            raise ValueError(f"Tracking variant {variant!r} requires --tracking-manifest")
        manifest_path = Path(manifest).resolve()
        try:
            payload = json.loads(manifest_path.read_text())
            if payload.get("schema_version") != 1 or not isinstance(payload.get("recordings"), dict):
                raise ValueError("expected schema_version 1 and recordings object")
            entries = {}
            for key, variants in payload["recordings"].items():
                if not Path(key).is_absolute() or not isinstance(variants, dict):
                    raise ValueError("recordings keys must be absolute video paths, values variant objects")
                canonical = str(Path(key).resolve())
                if canonical in entries:
                    raise ValueError(f"duplicate recording identity: {key}")
                entries[canonical] = variants
            entry = entries.get(video, {})
            if variant in entry:
                selected = entry[variant]
                if not isinstance(selected, str) or not selected:
                    raise ValueError(f"invalid companion path for {video}, variant {variant!r}")
                selected = manifest_path.parent / selected
                if Path(selected).suffix != ".trx":
                    raise ValueError(f"companion must have .trx extension: {selected}")
                resolved = variant
            elif not fallback:
                raise ValueError(f"no entry for {video}, variant {variant!r}; add it or use --tracking-fallback")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"Tracking manifest {manifest_path}: {exc}") from exc
    try:
        identity = file_identity(selected)
    except OSError as exc:
        raise ValueError(f"Selected tracking for {video}, variant {variant!r}: {selected}: {exc}; repair the manifest/file") from exc
    return dict(video=video, requested_variant=variant, resolved_variant=resolved,
                trajectory=identity)


def tracking_cache_key(selection, data_path, original_trx):
    """Include acquisition inputs as well as selected trajectory and variant."""
    return json.dumps(
        dict(selection=selection, video=file_identity(selection["video"]),
             data=file_identity(data_path), original=file_identity(original_trx)),
        sort_keys=True,
    )


def validate_companion(companion, original, *, video, original_trx, data_path):
    """Check provenance before admitting an offline payload to the analysis cache."""
    try:
        sources = companion["tracking_metadata"]["source_recording"]
        for name, path in (("video", video), ("trx", original_trx), ("data", data_path)):
            expected, actual = sources[name], file_identity(path)
            for key in ("path", "size_bytes", "mtime_ns"):
                wanted = str(Path(expected[key]).resolve()) if key == "path" else expected[key]
                if wanted != actual[key]:
                    raise ValueError(f"source {name} {key} mismatch: expected {wanted!r}, found {actual[key]!r}")
            if "sha256" in expected and sha256(path) != expected["sha256"]:
                raise ValueError(f"source {name} sha256 mismatch")
        shape = np.asarray(original["x"]).shape
        if len(shape) != 2:
            raise ValueError(f"original fly/frame shape is invalid: {shape}")
        for field in ("x", "y", "w", "h", "theta"):
            if (np.asarray(companion[field]).shape != shape
                    or np.asarray(original[field]).shape != shape):
                raise ValueError(f"{field} fly/frame shape mismatch (expected {shape})")
        if (np.asarray(companion["ts"]).shape != (shape[1],)
                or not np.array_equal(companion["ts"], original["ts"])):
            raise ValueError("timestamps mismatch; companion must retain exact original timestamps")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid companion for {video}: {exc}; regenerate it from this original recording") from exc


def load_tracking_data(video, data_path, original_trx, opts, cache, unpickle):
    selection = select_tracking(
        video, original_trx, variant=getattr(opts, "tracking_variant", "original"),
        manifest=getattr(opts, "tracking_manifest", None),
        fallback=getattr(opts, "tracking_fallback", False))
    key = tracking_cache_key(selection, data_path, original_trx)
    if key not in cache:
        data, original = unpickle(data_path), unpickle(original_trx)
        selected = original
        if selection["resolved_variant"] != "original":
            try:
                selected = unpickle(selection["trajectory"]["path"])
                validate_companion(selected, original, video=video,
                                   original_trx=original_trx, data_path=data_path)
            except Exception as exc:
                raise ValueError(f"Cannot use companion {selection['trajectory']['path']}: {exc}") from exc
        identity = dict(selection["trajectory"], sha256=sha256(selection["trajectory"]["path"]))
        cache[key] = data, selected, identity
    data, selected, identity = cache[key]
    selection["trajectory"] = identity
    return data, selected, selection
