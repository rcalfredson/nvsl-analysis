from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TORTUOSITY_SCALAR_EXPORT_KEYS = (
    "panel_labels",
    "per_unit_values_panel",
    "per_unit_ids_panel",
    "mean",
    "ci_lo",
    "ci_hi",
    "n_units_panel",
    "meta_json",
)

TORTUOSITY_WALL_SCATTER_KEYS = (
    "wall_frac",
    "wall_pct",
    "tortuosity",
    "s",
    "e",
    "metric_s",
    "metric_e",
    "n_wall_frames",
    "n_metric_frames",
    "b_idx",
    "video_id",
    "unit_id",
    "fly_id",
    "trx_idx",
    "role_idx",
    "fly_role",
    "meta_json",
)


def _where(path: str | Path | None) -> str:
    return str(path) if path is not None else "<between-reward tortuosity artifact>"


def _as_meta(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if raw is None:
        return {}
    return json.loads(str(raw))


def _require_keys(mapping, keys: tuple[str, ...], *, where: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{where} is missing tortuosity export keys: {missing}")


def _validate_nonnegative_no_inf(arr, key: str, *, where: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if np.any(np.isinf(out)):
        raise ValueError(f"{where} has infinite values in {key}")
    finite = np.isfinite(out)
    if np.any(out[finite] < 0.0):
        raise ValueError(f"{where} has negative values in {key}")
    return out


def _validate_count_array(arr, key: str, *, where: str) -> np.ndarray:
    out = np.asarray(arr)
    vals = out.astype(float)
    if np.any(~np.isfinite(vals)):
        raise ValueError(f"{where} has non-finite values in {key}")
    if np.any(vals < 0):
        raise ValueError(f"{where} has negative values in {key}")
    if np.any(vals != np.floor(vals)):
        raise ValueError(f"{where} has non-integer values in {key}")
    return vals.astype(int, copy=False)


def _object_panel_values(
    arr, *, where: str, key: str, n_panels: int
) -> list[np.ndarray]:
    panels = np.asarray(arr, dtype=object)
    if panels.ndim == 0:
        raise ValueError(f"{where} has scalar {key}")
    if panels.shape[0] != n_panels:
        raise ValueError(
            f"{where} has {key}.shape={panels.shape}, expected first dimension "
            f"{n_panels}"
        )
    return [np.asarray(panels[idx]).reshape(-1) for idx in range(n_panels)]


def validate_between_reward_tortuosity_scalar_export(
    artifact: dict,
    *,
    path: str | Path | None = None,
) -> None:
    """Validate Panel-26-style per-fly mean tortuosity scalar NPZ contents."""
    where = _where(path)
    _require_keys(artifact, TORTUOSITY_SCALAR_EXPORT_KEYS, where=where)

    labels = np.asarray(artifact["panel_labels"], dtype=object).reshape(-1)
    n_panels = int(labels.size)
    values_by_panel = _object_panel_values(
        artifact["per_unit_values_panel"],
        where=where,
        key="per_unit_values_panel",
        n_panels=n_panels,
    )
    ids_by_panel = _object_panel_values(
        artifact["per_unit_ids_panel"],
        where=where,
        key="per_unit_ids_panel",
        n_panels=n_panels,
    )
    if len(values_by_panel) != n_panels or len(ids_by_panel) != n_panels:
        raise ValueError(f"{where} has panel/value/id length mismatch")

    mean = _validate_nonnegative_no_inf(artifact["mean"], "mean", where=where)
    ci_lo = _validate_nonnegative_no_inf(artifact["ci_lo"], "ci_lo", where=where)
    ci_hi = _validate_nonnegative_no_inf(artifact["ci_hi"], "ci_hi", where=where)
    n_units = _validate_count_array(
        artifact["n_units_panel"], "n_units_panel", where=where
    )
    for key, arr in (
        ("mean", mean),
        ("ci_lo", ci_lo),
        ("ci_hi", ci_hi),
        ("n_units_panel", n_units),
    ):
        if np.asarray(arr).shape != (n_panels,):
            raise ValueError(
                f"{where} has {key}.shape={np.asarray(arr).shape}, "
                f"expected {(n_panels,)}"
            )

    for idx, (vals_raw, ids_raw) in enumerate(zip(values_by_panel, ids_by_panel)):
        vals = _validate_nonnegative_no_inf(
            vals_raw, f"per_unit_values_panel[{idx}]", where=where
        )
        ids = np.asarray(ids_raw, dtype=object).reshape(-1)
        if ids.shape[0] != vals.shape[0]:
            raise ValueError(
                f"{where} panel {idx} has {ids.shape[0]} ids but "
                f"{vals.shape[0]} tortuosity values"
            )
        finite_n = int(np.isfinite(vals).sum())
        if int(n_units[idx]) != finite_n:
            raise ValueError(
                f"{where} panel {idx} has n_units_panel={int(n_units[idx])} "
                f"but {finite_n} finite tortuosity values"
            )
        if finite_n == 0 and np.isfinite(mean[idx]):
            raise ValueError(f"{where} has finite mean for empty panel {idx}")
        if finite_n > 0:
            expected = float(np.nanmean(vals))
            if not np.isfinite(mean[idx]) or abs(float(mean[idx]) - expected) > 1e-10:
                raise ValueError(f"{where} has inconsistent mean for panel {idx}")

    meta = _as_meta(artifact["meta_json"])
    if meta and meta.get("metric") not in {None, "between_reward_tortuosity_mean"}:
        raise ValueError(f"{where} has unexpected tortuosity metric={meta.get('metric')!r}")
    if meta.get("metric_mode", "path_over_max_radius") == "path_over_max_radius":
        min_radius = float(meta.get("min_radius_mm", 0.0) or 0.0)
        if not np.isfinite(min_radius) or min_radius < 0.0:
            raise ValueError(f"{where} has invalid min_radius_mm={min_radius}")


def validate_between_reward_tortuosity_graphpad_columns(
    headers,
    columns,
    *,
    path: str | Path | None = None,
) -> None:
    """Validate wide GraphPad columns produced from tortuosity scalar exports."""
    where = _where(path)
    header_list = [str(h) for h in headers]
    column_list = [np.asarray(col, dtype=float).reshape(-1) for col in columns]
    if len(header_list) != len(column_list):
        raise ValueError(f"{where} has GraphPad header/column count mismatch")
    if not header_list:
        raise ValueError(f"{where} has no GraphPad tortuosity columns")
    if any(not h.strip() for h in header_list):
        raise ValueError(f"{where} has blank GraphPad tortuosity column header")
    for header, col in zip(header_list, column_list):
        _validate_nonnegative_no_inf(col, f"GraphPad column {header!r}", where=where)


def validate_between_reward_tortuosity_distance_box_result(
    result,
    *,
    path: str | Path | None = None,
) -> None:
    """Validate binned tortuosity-by-max-radius result objects or loaded exports."""
    where = _where(path)
    edges = np.asarray(result.x_edges_mm, dtype=float).reshape(-1)
    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{where} has invalid x_edges_mm")
    n_bins = int(edges.size - 1)

    for key in ("q1", "median", "q3", "whisker_low", "whisker_high"):
        arr = _validate_nonnegative_no_inf(getattr(result, key), key, where=where)
        if arr.shape != (n_bins,):
            raise ValueError(f"{where} has {key}.shape={arr.shape}, expected {(n_bins,)}")

    n_segments = _validate_count_array(result.n_segments, "n_segments", where=where)
    n_units = _validate_count_array(result.n_units, "n_units", where=where)
    if n_segments.shape != (n_bins,) or n_units.shape != (n_bins,):
        raise ValueError(f"{where} has count arrays with incorrect bin count")

    values_by_bin = np.asarray(result.values_by_bin, dtype=object).reshape(-1)
    if values_by_bin.shape[0] != n_bins:
        raise ValueError(f"{where} has values_by_bin length {values_by_bin.shape[0]}")
    for idx, raw in enumerate(values_by_bin):
        vals = _validate_nonnegative_no_inf(raw, f"values_by_bin[{idx}]", where=where)
        if np.asarray(vals).ndim != 1:
            vals = np.asarray(vals, dtype=float).reshape(-1)
        if int(n_segments[idx]) == 0 and np.any(np.isfinite(vals)):
            raise ValueError(f"{where} has finite tortuosity values in empty bin {idx}")


def validate_between_reward_tortuosity_wall_scatter_export(
    artifact: dict,
    *,
    path: str | Path | None = None,
) -> None:
    """Validate segment-level tortuosity-vs-wall-contact NPZ contents."""
    where = _where(path)
    _require_keys(artifact, TORTUOSITY_WALL_SCATTER_KEYS, where=where)

    tort = _validate_nonnegative_no_inf(artifact["tortuosity"], "tortuosity", where=where)
    n_rows = int(tort.size)
    wall_frac = _validate_nonnegative_no_inf(artifact["wall_frac"], "wall_frac", where=where)
    wall_pct = _validate_nonnegative_no_inf(artifact["wall_pct"], "wall_pct", where=where)
    for key in TORTUOSITY_WALL_SCATTER_KEYS:
        if key == "meta_json":
            continue
        if np.asarray(artifact[key]).reshape(-1).shape[0] != n_rows:
            raise ValueError(f"{where} has row-count mismatch for {key}")
    if np.any(wall_frac > 1.0):
        raise ValueError(f"{where} has wall_frac values above 1")
    if np.any(np.abs(wall_pct - 100.0 * wall_frac) > 1e-10):
        raise ValueError(f"{where} has inconsistent wall_pct values")

    n_wall = _validate_count_array(artifact["n_wall_frames"], "n_wall_frames", where=where)
    n_metric = _validate_count_array(
        artifact["n_metric_frames"], "n_metric_frames", where=where
    )
    if np.any(n_metric <= 0):
        raise ValueError(f"{where} has nonpositive n_metric_frames")
    if np.any(n_wall > n_metric):
        raise ValueError(f"{where} has n_wall_frames greater than n_metric_frames")
