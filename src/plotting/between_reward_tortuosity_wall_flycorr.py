from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.plotting.overlay_training_metric_scalar_bars import ExportedTrainingScalarBars
from src.utils.util import meanConfInt


def load_wall_scatter_bundle(label: str, path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    meta_json = z["meta_json"].item() if "meta_json" in z.files else "{}"
    if isinstance(meta_json, (bytes, bytearray)):
        meta_json = meta_json.decode("utf-8")
    meta = json.loads(str(meta_json))
    out = {
        "group": label,
        "path": path,
        "wall_pct": np.asarray(z["wall_pct"], dtype=float),
        "tortuosity": np.asarray(z["tortuosity"], dtype=float),
        "meta": meta,
    }
    for key in ("unit_id", "video_id", "fly_id", "trx_idx", "role_idx", "fly_role"):
        if key in z.files:
            out[key] = np.asarray(z[key], dtype=object)
    return out


def transformed_tortuosity(y: np.ndarray, *, y_transform: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y_transform = str(y_transform or "log10").lower()
    if y_transform == "log10":
        return np.where(y > 0, np.log10(y), np.nan)
    return y


def bundle_fly_keys(bundle: dict) -> np.ndarray:
    unit = bundle.get("unit_id")
    if unit is not None:
        return np.asarray(unit, dtype=object).astype(str)

    wall_pct = np.asarray(bundle["wall_pct"], dtype=float)
    video = np.asarray(bundle.get("video_id", np.full(wall_pct.shape, "unknown")), dtype=object)
    trx = np.asarray(bundle.get("trx_idx", np.full(wall_pct.shape, -1)), dtype=object)
    role = np.asarray(bundle.get("role_idx", np.full(wall_pct.shape, -1)), dtype=object)
    return np.asarray(
        [f"{v}|trx_idx={t}|role_idx={r}" for v, t, r in zip(video, trx, role)],
        dtype=object,
    )


def _corr(x: np.ndarray, y: np.ndarray, *, method: str) -> tuple[float, float]:
    method = str(method or "pearson").lower()
    if method == "spearman":
        r, p = stats.spearmanr(x, y, nan_policy="omit")
    else:
        r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def compute_fly_correlations(
    bundle: dict,
    *,
    corr_method: str = "pearson",
    y_transform: str = "log10",
    min_segments_per_fly: int = 5,
    min_wall_range_pct: float = 0.0,
    xmax: float | None = None,
) -> list[dict]:
    x_all = np.asarray(bundle["wall_pct"], dtype=float)
    y_all = transformed_tortuosity(bundle["tortuosity"], y_transform=y_transform)
    keys = bundle_fly_keys(bundle)
    if keys.shape[0] != x_all.shape[0]:
        raise ValueError(f"fly key length does not match wall_pct in {bundle['path']}")
    keys_str = keys.astype(str)

    base_mask = np.isfinite(x_all) & np.isfinite(y_all)
    if xmax is not None:
        base_mask &= x_all <= float(xmax)

    rows: list[dict] = []
    for key in sorted(set(keys_str[base_mask])):
        mask = base_mask & (keys_str == key)
        x = x_all[mask]
        y = y_all[mask]
        n = int(x.size)
        if n < int(max(3, min_segments_per_fly)):
            continue
        wall_range = float(np.nanmax(x) - np.nanmin(x))
        if wall_range < float(max(0.0, min_wall_range_pct)):
            continue
        if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
            continue
        r, p = _corr(x, y, method=corr_method)
        if not np.isfinite(r):
            continue
        r_clip = float(np.clip(r, -0.999999, 0.999999))
        rows.append(
            {
                "group": str(bundle["group"]),
                "unit_id": str(key),
                "r": float(r),
                "fisher_z": float(np.arctanh(r_clip)),
                "p": float(p),
                "n_segments": n,
                "mean_wall_pct": float(np.nanmean(x)),
                "mean_y": float(np.nanmean(y)),
                "wall_range_pct": wall_range,
            }
        )
    return rows


def rows_to_exported_scalar(
    *,
    group: str,
    rows: list[dict],
    panel_label: str,
    plot_value: str,
    meta: dict,
) -> ExportedTrainingScalarBars:
    plot_value = str(plot_value or "fisher_z").lower()
    vals = np.asarray([float(r[plot_value]) for r in rows], dtype=float)
    ids = np.asarray([str(r["unit_id"]) for r in rows], dtype=object)
    finite = np.isfinite(vals)
    vals = vals[finite]
    ids = ids[finite]
    if vals.size:
        mean, lo, hi, n = meanConfInt(vals, conf=0.95, asDelta=False)
    else:
        mean, lo, hi, n = np.nan, np.nan, np.nan, 0

    return ExportedTrainingScalarBars(
        group=group,
        panel_labels=[panel_label],
        per_unit_values_panel=np.asarray([vals], dtype=object),
        per_unit_ids_panel=np.asarray([ids], dtype=object),
        mean=np.asarray([float(mean)], dtype=float),
        ci_lo=np.asarray([float(lo)], dtype=float),
        ci_hi=np.asarray([float(hi)], dtype=float),
        n_units_panel=np.asarray([int(n)], dtype=int),
        meta=meta,
    )


def build_flycorr_exports(
    bundles: list[dict],
    *,
    corr_method: str = "pearson",
    y_transform: str = "log10",
    plot_value: str = "fisher_z",
    min_segments_per_fly: int = 5,
    min_wall_range_pct: float = 0.0,
    xmax: float | None = None,
) -> tuple[list[ExportedTrainingScalarBars], list[dict]]:
    rows_by_group = [
        compute_fly_correlations(
            bundle,
            corr_method=corr_method,
            y_transform=y_transform,
            min_segments_per_fly=min_segments_per_fly,
            min_wall_range_pct=min_wall_range_pct,
            xmax=xmax,
        )
        for bundle in bundles
    ]

    labels = [
        str((bundle.get("meta") or {}).get("training_label") or "per-fly correlations")
        for bundle in bundles
    ]
    panel_label = labels[0] if len(set(labels)) == 1 else "per-fly correlations"
    y_name = (
        "log10(tortuosity)" if str(y_transform).lower() == "log10" else "tortuosity"
    )
    y_label = (
        f"Per-fly Fisher z({corr_method} r): wall % vs {y_name}"
        if str(plot_value).lower() == "fisher_z"
        else f"Per-fly {corr_method} r: wall % vs {y_name}"
    )

    xs = []
    for bundle, rows in zip(bundles, rows_by_group):
        meta0 = dict(bundle.get("meta") or {})
        meta = {
            "log_tag": "btw_rwd_tortuosity_wall_flycorr",
            "metric": "between_reward_tortuosity_wall_flycorr",
            "metric_palette_family": "between_reward_distance",
            "source_metric": meta0.get("metric"),
            "source_path": bundle.get("path"),
            "training_label": meta0.get("training_label"),
            "segment_scope": meta0.get("segment_scope"),
            "corr_method": str(corr_method),
            "y_transform": str(y_transform),
            "plot_value": str(plot_value),
            "min_segments_per_fly": int(min_segments_per_fly),
            "min_wall_range_pct": float(min_wall_range_pct),
            "xmax": None if xmax is None else float(xmax),
            "pool_trainings": False,
            "ci_conf": 0.95,
            "y_label": y_label,
        }
        xs.append(
            rows_to_exported_scalar(
                group=str(bundle["group"]),
                rows=rows,
                panel_label=panel_label,
                plot_value=plot_value,
                meta=meta,
            )
        )
    all_rows = [r for rows in rows_by_group for r in rows]
    return xs, all_rows


def write_flycorr_npz(xs: list[ExportedTrainingScalarBars], out_npz: str) -> None:
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        groups=np.asarray([x.group for x in xs], dtype=object),
        panel_labels=np.asarray(xs[0].panel_labels if xs else [], dtype=object),
        per_unit_values_by_group=np.asarray(
            [x.per_unit_values_panel[0] for x in xs], dtype=object
        ),
        per_unit_ids_by_group=np.asarray(
            [x.per_unit_ids_panel[0] for x in xs], dtype=object
        ),
        n_units=np.asarray([x.n_units_panel[0] for x in xs], dtype=int),
        mean=np.asarray([x.mean[0] for x in xs], dtype=float),
        ci_lo=np.asarray([x.ci_lo[0] for x in xs], dtype=float),
        ci_hi=np.asarray([x.ci_hi[0] for x in xs], dtype=float),
        meta_json=json.dumps([x.meta for x in xs], sort_keys=True),
    )


def write_flycorr_csv(rows: list[dict], out_csv: str) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "unit_id",
        "r",
        "fisher_z",
        "p",
        "n_segments",
        "mean_wall_pct",
        "mean_y",
        "wall_range_pct",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
