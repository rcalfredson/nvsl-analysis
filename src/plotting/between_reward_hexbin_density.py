# src/plotting/between_reward_hexbin_density
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.between_reward_segment_binning import (
    sync_bucket_window,
    wall_contact_mask,
    build_nonwalk_mask,
    video_base,
    fly_role_name,
)
from src.plotting.between_reward_segment_metrics import dist_traveled_mm_masked

# ----------------------------
# Explicit axis limits (linear units)
# ----------------------------
DMAX_MM_MIN = 0.0
DMAX_MM_MAX = 33.0

LTOTAL_MM_MIN = 0.0
LTOTAL_MM_MAX = 1600.0

LRETURN_MM_MIN = 0.0
LRETURN_MM_MAX = 800.0


@dataclass
class HexBinGridSpec:
    gridside: int = 45
    # extent is always interpreted in *plot-space* (i.e., after transforms):
    # (xmin, xmax, ymin, ymax)
    extent: Optional[tuple[float, float, float, float]] = None
    mincnt: int = 1


@dataclass
class BetweenRewardHexbinConfig:
    training_index: int
    skip_first_sync_buckets: int = 0
    use_reward_exclusion_mask: bool = True

    # segment filters
    exclude_wall_contact: bool = False
    exclude_nonwalking_frames: bool = False
    min_walk_frames: int = 2
    per_segment_min_meddist_mm: float = 0.0

    # which x-axis metric
    x_mode: str = "Ltotal"  # "Ltotal" or "Lreturn"

    # transforms
    log1p_x: bool = True  # default True for Ltotal/Lreturn comparisons
    log1p_y: bool = False  # Dmax stays linear by default

    # hexbin settings
    hex: HexBinGridSpec = field(default_factory=HexBinGridSpec)


@dataclass(frozen=True)
class BetweenRewardHexbinResult:
    """
    Canonical hexbin-density result (in plot-space).
    density is a 1D vector aligned to bin_centers (same order).
    """

    training_index: int
    x_mode: str
    log1p_x: bool
    log1p_y: bool

    gridside: int
    extent: tuple[float, float, float, float]  # plot-space extent
    mincnt: int

    bin_centers: np.ndarray  # (M, 2) offsets from matplotlib hexbin
    mean_density: np.ndarray  # (M,) mean per-fly normalized density

    n_units: int
    meta: dict
    unit_info: list[dict]

    # optional (nice to keep for later overlays/stats)
    per_unit_density: np.ndarray | None = None  # (N, M)

    def validate(self) -> None:
        bc = np.asarray(self.bin_centers, float)
        md = np.asarray(self.mean_density, float)
        if bc.ndim != 2 or bc.shape[1] != 2:
            raise ValueError("bin_centers must be shape (M, 2)")
        if md.ndim != 1 or md.size != bc.shape[0]:
            raise ValueError("mean_density must be shape (M,) matching bin_centers")
        if self.per_unit_density is not None:
            pud = np.asarray(self.per_unit_density, float)
            if pud.ndim != 2 or pud.shape[1] != bc.shape[0]:
                raise ValueError("per_unit_density must be shape (N, M)")
        ex = tuple(self.extent)
        if len(ex) != 4 or not np.all(np.isfinite(ex)):
            raise ValueError("extent must be 4 finite floats")

    def save_npz(self, path: str) -> None:
        self.validate()
        np.savez_compressed(
            path,
            training_index=int(self.training_index),
            x_mode=str(self.x_mode),
            log1p_x=bool(self.log1p_x),
            log1p_y=bool(self.log1p_y),
            gridside=int(self.gridside),
            extent=np.asarray(self.extent, float),
            mincnt=int(self.mincnt),
            bin_centers=np.asarray(self.bin_centers, float),
            mean_density=np.asarray(self.mean_density, float),
            n_units=int(self.n_units),
            meta=np.asarray([self.meta], dtype=object),
            unit_info=np.asarray([self.unit_info], dtype=object),
            per_unit_density=(
                np.asarray(self.per_unit_density, float)
                if self.per_unit_density is not None
                else np.asarray([], float)
            ),
        )

    @staticmethod
    def load_npz(path: str) -> "BetweenRewardHexbinResult":
        z = np.load(path, allow_pickle=True)
        meta = {}
        unit_info: list[dict] = []
        try:
            meta_obj = z["meta"]
            meta = meta_obj.item() if hasattr(meta_obj, "item") else {}
        except Exception:
            meta = {}
        try:
            ui_obj = z["unit_info"]
            unit_info = ui_obj.item() if hasattr(ui_obj, "item") else []
        except Exception:
            unit_info = []

        per_unit_density = None
        if "per_unit_density" in z:
            pud = np.asarray(z["per_unit_density"], float)
            if pud.size > 0:
                per_unit_density = pud

        res = BetweenRewardHexbinResult(
            training_index=int(z["training_index"]),
            x_mode=str(z["x_mode"]),
            log1p_x=bool(z["log1p_x"]),
            log1p_y=bool(z["log1p_y"]),
            gridside=int(z["gridside"]),
            extent=tuple(np.asarray(z["extent"], float).tolist()),
            mincnt=int(z["mincnt"]),
            bin_centers=np.asarray(z["bin_centers"], float),
            mean_density=np.asarray(z["mean_density"], float),
            n_units=int(z["n_units"]),
            meta=dict(meta) if isinstance(meta, dict) else {},
            unit_info=list(unit_info) if isinstance(unit_info, list) else [],
            per_unit_density=per_unit_density,
        )
        res.validate()
        return res


def _hexbin_counts_for_points(
    pts_xy: np.ndarray,
    *,
    gridside: int,
    extent: tuple[float, float, float, float],
    mincnt: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute matplotlib hexbin bins for pts_xy (Nx2), returning:
      (bin_centers (M,2), counts (M,))
    Only non-empty bins are returned by matplotlib.
    """
    pts = np.asarray(pts_xy, float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), float), np.zeros((0,), float)

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    try:
        hb = ax.hexbin(
            pts[:, 0],
            pts[:, 1],
            gridsize=int(gridside),
            extent=extent,
            mincnt=int(mincnt),
            linewidths=0.0,
        )
        centers = np.asarray(hb.get_offsets(), float)  # (M,2)
        counts = np.asarray(hb.get_array(), float)  # (M,)
    finally:
        plt.close(fig)
    return centers, counts


def compute_mean_hexbin_density_union(
    per_fly_points: list[np.ndarray],
    *,
    cfg: "BetweenRewardHexbinConfig",
    meta: dict,
    unit_info: list[dict],
    keep_per_unit: bool = True,
    normalize_per_fly: bool = True,
    cbar_max: float | None = None,
    log_tag: str = "btw_rwd_hex",
) -> BetweenRewardHexbinResult:
    """
    Union-bins version:
      - canonical bin set is the union of all bins touched by any fly.
      - preserves mass for flies that visit bins absent in earlier flies.
    """
    extent = _resolved_extent(cfg)
    gridside = int(cfg.hex.gridside)
    mincnt = int(cfg.hex.mincnt)

    # First pass: compute each fly's (centers, counts) and build union bin index
    fly_hb: list[tuple[np.ndarray, np.ndarray]] = []
    union_map: dict[tuple[float, float], int] = {}
    union_centers_list: list[tuple[float, float]] = []

    def _get_union_index(cx: float, cy: float) -> int:
        key = (round(float(cx), 10), round(float(cy), 10))
        j = union_map.get(key)
        if j is None:
            j = len(union_centers_list)
            union_map[key] = j
            union_centers_list.append((float(key[0]), float(key[1])))
        return j

    n_used = 0
    for pts in per_fly_points:
        pts = np.asarray(pts, float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
            fly_hb.append((np.zeros((0, 2), float), np.zeros((0,), float)))
            continue

        centers, counts = _hexbin_counts_for_points(
            pts, gridside=gridside, extent=extent, mincnt=mincnt
        )
        centers = np.asarray(centers, float)
        counts = np.asarray(counts, float)
        fly_hb.append((centers, counts))

        if centers.shape[0] > 0:
            n_used += 1
            for c in centers:
                _get_union_index(c[0], c[1])

    M = len(union_centers_list)
    if M == 0 or n_used == 0:
        return BetweenRewardHexbinResult(
            training_index=int(cfg.training_index),
            x_mode=str(cfg.x_mode),
            log1p_x=bool(cfg.log1p_x),
            log1p_y=bool(cfg.log1p_y),
            gridside=gridside,
            extent=extent,
            mincnt=mincnt,
            bin_centers=np.zeros((0, 2), float),
            mean_density=np.zeros((0,), float),
            n_units=int(n_used),
            meta=dict(meta),
            unit_info=list(unit_info),
            per_unit_density=None,
        )

    # Canonical centers array (union)
    bin_centers = np.asarray(union_centers_list, float)  # (M,2)

    # Second pass: build per-fly rows in union space
    rows: list[np.ndarray] = []
    for centers, counts in fly_hb:
        if centers.shape[0] == 0:
            continue

        row = np.zeros((M,), float)
        for c, w in zip(centers, counts):
            j = _get_union_index(c[0], c[1])
            row[j] = float(w)

        s = float(np.sum(row))
        if normalize_per_fly and s > 0:
            row = row / s
        rows.append(row)

    if not rows:
        mean_density = np.zeros((M,), float)
        per_unit_density = None
    else:
        mat = np.vstack(rows)
        mean_density = mat.mean(axis=0)
        per_unit_density = mat if keep_per_unit else None

    cbar_max_val = None
    if cbar_max is not None:
        try:
            v = float(cbar_max)
            if np.isfinite(v) and v > 0:
                cbar_max_val = v
        except Exception:
            cbar_max_val = None

    res_meta = dict(meta)
    res_meta.update(
        dict(
            log_tag=str(log_tag),
            extent=tuple(float(v) for v in extent),
            gridside=gridside,
            mincnt=mincnt,
            normalize_per_fly=bool(normalize_per_fly),
            n_units_used=int(len(rows)),
            n_bins=int(M),
            bin_policy="union",
            cbar_max=cbar_max_val,
        )
    )

    return BetweenRewardHexbinResult(
        training_index=int(cfg.training_index),
        x_mode=str(cfg.x_mode),
        log1p_x=bool(cfg.log1p_x),
        log1p_y=bool(cfg.log1p_y),
        gridside=gridside,
        extent=extent,
        mincnt=mincnt,
        bin_centers=bin_centers,
        mean_density=np.asarray(mean_density, float),
        n_units=int(len(rows)),
        meta=res_meta,
        unit_info=list(unit_info),
        per_unit_density=per_unit_density,
    )


def plot_mean_hexbin_density(
    res: BetweenRewardHexbinResult,
    *,
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_max: float | None = None,
    customizer=None,
    image_format: str = "png",
):
    """
    Plot mean density using true matplotlib hexbin polygons.
    """
    res.validate()

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))

    if res.bin_centers.shape[0] == 0 or res.mean_density.size == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        # Feed bin centers back into hexbin; each center falls in its own bin.
        hb = ax.hexbin(
            res.bin_centers[:, 0],
            res.bin_centers[:, 1],
            C=res.mean_density,
            reduce_C_function=np.mean,
            gridsize=int(res.gridside),
            extent=res.extent,
            mincnt=1,  # ensure every provided center shows
            linewidths=0.0,
        )

        if cbar_max is not None:
            vmax = float(cbar_max)
            if np.isfinite(vmax) and vmax > 0:
                hb.set_clim(0.0, vmax)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("mean per-fly density")

        ax.set_xlim(float(res.extent[0]), float(res.extent[1]))
        ax.set_ylim(float(res.extent[2]), float(res.extent[3]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # annotate n
        ax.text(
            0.98,
            0.02,
            f"n={res.n_units}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=(
                getattr(customizer, "in_plot_font_size", 10) if customizer else 10
            ),
            color=".2",
        )

    fig.tight_layout()
    root, ext = os.path.splitext(out_path)
    if not ext:
        out_path = root + "." + str(image_format).lstrip(".")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_between_reward_hexbin_density(
    vas: list["VideoAnalysis"],
    *,
    cfg: BetweenRewardHexbinConfig,
    opts,
    out_npz: str | None = None,
    out_png: str | None = None,
    customizer=None,
    keep_per_unit: bool = True,
    normalize_per_fly: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cbar_max: float | None = None,
    image_format: str = "png",
    log_tag: str = "btw_rwd_hex",
) -> BetweenRewardHexbinResult:
    per_fly_points, unit_info, meta = collect_per_fly_segment_points(
        vas, cfg=cfg, opts=opts, log_tag=log_tag
    )
    res = compute_mean_hexbin_density_union(
        per_fly_points,
        cfg=cfg,
        meta=meta,
        unit_info=unit_info,
        keep_per_unit=keep_per_unit,
        normalize_per_fly=normalize_per_fly,
        cbar_max=cbar_max,
        log_tag=log_tag,
    )

    if cbar_max is None:
        cbar_max = res.meta.get("cbar_max", None)

    if out_npz:
        res.save_npz(out_npz)
    if out_png:
        t0, x0, y0 = _default_labels(cfg)
        plot_mean_hexbin_density(
            res,
            out_path=out_png,
            title=(title or t0),
            xlabel=(xlabel or x0),
            ylabel=(ylabel or y0),
            cbar_max=cbar_max,
            customizer=customizer,
            image_format=image_format,
        )
    return res


def _pretty_x_mode(x_mode: str) -> str:
    m = _x_mode_norm(x_mode)
    if m == "lreturn":
        return "return segment"  # human-facing
    return "total segment"


def _axis_label_x(cfg: BetweenRewardHexbinConfig) -> str:
    m = _x_mode_norm(cfg.x_mode)
    if m == "lreturn":
        base = "Distance traveled after farthest point [mm]"
    else:
        base = "Total distance traveled [mm]"
    if cfg.log1p_x:
        return base + " (log1p)"
    return base


def _axis_label_y(cfg: BetweenRewardHexbinConfig) -> str:
    base = "Max distance from reward [mm]"
    if cfg.log1p_y:
        return base + " (log1p)"
    return base


def _default_labels(cfg: BetweenRewardHexbinConfig) -> tuple[str, str, str]:
    # Training in title is 1-based for humans
    tnum = int(cfg.training_index) + 1
    xdesc = _pretty_x_mode(cfg.x_mode)

    title = f"Between-reward density (Training {tnum}, x={xdesc})"
    xlabel = _axis_label_x(cfg)
    ylabel = _axis_label_y(cfg)
    return title, xlabel, ylabel


def _x_mode_norm(x_mode: str) -> str:
    return str(x_mode or "").strip().lower()


def _transform_val(v: float, *, log1p: bool) -> float:
    v = float(v)
    if not np.isfinite(v):
        return np.nan
    if not log1p:
        return v
    # Safety: clamp negatives before log1p (shouldn't happen for lengths anyway)
    return float(np.log1p(max(0.0, v)))


def _default_extent(
    cfg: BetweenRewardHexbinConfig,
) -> tuple[float, float, float, float]:
    """
    Returns default extent in *plot-space* (after applying cfg.log1p_x/y).
    """
    mode = _x_mode_norm(cfg.x_mode)

    # Y is Dmax (mm)
    ymin_lin, ymax_lin = DMAX_MM_MIN, DMAX_MM_MAX
    ymin = _transform_val(ymin_lin, log1p=cfg.log1p_y)
    ymax = _transform_val(ymax_lin, log1p=cfg.log1p_y)

    # X is Ltotal or Lreturn (mm)
    if mode == "lreturn":
        xmin_lin, xmax_lin = LRETURN_MM_MIN, LRETURN_MM_MAX
    else:
        xmin_lin, xmax_lin = LTOTAL_MM_MIN, LTOTAL_MM_MAX

    xmin = _transform_val(xmin_lin, log1p=cfg.log1p_x)
    xmax = _transform_val(xmax_lin, log1p=cfg.log1p_x)

    # basic sanity
    if not (np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin):
        xmin, xmax = 0.0, 1.0
    if not (np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin):
        ymin, ymax = 0.0, 1.0

    return (float(xmin), float(xmax), float(ymin), float(ymax))


def _resolved_extent(
    cfg: BetweenRewardHexbinConfig,
) -> tuple[float, float, float, float]:
    if cfg.hex.extent is not None:
        ex = tuple(float(v) for v in cfg.hex.extent)
        if len(ex) != 4:
            raise ValueError("cfg.hex.extent must be (xmin, xmax, ymin, ymax)")
        return ex
    return _default_extent(cfg)


def collect_per_fly_segment_points(
    vas: list["VideoAnalysis"],
    *,
    cfg: BetweenRewardHexbinConfig,
    opts,
    log_tag: str = "btw_rwd_hex",
) -> tuple[list[np.ndarray], list[dict], dict]:
    """
    Return (per_fly_points, unit_info, meta).

    per_fly_points[i] is an (N_i, 2) array of points for fly-unit i:
        columns: [x, y] where y is Dmax and x is Ltotal or Lreturn,
                 both in *plot-space* (after optional transforms).

    meta includes resolved plot-space extent for consistent binning/comparisons.
    """
    warned_missing_wc = [False]
    per_fly_points: list[np.ndarray] = []
    unit_info: list[dict] = []

    t_idx = int(cfg.training_index)

    # Resolve and record shared extent (plot-space)
    extent = _resolved_extent(cfg)

    # If user asked for log-scale X comparisons but forgot log1p_x,
    # we'll still honor their config.
    mode = _x_mode_norm(cfg.x_mode)

    for va in vas:
        if getattr(va, "_skipped", False):
            continue
        if getattr(va, "trx", None) is None or len(va.trx) == 0:
            continue
        if va.trx[0].bad():
            continue

        trns = getattr(va, "trns", [])
        if t_idx < 0 or t_idx >= len(trns):
            continue

        px_per_mm = va.ct.pxPerMmFloor() * va.xf.fctr
        trn = trns[t_idx]

        for role_idx, trx_idx in enumerate(va.flies):
            if not va.noyc and role_idx != 0:
                continue
            if va._bad(trx_idx):
                continue

            fi, df, n_buckets, complete = sync_bucket_window(
                va,
                trn,
                t_idx=t_idx,
                f=trx_idx,
                skip_first=int(cfg.skip_first_sync_buckets),
                use_exclusion_mask=bool(cfg.use_reward_exclusion_mask),
            )
            if n_buckets <= 0:
                continue

            n_frames = int(max(1, n_buckets * df))

            exclude_wall = bool(
                getattr(opts, "com_exclude_wall_contact", False)
            ) or bool(cfg.exclude_wall_contact)
            exclude_nonwalk = bool(
                getattr(opts, "btw_rwd_conditioned_exclude_nonwalking_frames", False)
            ) or bool(cfg.exclude_nonwalking_frames)
            min_walk_frames = int(
                getattr(
                    opts, "btw_rwd_conditioned_min_walk_frames", cfg.min_walk_frames
                )
                or cfg.min_walk_frames
            )
            min_med_mm = float(
                getattr(
                    opts,
                    "com_per_segment_min_meddist_mm",
                    cfg.per_segment_min_meddist_mm,
                )
                or cfg.per_segment_min_meddist_mm
            )

            wc = wall_contact_mask(
                opts,
                va,
                trx_idx,
                fi=fi,
                n_frames=n_frames,
                log_tag=log_tag,
                warned_missing_wc=warned_missing_wc,
            )
            nonwalk_mask = build_nonwalk_mask(opts, va, trx_idx, fi, n_frames)

            traj = va.trx[trx_idx]
            pts: list[tuple[float, float]] = []

            dist_stats = ("median", "max")
            for seg in va._iter_between_reward_segment_com(
                trn,
                trx_idx,
                fi=fi,
                df=df,
                n_buckets=n_buckets,
                complete=complete,
                relative_to_reward=True,
                per_segment_min_meddist_mm=min_med_mm,
                exclude_wall=exclude_wall,
                wc=wc,
                exclude_nonwalk=exclude_nonwalk,
                nonwalk_mask=nonwalk_mask,
                min_walk_frames=min_walk_frames,
                dist_stats=dist_stats,
                debug=False,
                yield_skips=False,
            ):
                Dmax = float(getattr(seg, "max_d_mm", np.nan))
                if not np.isfinite(Dmax):
                    continue

                s = int(seg.s)
                e = int(seg.e)
                if e <= s + 1:
                    continue

                apex = seg.max_d_i
                if apex is None:
                    continue
                apex = int(apex)

                # Segment total length (mm), masked
                Ltotal = dist_traveled_mm_masked(
                    traj=traj,
                    s=s,
                    e=e,
                    fi=fi,
                    nonwalk_mask=nonwalk_mask,
                    exclude_nonwalk=exclude_nonwalk,
                    px_per_mm=px_per_mm,
                    start_override=None,
                    min_keep_frames=min_walk_frames,
                )
                if not np.isfinite(Ltotal):
                    continue

                if mode == "lreturn":
                    Lreturn = dist_traveled_mm_masked(
                        traj=traj,
                        s=s,
                        e=e,
                        fi=fi,
                        nonwalk_mask=nonwalk_mask,
                        exclude_nonwalk=exclude_nonwalk,
                        px_per_mm=px_per_mm,
                        start_override=apex,
                        min_keep_frames=min_walk_frames,
                    )
                    if not np.isfinite(Lreturn):
                        continue
                    x_lin = float(Lreturn)
                else:
                    x_lin = float(Ltotal)

                y_lin = float(Dmax)

                # Apply transforms into plot-space
                x = _transform_val(x_lin, log1p=cfg.log1p_x)
                y = _transform_val(y_lin, log1p=cfg.log1p_y)

                # Optionally clamp to extent so later hexbins don't get tugged by outliers
                xmin, xmax, ymin, ymax = extent
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    continue
                pts.append((x, y))

            if pts:
                per_fly_points.append(np.asarray(pts, dtype=float))
                unit_info.append(
                    dict(
                        video_id=video_base(va),
                        fly_id=va.f,
                        trx_idx=int(trx_idx),
                        role_idx=int(role_idx),
                        fly_role=fly_role_name(role_idx),
                        training_index=int(cfg.training_index),
                    )
                )

    meta = dict(
        log_tag=str(log_tag),
        training_index=int(cfg.training_index),
        skip_first_sync_buckets=int(cfg.skip_first_sync_buckets),
        use_reward_exclusion_mask=bool(cfg.use_reward_exclusion_mask),
        x_mode=str(cfg.x_mode),
        log1p_x=bool(cfg.log1p_x),
        log1p_y=bool(cfg.log1p_y),
        extent=tuple(float(v) for v in extent),  # plot-space extent
        gridside=int(cfg.hex.gridside),
        mincnt=int(cfg.hex.mincnt),
        n_fly_units=int(len(per_fly_points)),
        units=dict(x="mm (or log1p(mm) if enabled)", y="mm (or log1p(mm) if enabled)"),
    )

    return per_fly_points, unit_info, meta
