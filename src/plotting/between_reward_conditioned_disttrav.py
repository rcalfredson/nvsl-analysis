from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

import src.utils.util as util
from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import maybe_sentence_case, writeImage

from src.plotting.between_reward_segment_binning import (
    x_edges as make_x_edges,
    sync_bucket_window,
    build_nonwalk_mask,
    wall_contact_mask,
)


@dataclass
class BetweenRewardConditionedDistTravConfig:
    """
    Distance-binned distance-traveled analysis for between-reward segments.

    Binning:
        - x-axis: max distance from reward center during segment (mm)

    Metrics per segment:
        - total distance traveled over [s, e) frames (step indices [s, e-1))
        - tail distance traveled from max-distance frame to end (same step semantics)

    Aggregation:
        - per-fly mean within each x-bin
        - then mean + CI across flies per x-bin
    """

    out_file: str

    # Which training to analyze (0-based index).
    training_index: int = 1  # Training 2 default (0-based)

    skip_first_sync_buckets: int = 0
    use_reward_exclusion_mask: bool = False

    # Distance binning in mm
    x_bin_width_mm: float = 2.0
    x_min_mm: float = 0.0
    x_max_mm: float = 20.0

    # Plot options
    ci_conf: float = 0.95
    ymax: float | None = None
    subset_label: str | None = None


@dataclass(frozen=True)
class BetweenRewardConditionedDistTravResult:
    x_edges: np.ndarray
    x_centers: np.ndarray

    mean_total: np.ndarray
    ci_lo_total: np.ndarray
    ci_hi_total: np.ndarray

    mean_tail: np.ndarray
    ci_lo_tail: np.ndarray
    ci_hi_tail: np.ndarray

    n_units: np.ndarray
    meta: dict

    per_unit_total: np.ndarray | None = None  # (N_units, B)
    per_unit_tail: np.ndarray | None = None  # (N_units, B)

    def validate(self) -> None:
        edges = np.asarray(self.x_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("x_edges must be 1D with >= 2 entries")
        B = int(edges.size - 1)

        def _check_1d(name: str) -> None:
            arr = np.asarray(getattr(self, name))
            if arr.ndim != 1 or arr.size != B:
                raise ValueError(f"{name} must be 1D with length {B}")

        for nm in (
            "x_centers",
            "mean_total",
            "ci_lo_total",
            "ci_hi_total",
            "mean_tail",
            "ci_lo_tail",
            "ci_hi_tail",
            "n_units",
        ):
            _check_1d(nm)

        if self.per_unit_total is not None:
            pu = np.asarray(self.per_unit_total, dtype=float)
            if pu.ndim != 2 or pu.shape[1] != B:
                raise ValueError(f"per_unit_total must be 2D with shape (N, {B})")
        if self.per_unit_tail is not None:
            pu = np.asarray(self.per_unit_tail, dtype=float)
            if pu.ndim != 2 or pu.shape[1] != B:
                raise ValueError(f"per_unit_tail must be 2D with shape (N, {B})")

    def save_npz(self, path: str) -> None:
        self.validate()
        kwargs = dict(
            x_edges=np.asarray(self.x_edges, dtype=float),
            x_centers=np.asarray(self.x_centers, dtype=float),
            mean_total=np.asarray(self.mean_total, dtype=float),
            ci_lo_total=np.asarray(self.ci_lo_total, dtype=float),
            ci_hi_total=np.asarray(self.ci_hi_total, dtype=float),
            mean_tail=np.asarray(self.mean_tail, dtype=float),
            ci_lo_tail=np.asarray(self.ci_lo_tail, dtype=float),
            ci_hi_tail=np.asarray(self.ci_hi_tail, dtype=float),
            n_units=np.asarray(self.n_units, dtype=int),
            meta=np.asarray([self.meta], dtype=object),
        )
        if self.per_unit_total is not None:
            kwargs["per_unit_total"] = np.asarray(self.per_unit_total, dtype=float)
        if self.per_unit_tail is not None:
            kwargs["per_unit_tail"] = np.asarray(self.per_unit_tail, dtype=float)
        np.savez_compressed(path, **kwargs)

    @staticmethod
    def load_npz(path: str) -> "BetweenRewardConditionedDistTravResult":
        z = np.load(path, allow_pickle=True)
        meta = {}
        if "meta" in z:
            try:
                meta_obj = z["meta"]
                meta = meta_obj.item() if hasattr(meta_obj, "item") else {}
            except Exception:
                meta = {}

        res = BetweenRewardConditionedDistTravResult(
            x_edges=np.asarray(z["x_edges"], dtype=float),
            x_centers=np.asarray(z["x_centers"], dtype=float),
            mean_total=np.asarray(z["mean_total"], dtype=float),
            ci_lo_total=np.asarray(z["ci_lo_total"], dtype=float),
            ci_hi_total=np.asarray(z["ci_hi_total"], dtype=float),
            mean_tail=np.asarray(z["mean_tail"], dtype=float),
            ci_lo_tail=np.asarray(z["ci_lo_tail"], dtype=float),
            ci_hi_tail=np.asarray(z["ci_hi_tail"], dtype=float),
            n_units=np.asarray(z["n_units"], dtype=int),
            meta=dict(meta) if isinstance(meta, dict) else {},
            per_unit_total=(
                np.asarray(z["per_unit_total"], dtype=float)
                if "per_unit_total" in z
                else None
            ),
            per_unit_tail=(
                np.asarray(z["per_unit_tail"], dtype=float)
                if "per_unit_tail" in z
                else None
            ),
        )
        res.validate()
        return res


class BetweenRewardConditionedDistTravPlotter:
    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardConditionedDistTravConfig,
    ):
        self.vas = vas
        self.opts = opts
        self.gls = gls
        self.customizer = customizer
        self.cfg = cfg
        self.log_tag = "btw_rwd_dist_binned_disttrav"

    def _x_edges(self) -> np.ndarray:
        return make_x_edges(
            x_bin_width_mm=float(self.cfg.x_bin_width_mm),
            x_min_mm=float(self.cfg.x_min_mm),
            x_max_mm=float(self.cfg.x_max_mm),
        )

    def _px_per_mm(self, va: "VideoAnalysis") -> float | None:
        try:
            px_per_mm = float(va.xf.fctr * va.ct.pxPerMmFloor())
            if np.isfinite(px_per_mm) and px_per_mm > 0:
                return px_per_mm
        except Exception:
            pass
        return None

    def _seg_keep_frames(
        self,
        *,
        traj,
        s: int,
        e: int,
        fi: int,
        nonwalk_mask,
        exclude_nonwalk: bool,
        min_keep_frames: int = 2,
    ) -> tuple[np.ndarray, int] | tuple[None, int]:
        """
        Return (keep_frames, L) for the effective segment window [s, s+L).

        keep_frames is length L aligned to absolute frames s..s+L-1.
        L may be less than (e-s) if we clamp to available nonwalk_mask window,
        mirroring the iterator behavior.
        """
        s = int(s)
        e = int(e)
        L0 = int(max(0, e - s))
        if L0 <= 0:
            return None, 0

        L = L0
        keep = np.ones((L0,), dtype=bool)

        if exclude_nonwalk and nonwalk_mask is not None:
            s2 = max(0, min(s - fi, len(nonwalk_mask)))
            e2 = max(0, min(e - fi, len(nonwalk_mask)))
            if e2 <= s2:
                return None, 0
            L = int(min(L0, e2 - s2))
            if L <= 0:
                return None, 0
            keep = keep[:L] & (~np.asarray(nonwalk_mask[s2 : s2 + L], dtype=bool))

        # Clamp L to available trajectory data (numpy slices clamp silently; we must match lengths)
        max_L = int(min(len(traj.x) - s, len(traj.y) - s))
        if max_L <= 0:
            return None, 0
        if L > max_L:
            L = max_L
            keep = keep[:L]

        # finite xy filter on the same effective window
        xs = np.asarray(traj.x[s : s + L], dtype=float)
        ys = np.asarray(traj.y[s : s + L], dtype=float)
        fin = np.isfinite(xs) & np.isfinite(ys)
        keep &= fin

        min_keep_frames = int(max(2, min_keep_frames))
        if int(np.sum(keep)) < min_keep_frames:
            return None, L
        return keep, L

    def _dist_traveled_mm_masked(
        self,
        *,
        va: "VideoAnalysis",
        traj,
        s: int,
        e: int,
        fi: int,
        nonwalk_mask,
        exclude_nonwalk: bool,
        start_override: int | None = None,
        min_keep_frames: int = 2,
    ) -> float:
        """
        Distance traveled in mm within [s, e) frames, using masked frames (walking+finite).

        Step semantics:
          - include step i if both frames i and i+1 are kept
          - steps range is within [s, s+L) effective window, so step indices [s, s+L-1)

        If start_override is provided, compute within [start_override, e) (still clamped to window)
        """
        keep, L = self._seg_keep_frames(
            traj=traj,
            s=s,
            e=e,
            fi=fi,
            nonwalk_mask=nonwalk_mask,
            exclude_nonwalk=exclude_nonwalk,
            min_keep_frames=min_keep_frames,
        )
        if keep is None or L < 2:
            return np.nan

        px_per_mm = self._px_per_mm(va)
        if px_per_mm is None:
            # should not happen in normal pipeline
            return np.nan

        # Determine start offset within the effective [s, s+L) window
        if start_override is None:
            off = 0
        else:
            start_override = int(start_override)
            if start_override <= s:
                off = 0
            elif start_override >= s + L:
                return 0.0
            else:
                off = int(start_override - s)

        # Ensure start frame is kept; if not, advance to next kept frame
        if not keep[off]:
            nxt = np.where(keep[off:])[0]
            if nxt.size == 0:
                return np.nan
            off = int(off + nxt[0])
            if off >= L - 1:
                return 0.0

        # Steps exist for indices 0..L-2 within this effective window
        keep_steps = keep[:-1] & keep[1:]  # length L-1
        if off > 0:
            keep_steps = keep_steps[off:]
            step_px = np.asarray(traj.d[(s + off) : (s + L - 1)], dtype=float)
        else:
            step_px = np.asarray(traj.d[s : (s + L - 1)], dtype=float)

        if step_px.size == 0:
            return 0.0

        n = int(min(keep_steps.size, step_px.size))
        if n <= 0:
            return 0.0
        keep_steps = keep_steps[:n]
        step_px = step_px[:n]

        dpx = float(np.sum(step_px[keep_steps]))

        return float(dpx / px_per_mm)

    def _collect_per_fly_binned_means(self) -> tuple[np.ndarray, np.ndarray, dict]:
        edges = self._x_edges()
        B = int(max(1, edges.size - 1))

        exclude_wall = bool(getattr(self.opts, "com_exclude_wall_contact", False))
        min_med_mm = float(
            getattr(self.opts, "com_per_segment_min_meddist_mm", 0.0) or 0.0
        )
        warned_missing_wc = [False]

        exclude_nonwalk = bool(
            getattr(self.opts, "btw_rwd_conditioned_exclude_nonwalking_frames", False)
        )
        min_walk_frames = int(
            getattr(self.opts, "btw_rwd_conditioned_min_walk_frames", 2) or 2
        )

        per_fly_total: list[np.ndarray] = []
        per_fly_tail: list[np.ndarray] = []

        t_idx = int(self.cfg.training_index)

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if getattr(va, "trx", None) is None or len(va.trx) == 0:
                continue
            if va.trx[0].bad():
                continue

            trns = getattr(va, "trns", [])
            if t_idx < 0 or t_idx >= len(trns):
                continue
            trn = trns[t_idx]

            for role_idx, trx_idx in enumerate(va.flies):
                if not va.noyc and role_idx != 0:
                    continue

                fi, df, n_buckets, complete = sync_bucket_window(
                    va,
                    trn,
                    t_idx=t_idx,
                    f=trx_idx,
                    skip_first=int(self.cfg.skip_first_sync_buckets),
                    use_exclusion_mask=bool(self.cfg.use_reward_exclusion_mask),
                )
                if n_buckets <= 0:
                    continue

                n_frames = int(max(1, n_buckets * df))
                wc = wall_contact_mask(
                    self.opts,
                    va,
                    trx_idx,
                    fi=fi,
                    n_frames=n_frames,
                    log_tag=self.log_tag,
                    warned_missing_wc=warned_missing_wc,
                )
                nonwalk_mask = build_nonwalk_mask(self.opts, va, trx_idx, fi, n_frames)

                traj = va.trx[trx_idx]

                bin_vals_total: list[list[float]] = [[] for _ in range(B)]
                bin_vals_tail: list[list[float]] = [[] for _ in range(B)]

                # We need max_d_mm and max_d_i
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
                    x = float(getattr(seg, "max_d_mm", np.nan))
                    if not np.isfinite(x):
                        continue

                    s = int(getattr(seg, "s", -1))
                    e = int(getattr(seg, "e", -1))
                    if e <= s + 1:
                        continue

                    max_i = getattr(seg, "max_d_i", None)
                    if max_i is None:
                        continue
                    max_i = int(max_i)

                    j = int(np.searchsorted(edges, x, side="right") - 1)
                    if j < 0 or j >= B:
                        continue

                    dt_total = self._dist_traveled_mm_masked(
                        va=va,
                        traj=traj,
                        s=s,
                        e=e,
                        fi=fi,
                        nonwalk_mask=nonwalk_mask,
                        exclude_nonwalk=exclude_nonwalk,
                        start_override=None,
                        min_keep_frames=min_walk_frames,
                    )
                    dt_tail = self._dist_traveled_mm_masked(
                        va=va,
                        traj=traj,
                        s=s,
                        e=e,
                        fi=fi,
                        nonwalk_mask=nonwalk_mask,
                        exclude_nonwalk=exclude_nonwalk,
                        start_override=max_i,
                        min_keep_frames=min_walk_frames,
                    )

                    if np.isfinite(dt_total):
                        bin_vals_total[j].append(float(dt_total))
                    if np.isfinite(dt_tail):
                        bin_vals_tail[j].append(float(dt_tail))

                vec_total = np.full((B,), np.nan, dtype=float)
                vec_tail = np.full((B,), np.nan, dtype=float)

                for j in range(B):
                    if bin_vals_total[j]:
                        vv = np.asarray(bin_vals_total[j], dtype=float)
                        vv = vv[np.isfinite(vv)]
                        if vv.size:
                            vec_total[j] = float(np.mean(vv))
                    if bin_vals_tail[j]:
                        vv = np.asarray(bin_vals_tail[j], dtype=float)
                        vv = vv[np.isfinite(vv)]
                        if vv.size:
                            vec_tail[j] = float(np.mean(vv))

                if np.any(np.isfinite(vec_total)) or np.any(np.isfinite(vec_tail)):
                    per_fly_total.append(vec_total)
                    per_fly_tail.append(vec_tail)

        if not per_fly_total:
            meta = {
                "log_tag": self.log_tag,
                "training_index": int(self.cfg.training_index),
                "skip_first_sync_buckets": int(self.cfg.skip_first_sync_buckets),
                "use_reward_exclusion_mask": bool(self.cfg.use_reward_exclusion_mask),
                "x_bin_width_mm": float(self.cfg.x_bin_width_mm),
                "x_min_mm": float(self.cfg.x_min_mm),
                "x_max_mm": float(self.cfg.x_max_mm),
                "ci_conf": float(self.cfg.ci_conf),
                "n_fly_units": 0,
                "exclude_wall_contact": bool(
                    getattr(self.opts, "com_exclude_wall_contact", False)
                ),
                "exclude_nonwalking_frames": bool(exclude_nonwalk),
                "min_walk_frames": int(min_walk_frames),
                "units": "mm",
            }
            return np.empty((0, B), dtype=float), np.empty((0, B), dtype=float), meta

        Y_total = np.stack(per_fly_total, axis=0)
        Y_tail = np.stack(per_fly_tail, axis=0)

        meta = {
            "log_tag": self.log_tag,
            "training_index": int(self.cfg.training_index),
            "skip_first_sync_buckets": int(self.cfg.skip_first_sync_buckets),
            "use_reward_exclusion_mask": bool(self.cfg.use_reward_exclusion_mask),
            "x_bin_width_mm": float(self.cfg.x_bin_width_mm),
            "x_min_mm": float(self.cfg.x_min_mm),
            "x_max_mm": float(self.cfg.x_max_mm),
            "ci_conf": float(self.cfg.ci_conf),
            "n_fly_units": int(Y_total.shape[0]),
            "exclude_wall_contact": bool(
                getattr(self.opts, "com_exclude_wall_contact", False)
            ),
            "exclude_nonwalking_frames": bool(exclude_nonwalk),
            "min_walk_frames": int(min_walk_frames),
            "units": "mm",
        }
        return Y_total, Y_tail, meta

    def compute_result(self) -> BetweenRewardConditionedDistTravResult:
        edges = self._x_edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        B = int(max(1, edges.size - 1))

        Y_total, Y_tail, meta = self._collect_per_fly_binned_means()

        mean_total = np.full((B,), np.nan, dtype=float)
        lo_total = np.full((B,), np.nan, dtype=float)
        hi_total = np.full((B,), np.nan, dtype=float)

        mean_tail = np.full((B,), np.nan, dtype=float)
        lo_tail = np.full((B,), np.nan, dtype=float)
        hi_tail = np.full((B,), np.nan, dtype=float)

        n_units = np.zeros((B,), dtype=int)

        if Y_total.size:
            for j in range(B):
                m, lo, hi, n = util.meanConfInt(
                    Y_total[:, j], conf=float(self.cfg.ci_conf)
                )
                mean_total[j], lo_total[j], hi_total[j] = float(m), float(lo), float(hi)
                n_units[j] = int(n)

                m, lo, hi, _n = util.meanConfInt(
                    Y_tail[:, j], conf=float(self.cfg.ci_conf)
                )
                mean_tail[j], lo_tail[j], hi_tail[j] = float(m), float(lo), float(hi)

        return BetweenRewardConditionedDistTravResult(
            x_edges=np.asarray(edges, dtype=float),
            x_centers=np.asarray(centers, dtype=float),
            mean_total=mean_total,
            ci_lo_total=lo_total,
            ci_hi_total=hi_total,
            mean_tail=mean_tail,
            ci_lo_tail=lo_tail,
            ci_hi_tail=hi_tail,
            n_units=n_units,
            meta=meta,
            per_unit_total=(Y_total if Y_total.size else None),
            per_unit_tail=(Y_tail if Y_tail.size else None),
        )

    def plot(self) -> None:
        res = self.compute_result()
        res.validate()

        x = np.asarray(res.x_centers, dtype=float)
        edges = np.asarray(res.x_edges, dtype=float)
        widths = edges[1:] - edges[:-1]

        # derive two output paths from cfg.out_file
        out0 = str(self.cfg.out_file)
        root, ext = os.path.splitext(out0)
        if not ext:
            # if user passed something without an extension, fall back to opts.imageFormat
            ext = "." + str(getattr(self.opts, "imageFormat", "png")).lstrip(".")
        out_total = f"{root}_total{ext}"
        out_tail = f"{root}_tail{ext}"

        def _plot_one(*, y, lo, hi, title, ylabel, out_file) -> None:
            fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2))

            if not np.any(np.isfinite(y)):
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
            else:
                fin = np.isfinite(x) & np.isfinite(y) & np.isfinite(widths)

                # Histogram-like bars: one bar per distance bin
                ax.bar(
                    x[fin],
                    y[fin],
                    width=0.92 * widths[fin],
                    align="center",
                    alpha=0.75,
                    linewidth=0.8,
                )

                fin_ci = fin & np.isfinite(lo) & np.isfinite(hi)
                if fin_ci.any():
                    yerr = np.vstack([y[fin_ci] - lo[fin_ci], hi[fin_ci] - y[fin_ci]])
                    ax.errorbar(
                        x[fin_ci],
                        y[fin_ci],
                        yerr=yerr,
                        fmt="none",
                        elinewidth=1.1,
                        capsize=2.0,
                        alpha=0.9,
                    )

                ax.set_xlabel(
                    maybe_sentence_case("max distance from reward center [mm] (binned)")
                )
                ax.set_ylabel(maybe_sentence_case(ylabel))

                if edges.size >= 2 and np.all(np.isfinite(edges[[0, -1]])):
                    ax.set_xlim(float(edges[0]), float(edges[-1]))

                ax.set_ylim(bottom=0)
                if self.cfg.ymax is not None:
                    ax.set_ylim(top=float(self.cfg.ymax))
                else:
                    y_top = (
                        np.nanmax(hi) if np.isfinite(np.nanmax(hi)) else np.nanmax(y)
                    )
                    if np.isfinite(y_top):
                        ax.set_ylim(top=float(y_top) * 1.12)

                ylim0, ylim1 = ax.get_ylim()
                y_off = 0.04 * (ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 0.0
                for xi, hi_i, ni in zip(x, hi, res.n_units):
                    if np.isfinite(xi) and np.isfinite(hi_i) and int(ni) > 0:
                        util.pltText(
                            float(xi),
                            float(hi_i + y_off),
                            f"{int(ni)}",
                            ha="center",
                            size=self.customizer.in_plot_font_size,
                            color=".2",
                        )
                ax.set_title(maybe_sentence_case(title))

                # small annotation line
                parts = [f"T{int(self.cfg.training_index) + 1}"]
                if int(self.cfg.skip_first_sync_buckets) > 0:
                    parts.append(
                        f"skip first {int(self.cfg.skip_first_sync_buckets)} bucket(s)"
                    )
                if self.cfg.subset_label:
                    parts.append(str(self.cfg.subset_label))
                fig.text(
                    0.02,
                    0.98,
                    " | ".join(parts),
                    ha="left",
                    va="top",
                    fontsize=self.customizer.in_plot_font_size,
                    color="0",
                )

            if self.customizer.font_size_customized:
                self.customizer.adjust_padding_proportionally()
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            writeImage(out_file, format=self.opts.imageFormat)
            plt.close(fig)
            print(f"[{self.log_tag}] wrote {out_file}")

        _plot_one(
            y=np.asarray(res.mean_total, dtype=float),
            lo=np.asarray(res.ci_lo_total, dtype=float),
            hi=np.asarray(res.ci_hi_total, dtype=float),
            title="between-reward distance traveled vs max distance-from-reward (total)",
            ylabel="mean distance traveled per fly [mm]",
            out_file=out_total,
        )

        _plot_one(
            y=np.asarray(res.mean_tail, dtype=float),
            lo=np.asarray(res.ci_lo_tail, dtype=float),
            hi=np.asarray(res.ci_hi_tail, dtype=float),
            title="between-reward distance traveled vs max distance-from-reward (max→end)",
            ylabel="mean distance traveled per fly [mm]",
            out_file=out_tail,
        )
