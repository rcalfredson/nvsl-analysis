from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.between_reward_segment_metrics import dist_traveled_mm_masked
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.training_metric_histogram import (
    TrainingMetricHistogramConfig,
    TrainingMetricHistogramPlotter,
)


@dataclass
class BetweenRewardNormalizedDistanceHistogramConfig(TrainingMetricHistogramConfig):
    metric_palette_family: str | None = "between_reward_distance"
    normalize_by: str = "mean"  # "mean" or "median"
    transform: str = "none"  # "none" or "log10"
    exclude_wall_contact: bool = False
    exclude_nonwalking_frames: bool = False
    min_walk_frames: int = 2


class BetweenRewardNormalizedDistanceHistogramPlotter(
    TrainingMetricHistogramPlotter
):
    """
    Histogram of per-fly normalized between-reward path lengths.

    For each fly/training:
      1. collect between-reward segment distances (mm)
      2. divide by that fly's own mean or median segment distance
      3. optionally apply log10
      4. histogram the transformed values per fly
      5. average the per-fly histograms across flies

    This keeps the fly as the unit of analysis while still showing a
    distribution over between-reward segments.
    """

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardNormalizedDistanceHistogramConfig,
    ):
        transform = str(getattr(cfg, "transform", "none") or "none").strip().lower()
        x_label = (
            r"$\log_{10}$(normalized distance traveled between rewards)"
            if transform == "log10"
            else "Normalized distance traveled between rewards"
        )
        title_suffix = (
            "log10 scale"
            if transform == "log10"
            else f"normalized by per-fly {cfg.normalize_by}"
        )
        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="btw_rwd_norm_dist",
            x_label=x_label,
            base_title=f"Between-reward distance histogram ({title_suffix})",
        )

    @staticmethod
    def _slice_start_stop(sl) -> tuple[int, int]:
        a = 0 if getattr(sl, "start", None) is None else int(sl.start)
        b = 0 if getattr(sl, "stop", None) is None else int(sl.stop)
        return a, b

    @classmethod
    def _any_overlap_with_wall_regions(cls, wall_regions, s: int, e: int) -> bool:
        if not wall_regions:
            return False
        s = int(s)
        e = int(e)
        for sl in wall_regions:
            a, b = cls._slice_start_stop(sl)
            if min(b, e) > max(a, s):
                return True
        return False

    @staticmethod
    def _filter_on_to_window(on, fi: int, end: int) -> list[int]:
        out = []
        for x in on:
            try:
                t = int(x)
            except Exception:
                continue
            if fi <= t < end:
                out.append(t)
        return out

    @staticmethod
    def _build_nonwalk_mask(
        va: "VideoAnalysis",
        trx_idx: int,
        *,
        fi: int,
        n_frames: int,
        enabled: bool,
    ):
        if not enabled:
            return None
        traj = va.trx[trx_idx]
        walking = getattr(traj, "walking", None)
        if walking is None:
            return None

        s0 = max(0, min(int(fi), len(walking)))
        e0 = max(0, min(int(fi + n_frames), len(walking)))

        wwin = np.zeros((int(max(1, n_frames)),), dtype=bool)
        if e0 > s0:
            wseg = np.asarray(walking[s0:e0], dtype=float)
            wseg = np.where(np.isfinite(wseg), wseg, 0.0)
            wwin[: len(wseg)] = wseg > 0
        return ~wwin

    def _normalize_values(self, vals_mm: np.ndarray) -> np.ndarray:
        vals = np.asarray(vals_mm, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return vals

        mode = str(getattr(self.cfg, "normalize_by", "mean") or "mean").lower()
        if mode == "median":
            denom = float(np.nanmedian(vals))
        else:
            denom = float(np.nanmean(vals))
        if not np.isfinite(denom) or denom <= 0:
            return np.asarray([], dtype=float)

        out = vals / denom
        transform = str(getattr(self.cfg, "transform", "none") or "none").lower()
        if transform == "log10":
            out = out[out > 0]
            if out.size == 0:
                return np.asarray([], dtype=float)
            out = np.log10(out)

        out = out[np.isfinite(out)]
        return out.astype(float, copy=False)

    def _collect_values_for_fly_training(
        self,
        va: "VideoAnalysis",
        *,
        t_idx: int,
        trn,
        f: int,
    ) -> np.ndarray:
        traj = va.trx[f]
        wall_regions = None
        if self.cfg.exclude_wall_contact:
            try:
                wall_regions = traj.boundary_event_stats["wall"]["all"]["edge"][
                    "boundary_contact_regions"
                ]
            except (KeyError, TypeError, AttributeError):
                wall_regions = None

        skip_first, keep_first = self._effective_sync_bucket_window()
        fi, df, n_buckets, complete = sync_bucket_window(
            va,
            trn,
            t_idx=t_idx,
            f=f,
            skip_first=skip_first,
            keep_first=keep_first,
            use_exclusion_mask=False,
        )
        if n_buckets <= 0:
            return np.asarray([], dtype=float)

        end = int(fi + n_buckets * df)
        on = va._getOn(trn, False, f=f)
        if on is None or len(on) < 2:
            return np.asarray([], dtype=float)

        on_win = self._filter_on_to_window(on, fi=fi, end=end)
        if len(on_win) < 2:
            return np.asarray([], dtype=float)

        px_per_mm = float(traj.pxPerMmFloor * va.xf.fctr)
        if not np.isfinite(px_per_mm) or px_per_mm <= 0:
            return np.asarray([], dtype=float)

        nonwalk_mask = self._build_nonwalk_mask(
            va,
            f,
            fi=fi,
            n_frames=int(max(1, n_buckets * df)),
            enabled=bool(getattr(self.cfg, "exclude_nonwalking_frames", False)),
        )

        vals_mm: list[float] = []
        min_walk_frames = int(max(2, getattr(self.cfg, "min_walk_frames", 2) or 2))

        for s, e in zip(on_win[:-1], on_win[1:]):
            s = int(s)
            e = int(e)
            if e <= s:
                continue
            b_idx = int((s - fi) // df)
            if b_idx < 0 or b_idx >= n_buckets:
                continue
            if not complete[b_idx]:
                continue
            if self.cfg.exclude_wall_contact and self._any_overlap_with_wall_regions(
                wall_regions, s, e
            ):
                continue

            d_mm = dist_traveled_mm_masked(
                traj=traj,
                s=s,
                e=e,
                fi=fi,
                nonwalk_mask=nonwalk_mask,
                exclude_nonwalk=bool(
                    getattr(self.cfg, "exclude_nonwalking_frames", False)
                ),
                px_per_mm=px_per_mm,
                start_override=None,
                min_keep_frames=min_walk_frames,
            )
            if np.isfinite(d_mm):
                vals_mm.append(float(d_mm))

        if not vals_mm:
            return np.asarray([], dtype=float)
        return self._normalize_values(np.asarray(vals_mm, dtype=float))

    def _collect_values_by_training_per_fly(self) -> list[list[tuple[str, np.ndarray]]]:
        n_trn = self._n_trainings()

        if self.cfg.pool_trainings:
            pooled_panel: list[tuple[str, np.ndarray]] = []
        else:
            all_by_trn: list[list[tuple[str, np.ndarray]]] = [[] for _ in range(n_trn)]

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if va.trx[0].bad():
                continue

            pooled_for_this_va: list[np.ndarray] = []

            for t_idx, trn in enumerate(getattr(va, "trns", [])):
                if t_idx >= n_trn:
                    break
                for f in va.flies:
                    if not va.noyc and f != 0:
                        continue

                    vals = self._collect_values_for_fly_training(
                        va, t_idx=t_idx, trn=trn, f=f
                    )
                    if vals.size == 0:
                        continue

                    unit_id = self._unit_id(va, f=f)
                    if self.cfg.pool_trainings:
                        pooled_for_this_va.append(vals)
                    else:
                        all_by_trn[t_idx].append((unit_id, vals))

            if self.cfg.pool_trainings and pooled_for_this_va:
                unit_id = self._unit_id(va, f=0)
                pooled_panel.append((unit_id, np.concatenate(pooled_for_this_va)))

        if self.cfg.pool_trainings:
            return [pooled_panel]
        return all_by_trn

    def _collect_values_by_training(self) -> list[np.ndarray]:
        by_trn_by_fly = self._collect_values_by_training_per_fly()
        out: list[np.ndarray] = []
        for panel in by_trn_by_fly:
            vals = []
            for _uid, arr in panel:
                if arr is None:
                    continue
                vv = np.asarray(arr, dtype=float)
                vv = vv[np.isfinite(vv)]
                if vv.size:
                    vals.append(vv)
            out.append(
                np.concatenate(vals, axis=0) if vals else np.asarray([], dtype=float)
            )
        return out
