from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.between_reward_segment_metrics import tortuosity_metric_masked
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.training_metric_histogram import (
    TrainingMetricHistogramConfig,
    TrainingMetricHistogramPlotter,
)
from src.plotting.wall_contact_utils import build_wall_contact_mask_for_window


@dataclass
class BetweenRewardTortuosityHistogramConfig(TrainingMetricHistogramConfig):
    metric_palette_family: str | None = "between_reward_distance"
    metric_mode: str = "path_over_max_radius"
    transform: str = "none"  # "none" or "log10"
    exclude_wall_contact: bool = False
    exclude_nonwalking_frames: bool = False
    exclude_reward_endpoints: bool = False
    min_walk_frames: int = 2
    min_displacement_mm: float = 0.0
    min_radius_mm: float = 0.0


class BetweenRewardTortuosityHistogramPlotter(TrainingMetricHistogramPlotter):
    """
    Histogram of per-between-reward-segment tortuosity-like metrics.

    Segment membership and training assignment follow
    VideoAnalysis._iter_between_reward_segment_com(...), while the metric itself
    is computed on the same frame window with optional wall/nonwalking masking.
    """

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardTortuosityHistogramConfig,
    ):
        mode = str(getattr(cfg, "metric_mode", "path_over_max_radius") or "").lower()
        transform = str(getattr(cfg, "transform", "none") or "none").lower()

        if mode == "straightness":
            x_label = "Between-reward straightness"
            base_title = "Between-reward straightness"
        elif mode == "excess_path":
            x_label = "Between-reward excess path ratio"
            base_title = "Between-reward excess path ratio"
        elif mode == "path_over_displacement":
            x_label = "Between-reward chord tortuosity"
            base_title = "Between-reward chord tortuosity"
        else:
            x_label = "Between-reward tortuosity"
            base_title = "Between-reward tortuosity (path/max radius)"

        if transform == "log10":
            x_label = rf"$\log_{{10}}$({x_label.lower()})"
            base_title = f"{base_title} (log10)"

        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="btw_rwd_tortuosity",
            x_label=x_label,
            base_title=base_title,
        )

    def _wall_contact_mask_for_window(
        self,
        va: "VideoAnalysis",
        f: int,
        *,
        fi: int,
        n_frames: int,
        warned_missing_wc: list[bool],
    ):
        return build_wall_contact_mask_for_window(
            va,
            f,
            fi=fi,
            n_frames=int(max(1, n_frames)),
            enabled=bool(getattr(self.cfg, "exclude_wall_contact", False)),
            warned_missing_wc=warned_missing_wc,
            log_tag=self.log_tag,
        )

    def _transform_values(self, vals: list[float]) -> np.ndarray:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.asarray([], dtype=float)

        transform = str(getattr(self.cfg, "transform", "none") or "none").lower()
        if transform == "log10":
            arr = arr[arr > 0]
            if arr.size == 0:
                return np.asarray([], dtype=float)
            arr = np.log10(arr)

        return arr[np.isfinite(arr)].astype(float, copy=False)

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

    def _collect_values_for_fly_training(
        self,
        va: "VideoAnalysis",
        *,
        t_idx: int,
        trn,
        f: int,
        warned_missing_wc: list[bool],
    ) -> np.ndarray:
        traj = va.trx[f]
        skip_first = self._effective_skip_first_sync_buckets()
        keep_first = self._effective_keep_first_sync_buckets()
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

        n_frames = int(max(1, n_buckets * df))
        px_per_mm = float(traj.pxPerMmFloor * va.xf.fctr)
        if not np.isfinite(px_per_mm) or px_per_mm <= 0:
            return np.asarray([], dtype=float)
        wc = self._wall_contact_mask_for_window(
            va, f, fi=fi, n_frames=n_frames, warned_missing_wc=warned_missing_wc
        )
        nonwalk_mask = self._build_nonwalk_mask(
            va,
            f,
            fi=fi,
            n_frames=n_frames,
            enabled=bool(getattr(self.cfg, "exclude_nonwalking_frames", False)),
        )

        vals: list[float] = []
        min_walk_frames = int(max(2, getattr(self.cfg, "min_walk_frames", 2) or 2))
        mode = str(
            getattr(self.cfg, "metric_mode", "path_over_max_radius")
            or "path_over_max_radius"
        )

        try:
            cx, cy, _ = trn.circles(f)[0]
            reward_center_xy = (float(cx), float(cy))
        except Exception:
            reward_center_xy = None

        for seg in va._iter_between_reward_segment_com(
            trn,
            f,
            fi=fi,
            df=df,
            n_buckets=n_buckets,
            complete=complete,
            relative_to_reward=True,
            per_segment_min_meddist_mm=0.0,
            exclude_wall=bool(getattr(self.cfg, "exclude_wall_contact", False)),
            wc=wc,
            exclude_nonwalk=bool(
                getattr(self.cfg, "exclude_nonwalking_frames", False)
            ),
            nonwalk_mask=nonwalk_mask,
            min_walk_frames=min_walk_frames,
            exclude_reward_endpoints=bool(
                getattr(self.cfg, "exclude_reward_endpoints", False)
            ),
            debug=False,
            yield_skips=False,
        ):
            s = int(seg.s) + (
                1 if bool(getattr(self.cfg, "exclude_reward_endpoints", False)) else 0
            )
            e = int(seg.e) - (
                1 if bool(getattr(self.cfg, "exclude_reward_endpoints", False)) else 0
            )
            if e <= s:
                continue

            val = tortuosity_metric_masked(
                traj=traj,
                s=s,
                e=e,
                fi=fi,
                nonwalk_mask=nonwalk_mask,
                exclude_nonwalk=bool(
                    getattr(self.cfg, "exclude_nonwalking_frames", False)
                ),
                px_per_mm=px_per_mm,
                mode=mode,
                reward_center_xy=reward_center_xy,
                min_keep_frames=min_walk_frames,
                min_displacement_mm=float(
                    getattr(self.cfg, "min_displacement_mm", 0.0) or 0.0
                ),
                min_radius_mm=float(
                    getattr(self.cfg, "min_radius_mm", 0.0) or 0.0
                ),
            )
            if np.isfinite(val):
                vals.append(float(val))

        return self._transform_values(vals)

    def _collect_values_by_training_per_fly(self) -> list[list[tuple[str, np.ndarray]]]:
        n_trn = self._n_trainings()
        warned_missing_wc = [False]

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
                        va,
                        t_idx=t_idx,
                        trn=trn,
                        f=f,
                        warned_missing_wc=warned_missing_wc,
                    )
                    if vals.size == 0:
                        continue

                    unit_id = self._unit_id(va, f=f)
                    if self.cfg.pool_trainings:
                        pooled_for_this_va.append(vals)
                    else:
                        all_by_trn[t_idx].append((unit_id, vals))

            if self.cfg.pool_trainings and pooled_for_this_va:
                pooled_panel.append(
                    (self._unit_id(va, f=0), np.concatenate(pooled_for_this_va))
                )

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
