from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.analysis.between_reward_filters import (
    min_between_reward_sync_bucket_trajectories,
)
from src.analysis.sync_bucket_presence_filters import (
    exp_target_sync_bucket_filter_result,
)
from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.training_metric_histogram import (
    TrainingMetricHistogramConfig,
    TrainingMetricHistogramPlotter,
)


DEFAULT_MAX_DISTANCE_BIN_EDGES_MM = tuple(float(x) for x in range(0, 31, 5))


@dataclass
class BetweenRewardMaxDistanceHistogramConfig(TrainingMetricHistogramConfig):
    metric_palette_family: str | None = "between_reward_distance"


class BetweenRewardMaxDistanceHistogramPlotter(TrainingMetricHistogramPlotter):
    """Per-fly histograms of trajectory maximum distance from the reward center."""

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardMaxDistanceHistogramConfig,
    ):
        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="btw_rwd_maxdist",
            x_label="Maximum distance from reward circle center (mm)",
            base_title="Between-reward trajectory maximum distance",
        )

    def _collect_fly_training_values(self, va, *, t_idx: int, trn, f: int):
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

        values = []
        for seg in va._iter_between_reward_segment_com(
            trn,
            f,
            fi=fi,
            df=df,
            n_buckets=n_buckets,
            complete=complete,
            relative_to_reward=True,
            per_segment_min_meddist_mm=0.0,
            exclude_wall=False,
            wc=None,
            exclude_nonwalk=False,
            nonwalk_mask=None,
            min_walk_frames=2,
            dist_stats=("max",),
            debug=False,
            yield_skips=False,
        ):
            value = float(getattr(seg, "max_d_mm", np.nan))
            if np.isfinite(value):
                values.append(value)
        return np.asarray(values, dtype=float)

    def _collect_values_by_training_per_fly(
        self,
    ) -> list[list[tuple[str, np.ndarray]]]:
        n_trn = self._n_trainings()
        min_trajectories = min_between_reward_sync_bucket_trajectories(self.opts)
        by_training: list[list[tuple[str, np.ndarray]]] = [
            [] for _ in range(n_trn)
        ]
        pooled: list[tuple[str, np.ndarray]] = []

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if not getattr(va, "trx", None) or va.trx[0].bad():
                continue
            if not exp_target_sync_bucket_filter_result(va, self.opts).eligible:
                continue

            pooled_values = []
            for t_idx, trn in enumerate(getattr(va, "trns", [])):
                if t_idx >= n_trn:
                    break
                for f in va.flies:
                    if not va.noyc and f != 0:
                        continue
                    values = self._collect_fly_training_values(
                        va, t_idx=t_idx, trn=trn, f=f
                    )
                    if self.cfg.pool_trainings:
                        if values.size:
                            pooled_values.append(values)
                    elif values.size >= min_trajectories:
                        by_training[t_idx].append((self._unit_id(va, f=f), values))

            if self.cfg.pool_trainings and pooled_values:
                values = np.concatenate(pooled_values)
                if values.size >= min_trajectories:
                    pooled.append((self._unit_id(va, f=0), values))

        return [pooled] if self.cfg.pool_trainings else by_training

    def _collect_values_by_training(self) -> list[np.ndarray]:
        per_fly = self._collect_values_by_training_per_fly()
        return [
            np.concatenate([np.asarray(item[1], dtype=float) for item in panel])
            if panel
            else np.asarray([], dtype=float)
            for panel in per_fly
        ]

    def compute_histograms(self) -> dict:
        data = super().compute_histograms()
        results = [
            exp_target_sync_bucket_filter_result(va, self.opts)
            for va in self.vas
            if not getattr(va, "_skipped", False)
        ]
        data["meta"].update(
            {
                "metric": "between_reward_max_distance_hist",
                "distance_reference": "reward_circle_center",
                "min_trajectories_per_fly_window": (
                    min_between_reward_sync_bucket_trajectories(self.opts)
                ),
                "min_trajectories_applied_before_range_clipping": True,
                "exp_target_sync_bucket_filter_enabled": bool(
                    getattr(self.opts, "require_exp_target_sync_bucket", False)
                ),
                "exp_target_sync_bucket_filter_training": int(
                    getattr(
                        self.opts, "exp_target_sync_bucket_filter_training", 2
                    )
                    or 2
                ),
                "exp_target_sync_bucket_filter_sync_bucket": int(
                    getattr(
                        self.opts, "exp_target_sync_bucket_filter_sync_bucket", 5
                    )
                    or 5
                ),
                "exp_target_sync_bucket_filter_eligible_count": int(
                    sum(result.eligible for result in results)
                ),
                "exp_target_sync_bucket_filter_total_count": len(results),
            }
        )
        return data
