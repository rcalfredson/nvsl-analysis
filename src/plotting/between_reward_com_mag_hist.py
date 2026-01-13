from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.training_metric_histogram import (
    TrainingMetricHistogramConfig,
    TrainingMetricHistogramPlotter,
)
from src.plotting.wall_contact_utils import build_wall_contact_mask_for_window


@dataclass
class BetweenRewardCOMMagHistogramConfig(TrainingMetricHistogramConfig):
    pass


class BetweenRewardCOMMagHistogramPlotter(TrainingMetricHistogramPlotter):
    """
    Histogram of per-between-reward-segment COM magnitudes (mm), pooled per training.
    Experimental fly only in multi-fly recordings (f==0 when exp+yoked).

    Uses VideoAnalysis._iter_between_reward_segment_com(...) with a single
    bucket spanning the whole training to reuse the existing segment logic.
    """

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardCOMMagHistogramConfig,
    ):
        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="btw_rwd_com_mag",
            x_label="COM distance to\nreward circle [mm]",
            base_title="Between-reward COM magnitude (experimental flies only)",
        )

    def _wall_contact_mask_for_training(
        self,
        va: "VideoAnalysis",
        f: int,
        trn,
        *,
        exclude_wall: bool,
        warned_missing_wc: list[bool],
    ) -> Optional[np.ndarray]:
        """
        Return a per-frame wall-contact boolean mask for the training window [trn.start, trn.stop),
        or None if unavailable / not requested.

        warned_missing_wc is a 1-item list used as a mutable "warn once" flag.
        """
        if not exclude_wall:
            return None
        fi = int(trn.start)
        df = int(max(1, trn.stop - trn.start))
        return build_wall_contact_mask_for_window(
            va,
            f,
            fi=fi,
            n_frames=df,
            enabled=True,
            warned_missing_wc=warned_missing_wc,
            log_tag="btw_rwd_com_mag",
        )

    def _collect_values_by_training(self) -> list[np.ndarray]:
        n_trn = self._n_trainings()
        all_by_trn: list[list[float]] = [[] for _ in range(n_trn)]

        exclude_wall = bool(getattr(self.opts, "com_exclude_wall_contact", False))
        min_med_mm = float(
            getattr(self.opts, "com_per_segment_min_meddist_mm", 0.0) or 0.0
        )
        warned_missing_wc = [False]

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if va.trx[0].bad():
                continue

            for t_idx, trn in enumerate(getattr(va, "trns", [])):
                if t_idx >= n_trn:
                    break
                for f in va.flies:
                    if not va.noyc and f != 0:
                        continue

                    # One "bucket" spanning the whole training:
                    fi = int(trn.start)
                    df = int(max(1, trn.stop - trn.start))
                    n_buckets = 1
                    complete = [True]
                    wc = self._wall_contact_mask_for_training(
                        va,
                        f,
                        trn,
                        exclude_wall=exclude_wall,
                        warned_missing_wc=warned_missing_wc,
                    )

                    for seg in va._iter_between_reward_segment_com(
                        trn,
                        f,
                        fi=fi,
                        df=df,
                        n_buckets=n_buckets,
                        complete=complete,
                        relative_to_reward=True,
                        per_segment_min_meddist_mm=min_med_mm,
                        exclude_wall=exclude_wall,
                        wc=wc,
                        debug=False,
                        yield_skips=False,
                    ):
                        all_by_trn[t_idx].append(float(seg.mag_mm))

        return [np.asarray(xs, dtype=float) for xs in all_by_trn]

    def _collect_values_by_training_per_fly(self) -> list[list[np.ndarray]]:
        """
        Per-fly collection for histogram aggregation (fly/video is the unit).

        Returns:
          - if cfg.pool_trainings is False:
              list of length n_trainings; each element is a list of 1D arrays,
              one per fly (per VideoAnalysis unit, exp fly only) for that training.
          - if cfg.pool_trainings is True:
              a single-panel list [pooled_panel], where pooled_panel is a list of
              1D arrays, one per fly, formed by concatenating that fly's values
              across all trainings.
        """
        n_trn = self._n_trainings()

        exclude_wall = bool(getattr(self.opts, "com_exclude_wall_contact", False))
        min_med_mm = float(
            getattr(self.opts, "com_per_segment_min_meddist_mm", 0.0) or 0.0
        )
        warned_missing_wc = [False]

        if self.cfg.pool_trainings:
            pooled_panel: list[np.ndarray] = []
        else:
            all_by_trn: list[list[np.ndarray]] = [[] for _ in range(n_trn)]

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if va.trx[0].bad():
                continue

            pooled_for_this_fly: list[np.ndarray] = []

            for t_idx, trn in enumerate(getattr(va, "trns", [])):
                if t_idx >= n_trn:
                    break

                for f in va.flies:
                    if not va.noyc and f != 0:
                        continue

                    # One "bucket" spanning the whole training:
                    fi = int(trn.start)
                    df = int(max(1, trn.stop - trn.start))
                    n_buckets = 1
                    complete = [True]

                    wc = self._wall_contact_mask_for_training(
                        va,
                        f,
                        trn,
                        exclude_wall=exclude_wall,
                        warned_missing_wc=warned_missing_wc,
                    )

                    mags: list[float] = []
                    for seg in va._iter_between_reward_segment_com(
                        trn,
                        f,
                        fi=fi,
                        df=df,
                        n_buckets=n_buckets,
                        complete=complete,
                        relative_to_reward=True,
                        per_segment_min_meddist_mm=min_med_mm,
                        exclude_wall=exclude_wall,
                        wc=wc,
                        debug=False,
                        yield_skips=False,
                    ):
                        mags.append(float(seg.mag_mm))

                    if not mags:
                        continue
                    arr = np.asarray(mags, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        continue

                    if self.cfg.pool_trainings:
                        pooled_for_this_fly.append(arr.astype(float, copy=False))
                    else:
                        all_by_trn[t_idx].append(arr.astype(float, copy=False))

            if self.cfg.pool_trainings and pooled_for_this_fly:
                pooled_panel.append(np.concatenate(pooled_for_this_fly, axis=0))

        if self.cfg.pool_trainings:
            return [pooled_panel]
        return all_by_trn
