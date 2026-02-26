# src/plotting/between_reward_distance_hist.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.training_metric_histogram import (
    TrainingMetricHistogramConfig,
    TrainingMetricHistogramPlotter,
)


@dataclass
class BetweenRewardDistanceHistogramConfig(TrainingMetricHistogramConfig):
    # If True: drop any between-reward segment whose frame range overlaps any
    # wall-contact region (segment-wise exclusion).
    exclude_wall_contact: bool = False


class BetweenRewardDistanceHistogramPlotter(TrainingMetricHistogramPlotter):
    """
    Aggregate between-reward distances across VideoAnalysis instances and
    plot histograms separated by training.

    Uses only experimental flies (f == 0 in multi-fly recordings).
    """

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardDistanceHistogramConfig,
    ):
        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="btw_rwd_dists",
            x_label="Distance traveled between rewards (mm)",
            base_title="Lengths of between-rewards trajectories",
        )

    # ---------- helpers ----------

    @staticmethod
    def _slice_start_stop(sl) -> tuple[int, int]:
        a = 0 if getattr(sl, "start", None) is None else int(sl.start)
        b = 0 if getattr(sl, "stop", None) is None else int(sl.stop)
        return a, b

    @classmethod
    def _any_overlap_with_wall_regions(
        cls, wall_regions: Optional[Sequence[slice]], s: int, e: int
    ) -> bool:
        """
        Return True if ANY wall-contact region overlaps [s, e)
        """
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
        # Keep only reward frames in [fi, end)
        out = []
        for x in on:
            try:
                t = int(x)
            except Exception:
                continue
            if fi <= t < end:
                out.append(t)
        return out

    # ---------- data collection ----------

    def _collect_values_by_training(self) -> list[np.ndarray]:
        """Return a list of length n_trainings, each a 1D array of distances."""
        n_trn = self._n_trainings()
        all_by_trn: list[list[float]] = [[] for _ in range(n_trn)]

        for va in self.vas:
            # Skip VAs that were skipped or have bad main trajectory
            if getattr(va, "_skipped", False):
                continue
            if va.trx[0].bad():
                continue

            for t_idx, trn in enumerate(getattr(va, "trns", [])):
                if t_idx >= n_trn:
                    break
                # Loop over flies, but only use experimental flies
                for f in va.flies:
                    if not va.noyc and f != 0:
                        # multi-fly (exp + yoked); keep only experimental fly
                        continue

                    trj = va.trx[f]
                    wall_regions = None
                    if self.cfg.exclude_wall_contact:
                        try:
                            wall_regions = trj.boundary_event_stats["wall"]["all"][
                                "edge"
                            ]["boundary_contact_regions"]
                        except (KeyError, TypeError, AttributeError):
                            wall_regions = None

                    # --- Determine included sync-bucket window for this training/fly
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
                        continue

                    # End frame of included window (exclusive)
                    end = int(fi + n_buckets * df)

                    # False → use actual rewards, not "calculated" rewards
                    on = va._getOn(trn, False, f=f)
                    if on is None or len(on) < 2:
                        continue

                    on_win = self._filter_on_to_window(on, fi=fi, end=end)
                    if len(on_win) < 2:
                        continue

                    dists_px = va._distTrav(f, on_win)
                    if not dists_px:
                        continue

                    px_per_mm = trj.pxPerMmFloor * va.xf.fctr
                    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
                        continue
                    dists_mm = np.asarray(dists_px, dtype=float) / float(px_per_mm)
                    dists_mm = dists_mm[np.isfinite(dists_mm)]
                    if dists_mm.size == 0:
                        continue

                    n_seg = min(dists_mm.size, len(on) - 1)
                    if n_seg <= 0:
                        continue
                    for i in range(n_seg):
                        s = int(on_win[i])
                        e = int(on_win[i + 1])
                        if (
                            self.cfg.exclude_wall_contact
                            and self._any_overlap_with_wall_regions(wall_regions, s, e)
                        ):
                            continue
                        all_by_trn[t_idx].append(float(dists_mm[i]))

        return [np.asarray(xs, dtype=float) for xs in all_by_trn]

    def _collect_values_by_training_per_fly(self) -> list[list[tuple[str, np.ndarray]]]:
        """
        Per-fly collection for histogram aggregation.

        Returns:
          - if cfg.pool_trainings is False:
              list of length n_trainings; each element is a list of 1D arrays,
              one per fly (per VideoAnalysis unit, exp fly only) for that training.
          - if cfg.pool_trainings is True:
              a single-panel list [pooled_panel], where pooled_panel is a list of
              1D arrays, one per fly, formed by concatenating that fly's values
              across all trainings. (This preserves "fly is the unit" weighting.)

        NOTE: "per fly" here corresponds to one VideoAnalysis unit (and exp fly
        only in multi-fly recordings).
        """
        n_trn = self._n_trainings()

        # Panel structure depends on pool_trainings
        if self.cfg.pool_trainings:
            pooled_panel: list[np.ndarray] = []
        else:
            all_by_trn: list[list[np.ndarray]] = [[] for _ in range(n_trn)]

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if va.trx[0].bad():
                continue

            # In pooled mode, accumulate across trainings for this VA/fly
            pooled_for_this_fly: list[np.ndarray] = []

            for t_idx, trn in enumerate(getattr(va, "trns", [])):
                if t_idx >= n_trn:
                    break

                for f in va.flies:
                    if not va.noyc and f != 0:
                        continue

                    trj = va.trx[f]

                    # --- Determine included sync-bucket window for this training/fly
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
                        continue

                    # End frame of included window (exclusive)
                    end = int(fi + n_buckets * df)

                    on = va._getOn(trn, False, f=f)
                    if on is None or len(on) < 2:
                        continue

                    on_win = self._filter_on_to_window(on, fi=fi, end=end)
                    if len(on_win) < 2:
                        continue

                    wall_regions = None
                    if self.cfg.exclude_wall_contact:
                        try:
                            wall_regions = trj.boundary_event_stats["wall"]["all"][
                                "edge"
                            ]["boundary_contact_regions"]
                        except (KeyError, TypeError, AttributeError):
                            wall_regions = None

                    dists_px = va._distTrav(f, on_win)
                    if not dists_px:
                        continue

                    px_per_mm = float(trj.pxPerMmFloor * va.xf.fctr)
                    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
                        continue

                    dists_mm_all = np.asarray(dists_px, dtype=float) / float(px_per_mm)
                    dists_mm_all = dists_mm_all[np.isfinite(dists_mm_all)]
                    if dists_mm_all.size == 0:
                        continue

                    n_seg = min(dists_mm_all.size, len(on_win) - 1)
                    if n_seg <= 0:
                        continue

                    kept: list[float] = []
                    for i in range(n_seg):
                        s = int(on_win[i])
                        e = int(on_win[i + 1])
                        if (
                            self.cfg.exclude_wall_contact
                            and self._any_overlap_with_wall_regions(wall_regions, s, e)
                        ):
                            continue
                        kept.append(float(dists_mm_all[i]))
                    if not kept:
                        continue

                    dists_mm = np.asarray(kept, dtype=float)

                    if self.cfg.pool_trainings:
                        pooled_for_this_fly.append(dists_mm.astype(float, copy=False))
                    else:
                        unit_id = self._unit_id(va, f=f)
                        all_by_trn[t_idx].append(
                            (unit_id, dists_mm.astype(float, copy=False))
                        )
            if self.cfg.pool_trainings:
                if pooled_for_this_fly:
                    pooled_panel.append(np.concatenate(pooled_for_this_fly, axis=0))

        if self.cfg.pool_trainings:
            return [pooled_panel]
        return all_by_trn
