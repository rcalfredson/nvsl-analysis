# src/plotting/between_reward_distance_hist.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.training_metric_histogram import (
    TrainingMetricHistogramConfig,
    TrainingMetricHistogramPlotter,
)


@dataclass
class BetweenRewardDistanceHistogramConfig(TrainingMetricHistogramConfig):
    pass


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
            x_label="distance between rewards (mm)",
            base_title="Between-reward distances (experimental flies only)",
        )

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

                    # False → use actual rewards, not "calculated" rewards
                    on = va._getOn(trn, False, f=f)
                    if on is None or len(on) < 2:
                        continue

                    dists_px = va._distTrav(f, on)
                    if not dists_px:
                        continue

                    trj = va.trx[f]
                    px_per_mm = trj.pxPerMmFloor * va.xf.fctr
                    dists_mm = np.array(dists_px, dtype=float) / px_per_mm

                    all_by_trn[t_idx].extend(dists_mm)

        return [np.asarray(xs, dtype=float) for xs in all_by_trn]
