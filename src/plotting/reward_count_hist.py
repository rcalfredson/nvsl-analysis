# src/plotting/reward_count_hist.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.training_metric_histogram import (
    TrainingMetricHistogramConfig,
    TrainingMetricHistogramPlotter,
)


@dataclass
class RewardCountHistogramConfig(TrainingMetricHistogramConfig):
    pass


class RewardCountHistogramPlotter(TrainingMetricHistogramPlotter):
    """
    Histogram of total reward counts per fly, by training.

    Uses only experimental fly (f==0) in multi-fly recordings.
    Counts are computed within the included sync-bucket window.
    """

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: RewardCountHistogramConfig,
    ):
        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="reward_count",
            x_label="Total rewards in selected window",
            base_title="Reward totals (per fly)",
        )

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

    def _collect_values_by_training(self) -> list[np.ndarray]:
        """
        Return list length n_trainings; each element is a 1D array of per-fly reward counts.
        """
        n_trn = self._n_trainings()
        all_by_trn: list[list[float]] = [[] for _ in range(n_trn)]

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if va.trx[0].bad():
                continue

            for t_idx, trn in enumerate(getattr(va, "trns", [])):
                if t_idx >= n_trn:
                    break

                # Experimental fly only
                for f in va.flies:
                    if not va.noyc and f != 0:
                        continue

                    # Determine included sync-bucket window
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

                    end = int(fi + n_buckets * df)

                    # actual rewards (calc=False)
                    on = va._getOn(trn, False, f=f)
                    if on is None:
                        continue

                    on_win = self._filter_on_to_window(on, fi=fi, end=end)

                    # total rewards received within window
                    n_rewards = float(len(on_win))
                    all_by_trn[t_idx].append(n_rewards)

        return [np.asarray(xs, dtype=float) for xs in all_by_trn]

    def _collect_values_by_training_per_fly(self):
        """
        Return list length n_trainings; each element is a list of (unit_id, value)
        where values is a 1D float array of raw values for that unit in that training.

        For reward totals: each unit contributes exactly one value (count) per training,
        so values is a length-1 array.
        """
        n_trn = self._n_trainings()
        out: list[list[object]] = [[] for _ in range(n_trn)]

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if va.trx[0].bad():
                continue

            trns = getattr(va, "trns", [])
            for t_idx, trn in enumerate(trns[:n_trn]):

                # Experimental fly only
                for f in va.flies:
                    if not va.noyc and f != 0:
                        continue

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

                    end = int(fi + n_buckets * df)

                    on = va._getOn(trn, False, f=f)
                    if on is None:
                        continue

                    on_win = self._filter_on_to_window(on, fi=fi, end=end)
                    n_rewards = float(len(on_win))

                    unit_id = self._unit_id(va, f=f)
                    out[t_idx].append((unit_id, np.asarray([n_rewards], dtype=float)))

        return out
