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

    def _collect_values_by_training(self) -> list[np.ndarray]:
        n_trn = self._n_trainings()
        all_by_trn: list[list[float]] = [[] for _ in range(n_trn)]

        exclude_wall = bool(getattr(self.opts, "com_exclude_wall_contact", False))
        min_med_mm = float(
            getattr(self.opts, "com_per_segment_min_meddist_mm", 0.0) or 0.0
        )
        warned_missing_wc = False

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

                    wc = None
                    # One "bucket" spanning the whole training:
                    fi = int(trn.start)
                    df = int(max(1, trn.stop - trn.start))
                    n_buckets = 1
                    complete = [True]
                    if exclude_wall:
                        try:
                            leaf = va.trx[f].boundary_event_stats["wall"]["all"]["edge"]

                            regions = leaf.get("boundary_contact_regions", None)

                            if regions is not None:
                                # Build a per-frame mask for just this training window [fi, fi+df)
                                # Note: df here is window length in frames (see note below)
                                wc = np.zeros(df, dtype=bool)

                                for a, b in regions:
                                    s = max(int(a), fi)
                                    e = min(int(b), fi + df)
                                    if e > s:
                                        wc[s - fi : e - fi] = True
                            else:
                                bc = leaf.get("boundary_contact", None)
                                if bc is not None:
                                    wc = np.asarray(bc[fi : fi + df], dtype=bool)
                                else:
                                    wc = None
                        except Exception:
                            wc = None
                            if not warned_missing_wc:
                                print(
                                    "[btw_rwd_com_mag] warning: can't load wall-contact data; "
                                    "--com-exclude-wall-contact will be ignored for some videos."
                                )
                                warned_missing_wc = True

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
