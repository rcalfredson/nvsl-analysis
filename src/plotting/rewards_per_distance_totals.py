from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.training_metric_scalar_bars import (
    TrainingMetricScalarBarsConfig,
    TrainingMetricScalarBarsPlotter,
)
from src.utils import util
from src.utils.constants import RI_START


@dataclass
class RewardsPerDistanceTotalsConfig(TrainingMetricScalarBarsConfig):
    pass


class RewardsPerDistancePerFlyCollector:
    """
    Collect one rewards-per-distance scalar per experimental fly/training.

    The scalar is denominator-aware over the selected sync-bucket window:
      total calculated rewards / total distance traveled
    rather than an average of per-bucket ratios.
    """

    @staticmethod
    def _unit_id(va, *, f: int) -> str:
        video_fn = getattr(va, "fn", None)
        base = os.path.basename(str(video_fn)) if video_fn else "unknown_video"
        va_id = int(getattr(va, "f", 0) or 0)
        return f"{base}|va_tag={va_id}|trx_idx={int(f)}"

    def _effective_keep_first_sync_buckets(self) -> int:
        ckeep = int(getattr(self.cfg, "keep_first_sync_buckets", 0) or 0)
        return 0 if ckeep < 0 else ckeep

    def _effective_skip_first_sync_buckets(self) -> int:
        cskip = int(getattr(self.cfg, "skip_first_sync_buckets", 0) or 0)
        return 0 if cskip < 0 else cskip

    def _effective_sync_bucket_window(self) -> tuple[int, int]:
        return (
            self._effective_skip_first_sync_buckets(),
            self._effective_keep_first_sync_buckets(),
        )

    @staticmethod
    def _count_calc_rewards_in_window(
        va, trn, *, f: int, start: int, stop: int
    ) -> float:
        try:
            fi_count = util.none2val(va._idxSync(RI_START, trn, start, stop), stop)
            return float(
                va._countOn(
                    max(start, int(fi_count)),
                    stop,
                    calc=True,
                    ctrl=False,
                    f=f,
                )
            )
        except Exception:
            try:
                on = np.asarray(
                    va._getOn(trn, calc=True, ctrl=False, f=f),
                    dtype=float,
                )
            except Exception:
                return np.nan
            on = on[np.isfinite(on)]
            return float(np.count_nonzero((on >= start) & (on < stop)))

    @staticmethod
    def _distance_traveled_m(va, *, f: int, start: int, stop: int) -> float:
        try:
            traj = va.trx[f]
            px_per_mm = float(va.xf.fctr) * float(va.ct.pxPerMmFloor())
        except Exception:
            return np.nan
        if not np.isfinite(px_per_mm) or px_per_mm <= 0:
            return np.nan
        if stop <= start:
            return np.nan
        try:
            dist_px = float(traj.distTrav(int(start), int(stop)))
        except Exception:
            return np.nan
        if not np.isfinite(dist_px):
            return np.nan
        return float(dist_px / px_per_mm / 1000.0)

    @staticmethod
    def _all_selected_buckets_valid(
        va,
        *,
        f: int,
        t_idx: int,
        skip_first: int,
        n_buckets: int,
    ) -> bool:
        for j in range(int(n_buckets)):
            b_idx = int(skip_first) + j
            try:
                if va.is_excluded_pair(f, t_idx, b_idx):
                    return False
            except Exception:
                return False
        return True

    def _collect_reward_distance_totals_by_training_per_fly(self):
        n_trn = self._n_trainings()
        out: list[list[tuple[str, float]]] = [[] for _ in range(n_trn)]
        skip_first, keep_first = self._effective_sync_bucket_window()

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if getattr(va, "trx", None) is None or len(va.trx) == 0:
                continue
            if va.trx[0].bad():
                continue

            trns = getattr(va, "trns", [])
            for t_idx, trn in enumerate(trns[:n_trn]):
                f = 0
                fi, df, n_buckets, _complete = sync_bucket_window(
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
                if not self._all_selected_buckets_valid(
                    va,
                    f=f,
                    t_idx=t_idx,
                    skip_first=skip_first,
                    n_buckets=n_buckets,
                ):
                    continue

                start = int(fi)
                stop = int(fi + n_buckets * df)
                n_rewards = self._count_calc_rewards_in_window(
                    va, trn, f=f, start=start, stop=stop
                )
                dist_m = self._distance_traveled_m(va, f=f, start=start, stop=stop)
                if (
                    not np.isfinite(n_rewards)
                    or not np.isfinite(dist_m)
                    or dist_m <= 0
                ):
                    continue

                unit_id = self._unit_id(va, f=f)
                out[t_idx].append((unit_id, float(n_rewards / dist_m)))

        return out


class RewardsPerDistanceTotalsPlotter(
    TrainingMetricScalarBarsPlotter, RewardsPerDistancePerFlyCollector
):
    def __init__(self, vas, opts, gls, customizer, cfg: RewardsPerDistanceTotalsConfig):
        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="rpd_total",
            y_label="Rewards per\ndistance traveled (m$^{-1}$)",
            base_title="Rewards per distance traveled",
        )

    def _collect_values_by_training_per_fly_scalar(self):
        return self._collect_reward_distance_totals_by_training_per_fly()
