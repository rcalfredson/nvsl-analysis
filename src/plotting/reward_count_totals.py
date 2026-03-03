from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.plotting.training_metric_scalar_bars import (
    TrainingMetricScalarBarsConfig,
    TrainingMetricScalarBarsPlotter,
)
from src.plotting.reward_count_collectors import RewardCountPerFlyCollector


@dataclass
class RewardCountTotalsConfig(TrainingMetricScalarBarsConfig):
    pass


class RewardCountTotalsPlotter(
    TrainingMetricScalarBarsPlotter, RewardCountPerFlyCollector
):
    def __init__(self, vas, opts, gls, customizer, cfg: RewardCountTotalsConfig):
        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="reward_count_total",
            y_label="Total rewards per fly",
            base_title="Total reward counts per fly",
        )

    def _collect_values_by_training_per_fly_scalar(self):
        return self._collect_reward_totals_by_training_per_fly()
