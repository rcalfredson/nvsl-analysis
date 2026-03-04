from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.plotting.training_metric_scalar_bars import (
    TrainingMetricScalarBarsConfig,
    TrainingMetricScalarBarsPlotter,
)
from src.plotting.btw_rwd_return_leg_dist_collectors import ReturnLegDistPerFlyCollector


@dataclass
class ReturnLegDistTotalsConfig(TrainingMetricScalarBarsConfig):
    pass


class ReturnLegDistTotalsPlotter(
    TrainingMetricScalarBarsPlotter, ReturnLegDistPerFlyCollector
):
    def __init__(self, vas, opts, gls, customizer, cfg: ReturnLegDistTotalsConfig):
        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="btw_rwd_return_leg_dist_total",
            y_label="Mean return-leg distance per segment [mm]",
            base_title="Between-reward return-leg distance (per fly)",
        )

    def _collect_values_by_training_per_fly_scalar(self):
        return self._collect_return_leg_means_by_training_per_fly()
