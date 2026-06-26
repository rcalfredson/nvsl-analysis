from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Sequence

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
    sli_values: Sequence[float] | None = None
    sli_exp_values: Sequence[float] | None = None
    sli_ctrl_values: Sequence[float] | None = None


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
        va, trn, *, f: int, ctrl: bool, start: int, stop: int
    ) -> float:
        try:
            fi_count = util.none2val(va._idxSync(RI_START, trn, start, stop), stop)
            return float(
                va._countOn(
                    max(start, int(fi_count)),
                    stop,
                    calc=True,
                    ctrl=bool(ctrl),
                    f=f,
                )
            )
        except Exception:
            try:
                on = np.asarray(
                    va._getOn(trn, calc=True, ctrl=bool(ctrl), f=f),
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
        self._rpd_total_diagnostics_by_training = [dict() for _ in range(n_trn)]
        skip_first, keep_first = self._effective_sync_bucket_window()

        sli_values = getattr(self.cfg, "sli_values", None)
        sli_exp_values = getattr(self.cfg, "sli_exp_values", None)
        sli_ctrl_values = getattr(self.cfg, "sli_ctrl_values", None)

        for va_idx, va in enumerate(self.vas):
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
                    va, trn, f=f, ctrl=False, start=start, stop=stop
                )
                exp_control_rewards = self._count_calc_rewards_in_window(
                    va, trn, f=0, ctrl=True, start=start, stop=stop
                )
                yok_calc_rewards = self._count_calc_rewards_in_window(
                    va, trn, f=1, ctrl=False, start=start, stop=stop
                )
                yok_control_rewards = self._count_calc_rewards_in_window(
                    va, trn, f=1, ctrl=True, start=start, stop=stop
                )
                dist_m = self._distance_traveled_m(va, f=f, start=start, stop=stop)
                if (
                    not np.isfinite(n_rewards)
                    or not np.isfinite(dist_m)
                    or dist_m <= 0
                ):
                    continue

                unit_id = self._unit_id(va, f=f)
                value = float(n_rewards / dist_m)
                out[t_idx].append((unit_id, value))
                self._rpd_total_diagnostics_by_training[t_idx][unit_id] = {
                    "rewards": float(n_rewards),
                    "exp_calc_rewards": float(n_rewards),
                    "exp_control_rewards": float(exp_control_rewards),
                    "yok_calc_rewards": float(yok_calc_rewards),
                    "yok_control_rewards": float(yok_control_rewards),
                    "distance_m": float(dist_m),
                    "start_frame": int(start),
                    "stop_frame": int(stop),
                    "n_buckets": int(n_buckets),
                    "value": value,
                    "sli": (
                        float(sli_values[va_idx])
                        if sli_values is not None and va_idx < len(sli_values)
                        else np.nan
                    ),
                    "sli_exp": (
                        float(sli_exp_values[va_idx])
                        if (
                            sli_exp_values is not None
                            and va_idx < len(sli_exp_values)
                        )
                        else np.nan
                    ),
                    "sli_ctrl": (
                        float(sli_ctrl_values[va_idx])
                        if (
                            sli_ctrl_values is not None
                            and va_idx < len(sli_ctrl_values)
                        )
                        else np.nan
                    ),
                }

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

    def compute_scalar_panels(self) -> dict:
        data = super().compute_scalar_panels()
        diagnostics_by_training = getattr(
            self,
            "_rpd_total_diagnostics_by_training",
            None,
        )
        if not diagnostics_by_training or not data.get("panel_labels"):
            return data

        training_info = data.get("meta", {}).get("training_selection", {})
        effective = training_info.get("trainings_effective")
        if effective:
            training_indices = [int(x) - 1 for x in effective]
        else:
            training_indices = list(range(len(data["panel_labels"])))

        fields = {
            "rpd_total_rewards_panel": [],
            "rpd_total_distance_m_panel": [],
            "rpd_total_start_frame_panel": [],
            "rpd_total_stop_frame_panel": [],
            "rpd_total_n_buckets_panel": [],
            "rpd_total_exp_calc_rewards_panel": [],
            "rpd_total_exp_control_rewards_panel": [],
            "rpd_total_yok_calc_rewards_panel": [],
            "rpd_total_yok_control_rewards_panel": [],
            "rpd_total_sli_panel": [],
            "rpd_total_sli_exp_panel": [],
            "rpd_total_sli_ctrl_panel": [],
        }
        source_keys = {
            "rpd_total_rewards_panel": "rewards",
            "rpd_total_distance_m_panel": "distance_m",
            "rpd_total_start_frame_panel": "start_frame",
            "rpd_total_stop_frame_panel": "stop_frame",
            "rpd_total_n_buckets_panel": "n_buckets",
            "rpd_total_exp_calc_rewards_panel": "exp_calc_rewards",
            "rpd_total_exp_control_rewards_panel": "exp_control_rewards",
            "rpd_total_yok_calc_rewards_panel": "yok_calc_rewards",
            "rpd_total_yok_control_rewards_panel": "yok_control_rewards",
            "rpd_total_sli_panel": "sli",
            "rpd_total_sli_exp_panel": "sli_exp",
            "rpd_total_sli_ctrl_panel": "sli_ctrl",
        }

        for panel_idx, ids in enumerate(data["per_unit_ids_panel"]):
            t_idx = (
                training_indices[panel_idx]
                if panel_idx < len(training_indices)
                else panel_idx
            )
            diag_by_id = (
                diagnostics_by_training[t_idx]
                if 0 <= t_idx < len(diagnostics_by_training)
                else {}
            )
            for field, source_key in source_keys.items():
                values = []
                for uid in ids:
                    diag = diag_by_id.get(str(uid), {})
                    values.append(float(diag.get(source_key, np.nan)))
                fields[field].append(np.asarray(values, dtype=float))

        for field, values in fields.items():
            data[field] = np.asarray(values, dtype=object)
        return data

    def export_npz(self, out_npz: str) -> None:
        data = self.compute_scalar_panels()
        if not data["panel_labels"]:
            print(f"[{self.log_tag}] no data found; skipping export.")
            return
        np.savez_compressed(
            out_npz,
            panel_labels=np.asarray(data["panel_labels"], dtype=object),
            per_unit_values_panel=data["per_unit_values_panel"],
            per_unit_ids_panel=data["per_unit_ids_panel"],
            mean=data["mean"],
            ci_lo=data["ci_lo"],
            ci_hi=data["ci_hi"],
            n_units_panel=data["n_units_panel"],
            meta_json=json.dumps(data["meta"], sort_keys=True),
            rpd_total_rewards_panel=data.get(
                "rpd_total_rewards_panel",
                np.asarray([], dtype=object),
            ),
            rpd_total_distance_m_panel=data.get(
                "rpd_total_distance_m_panel",
                np.asarray([], dtype=object),
            ),
            rpd_total_start_frame_panel=data.get(
                "rpd_total_start_frame_panel",
                np.asarray([], dtype=object),
            ),
            rpd_total_stop_frame_panel=data.get(
                "rpd_total_stop_frame_panel",
                np.asarray([], dtype=object),
            ),
            rpd_total_n_buckets_panel=data.get(
                "rpd_total_n_buckets_panel",
                np.asarray([], dtype=object),
            ),
            rpd_total_exp_calc_rewards_panel=data.get(
                "rpd_total_exp_calc_rewards_panel",
                np.asarray([], dtype=object),
            ),
            rpd_total_exp_control_rewards_panel=data.get(
                "rpd_total_exp_control_rewards_panel",
                np.asarray([], dtype=object),
            ),
            rpd_total_yok_calc_rewards_panel=data.get(
                "rpd_total_yok_calc_rewards_panel",
                np.asarray([], dtype=object),
            ),
            rpd_total_yok_control_rewards_panel=data.get(
                "rpd_total_yok_control_rewards_panel",
                np.asarray([], dtype=object),
            ),
            rpd_total_sli_panel=data.get(
                "rpd_total_sli_panel",
                np.asarray([], dtype=object),
            ),
            rpd_total_sli_exp_panel=data.get(
                "rpd_total_sli_exp_panel",
                np.asarray([], dtype=object),
            ),
            rpd_total_sli_ctrl_panel=data.get(
                "rpd_total_sli_ctrl_panel",
                np.asarray([], dtype=object),
            ),
        )
        print(f"[{self.log_tag}] wrote scalar export {out_npz}")
