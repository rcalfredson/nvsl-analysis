from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.between_reward_segment_metrics import tortuosity_metric_masked
from src.plotting.palettes import (
    NEUTRAL_DARK,
    group_metric_edge_color,
    group_metric_fill_color,
)
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.training_metric_scalar_bars import (
    TrainingMetricScalarBarsConfig,
    TrainingMetricScalarBarsPlotter,
)
from src.plotting.wall_contact_utils import build_wall_contact_mask_for_window
from src.utils.common import writeImage


@dataclass
class BetweenRewardTortuosityMeanSwarmConfig(TrainingMetricScalarBarsConfig):
    metric_palette_family: str | None = "between_reward_distance"
    metric_mode: str = "path_over_max_radius"
    segment_scope: str = "full"  # "full" or "return_leg"
    exclude_wall_contact: bool = False
    exclude_nonwalking_frames: bool = False
    exclude_reward_endpoints: bool = False
    min_walk_frames: int = 2
    min_segments_per_fly: int = 1
    min_displacement_mm: float = 0.0
    min_radius_mm: float = 0.0


class BetweenRewardTortuosityMeanSwarmPlotter(TrainingMetricScalarBarsPlotter):
    """
    One mean tortuosity value per fly/unit, plotted as per-training swarms.

    The segment iterator and masking semantics match the existing tortuosity
    histogram and max-distance-bin boxplot code.  For segment_scope="return_leg",
    the metric window starts at the segment's max-distance frame.
    """

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardTortuosityMeanSwarmConfig,
    ):
        self.segment_scope = self._normalize_segment_scope(cfg.segment_scope)
        y_label = self._y_label(cfg.metric_mode, self.segment_scope)
        base_title = self._base_title(cfg.metric_mode, self.segment_scope)
        super().__init__(
            vas=vas,
            opts=opts,
            gls=gls,
            customizer=customizer,
            cfg=cfg,
            log_tag="btw_rwd_tortuosity_mean_swarm",
            y_label=y_label,
            base_title=base_title,
        )

    @staticmethod
    def _normalize_segment_scope(scope: str | None) -> str:
        scope = str(scope or "full").strip().lower().replace("-", "_")
        if scope in {"return", "return_leg", "tail"}:
            return "return_leg"
        return "full"

    @staticmethod
    def _metric_label(mode: str | None) -> str:
        mode = str(mode or "path_over_max_radius").strip().lower()
        if mode == "straightness":
            return "straightness"
        if mode == "excess_path":
            return "excess path ratio"
        if mode == "path_over_displacement":
            return "chord tortuosity"
        return "tortuosity"

    @classmethod
    def _y_label(cls, mode: str | None, scope: str) -> str:
        metric = cls._metric_label(mode)
        if scope == "return_leg":
            return f"Mean return-leg {metric} per fly"
        return f"Mean between-reward {metric} per fly"

    @classmethod
    def _base_title(cls, mode: str | None, scope: str) -> str:
        metric = cls._metric_label(mode)
        if scope == "return_leg":
            return f"Between-reward return-leg {metric} (per fly)"
        return f"Between-reward {metric} (per fly)"

    @staticmethod
    def _unit_id(va, *, f: int) -> str:
        fly_id = getattr(va, "f", None)
        try:
            fly_id = int(fly_id) if fly_id is not None else -1
        except Exception:
            fly_id = -1
        return (
            f"{getattr(va, 'fn', 'unknown_video')}|fly_id={int(fly_id)}"
            f"|trx_idx={int(f)}"
        )

    @staticmethod
    def _build_nonwalk_mask(va, f: int, *, fi: int, n_frames: int, enabled: bool):
        if not enabled:
            return None
        traj = va.trx[f]
        walking = getattr(traj, "walking", None)
        if walking is None:
            return None
        s0 = max(0, min(int(fi), len(walking)))
        e0 = max(0, min(int(fi + n_frames), len(walking)))
        wwin = np.zeros((int(max(1, n_frames)),), dtype=bool)
        if e0 > s0:
            wseg = np.asarray(walking[s0:e0], dtype=float)
            wseg = np.where(np.isfinite(wseg), wseg, 0.0)
            wwin[: len(wseg)] = wseg > 0
        return ~wwin

    def _collect_values_by_training_per_fly_scalar(
        self,
    ) -> list[list[tuple[str, float]]]:
        n_trn = self._n_trainings()
        out: list[list[tuple[str, float]]] = [[] for _ in range(n_trn)]
        warned_missing_wc = [False]
        min_walk_frames = int(max(2, getattr(self.cfg, "min_walk_frames", 2) or 2))
        min_segments = int(max(1, getattr(self.cfg, "min_segments_per_fly", 1) or 1))
        mode = str(
            getattr(self.cfg, "metric_mode", "path_over_max_radius")
            or "path_over_max_radius"
        )
        needs_max_frame = self.segment_scope == "return_leg"

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if getattr(va, "trx", None) is None or len(va.trx) == 0:
                continue
            if va.trx[0].bad():
                continue

            for t_idx, trn in enumerate(getattr(va, "trns", [])[:n_trn]):
                for f in va.flies:
                    if not va.noyc and f != 0:
                        continue

                    fi, df, n_buckets, complete = sync_bucket_window(
                        va,
                        trn,
                        t_idx=t_idx,
                        f=f,
                        skip_first=int(
                            max(0, getattr(self.cfg, "skip_first_sync_buckets", 0) or 0)
                        ),
                        keep_first=int(
                            max(0, getattr(self.cfg, "keep_first_sync_buckets", 0) or 0)
                        ),
                        use_exclusion_mask=False,
                    )
                    if n_buckets <= 0:
                        continue

                    traj = va.trx[f]
                    px_per_mm = float(traj.pxPerMmFloor * va.xf.fctr)
                    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
                        continue

                    n_frames = int(max(1, n_buckets * df))
                    wc = build_wall_contact_mask_for_window(
                        va,
                        f,
                        fi=fi,
                        n_frames=n_frames,
                        enabled=bool(getattr(self.cfg, "exclude_wall_contact", False)),
                        warned_missing_wc=warned_missing_wc,
                        log_tag=self.log_tag,
                    )
                    nonwalk_mask = self._build_nonwalk_mask(
                        va,
                        f,
                        fi=fi,
                        n_frames=n_frames,
                        enabled=bool(
                            getattr(self.cfg, "exclude_nonwalking_frames", False)
                        ),
                    )

                    try:
                        cx, cy, _ = trn.circles(f)[0]
                        reward_center_xy = (float(cx), float(cy))
                    except Exception:
                        reward_center_xy = None

                    vals: list[float] = []
                    for seg in va._iter_between_reward_segment_com(
                        trn,
                        f,
                        fi=fi,
                        df=df,
                        n_buckets=n_buckets,
                        complete=complete,
                        relative_to_reward=True,
                        per_segment_min_meddist_mm=0.0,
                        exclude_wall=bool(
                            getattr(self.cfg, "exclude_wall_contact", False)
                        ),
                        wc=wc,
                        exclude_nonwalk=bool(
                            getattr(self.cfg, "exclude_nonwalking_frames", False)
                        ),
                        nonwalk_mask=nonwalk_mask,
                        min_walk_frames=min_walk_frames,
                        exclude_reward_endpoints=bool(
                            getattr(self.cfg, "exclude_reward_endpoints", False)
                        ),
                        dist_stats=("max",) if needs_max_frame else (),
                        debug=False,
                        yield_skips=False,
                    ):
                        endpoint_offset = (
                            1
                            if bool(
                                getattr(self.cfg, "exclude_reward_endpoints", False)
                            )
                            else 0
                        )
                        s = int(seg.s) + endpoint_offset
                        e = int(seg.e) - endpoint_offset
                        if e <= s:
                            continue

                        if needs_max_frame:
                            max_i = getattr(seg, "max_d_i", None)
                            if max_i is None:
                                continue
                            s_metric = max(s, int(max_i))
                        else:
                            s_metric = s
                        if e <= s_metric:
                            continue

                        val = tortuosity_metric_masked(
                            traj=traj,
                            s=s_metric,
                            e=e,
                            fi=fi,
                            nonwalk_mask=nonwalk_mask,
                            exclude_nonwalk=bool(
                                getattr(
                                    self.cfg, "exclude_nonwalking_frames", False
                                )
                            ),
                            px_per_mm=px_per_mm,
                            mode=mode,
                            reward_center_xy=reward_center_xy,
                            min_keep_frames=min_walk_frames,
                            min_displacement_mm=float(
                                getattr(self.cfg, "min_displacement_mm", 0.0) or 0.0
                            ),
                            min_radius_mm=float(
                                getattr(self.cfg, "min_radius_mm", 0.0) or 0.0
                            ),
                        )
                        if np.isfinite(val):
                            vals.append(float(val))

                    if len(vals) < min_segments:
                        continue
                    out[t_idx].append(
                        (
                            self._unit_id(va, f=f),
                            float(np.nanmean(np.asarray(vals, dtype=float))),
                        )
                    )
        return out

    def compute_scalar_panels(self) -> dict:
        data = super().compute_scalar_panels()
        meta = dict(data.get("meta") or {})
        meta.update(
            {
                "metric": "between_reward_tortuosity_mean",
                "metric_mode": str(
                    getattr(self.cfg, "metric_mode", "path_over_max_radius")
                    or "path_over_max_radius"
                ),
                "segment_scope": str(self.segment_scope),
                "exclude_wall_contact": bool(
                    getattr(self.cfg, "exclude_wall_contact", False)
                ),
                "exclude_nonwalking_frames": bool(
                    getattr(self.cfg, "exclude_nonwalking_frames", False)
                ),
                "exclude_reward_endpoints": bool(
                    getattr(self.cfg, "exclude_reward_endpoints", False)
                ),
                "min_walk_frames": int(getattr(self.cfg, "min_walk_frames", 2) or 2),
                "min_segments_per_fly": int(
                    getattr(self.cfg, "min_segments_per_fly", 1) or 1
                ),
                "min_displacement_mm": float(
                    getattr(self.cfg, "min_displacement_mm", 0.0) or 0.0
                ),
                "min_radius_mm": float(
                    getattr(self.cfg, "min_radius_mm", 0.0) or 0.0
                ),
            }
        )
        data["meta"] = meta
        return data

    def plot_swarms(self) -> None:
        data = self.compute_scalar_panels()
        labels = data["panel_labels"]
        if not labels:
            print(f"[{self.log_tag}] no data found; skipping plot.")
            return

        customizer = self.customizer
        fig, ax = plt.subplots(1, 1, figsize=(max(5.5, 1.05 * len(labels)), 4.0))
        x = np.arange(len(labels), dtype=float)
        fill = group_metric_fill_color(0, self.cfg.metric_palette_family)
        edge = group_metric_edge_color(0, self.cfg.metric_palette_family)
        rng = np.random.default_rng(0)

        for i in range(len(labels)):
            vals = np.asarray(data["per_unit_values_panel"][i], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.14, 0.14, size=vals.size)
            ax.scatter(
                np.full(vals.size, x[i]) + jitter,
                vals,
                s=22,
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.7,
                alpha=0.78,
                zorder=3,
            )
            mean = float(np.nanmean(vals))
            ax.plot(
                [x[i] - 0.22, x[i] + 0.22],
                [mean, mean],
                color=NEUTRAL_DARK,
                linewidth=1.4,
                zorder=4,
            )
            ax.text(
                x[i],
                mean,
                f"n={vals.size}",
                ha="center",
                va="bottom",
                fontsize=max(7, min(float(customizer.in_plot_font_size) - 5.0, 9.0)),
                color="0.25",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(self.y_label)
        ax.set_ylim(bottom=0)
        if self.cfg.ymax is not None:
            ax.set_ylim(top=float(self.cfg.ymax))
        if self.cfg.show_suptitle:
            title = self.base_title
            if self.cfg.subset_label:
                title = f"{title}\n{self.cfg.subset_label}"
            ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        writeImage(self.cfg.out_file, format=self.opts.imageFormat)
        plt.close(fig)
        print(f"[{self.log_tag}] wrote {self.cfg.out_file}")
