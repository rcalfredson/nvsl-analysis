from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence
import json

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import writeImage
from src.utils.util import meanConfInt


@dataclass
class TrainingMetricScalarBarsConfig:
    out_file: str
    pool_trainings: bool = False
    trainings: Sequence[int] | None = None  # 1-based
    subset_label: str | None = None
    ymax: float | None = None

    # Sync-bucket windowing knobs (kept here for meta + shared call-site symmetry)
    skip_first_sync_buckets: int = 0
    keep_first_sync_buckets: int = 0

    # Export/plot extras
    ci: bool = False
    ci_conf: float = 0.95
    show_points: bool = False
    show_suptitle: bool = False


class TrainingMetricScalarBarsPlotter:
    """
    Generic per-training bar plotter for one scalar value per fly/unit.
    Subclasses implement _collect_values_by_training_per_fly_scalar().
    """

    def __init__(
        self,
        vas,
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: TrainingMetricScalarBarsConfig,
        *,
        log_tag: str,
        y_label: str,
        base_title: str,
    ):
        self.vas = vas
        self.opts = opts
        self.gls = gls
        self.customizer = customizer
        self.cfg = cfg
        self.log_tag = log_tag
        self.y_label = y_label
        self.base_title = base_title

    def _n_trainings(self) -> int:
        return max((len(getattr(va, "trns", [])) for va in self.vas), default=0)

    def _training_labels(self, n_trn: int) -> list[str]:
        labels = []
        for i in range(n_trn):
            names = []
            for va in self.vas:
                trns = getattr(va, "trns", None)
                if not trns or i >= len(trns):
                    continue
                try:
                    names.append(trns[i].name())
                except Exception:
                    continue
            labels.append(
                max(set(names), key=names.count) if names else f"training {i+1}"
            )
        return labels

    def _selected_training_indices(self, n_panels: int):
        t = self.cfg.trainings
        if not t or self.cfg.pool_trainings:
            return None, {
                "trainings_ignored": bool(self.cfg.pool_trainings),
                "trainings_user": list(t) if t else None,
            }

        keep = []
        dropped = []
        seen = set()
        for x in t:
            try:
                idx0 = int(x) - 1
            except Exception:
                continue
            if idx0 < 0 or idx0 >= n_panels:
                dropped.append(int(x))
                continue
            if idx0 not in seen:
                keep.append(idx0)
                seen.add(idx0)

        keep.sort()
        return keep, {
            "trainings_user": list(t),
            "trainings_effective": [i + 1 for i in keep],
            "trainings_dropped_out_of_range": sorted(set(dropped)),
            "trainings_ignored": False,
        }

    def _collect_values_by_training_per_fly_scalar(
        self,
    ) -> list[list[tuple[str, float]]]:
        """
        Return list length n_training:
          out[t] = list of (unit_id, scalar_value)
        """
        raise NotImplementedError

    def compute_scalar_panels(self) -> dict[str, Any]:
        vals_by_trn = self._collect_values_by_training_per_fly_scalar()
        if not vals_by_trn or not any(len(v) for v in vals_by_trn):
            return {
                "panel_labels": [],
                "per_unit_values_panel": np.asarray([], dtype=object),
                "per_unit_ids_panel": np.asarray([], dtype=object),
                "mean": np.asarray([], dtype=float),
                "ci_lo": np.asarray([], dtype=float),
                "ci_hi": np.asarray([], dtype=float),
                "n_units_panel": np.asarray([], dtype=int),
                "meta": {},
            }

        if self.cfg.pool_trainings:
            acc = defaultdict(float)
            for panel in vals_by_trn:
                for uid, v in panel:
                    if v is None:
                        continue
                    try:
                        vv = float(v)
                    except Exception:
                        continue
                    if np.isfinite(vv):
                        acc[uid] += vv

            pooled_ids = sorted(acc.keys())
            vals_by_panel = [[(uid, float(acc[uid])) for uid in pooled_ids]]
            panel_labels = ["All trainings combined"]
            sel_info = {
                "trainings_ignored": True,
                "trainings_user": (
                    list(self.cfg.trainings) if self.cfg.trainings else None
                ),
            }
        else:
            vals_by_panel = vals_by_trn
            panel_labels = self._training_labels(len(vals_by_panel))
            keep, sel_info = self._selected_training_indices(len(vals_by_panel))
            if keep is not None:
                vals_by_panel = [vals_by_panel[i] for i in keep]
                panel_labels = [panel_labels[i] for i in keep]
        # stats per panel
        means = []
        ci_lo = []
        ci_hi = []
        n_units_panel = []
        per_unit_values_panel = []
        per_unit_ids_panel = []

        for panel in vals_by_panel:
            pairs = []
            for uid, v in panel:
                try:
                    vv = float(v)
                except Exception:
                    continue
                if np.isfinite(vv):
                    pairs.append((uid, vv))

            ids = np.asarray([u for u, _ in pairs], dtype=object)
            x = np.asarray([vv for _, vv in pairs], dtype=float)
            per_unit_values_panel.append(x)
            per_unit_ids_panel.append(np.asarray(ids, dtype=object))

            if x.size == 0:
                means.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
                n_units_panel.append(0)
                continue
            m, lo, hi, n = (
                meanConfInt(x, conf=float(self.cfg.ci_conf))
                if self.cfg.ci
                else (np.mean(x), np.nan, np.nan, int(x.size))
            )
            means.append(float(m))
            ci_lo.append(float(lo))
            ci_hi.append(float(hi))
            n_units_panel.append(int(n))

        meta = {
            "log_tag": self.log_tag,
            "y_label": self.y_label,
            "base_title": self.base_title,
            "pool_trainings": bool(self.cfg.pool_trainings),
            "subset_label": self.cfg.subset_label,
            "ci": bool(self.cfg.ci),
            "ci_conf": float(self.cfg.ci_conf),
            "skip_first_sync_buckets": int(
                getattr(self.cfg, "skip_first_sync_buckets", 0)
            ),
            "keep_first_sync_buckets": int(
                getattr(self.cfg, "keep_first_sync_buckets", 0)
            ),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "training_selection": sel_info,
        }

        return {
            "panel_labels": panel_labels,
            "per_unit_values_panel": np.asarray(per_unit_values_panel, dtype=object),
            "per_unit_ids_panel": np.asarray(per_unit_ids_panel, dtype=object),
            "mean": np.asarray(means, dtype=float),
            "ci_lo": np.asarray(ci_lo, dtype=float),
            "ci_hi": np.asarray(ci_hi, dtype=float),
            "n_units_panel": np.asarray(n_units_panel, dtype=int),
            "meta": meta,
        }

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
        )
        print(f"[{self.log_tag}] wrote scalar export {out_npz}")

    def plot_bars(self) -> None:
        data = self.compute_scalar_panels()
        labels = data["panel_labels"]
        if not labels:
            print(f"[{self.log_tag}] no data found; skipping plot.")
            return

        means = data["mean"]
        fig, ax = plt.subplots(1, 1, figsize=(max(6.0, 1.2 * len(labels)), 4.0))

        x = np.arange(len(labels))
        ax.bar(x, means)

        if self.cfg.ci:
            lo = data["ci_lo"]
            hi = data["ci_hi"]
            yerr = np.vstack([means - lo, hi - means])
            yerr = np.where(np.isfinite(yerr), yerr, 0)
            ax.errorbar(
                x,
                means,
                yerr=yerr,
                fmt="none",
                capsize=3,
                ecolor="0.2",
                elinewidth=1.0,
                zorder=3,
            )

        if self.cfg.show_points:
            for i in range(len(labels)):
                vals = np.asarray(data["per_unit_values_panel"][i], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    # light jitter
                    jitter = (np.random.rand(vals.size) - 0.5) * 0.2
                    ax.plot(
                        np.full(vals.size, x[i]) + jitter,
                        vals,
                        "o",
                        markersize=3,
                        alpha=0.6,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(self.y_label)
        ax.set_ylim(bottom=0)
        if self.cfg.ymax is not None:
            ax.set_ylim(top=self.cfg.ymax)

        if self.cfg.show_suptitle:
            title = self.base_title
            if self.cfg.subset_label:
                title = f"{title}\n{self.cfg.subset_label}"
            ax.set_title(title)

        fig.tight_layout()
        writeImage(self.cfg.out_file, format=self.opts.imageFormat)
        plt.close(fig)
        print(f"[{self.log_tag}] wrote {self.cfg.out_file}")
