from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence
import json
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import writeImage


@dataclass
class TrainingMetricHistogramConfig:
    out_file: str
    bins: int = 30
    xmax: float | None = None
    normalize: bool = False
    pool_trainings: bool = False
    subset_label: str | None = None
    ymax: float | None = None


class TrainingMetricHistogramPlotter:
    """
    Generic "one panel per training" histogram plotter.
    Subclasses only need to implement _collect_values_by_training().
    """

    def _n_trainings(self) -> int:
        return max((len(getattr(va, "trns", [])) for va in self.vas), default=0)

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: TrainingMetricHistogramConfig,
        *,
        log_tag: str,
        x_label: str,
        base_title: str,
    ):
        self.vas = vas
        self.opts = opts
        self.gls = gls
        self.customizer = customizer
        self.cfg = cfg
        self.log_tag = log_tag
        self.x_label = x_label
        self.base_title = base_title

    def _training_labels(self, n_trn: int) -> list[str]:
        """
        Build per-training labels without assuming share Training objects.
        Uses the most common Training.name() at each training index across VAs.
        """
        labels: list[str] = []
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
            if names:
                labels.append(Counter(names).most_common(1)[0][0])
            else:
                labels.append(f"training {i + 1}")
        return labels

    def _collect_values_by_training(self) -> list[np.ndarray]:
        raise NotImplementedError

    def _effective_xmax(self, vals_by_panel: list[np.ndarray]) -> float | None:
        """
        Determine a deterministic xmax to use for bin edges.

        - If cfg.xmax is provided, use it.
        - Otherwise use the maximum observed value across all panels (if any).

        Note: overlays across groups will only align if they share the same
        effective xmax, which in practice means explicitly specifying --*-max.
        """
        if self.cfg.xmax is not None:
            try:
                return float(self.cfg.xmax)
            except Exception:
                return None
        mx = None
        for v in vals_by_panel:
            if v is None or v.size == 0:
                continue
            try:
                v_max = float(np.nanmax(v))
            except Exception:
                continue
            if not np.isfinite(v_max):
                continue
            if mx is None or v_max > mx:
                mx = v_max
        return mx

    def compute_histograms(self) -> dict:
        """
        Compute binned histograms for each panel (training or pooled).

        Returns a dict with:
          - panel_labels: list[str]
          - counts: np.ndarray shape (n_panels, bins)
          - bin_edges: np.ndarray shape (n_panels, bins+1)
          - n_raw, n_used, n_dropped: np.ndarray shape (n_panels,)
          - meta: dict
        """
        vals_by_trn = self._collect_values_by_training()

        if not any(len(v) for v in vals_by_trn):
            return {
                "panel_labels": [],
                "counts": np.zeros((0, self.cfg.bins), dtype=int),
                "bin_edges": np.zeros((0, self.cfg.bins + 1), dtype=float),
                "n_raw": np.zeros((0,), dtype=int),
                "n_used": np.zeros((0,), dtype=int),
                "n_dropped": np.zeros((0,), dtype=int),
                "meta": {
                    "log_tag": self.log_tag,
                    "x_label": self.x_label,
                    "base_title": self.base_title,
                    "bins": int(self.cfg.bins),
                    "xmax_user": self.cfg.xmax,
                    "xmax_effective": None,
                    "pool_trainings": bool(self.cfg.pool_trainings),
                    "subset_label": self.cfg.subset_label,
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                },
            }

        if self.cfg.pool_trainings:
            pooled = np.concatenate([v for v in vals_by_trn if v.size > 0])
            vals_by_panel = [pooled]
            panel_labels = ["all trainings combined"]
        else:
            vals_by_panel = vals_by_trn
            panel_labels = self._training_labels(len(vals_by_trn))

        eff_xmax = self._effective_xmax(vals_by_panel)

        # Guard against degenerate histogram ranges (e.g., all values == 0 → xmax == 0)
        # np.histogram(..., range=(0, 0)) is problematic, so fall back to data-driven bins.
        if eff_xmax is not None and eff_xmax <= 0:
            eff_xmax = None

        counts_list: list[np.ndarray] = []
        edges_list: list[np.ndarray] = []
        n_raw_list: list[int] = []
        n_used_list: list[int] = []
        n_dropped_list: list[int] = []

        for vals, label in zip(vals_by_panel, panel_labels):
            if vals is None or vals.size == 0:
                counts_list.append(np.zeros((self.cfg.bins,), dtype=int))
                hi = float(eff_xmax) if eff_xmax is not None else 1.0
                edges_list.append(np.linspace(0, hi, self.cfg.bins + 1, dtype=float))
                n_raw_list.append(0)
                n_used_list.append(0)
                n_dropped_list.append(0)
                continue
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            n_raw = int(vals.size)

            if eff_xmax is not None:
                vals = vals[vals <= eff_xmax]
            n_used = int(vals.size)
            n_dropped = int(n_raw - n_used)

            if n_used == 0:
                counts = np.zeros((self.cfg.bins,), dtype=int)
                hi = float(eff_xmax) if eff_xmax is not None else 1.0
                edges = np.linspace(0, hi, self.cfg.bins + 1, dtype=float)
            else:
                if eff_xmax is not None:
                    # Enforce shared edges via an explicit range.
                    counts, edges = np.histogram(
                        vals, bins=self.cfg.bins, range=(0.0, float(eff_xmax))
                    )
                else:
                    # Fallback: data-driven edges (will likely not align across groups).
                    counts, edges = np.histogram(vals, bins=self.cfg.bins)

            counts_list.append(counts.astype(int, copy=False))
            edges_list.append(edges.astype(float, copy=False))
            n_raw_list.append(n_raw)
            n_used_list.append(n_used)
            n_dropped_list.append(n_dropped)

            if n_dropped > 0 and self.cfg.xmax is not None:
                print(
                    f"[{self.log_tag}] {label}: dropped {n_dropped} values above {self.cfg.xmax}"
                )

        counts_arr = (
            np.stack(counts_list, axis=0)
            if counts_list
            else np.zeros((0, self.cfg.bins), dtype=int)
        )
        edges_arr = (
            np.stack(edges_list, axis=0)
            if edges_list
            else np.zeros((0, self.cfg.bins + 1), dtype=float)
        )

        meta = {
            "log_tag": self.log_tag,
            "x_label": self.x_label,
            "base_title": self.base_title,
            "bins": int(self.cfg.bins),
            "xmax_user": self.cfg.xmax,
            "xmax_effective": eff_xmax,
            "pool_trainings": bool(self.cfg.pool_trainings),
            "subset_label": self.cfg.subset_label,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "panel_labels": panel_labels,
            "counts": counts_arr,
            "bin_edges": edges_arr,
            "n_raw": np.asarray(n_raw_list, dtype=int),
            "n_used": np.asarray(n_used_list, dtype=int),
            "n_dropped": np.asarray(n_dropped_list, dtype=int),
            "meta": meta,
        }

    def export_histograms_npz(self, out_npz: str) -> None:
        data = self.compute_histograms()
        if not data["panel_labels"]:
            print(f"[{self.log_tag}] no data found; skipping export.")
            return
        eff = data["meta"].get("xmax_effective", None)
        if self.cfg.xmax is None and eff is not None:
            print(
                f"[{self.log_tag}] NOTE: exporting histogram with data-derived xmax_effective={eff:.6g}. "
                "For overlay plots across groups, consider setting an explicit --*-max so bin edges match. "
                f"Current bins: {self.cfg.bins}"
            )
        np.savez_compressed(
            out_npz,
            panel_labels=np.asarray(data["panel_labels"], dtype=object),
            counts=data["counts"],
            bin_edges=data["bin_edges"],
            n_raw=data["n_raw"],
            n_used=data["n_used"],
            n_dropped=data["n_dropped"],
            meta_json=json.dumps(data["meta"], sort_keys=True),
        )
        print(f"[{self.log_tag}] wrote histogram export {out_npz}")

    def plot_histograms(self) -> None:
        data = self.compute_histograms()
        panel_labels: list[str] = data["panel_labels"]
        counts_arr: np.ndarray = data["counts"]
        edges_arr: np.ndarray = data["bin_edges"]
        if not panel_labels:
            print(f"[{self.log_tag}] no data found; skipping plot.")
            return

        n_trn = len(panel_labels)
        fig, axes = plt.subplots(
            1,
            n_trn,
            figsize=(4.0 * n_trn if n_trn > 1 else 6.5, 4.0),
            squeeze=False,
            sharey=True,
        )
        axes = axes[0]

        for idx, (ax, label) in enumerate(zip(axes, panel_labels)):
            counts = counts_arr[idx]
            edges = edges_arr[idx]

            if counts.sum() == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue

            bin_widths = np.diff(edges)
            lefts = edges[:-1]
            if self.cfg.normalize:
                total = counts.sum()
                if total == 0:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    continue
                y = counts / total
                ax.bar(lefts, y, width=bin_widths, align="edge")
            else:
                ax.bar(lefts, counts, width=bin_widths, align="edge")

            # Keep xlim consistent with effective edges
            if edges.size >= 2:
                ax.set_xlim(float(edges[0]), float(edges[-1]))
            if self.cfg.ymax is not None:
                ax.set_ylim(top=self.cfg.ymax)

            ax.set_title(label)
            ax.set_xlabel(self.x_label)
            if idx == 0:
                ax.set_ylabel(
                    "proportion of segments" if self.cfg.normalize else "# segments"
                )
        title = self.base_title
        if self.cfg.subset_label:
            title = f"{title}\n{self.cfg.subset_label}"
        fig.suptitle(title)
        fig.tight_layout()

        writeImage(self.cfg.out_file, format=self.opts.imageFormat)
        plt.close(fig)
        print(f"[{self.log_tag}] wrote {self.cfg.out_file}")
