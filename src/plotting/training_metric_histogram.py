from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence
import json
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import writeImage
from src.utils.util import meanConfInt


@dataclass
class TrainingMetricHistogramConfig:
    out_file: str
    bins: int = 30
    xmax: float | None = None
    normalize: bool = False
    pool_trainings: bool = False
    subset_label: str | None = None
    ymax: float | None = None
    # If True: compute histograms per fly first, then aggregate across flies.
    # This avoid overweighting flies/videos with many segments.
    per_fly: bool = False
    # If True and (per_fly=True): compute per-bin confidence intervals across flies.
    ci: bool = False
    ci_conf: float = 0.95
    # Trainings to include when pool_trainings=False.
    # Using 1-based indexing (training 1 == first training).
    trainings: Sequence[int] | None = None


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

    def _selected_training_indices(
        self, n_panels: int
    ) -> tuple[list[int] | None, dict[str, Any]]:
        """
        Return (keep, info):
        - keep is a sorted list of 0-based indices to keep, or None to keep all.
        - info contains:
            trainings_user: list[int] | None (1-based as provided)
            trainings_effective: list[int] | None (1-based after bounds filtering)
            trainings_ignored: bool
            trainings_dropped_out_of_range: list[int] (1-based)
        """
        info = {
            "trainings_user": list(self.cfg.trainings) if self.cfg.trainings else None,
            "trainings_effective": None,
            "trainings_ignored": False,
            "trainings_dropped_out_of_range": [],
        }

        t = self.cfg.trainings
        if not t:
            return None, info

        if self.cfg.pool_trainings:
            # decision: ignore in pooled mode
            info["trainings_ignored"] = True
            return None, info

        keep: list[int] = []
        seen: set[int] = set()
        dropped: list[int] = []

        for x in t:
            try:
                idx0 = int(x) - 1  # 1-based -> 0-based
            except Exception:
                continue
            if idx0 < 0 or idx0 >= n_panels:
                dropped.append(int(x))
                continue
            if idx0 not in seen:
                keep.append(idx0)
                seen.add(idx0)

        keep.sort()
        dropped = sorted(set(dropped))
        info["trainings_dropped_out_of_range"] = dropped
        info["trainings_effective"] = [i + 1 for i in keep]  # back to 1-based

        return keep if keep else [], info

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

    def _collect_values_by_training_per_fly(self) -> list[list[np.ndarray]]:
        """
        Return a list of length n_training, where each element is a list of 1D arrays,
        one per fly (or per VideoAnalysis unit), containing raw values for that training.

        Subclasses should override this when cfg.per_fly is enabled.
        """
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

        Two modes:
          (A): pooled mode (cfg.per_fly=False): identical to the original behavior.
               Returns pooled counts (or pooled proportions if normalize=True).

          (B): per-fly mode (cfg.per_fly=True): computes per-fly histograms first,
               then aggregates across flies.
               If cfg.ci=True, returns per-bin mean + confidence interval across flies.

        Returns a dict with:
          - panel_labels: list[str]
          - bin_edges: np.ndarray shape (n_panels, bins+1)
          - n_raw, n_used, n_dropped: np.ndarray shape (n_panels,)
          - meta: dict
        Plus either:
          - counts (pooled mode): np.ndarray shape (n_panels, bins)
        Or:
          - mean, ci_lo, ci_hi, n_units (per-fly mode): np.ndarray shapes (n_panels, bins, )
        """
        sel_info = {}
        if self.cfg.per_fly:
            vals_by_trn_by_fly = self._collect_values_by_training_per_fly()
            if not any(len(vlist) for vlist in vals_by_trn_by_fly):
                return {
                    "panel_labels": [],
                    "bin_edges": np.zeros((0, self.cfg.bins + 1), dtype=float),
                    "mean": np.zeros((0, self.cfg.bins), dtype=float),
                    "ci_lo": np.zeros((0, self.cfg.bins), dtype=float),
                    "ci_hi": np.zeros((0, self.cfg.bins), dtype=float),
                    "n_units": np.zeros((0, self.cfg.bins), dtype=int),
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
                        "per_fly": True,
                        "ci": bool(self.cfg.ci),
                        "ci_conf": float(self.cfg.ci_conf),
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                    },
                }
        else:
            vals_by_trn = self._collect_values_by_training()

        if not self.cfg.per_fly and not any(len(v) for v in vals_by_trn):
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
                    "per_fly": False,
                    "ci": False,
                    "ci_conf": float(self.cfg.ci_conf),
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                },
            }

        if self.cfg.per_fly:
            if self.cfg.pool_trainings:
                # Two possibilities:
                #  (1) Subclass already pooled per-fly across trainings and returned a single panel.
                #  (2) Subclass returned per-training lists; pool by treating each (fly, training)
                #      distribution as its own unit.
                if len(vals_by_trn_by_fly) == 1:
                    vals_by_panel_by_fly = vals_by_trn_by_fly
                else:
                    pooled_units: list[np.ndarray] = []
                    for vlist in vals_by_trn_by_fly:
                        pooled_units.extend(vlist)
                    vals_by_panel_by_fly = [pooled_units]
                panel_labels = ["All trainings combined"]
            else:
                vals_by_panel_by_fly = vals_by_trn_by_fly
                panel_labels = self._training_labels(len(vals_by_panel_by_fly))

                keep, sel_info = self._selected_training_indices(
                    len(vals_by_panel_by_fly)
                )
                if keep is not None:
                    if not keep:
                        # nothing selected: return empty payload (consistent with your "no data" style)
                        return {
                            "panel_labels": [],
                            "bin_edges": np.zeros((0, self.cfg.bins + 1), dtype=float),
                            "mean": np.zeros((0, self.cfg.bins), dtype=float),
                            "ci_lo": np.zeros((0, self.cfg.bins), dtype=float),
                            "ci_hi": np.zeros((0, self.cfg.bins), dtype=float),
                            "n_units": np.zeros((0, self.cfg.bins), dtype=int),
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
                                **sel_info,
                                "subset_label": self.cfg.subset_label,
                                "per_fly": True,
                                "ci": bool(self.cfg.ci),
                                "ci_conf": float(self.cfg.ci_conf),
                                "generated_utc": datetime.now(timezone.utc).isoformat(),
                            },
                        }

                    vals_by_panel_by_fly = [vals_by_panel_by_fly[i] for i in keep]
                    panel_labels = [panel_labels[i] for i in keep]

                # optional warnings (one-liners)
                if sel_info.get("trainings_ignored", False):
                    print(
                        f"[{self.log_tag}] NOTE: cfg.trainings ignored because pool_trainings=True"
                    )
                dropped = sel_info.get("trainings_dropped_out_of_range") or []
                if dropped:
                    print(
                        f"[{self.log_tag}] NOTE: dropped out-of-range trainings: {dropped}"
                    )

            # Determine eff_xmax across all values
            all_panels_flat: list[np.ndarray] = []
            for vlist in vals_by_panel_by_fly:
                for v in vlist:
                    if v is None:
                        continue
                    vv = np.asarray(v, dtype=float)
                    vv = vv[np.isfinite(vv)]
                    if vv.size:
                        all_panels_flat.append(vv)
            eff_xmax = self._effective_xmax(all_panels_flat)
        else:
            if self.cfg.pool_trainings:
                pooled = np.concatenate([v for v in vals_by_trn if v.size > 0])
                vals_by_panel = [pooled]
                panel_labels = ["All trainings combined"]
                keep, sel_info = self._selected_training_indices(
                    len(vals_by_panel)
                )  # will ignore
            else:
                vals_by_panel = vals_by_trn
                panel_labels = self._training_labels(len(vals_by_trn))

                keep, sel_info = self._selected_training_indices(len(vals_by_panel))
                if keep is not None:
                    if not keep:
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
                                **sel_info,
                                "subset_label": self.cfg.subset_label,
                                "per_fly": False,
                                "ci": False,
                                "ci_conf": float(self.cfg.ci_conf),
                                "generated_utc": datetime.now(timezone.utc).isoformat(),
                            },
                        }

                    vals_by_panel = [vals_by_panel[i] for i in keep]
                    panel_labels = [panel_labels[i] for i in keep]

            if sel_info.get("trainings_ignored", False):
                print(
                    f"[{self.log_tag}] NOTE: cfg.trainings ignored because pool_trainings=True"
                )
            dropped = sel_info.get("trainings_dropped_out_of_range") or []
            if dropped:
                print(
                    f"[{self.log_tag}] NOTE: dropped out-of-range trainings: {dropped}"
                )

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

        mean_list: list[np.ndarray] = []
        lo_list: list[np.ndarray] = []
        hi_list: list[np.ndarray] = []
        n_units_list: list[np.ndarray] = []

        # for vals, label in zip(vals_by_panel, panel_labels):
        if self.cfg.per_fly:
            for vlist, label in zip(vals_by_panel_by_fly, panel_labels):
                # Flatten for raw segment counting / xmax filtering diagnostics
                raw_all = []
                for v in vlist:
                    if v is None:
                        continue
                    vv = np.asarray(v, dtype=float)
                    vv = vv[np.isfinite(vv)]
                    if vv.size:
                        raw_all.append(vv)
                if not raw_all:
                    # No data: still emit deterministic edges
                    hi = float(eff_xmax) if eff_xmax is not None else 1.0
                    edges = np.linspace(0, hi, self.cfg.bins + 1, dtype=float)
                    edges_list.append(edges)
                    mean_list.append(np.full((self.cfg.bins,), np.nan, dtype=float))
                    lo_list.append(np.full((self.cfg.bins,), np.nan, dtype=float))
                    hi_list.append(np.full((self.cfg.bins,), np.nan, dtype=float))
                    n_units_list.append(np.zeros((self.cfg.bins,), dtype=int))
                    n_raw_list.append(0)
                    n_used_list.append(0)
                    n_dropped_list.append(0)
                    continue

                flat = np.concatenate(raw_all, axis=0)
                n_raw = int(flat.size)
                if eff_xmax is not None:
                    flat_used = flat[flat <= eff_xmax]
                else:
                    flat_used = flat

                n_used = int(flat_used.size)
                n_dropped = int(n_raw - n_used)

                # Shared edges per panel
                if eff_xmax is not None:
                    edges = np.linspace(
                        0.0, float(eff_xmax), self.cfg.bins + 1, dtype=float
                    )
                else:
                    # Data-driven edges per panel (may not align across groups)
                    edges = np.histogram_bin_edges(flat_used, bins=self.cfg.bins)
                edges_list.append(edges.astype(float, copy=False))

                # Per-fly histograms
                fly_hists: list[np.ndarray] = []
                for v in vlist:
                    if v is None:
                        continue
                    vv = np.asarray(v, dtype=float)
                    vv = vv[np.isfinite(vv)]
                    if vv.size == 0:
                        continue
                    if eff_xmax is not None:
                        vv = vv[vv <= eff_xmax]
                    if vv.size == 0:
                        continue
                    c, _ = np.histogram(vv, bins=edges)
                    c = c.astype(float, copy=False)
                    if self.cfg.normalize:
                        tot = float(np.sum(c))
                        if tot > 0:
                            c = c / tot
                        else:
                            c[:] = np.nan
                    fly_hists.append(c)
                if not fly_hists:
                    mean = np.full((self.cfg.bins,), np.nan, dtype=float)
                    lo = np.full((self.cfg.bins,), np.nan, dtype=float)
                    hi = np.full((self.cfg.bins,), np.nan, dtype=float)
                    n_units = np.zeros((self.cfg.bins,), dtype=int)
                else:
                    M = np.stack(fly_hists, axis=0)  # (n_units, bins)
                    mean = np.full((self.cfg.bins,), np.nan, dtype=float)
                    lo = np.full((self.cfg.bins), np.nan, dtype=float)
                    hi = np.full((self.cfg.bins), np.nan, dtype=float)
                    n_units = np.zeros((self.cfg.bins,), dtype=int)
                    for j in range(self.cfg.bins):
                        m, lo_j, hi_j, n_j = meanConfInt(
                            M[:, j], conf=float(self.cfg.ci_conf)
                        )
                        mean[j] = float(m)
                        lo[j] = float(lo_j)
                        hi[j] = float(hi_j)
                        n_units[j] = int(n_j)
                mean_list.append(mean)
                lo_list.append(lo)
                hi_list.append(hi)
                n_units_list.append(n_units)
                n_raw_list.append(n_raw)
                n_used_list.append(n_used)
                n_dropped_list.append(n_dropped)

                if n_dropped > 0 and self.cfg.xmax is not None:
                    print(
                        f"[{self.log_tag}] {label}: dropped {n_dropped} values above {self.cfg.xmax}"
                    )
        else:
            for vals, label in zip(vals_by_panel, panel_labels):
                if vals is None or vals.size == 0:
                    counts_list.append(np.zeros((self.cfg.bins,), dtype=int))
                    hi = float(eff_xmax) if eff_xmax is not None else 1.0
                    edges_list.append(
                        np.linspace(0, hi, self.cfg.bins + 1, dtype=float)
                    )
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

        if self.cfg.per_fly:
            mean_arr = (
                np.stack(mean_list, axis=0)
                if mean_list
                else np.zeros((0, self.cfg.bins), dtype=float)
            )
            lo_arr = (
                np.stack(lo_list, axis=0)
                if lo_list
                else np.zeros((0, self.cfg.bins), dtype=float)
            )
            hi_arr = (
                np.stack(hi_list, axis=0)
                if hi_list
                else np.zeros((0, self.cfg.bins), dtype=float)
            )
            n_units_arr = (
                np.stack(n_units_list, axis=0)
                if n_units_list
                else np.zeros((0, self.cfg.bins), dtype=int)
            )

        meta = {
            "log_tag": self.log_tag,
            "x_label": self.x_label,
            "base_title": self.base_title,
            "bins": int(self.cfg.bins),
            "xmax_user": self.cfg.xmax,
            "xmax_effective": eff_xmax,
            "pool_trainings": bool(self.cfg.pool_trainings),
            **sel_info,
            "subset_label": self.cfg.subset_label,
            "per_fly": bool(self.cfg.per_fly),
            "ci": bool(self.cfg.ci),
            "ci_conf": float(self.cfg.ci_conf),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }

        out = {
            "panel_labels": panel_labels,
            "counts": counts_arr,
            "bin_edges": edges_arr,
            "n_raw": np.asarray(n_raw_list, dtype=int),
            "n_used": np.asarray(n_used_list, dtype=int),
            "n_dropped": np.asarray(n_dropped_list, dtype=int),
            "meta": meta,
        }
        if self.cfg.per_fly:
            out.update(
                {
                    "mean": mean_arr,
                    "ci_lo": lo_arr,
                    "ci_hi": hi_arr,
                    "n_units": n_units_arr,
                }
            )
        else:
            out["counts"] = counts_arr
        return out

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
            bin_edges=data["bin_edges"],
            n_raw=data["n_raw"],
            n_used=data["n_used"],
            n_dropped=data["n_dropped"],
            # pooled-mode payload
            counts=data.get("counts", None),
            # per-fly-mode payload
            mean=data.get("mean", None),
            ci_lo=data.get("ci_lo", None),
            ci_hi=data.get("ci_hi", None),
            n_units=data.get("n_units", None),
            meta_json=json.dumps(data["meta"], sort_keys=True),
        )
        print(f"[{self.log_tag}] wrote histogram export {out_npz}")

    def plot_histograms(self) -> None:
        data = self.compute_histograms()
        panel_labels: list[str] = data["panel_labels"]
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
            edges = edges_arr[idx]

            bin_widths = np.diff(edges)
            lefts = edges[:-1]
            centers = lefts + 0.5 * bin_widths

            if self.cfg.per_fly:
                y = np.asarray(data["mean"][idx], dtype=float)
                if not np.any(np.isfinite(y)):
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    continue
                ax.bar(lefts, y, width=bin_widths, align="edge")
                if self.cfg.ci:
                    lo = np.asarray(data["ci_lo"][idx], dtype=float)
                    hi = np.asarray(data["ci_hi"][idx], dtype=float)
                    # yerr expects non-negative deltas
                    yerr = np.vstack([y - lo, hi - y])
                    # Protect against NaNs so matplotlib doesn't complain
                    yerr = np.where(np.isfinite(yerr), yerr, 0)
                    ax.errorbar(
                        centers,
                        y,
                        yerr=yerr,
                        fmt="none",
                        capsize=2,
                        ecolor="0.2",
                        elinewidth=1.0,
                        zorder=3,
                    )
            else:
                counts = np.asarray(data["counts"][idx], dtype=float)
                if np.sum(counts) == 0:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    continue
                if self.cfg.normalize:
                    total = float(np.sum(counts))
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
            ax.set_ylim(bottom=0)
            if self.cfg.ymax is not None:
                ax.set_ylim(top=self.cfg.ymax)

            ax.set_title(label)
            ax.set_xlabel(self.x_label)
            if idx == 0:
                if self.cfg.per_fly:
                    ax.set_ylabel(
                        "Proportion"
                        if self.cfg.normalize
                        else "Mean # segments (per fly)"
                    )
                else:
                    ax.set_ylabel("Proportion" if self.cfg.normalize else "# segments")
        title = self.base_title
        if self.cfg.subset_label:
            title = f"{title}\n{self.cfg.subset_label}"
        fig.suptitle(title)
        fig.tight_layout()

        writeImage(self.cfg.out_file, format=self.opts.imageFormat)
        plt.close(fig)
        print(f"[{self.log_tag}] wrote {self.cfg.out_file}")
