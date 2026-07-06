# src/plotting/overlay_training_metric_hist.py

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter

from src.plotting.bin_edges import (
    is_grouped_edges,
    normalize_panel_edges,
    geom_from_edges,
)
from src.plotting.palettes import (
    NEUTRAL_DARK,
    group_metric_edge_color_for_label,
    group_metric_fill_color_for_label,
    normalize_metric_palette_family,
)
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.stats_bars import StatAnnotConfig, annotate_grouped_bars_per_bin
from src.utils.util import meanConfInt


@dataclass(frozen=True)
class ExportedTrainingHistogram:
    group: str
    panel_labels: list[str]
    # pooled-mode payload (may be empty/unused when per_fly=True)
    counts: np.ndarray  # (n_panels, bins)
    # per-fly-mode payload (None if unavailable)
    mean: np.ndarray | None  # (n_panels, bins)
    ci_lo: np.ndarray | None  # (n_panels, bins)
    ci_hi: np.ndarray | None  # (n_panels, bins)
    n_units: np.ndarray | None  # (n_panels, bins) int
    n_units_panel: np.ndarray | None  # (n_panels,) int
    per_unit_panel: (
        np.ndarray | None
    )  # object array length n_panels; each entry (N_panel, bins)
    per_unit_ids_panel: (
        np.ndarray | None
    )  # object array length n_panels; each entry (N_panel,)
    overflow_mean: np.ndarray | None  # (n_panels,)
    overflow_ci_lo: np.ndarray | None  # (n_panels,)
    overflow_ci_hi: np.ndarray | None  # (n_panels,)
    overflow_n_units: np.ndarray | None  # (n_panels,)
    overflow_per_unit_panel: (
        np.ndarray | None
    )  # object array length n_panels; each entry (N_panel,)
    # bin_edges:
    #   - flat: (n_panels, bins+1) float array
    #   - grouped: object array length n_panels; each entry is list[np.ndarray] of 1D edges
    bin_edges: np.ndarray
    n_raw: np.ndarray | None  # (n_panels,)
    n_used: np.ndarray  # (n_panels,)
    n_dropped: np.ndarray | None  # (n_panels,)
    meta: dict

    @property
    def bins(self) -> int:
        return int(self.meta.get("bins"))

    @property
    def xmax_effective(self) -> float | None:
        return self.meta.get("xmax_effective", None)

    @property
    def pool_trainings(self) -> bool:
        return bool(self.meta.get("pool_trainings", False))

    @property
    def per_fly(self) -> bool:
        return bool(self.meta.get("per_fly", False))

    @property
    def normalize(self) -> bool:
        # present in histogram config, not always in meta historically
        return bool(self.meta.get("normalize", False))

    def validate(self) -> None:
        panel_labels = list(self.panel_labels)
        P = len(panel_labels)
        bins_meta = self.meta.get("bins", None)
        if bins_meta is None:
            raise ValueError("histogram export meta is missing bins")
        B = int(bins_meta)
        if B < 0:
            raise ValueError("histogram export bins must be nonnegative")

        def _check_1d_count(name: str, value, *, required: bool) -> np.ndarray | None:
            if value is None:
                if required:
                    raise ValueError(f"{name} is required")
                return None
            arr = np.asarray(value)
            if arr.ndim != 1 or arr.shape[0] != P:
                raise ValueError(f"{name} must be 1D with length {P}")
            arr_float = arr.astype(float)
            if np.any(~np.isfinite(arr_float)):
                raise ValueError(f"{name} must contain only finite values")
            if np.any(arr_float < 0):
                raise ValueError(f"{name} must be nonnegative")
            if np.any(arr_float != np.floor(arr_float)):
                raise ValueError(f"{name} must contain integer values")
            return arr_float.astype(int)

        n_raw = _check_1d_count("n_raw", self.n_raw, required=False)
        n_used = _check_1d_count("n_used", self.n_used, required=True)
        n_dropped = _check_1d_count("n_dropped", self.n_dropped, required=False)
        if n_raw is not None and n_dropped is not None:
            if np.any(n_raw != n_used + n_dropped):
                raise ValueError("n_raw must equal n_used + n_dropped")

        if np.asarray(self.bin_edges).shape[0] != P:
            raise ValueError(f"bin_edges first dimension must match {P} panels")

        def _panel_edges(panel_idx: int):
            return normalize_panel_edges(np.asarray(self.bin_edges, dtype=object)[panel_idx])

        def _validate_edges(panel_idx: int) -> int:
            edges = _panel_edges(panel_idx)
            if isinstance(edges, list):
                if not edges:
                    raise ValueError("grouped bin_edges must contain at least one group")
                total_bins = 0
                last_hi = None
                for gi, group in enumerate(edges):
                    group = np.asarray(group, dtype=float)
                    if group.ndim != 1 or group.size < 2:
                        raise ValueError(
                            f"bin_edges panel {panel_idx} group {gi} must have >= 2 edges"
                        )
                    if not np.all(np.isfinite(group)):
                        raise ValueError(
                            f"bin_edges panel {panel_idx} group {gi} must be finite"
                        )
                    if np.any(np.diff(group) <= 0):
                        raise ValueError(
                            f"bin_edges panel {panel_idx} group {gi} must be strictly increasing"
                        )
                    if last_hi is not None and float(group[0]) <= last_hi:
                        raise ValueError(
                            f"bin_edges panel {panel_idx} groups must be non-overlapping"
                        )
                    last_hi = float(group[-1])
                    total_bins += int(group.size - 1)
                return total_bins

            edges = np.asarray(edges, dtype=float)
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError(f"bin_edges panel {panel_idx} must have >= 2 edges")
            if not np.all(np.isfinite(edges)):
                raise ValueError(f"bin_edges panel {panel_idx} must be finite")
            if np.any(np.diff(edges) <= 0):
                raise ValueError(
                    f"bin_edges panel {panel_idx} must be strictly increasing"
                )
            return int(edges.size - 1)

        for p_idx in range(P):
            if _validate_edges(p_idx) != B:
                raise ValueError(f"bin_edges panel {p_idx} must define {B} bins")

        if self.per_fly:
            self._validate_per_fly_payload(P, B, n_used)
        else:
            self._validate_pooled_payload(P, B, n_used)

    def _validate_pooled_payload(self, P: int, B: int, n_used: np.ndarray) -> None:
        counts = np.asarray(self.counts)
        if counts.ndim != 2 or counts.shape != (P, B):
            raise ValueError(f"counts must have shape {(P, B)}")
        counts_float = counts.astype(float)
        if np.any(~np.isfinite(counts_float)):
            raise ValueError("counts must contain only finite values")
        if np.any(counts_float < 0):
            raise ValueError("counts must be nonnegative")
        if np.any(counts_float != np.floor(counts_float)):
            raise ValueError("counts must contain integer values")
        if np.any(np.sum(counts_float, axis=1).astype(int) != n_used):
            raise ValueError("pooled counts must sum to n_used per panel")

    def _validate_per_fly_payload(self, P: int, B: int, n_used: np.ndarray) -> None:
        for name in ("mean", "ci_lo", "ci_hi", "n_units", "n_units_panel"):
            if getattr(self, name) is None:
                raise ValueError(f"{name} is required when per_fly=True")

        mean = np.asarray(self.mean, dtype=float)
        ci_lo = np.asarray(self.ci_lo, dtype=float)
        ci_hi = np.asarray(self.ci_hi, dtype=float)
        n_units = np.asarray(self.n_units)
        n_units_panel = np.asarray(self.n_units_panel)
        if mean.shape != (P, B):
            raise ValueError(f"mean must have shape {(P, B)}")
        if ci_lo.shape != (P, B):
            raise ValueError(f"ci_lo must have shape {(P, B)}")
        if ci_hi.shape != (P, B):
            raise ValueError(f"ci_hi must have shape {(P, B)}")
        if n_units.shape != (P, B):
            raise ValueError(f"n_units must have shape {(P, B)}")
        if n_units_panel.shape != (P,):
            raise ValueError(f"n_units_panel must have shape {(P,)}")

        n_units_float = n_units.astype(float)
        n_units_panel_float = n_units_panel.astype(float)
        for name, arr_float in (
            ("n_units", n_units_float),
            ("n_units_panel", n_units_panel_float),
        ):
            if np.any(~np.isfinite(arr_float)):
                raise ValueError(f"{name} must contain only finite values")
            if np.any(arr_float < 0):
                raise ValueError(f"{name} must be nonnegative")
            if np.any(arr_float != np.floor(arr_float)):
                raise ValueError(f"{name} must contain integer values")

        for name, arr in (("mean", mean), ("ci_lo", ci_lo), ("ci_hi", ci_hi)):
            if np.any(np.isinf(arr)):
                raise ValueError(f"{name} must not contain infinite values")
            finite = np.isfinite(arr)
            if np.any(arr[finite] < 0.0):
                raise ValueError(f"{name} must be nonnegative where finite")

        finite_ci = np.isfinite(mean) & np.isfinite(ci_lo) & np.isfinite(ci_hi)
        if np.any(ci_lo[finite_ci] > mean[finite_ci]):
            raise ValueError("ci_lo must be <= mean where finite")
        if np.any(mean[finite_ci] > ci_hi[finite_ci]):
            raise ValueError("mean must be <= ci_hi where finite")

        empty_bins = n_units_float == 0
        for name, arr in (("mean", mean), ("ci_lo", ci_lo), ("ci_hi", ci_hi)):
            if np.any(np.isfinite(arr[empty_bins])):
                raise ValueError(f"{name} must be NaN where n_units == 0")

        per_unit_panel = self.per_unit_panel
        per_unit_ids_panel = self.per_unit_ids_panel
        if per_unit_panel is None or per_unit_ids_panel is None:
            raise ValueError(
                "per_unit_panel and per_unit_ids_panel are required when per_fly=True"
            )
        if np.asarray(per_unit_panel, dtype=object).shape[0] != P:
            raise ValueError("per_unit_panel first dimension must match panels")
        if np.asarray(per_unit_ids_panel, dtype=object).shape[0] != P:
            raise ValueError("per_unit_ids_panel first dimension must match panels")

        per_unit_panel_arr = np.asarray(per_unit_panel, dtype=object)
        per_unit_ids_panel_arr = np.asarray(per_unit_ids_panel, dtype=object)
        for p_idx in range(P):
            panel_values_raw = per_unit_panel_arr[p_idx]
            panel_ids_raw = per_unit_ids_panel_arr[p_idx]
            panel_values = np.asarray(panel_values_raw)
            panel_ids = np.asarray(
                [] if panel_ids_raw is None else panel_ids_raw, dtype=object
            ).reshape(-1)

            if int(n_units_panel_float[p_idx]) == 0:
                if panel_values_raw is not None and panel_values.size:
                    raise ValueError(
                        f"per_unit_panel panel {p_idx} must be empty when n_units_panel == 0"
                    )
                if panel_ids.size:
                    raise ValueError(
                        f"per_unit_ids_panel panel {p_idx} must be empty when n_units_panel == 0"
                    )
                continue

            panel_values = np.asarray(panel_values, dtype=float)
            if panel_values.ndim != 2 or panel_values.shape[1] != B:
                raise ValueError(
                    f"per_unit_panel panel {p_idx} must have shape (N, {B})"
                )
            if panel_values.shape[0] != int(n_units_panel_float[p_idx]):
                raise ValueError(
                    f"per_unit_panel panel {p_idx} row count must match n_units_panel"
                )
            if panel_ids.shape[0] != panel_values.shape[0]:
                raise ValueError(
                    f"per_unit_ids_panel panel {p_idx} length must match per_unit rows"
                )
            if np.any(np.isinf(panel_values)):
                raise ValueError(
                    f"per_unit_panel panel {p_idx} must not contain infinite values"
                )
            finite = np.isfinite(panel_values)
            if np.any(panel_values[finite] < 0.0):
                raise ValueError(
                    f"per_unit_panel panel {p_idx} must be nonnegative where finite"
                )
            observed_n_units = np.sum(np.isfinite(panel_values), axis=0)
            if np.any(observed_n_units != n_units[p_idx].astype(int)):
                raise ValueError(
                    f"n_units panel {p_idx} must match finite per-unit values"
                )


def _mean_ci_from_util(x: np.ndarray, conf: float) -> tuple[float, float, float, int]:
    """
    Mean and t CI across finite x, via src.utils.util.meanConfInt.
    Returns (mean, lo, hi, n).
    """
    m, lo, hi, n = meanConfInt(x, conf=float(conf), asDelta=False)
    return float(m), float(lo), float(hi), int(n)


def _maybe_none_array(x) -> np.ndarray | None:
    """
    np.savez will sometimes store None as a 0-d object array.
    Normalize those cases back to Python None.
    """
    if x is None:
        return None
    arr = np.asarray(x)
    # Common patterns: array(None, dtype=object) or array([None], dtype=object)
    if arr.dtype == object:
        try:
            if arr.shape == () and arr.item() is None:
                return None
            if arr.size == 1 and arr.ravel()[0] is None:
                return None
        except Exception:
            pass
    return arr


def _wrapped_xlabel_text(text: str) -> str:
    text = str(text)
    if "\n" in text:
        return text
    if " from " in text:
        return text.replace(" from ", "\nfrom ", 1)
    if " (" in text:
        return text.replace(" (", "\n(", 1)
    return text


def _wrapped_ylabel_text(text: str) -> str:
    text = str(text)
    if "\n" in text:
        return text
    for phrase in (
        " between-rewards ",
        " between-reward ",
        " of all ",
        " without wall contact",
        " per ",
    ):
        if phrase in text:
            before, after = text.split(phrase, 1)
            return f"{before}\n{phrase.strip()} {after}".rstrip()
    if ", " in text:
        return text.replace(", ", ",\n", 1)
    if " (" in text:
        return text.replace(" (", "\n(", 1)
    return text


def _ensure_xlabel_visible(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    if not axes:
        return

    labels = [ax.xaxis.get_label() for ax in axes if ax.xaxis.get_label().get_text()]
    if not labels:
        return

    pad_y_px = 6.0
    pad_x_px = max(
        18.0, max(0.9 * float(label.get_fontsize()) + 8.0 for label in labels)
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox

    def _labels_within_bounds() -> bool:
        for label in labels:
            bbox = label.get_window_extent(renderer=renderer)
            x_ok = (
                bbox.x0 >= fig_bbox.x0 + pad_x_px
                and bbox.x1 <= fig_bbox.x1 - pad_x_px
            )
            y_ok = bbox.y0 >= fig_bbox.y0 + pad_y_px
            if not (x_ok and y_ok):
                return False
        return True

    if _labels_within_bounds():
        return

    changed = False
    for label in labels:
        wrapped = _wrapped_xlabel_text(label.get_text())
        if wrapped != label.get_text():
            label.set_text(wrapped)
            changed = True

    if changed:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if _labels_within_bounds():
            return

    overflow_bottom_px = 0.0
    overflow_left_px = 0.0
    overflow_right_px = 0.0
    for label in labels:
        bbox = label.get_window_extent(renderer=renderer)
        overflow_bottom_px = max(
            overflow_bottom_px, max((fig_bbox.y0 + pad_y_px) - bbox.y0, 0.0)
        )
        overflow_left_px = max(
            overflow_left_px, max((fig_bbox.x0 + pad_x_px) - bbox.x0, 0.0)
        )
        overflow_right_px = max(
            overflow_right_px, max(bbox.x1 - (fig_bbox.x1 - pad_x_px), 0.0)
        )

    fig_h_px = max(fig.get_size_inches()[1] * fig.dpi, 1.0)
    fig_w_px = max(fig.get_size_inches()[0] * fig.dpi, 1.0)

    extra_bottom = float(overflow_bottom_px / fig_h_px) + 0.01
    extra_left = float(overflow_left_px / fig_w_px) + 0.005
    extra_right = float(overflow_right_px / fig_w_px) + 0.005

    new_bottom = min(fig.subplotpars.bottom + extra_bottom, 0.38)
    new_left = min(fig.subplotpars.left + extra_left, 0.20)
    new_right = max(fig.subplotpars.right - extra_right, 0.82)
    if new_right <= new_left:
        new_left = fig.subplotpars.left
        new_right = fig.subplotpars.right

    fig.subplots_adjust(bottom=new_bottom, left=new_left, right=new_right)
    fig.canvas.draw()


def _ensure_ylabel_visible(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    if not axes:
        return

    labels = [ax.yaxis.get_label() for ax in axes if ax.yaxis.get_label().get_text()]
    if not labels:
        return

    for label in labels:
        wrapped = _wrapped_ylabel_text(label.get_text())
        if wrapped != label.get_text():
            label.set_text(wrapped)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    pad_x_px = max(
        8.0, max(0.55 * float(label.get_fontsize()) + 4.0 for label in labels)
    )
    pad_y_px = max(
        10.0, max(0.45 * float(label.get_fontsize()) + 4.0 for label in labels)
    )

    def _labels_within_bounds() -> bool:
        for label in labels:
            bbox = label.get_window_extent(renderer=renderer)
            x_ok = bbox.x0 >= fig_bbox.x0 + pad_x_px
            y_ok = (
                bbox.y0 >= fig_bbox.y0 + pad_y_px
                and bbox.y1 <= fig_bbox.y1 - pad_y_px
            )
            if not (x_ok and y_ok):
                return False
        return True

    if _labels_within_bounds():
        return

    overflow_left_px = 0.0
    overflow_bottom_px = 0.0
    overflow_top_px = 0.0
    for label in labels:
        bbox = label.get_window_extent(renderer=renderer)
        overflow_left_px = max(
            overflow_left_px, max((fig_bbox.x0 + pad_x_px) - bbox.x0, 0.0)
        )
        overflow_bottom_px = max(
            overflow_bottom_px, max((fig_bbox.y0 + pad_y_px) - bbox.y0, 0.0)
        )
        overflow_top_px = max(
            overflow_top_px, max(bbox.y1 - (fig_bbox.y1 - pad_y_px), 0.0)
        )

    fig_h_px = max(fig.get_size_inches()[1] * fig.dpi, 1.0)
    fig_w_px = max(fig.get_size_inches()[0] * fig.dpi, 1.0)

    extra_left = float(overflow_left_px / fig_w_px) + 0.01
    extra_bottom = float(overflow_bottom_px / fig_h_px) + 0.005
    extra_top = float(overflow_top_px / fig_h_px) + 0.005

    new_left = min(fig.subplotpars.left + extra_left, 0.28)
    new_bottom = min(fig.subplotpars.bottom + extra_bottom, 0.22)
    new_top = max(fig.subplotpars.top - extra_top, 0.82)
    if new_top <= new_bottom:
        new_bottom = fig.subplotpars.bottom
        new_top = fig.subplotpars.top

    fig.subplots_adjust(left=new_left, bottom=new_bottom, top=new_top)
    fig.canvas.draw()


def _edges_equal(a_item, b_item) -> bool:
    a = normalize_panel_edges(a_item)
    b = normalize_panel_edges(b_item)
    if isinstance(a, list) != isinstance(b, list):
        return False
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(
            aa.shape == bb.shape and np.allclose(aa, bb, rtol=0, atol=1e-12)
            for aa, bb in zip(a, b)
        )
    return a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-12)


def _panel_n_label(h: ExportedTrainingHistogram, p_idx: int) -> str:
    if h.per_fly:
        if h.n_units_panel is not None and h.n_units_panel.shape[0] > p_idx:
            return str(int(h.n_units_panel[p_idx]))
        # fallback if older exports: try max n_units across bins
        if h.n_units is not None and h.n_units.shape[0] > p_idx:
            return str(int(np.nanmax(h.n_units[p_idx])))
        return "?"
    else:
        return str(int(h.n_used[p_idx]))


def load_export_npz(group: str, path: str) -> ExportedTrainingHistogram:
    d = np.load(path, allow_pickle=True)
    panel_labels = [str(x) for x in list(d["panel_labels"])]
    counts = np.asarray(d["counts"])
    mean = _maybe_none_array(d["mean"] if "mean" in d.files else None)
    ci_lo = _maybe_none_array(d["ci_lo"] if "ci_lo" in d.files else None)
    ci_hi = _maybe_none_array(d["ci_hi"] if "ci_hi" in d.files else None)
    n_units = _maybe_none_array(d["n_units"] if "n_units" in d.files else None)
    n_units_panel = _maybe_none_array(
        d["n_units_panel"] if "n_units_panel" in d.files else None
    )
    per_unit_panel = _maybe_none_array(
        d["per_unit_panel"] if "per_unit_panel" in d.files else None
    )
    per_unit_ids_panel = _maybe_none_array(
        d["per_unit_ids_panel"] if "per_unit_ids_panel" in d.files else None
    )
    overflow_mean = _maybe_none_array(
        d["overflow_mean"] if "overflow_mean" in d.files else None
    )
    overflow_ci_lo = _maybe_none_array(
        d["overflow_ci_lo"] if "overflow_ci_lo" in d.files else None
    )
    overflow_ci_hi = _maybe_none_array(
        d["overflow_ci_hi"] if "overflow_ci_hi" in d.files else None
    )
    overflow_n_units = _maybe_none_array(
        d["overflow_n_units"] if "overflow_n_units" in d.files else None
    )
    overflow_per_unit_panel = _maybe_none_array(
        d["overflow_per_unit_panel"] if "overflow_per_unit_panel" in d.files else None
    )
    bin_edges = np.asarray(d["bin_edges"])
    n_used = np.asarray(d["n_used"])
    meta_json = d["meta_json"].item() if "meta_json" in d.files else "{}"
    if isinstance(meta_json, (bytes, bytearray)):
        meta_json = meta_json.decode("utf-8")
    meta = json.loads(meta_json)
    hist = ExportedTrainingHistogram(
        group=group,
        panel_labels=panel_labels,
        counts=counts,
        mean=mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_units=n_units,
        n_units_panel=n_units_panel,
        per_unit_panel=per_unit_panel,
        per_unit_ids_panel=per_unit_ids_panel,
        overflow_mean=overflow_mean,
        overflow_ci_lo=overflow_ci_lo,
        overflow_ci_hi=overflow_ci_hi,
        overflow_n_units=overflow_n_units,
        overflow_per_unit_panel=overflow_per_unit_panel,
        bin_edges=bin_edges,
        n_raw=np.asarray(d["n_raw"]) if "n_raw" in d.files else None,
        n_used=n_used,
        n_dropped=np.asarray(d["n_dropped"]) if "n_dropped" in d.files else None,
        meta=meta,
    )
    hist.validate()
    return hist


def _fmt_mismatch(name: str, vals: list) -> str:
    uniq = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)
    return f"{name} differs across inputs: {uniq}"


def _metric_palette_family_from_hist_exports(
    hists: list[ExportedTrainingHistogram],
) -> str | None:
    families = []
    for h in hists:
        meta = dict(h.meta or {})
        family = normalize_metric_palette_family(
            meta.get("metric_palette_family") or meta.get("metric")
        )
        if family is not None:
            families.append(family)
    if not families:
        return None
    uniq = []
    for family in families:
        if family not in uniq:
            uniq.append(family)
    if len(uniq) == 1:
        return uniq[0]
    return None


def validate_alignment(hists: list[ExportedTrainingHistogram]) -> None:
    if len(hists) < 2:
        return

    bins = [h.bins for h in hists]
    if len(set(bins)) != 1:
        raise ValueError(_fmt_mismatch("bins", bins))

    per_fly = [h.per_fly for h in hists]
    if len(set(per_fly)) != 1:
        raise ValueError(_fmt_mismatch("per_fly", per_fly))

    pools = [h.pool_trainings for h in hists]
    if len(set(pools)) != 1:
        raise ValueError(_fmt_mismatch("pool_trainings", pools))

    xmax_eff = [h.xmax_effective for h in hists]
    # For numeric xmax, require close equality; for None, all must be None.
    if any(x is None for x in xmax_eff):
        if not all(x is None for x in xmax_eff):
            raise ValueError(_fmt_mismatch("xmax_effective", xmax_eff))
    else:
        # All numeric
        x0 = float(xmax_eff[0])
        for x in xmax_eff[1:]:
            if not np.isclose(float(x), x0, rtol=0, atol=1e-9):
                raise ValueError(_fmt_mismatch("xmax_effective", xmax_eff))

    # Panel labels must match exactly
    labels0 = hists[0].panel_labels
    for h in hists[1:]:
        if h.panel_labels != labels0:
            raise ValueError(
                _fmt_mismatch("panel_labels", [hh.panel_labels for hh in hists])
            )

    # Bin edges must match per panel
    edges0 = hists[0].bin_edges
    for h in hists[1:]:
        if len(h.bin_edges) != len(edges0):
            raise ValueError("bin_edges shape differs across inputs")
        for p_idx in range(len(edges0)):
            if not _edges_equal(h.bin_edges[p_idx], edges0[p_idx]):
                raise ValueError(
                    "bin_edges differ across inputs. Re-export with the same bin_edges/bin_edges_groups."
                )


def _panel_y(h: ExportedTrainingHistogram, p_idx: int) -> np.ndarray:
    """
    Return y values of length (bins,) for this panel.
    - per_fly=True: use mean (already PDF if normalize=True)
    - per_fly=False: use pooled counts
    """
    if h.per_fly:
        if h.mean is None:
            return np.full((h.bins,), np.nan, dtype=float)
        if h.mean.shape[0] <= p_idx:
            return np.full((h.bins,), np.nan, dtype=float)
        y = np.asarray(h.mean[p_idx], dtype=float)
        return y
    else:
        if h.counts.shape[0] <= p_idx:
            return np.zeros((h.bins,), dtype=float)
        return np.asarray(h.counts[p_idx], dtype=float)


def _panel_overflow_y(h: ExportedTrainingHistogram, p_idx: int) -> float:
    if h.overflow_mean is not None and np.asarray(h.overflow_mean).shape[0] > p_idx:
        value = float(np.asarray(h.overflow_mean, dtype=float)[p_idx])
        if np.isfinite(value):
            return value

    if h.n_raw is None or h.n_dropped is None:
        return np.nan
    n_raw = np.asarray(h.n_raw, dtype=float)
    n_dropped = np.asarray(h.n_dropped, dtype=float)
    if n_raw.shape[0] <= p_idx or n_dropped.shape[0] <= p_idx:
        return np.nan
    if n_raw[p_idx] <= 0:
        return np.nan
    return float(n_dropped[p_idx] / n_raw[p_idx])


def _panel_overflow_per_unit(
    h: ExportedTrainingHistogram,
    p_idx: int,
) -> np.ndarray | None:
    pu_obj = getattr(h, "overflow_per_unit_panel", None)
    if pu_obj is None or np.asarray(pu_obj, dtype=object).shape[0] <= p_idx:
        return None
    values = pu_obj[p_idx]
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1, 1)
    return arr


def _panel_overflow_ci(
    h: ExportedTrainingHistogram,
    p_idx: int,
) -> tuple[float, float] | None:
    if h.overflow_ci_lo is None or h.overflow_ci_hi is None:
        return None
    lo_arr = np.asarray(h.overflow_ci_lo, dtype=float)
    hi_arr = np.asarray(h.overflow_ci_hi, dtype=float)
    if lo_arr.shape[0] <= p_idx or hi_arr.shape[0] <= p_idx:
        return None
    lo = float(lo_arr[p_idx])
    hi = float(hi_arr[p_idx])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return lo, hi


def _restore_overflow_y_ticks(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(
        axis="y",
        which="both",
        right=True,
        labelright=True,
        left=False,
        labelleft=False,
    )


def _restore_fraction_y_tick_precision(axes: list[plt.Axes]) -> None:
    for ax in axes:
        ylim0, ylim1 = ax.get_ylim()
        if np.isfinite(ylim0) and np.isfinite(ylim1) and ylim1 <= 1.0:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def plot_overlays(
    hists: list[ExportedTrainingHistogram],
    *,
    mode: str,  # "pdf" or "cdf"
    layout: str = "overlay",  # "overlay" or "grouped"
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymax: float | None = None,
    stats: bool = False,
    stats_alpha: float = 0.05,
    stats_paired: bool = False,
    xmax_plot: float | None = None,
    overflow: bool = False,
    overflow_threshold: float | None = None,
    overflow_ymax: float | None = None,
    categorical_bin_ratio_max: float = 4.0,
    debug: bool = False,
    opts=None,
) -> plt.Figure:
    if opts is None:
        opts = SimpleNamespace(imageFormat="png", fontSize=None, fontFamily=None)

    if mode not in ("pdf", "cdf"):
        raise ValueError("mode must be 'pdf' or 'cdf'")
    if layout not in ("overlay", "grouped"):
        raise ValueError("layout must be 'overlay' or 'grouped'")
    if layout == "grouped" and mode != "pdf":
        raise ValueError("layout='grouped' is implemented for mode='pdf' only")
    if overflow and (mode != "pdf" or layout != "grouped"):
        raise ValueError("overflow subplot is implemented for grouped PDF plots only")

    validate_alignment(hists)
    metric_palette_family = _metric_palette_family_from_hist_exports(hists)
    customizer = PlotCustomizer()
    font_size = getattr(opts, "fontSize", None)
    if font_size is not None:
        customizer.update_font_size(font_size)
    customizer.update_font_family(getattr(opts, "fontFamily", None))
    font_scale = max(float(customizer.increase_factor), 1.0)
    annotation_font_size = max(
        7,
        min(float(customizer.in_plot_font_size) - 1.0, 11.0),
    )
    legend_font_size = max(
        8,
        min(float(customizer.in_plot_font_size), 14.0),
    )
    panel_labels = hists[0].panel_labels
    n_panels = len(panel_labels)
    show_overflow = bool(overflow)
    if show_overflow and n_panels != 1:
        raise ValueError("overflow subplot currently requires a single panel")

    # ---- figure sizing ----
    G = len(hists)
    bins = hists[0].bins

    if layout == "grouped" and mode == "pdf":
        # px heuristics
        px_per_group = 12  # tweakable: 10-14
        dpi = 100
        panel_width = (bins * G * px_per_group) / dpi
        panel_width = max(panel_width, 7.5)  # floor so small histograms aren't tiny
    else:
        panel_width = 4.0

    if layout == "grouped" and bins >= 10:
        width_scale = min(1.0 + 0.18 * (font_scale - 1.0), 1.25)
    else:
        width_scale = min(1.0 + 0.10 * (font_scale - 1.0), 1.18)
    fig_width = panel_width * n_panels * width_scale
    if show_overflow:
        fig_width *= 1.20
    fig_height = 4.5 * min(1.0 + 0.10 * (font_scale - 1.0), 1.20)
    capsize = 1.25 if layout == "grouped" else 2.0

    if show_overflow:
        fig = plt.figure(figsize=(fig_width, fig_height))
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.0, 0.52],
            height_ratios=[0.42, 0.58],
            wspace=0.08,
            hspace=0.34,
        )
        axes = np.asarray([fig.add_subplot(grid[:, 0])], dtype=object)
        legend_ax = fig.add_subplot(grid[0, 1])
        legend_ax.set_axis_off()
        overflow_ax = fig.add_subplot(grid[1, 1])
    else:
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(fig_width, fig_height),
            squeeze=False,
            sharey=True,
        )
        axes = axes[0]
        legend_ax = None
        overflow_ax = None

    edges = hists[0].bin_edges
    # For a step plot, we’ll use edges and a y of length bins+1
    for p_idx, (ax, plabel) in enumerate(zip(axes, panel_labels)):
        any_data = False
        keep_bins = None

        paired_plot_mode = (
            stats_paired and stats and mode == "pdf" and layout == "grouped"
        )

        paired_n_constant: int | None = None

        def _legend_label(h: ExportedTrainingHistogram) -> str:
            # If paired n varies by bin, we omit legend n and show per-bin n labels.
            # If paired n is constant across bins, put it back in the legend.
            if paired_plot_mode:
                if paired_n_constant is not None:
                    return f"{h.group} (n={paired_n_constant})"
                return f"{h.group}"
            return f"{h.group} (n={_panel_n_label(h, p_idx)})"

        e_item0 = normalize_panel_edges(hists[0].bin_edges[p_idx])
        e_item = e_item0

        # --- optional plot-time truncation ---
        if xmax_plot is not None and np.isfinite(xmax_plot):
            xcut = float(xmax_plot)

            if is_grouped_edges(e_item0):
                new_groups = []
                kept = 0
                for g in e_item0:
                    g = np.asarray(g, float)
                    # keep bins whose RIGHT edge <= xcut
                    right = g[1:]
                    k = int(np.sum(right <= xcut))
                    if k <= 0:
                        continue
                    new_groups.append(g[: k + 1])
                    kept += k
                if kept <= 0:
                    kept = 1
                    # keep first bin of the first group so geometry isn't empty
                    g0 = np.asarray(e_item0[0], float)
                    new_groups = [g0[:2]]
                e_item = new_groups
                keep_bins = kept
            else:
                e_flat = np.asarray(e_item0, float)
                keep_bins = int(
                    np.searchsorted(e_flat[1:], xcut, side="right")
                )  # count of bins
                keep_bins = max(1, min(keep_bins, len(e_flat) - 1))
                e_item = e_flat[: keep_bins + 1]

        # Precompute geometry for grouped bars (PDF only)
        if mode == "pdf":
            widths, centers, bin_ranges, B, x0, x1 = geom_from_edges(e_item)

            # Decide whether to switch to categorical spacing
            wpos = widths[np.isfinite(widths) & (widths > 0)]
            width_ratio = float(np.nanmax(wpos) / np.nanmin(wpos)) if wpos.size else 1.0
            categorical_x = (layout == "grouped") and (
                width_ratio > float(categorical_bin_ratio_max)
            )

        if mode == "pdf" and layout == "grouped":
            G = max(1, len(hists))

            if categorical_x:
                # Categorical x positions: 0,1,2... so bins have equal spacing
                centers_x = np.arange(B, dtype=float)

                group_band = 0.86
                bar_w = np.full((B,), group_band / G, dtype=float)
            else:
                # Proportional x positions: respect bin widths
                centers_x = centers

                group_band = 0.86 * widths
                bar_w = group_band / G

            gpos = (np.arange(G) - (G - 1) / 2.0)[:, None]  # (G,1)
            offsets = gpos * bar_w[None, :]

        xpos_by_group: list[np.ndarray] = []
        per_unit_by_group: list[np.ndarray | None] = []
        per_unit_ids_by_group: list[np.ndarray | None] = []
        hi_by_group: list[np.ndarray] = []
        group_names = [h.group for h in hists]

        # If we're in paired mode, we will recompute the displayed means/CI from paired-only
        # observations (paired within-bin by common unit IDs across all groups).
        paired_means: list[np.ndarray] | None = None
        paired_cilo: list[np.ndarray] | None = None
        paired_cihi: list[np.ndarray] | None = None
        paired_n_per_bin: np.ndarray | None = None

        if paired_plot_mode:
            # Require per-fly payloads and IDs
            if not all(h.per_fly for h in hists):
                raise ValueError("Paired stats require per_fly=True exports.")

            # Grab per-unit matrices/IDs for all groups for this panel (already aligned by keep_bins)
            pu_list: list[np.ndarray] = []
            id_list: list[np.ndarray] = []
            ci_conf = float(hists[0].meta.get("ci_conf", 0.95))

            for h in hists:
                pu_obj = getattr(h, "per_unit_panel", None)
                ids_obj = getattr(h, "per_unit_ids_panel", None)
                if pu_obj is None or ids_obj is None:
                    raise ValueError(
                        "Paired plotting requested but per_unit_panel/per_unit_ids_panel missing. "
                        "Re-export with per_unit_panel and per_unit_ids_panel enabled."
                    )
                if (
                    np.asarray(pu_obj).shape[0] <= p_idx
                    or np.asarray(ids_obj).shape[0] <= p_idx
                ):
                    raise ValueError(
                        f"Missing per-unit payload for panel={plabel} in group={h.group}."
                    )

                pu = np.asarray(pu_obj[p_idx], float)
                ids = np.asarray(ids_obj[p_idx], dtype=object).ravel()
                if ids.shape[0] != pu.shape[0]:
                    raise ValueError(
                        f"per_unit_ids_panel size mismatch for group={h.group}, panel={plabel}: "
                        f"ids={ids.shape[0]} vs per_unit_panel rows={pu.shape[0]}"
                    )
                if keep_bins is not None:
                    pu = pu[:, :keep_bins]
                pu_list.append(pu)
                id_list.append(ids)

            B_eff = int(pu_list[0].shape[1])

            # Per-bin common IDs across ALL groups
            common_ids_by_bin: list[list[str]] = []
            for j in range(B_eff):
                sets = []
                for pu, ids in zip(pu_list, id_list):
                    col = np.asarray(pu[:, j], float)
                    mask = np.isfinite(col)
                    keys = {str(i) for i in ids[mask] if i is not None}
                    sets.append(keys)
                common = set.intersection(*sets) if sets else set()
                common_ids_by_bin.append(sorted(common))

            # Build a stable union ID list (only those that are paired in at least one bin)
            union_ids: list[str] = []
            seen = set()
            for keys in common_ids_by_bin:
                for k in keys:
                    if k not in seen:
                        seen.add(k)
                        union_ids.append(k)
            union_ids = sorted(union_ids)

            # Create filtered per-unit matrices: rows are union_ids, values are present only if paired in that bin
            pu_filt_list: list[np.ndarray] = []
            for pu, ids in zip(pu_list, id_list):
                # map id -> row index in original matrix
                idx_map = {str(i): ii for ii, i in enumerate(ids) if i is not None}
                out = np.full((len(union_ids), B_eff), np.nan, dtype=float)
                for j in range(B_eff):
                    keys = common_ids_by_bin[j]
                    for r, k in enumerate(union_ids):
                        if k not in keys:
                            continue
                        ii = idx_map.get(k, None)
                        if ii is None:
                            continue
                        v = float(pu[ii, j])
                        if np.isfinite(v):
                            out[r, j] = v
                pu_filt_list.append(out)

            # Compute paired-only mean/CI per group per bin
            paired_means = []
            paired_cilo = []
            paired_cihi = []
            paired_n = np.zeros((B_eff,), dtype=int)
            for j in range(B_eff):
                paired_n[j] = int(len(common_ids_by_bin[j]))
            paired_n_per_bin = paired_n

            # Decide whether paired n is constant across bins (ignore zeros)
            nz = paired_n_per_bin[
                np.isfinite(paired_n_per_bin) & (paired_n_per_bin > 0)
            ]
            uniq = np.unique(nz) if nz.size else np.asarray([], int)
            if uniq.size == 1:
                paired_n_constant = int(uniq[0])

            for out in pu_filt_list:
                m = np.full((B_eff,), np.nan, dtype=float)
                lo = np.full((B_eff,), np.nan, dtype=float)
                hi = np.full((B_eff,), np.nan, dtype=float)
                for j in range(B_eff):
                    mm, l0, h0, _n = _mean_ci_from_util(out[:, j], conf=ci_conf)
                    m[j] = mm
                    lo[j] = l0
                    hi[j] = h0
                paired_means.append(m)
                paired_cilo.append(lo)
                paired_cihi.append(hi)

            # Overwrite what stats sees with paired-filtered matrices/IDs
            per_unit_by_group = [m for m in pu_filt_list]
            per_unit_ids_by_group = [
                np.asarray(union_ids, dtype=object) for _ in pu_filt_list
            ]

        for g_idx, h in enumerate(hists):
            # In paired plot mode, the displayed y comes from paired-only means.
            if paired_plot_mode and paired_means is not None:
                y_raw = paired_means[g_idx]
            else:
                y_raw = _panel_y(h, p_idx)

            # Determine total / skip logic depending on payload type
            if h.per_fly:
                # mean may contain NaNs for bins with no contributing units
                y_bins = np.asarray(y_raw, dtype=float)

                if keep_bins is not None:
                    y_bins = y_bins[:keep_bins]

                if not np.any(np.isfinite(y_bins)):
                    continue
                any_data = True
                # In per-fly mode, the exported mean is already a PDF if normalize=True.
                # If normalize was False, mean is "mean #segments per fly", which is not a PDF.
                # Overlay script is intended for PDF/CDF, so require normalized per-fly exports.
                if not h.normalize:
                    raise ValueError(
                        "Overlay plotting with per_fly=True requires normalize=True in the export "
                        "(so mean represents a probability distribution)."
                    )
            else:
                counts = np.asarray(y_raw, dtype=float)
                total = float(np.sum(counts))
                if total <= 0:
                    continue
                any_data = True
                if mode == "pdf":
                    y_bins = counts / total

                    if keep_bins is not None:
                        y_bins = y_bins[:keep_bins]

            # ---- plotting ----
            if mode == "pdf":
                if layout == "overlay":
                    if is_grouped_edges(e_item):
                        # plot each group as a separate step segment (no bridging across gaps)
                        pos = 0
                        for gi, g in enumerate(e_item):
                            nb = int(len(g) - 1)
                            if nb <= 0:
                                continue
                            y_seg = y_bins[pos : pos + nb]
                            if y_seg.size != nb:
                                break
                            y_step = np.concatenate([y_seg, [y_seg[-1]]])
                            ax.step(
                                g,
                                y_step,
                                where="post",
                                label=(_legend_label(h) if gi == 0 else None),
                            )
                            pos += nb
                    else:
                        y_step = np.concatenate([y_bins, [y_bins[-1]]])
                        ax.step(
                            e_item,
                            y_step,
                            where="post",
                            label=_legend_label(h),
                        )
                else:
                    # grouped / dodged bars
                    x = centers_x + offsets[g_idx]

                    # record x positions for bracket drawing
                    xpos_by_group.append(np.asarray(x, float))

                    # baseline for brackets (top of CI if available; else bar height)
                    if paired_plot_mode and paired_cihi is not None:
                        hi_by_group.append(np.asarray(paired_cihi[g_idx], float))
                    elif h.ci_hi is not None and h.ci_hi.shape[0] > p_idx:
                        tmp_hi = np.asarray(h.ci_hi[p_idx], float)
                        if keep_bins is not None:
                            tmp_hi = tmp_hi[:keep_bins]
                        hi_by_group.append(tmp_hi)
                    else:
                        hi_by_group.append(np.asarray(y_bins, float))

                    # record per-fly per-bin PDF values for stats
                    if (not paired_plot_mode) and h.per_fly:
                        pu_obj = getattr(h, "per_unit_panel", None)
                        pu = None
                        if pu_obj is not None and np.asarray(pu_obj).shape[0] > p_idx:
                            pu = pu_obj[p_idx]

                        ids_obj = getattr(h, "per_unit_ids_panel", None)
                        ids = None
                        if ids_obj is not None and np.asarray(ids_obj).shape[0] > p_idx:
                            ids = ids_obj[p_idx]

                        if pu is None:
                            # keep placeholder to error cleanly later if stats requested
                            per_unit_by_group.append(None)  # type: ignore[arg-type]
                            per_unit_ids_by_group.append(None)
                        else:
                            pu = np.asarray(pu, float)
                            if keep_bins is not None:
                                pu = pu[:, :keep_bins]
                            per_unit_by_group.append(pu)  # (N_panel, bins)

                            # ids must align with pu rows
                            if ids is None:
                                per_unit_ids_by_group.append(None)
                            else:
                                ids = np.asarray(ids, dtype=object).ravel()
                                if ids.shape[0] != pu.shape[0]:
                                    raise ValueError(
                                        f"per_unit_ids_panel size mismatch for group={h.group}, panel={plabel}: "
                                        f"ids={ids.shape[0]} vs per_unit_panel rows={pu.shape[0]}"
                                    )
                                per_unit_ids_by_group.append(ids)

                    # bar() ignores NaNs poorly; replace NaNs with 0-height bars
                    y_plot = np.where(np.isfinite(y_bins), y_bins, 0.0)
                    bar_color = group_metric_fill_color_for_label(
                        h.group, g_idx, metric_palette_family
                    )
                    edge_color = group_metric_edge_color_for_label(
                        h.group, g_idx, metric_palette_family
                    )
                    ax.bar(
                        x,
                        y_plot,
                        width=bar_w,
                        align="center",
                        label=_legend_label(h),
                        color=bar_color,
                        edgecolor=edge_color,
                        alpha=0.90,
                        linewidth=0.9,
                    )

                # CI whiskers: only meaningful for per-fly PDF overlays
                if h.per_fly and (
                    paired_plot_mode
                    or (
                        getattr(h, "ci_lo", None) is not None
                        and getattr(h, "ci_hi", None) is not None
                    )
                ):
                    if h.ci_lo.shape[0] <= p_idx or h.ci_hi.shape[0] <= p_idx:
                        continue
                    if (
                        paired_plot_mode
                        and paired_cilo is not None
                        and paired_cihi is not None
                    ):
                        lo = np.asarray(paired_cilo[g_idx], dtype=float)
                        hi = np.asarray(paired_cihi[g_idx], dtype=float)
                    else:
                        lo = np.asarray(h.ci_lo[p_idx], dtype=float)
                        hi = np.asarray(h.ci_hi[p_idx], dtype=float)
                    y = np.asarray(y_bins, dtype=float)

                    if keep_bins is not None:
                        lo = lo[:keep_bins]
                        hi = hi[:keep_bins]

                    # yerr expects non-negative deltas; guard NaNs
                    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
                    # whisker x positions depend on layout
                    if layout == "overlay":
                        xerr = centers
                    else:
                        xerr = centers_x + offsets[g_idx]
                    ax.errorbar(
                        xerr[mask],
                        y[mask],
                        yerr=np.vstack([y[mask] - lo[mask], hi[mask] - y[mask]]),
                        fmt="none",
                        ecolor=NEUTRAL_DARK,
                        capsize=capsize,
                        capthick=1.0,
                        elinewidth=1.0,
                        alpha=0.9,
                        zorder=3,
                    )
            else:
                # mode == "cdf"
                if h.per_fly:
                    # y_bins is already a PDF across bins (sum ~ 1, ignoring NaNs)
                    yy = np.where(np.isfinite(y_bins), y_bins, 0.0)
                    cdf = np.cumsum(yy)
                    # normalize in case of tiny numerical drift
                    if cdf.size and cdf[-1] > 0:
                        cdf = cdf / cdf[-1]
                else:
                    # pooled counts
                    cdf = np.cumsum(counts) / total
                    y_bins = counts / total

                if is_grouped_edges(e_item):
                    # Draw per-group, carrying forward cumulative value
                    pos = 0
                    cprev = 0.0
                    for gi, g in enumerate(e_item):
                        nb = int(len(g) - 1)
                        if nb <= 0:
                            continue
                        yy = np.where(
                            np.isfinite(y_bins[pos : pos + nb]),
                            y_bins[pos : pos + nb],
                            0.0,
                        )
                        cdf_seg = cprev + np.cumsum(yy)
                        y_step = np.concatenate([[cprev], cdf_seg])
                        ax.step(
                            g,
                            y_step,
                            where="post",
                            label=(_legend_label(h) if gi == 0 else None),
                            color=group_metric_edge_color_for_label(
                                h.group, g_idx, metric_palette_family
                            ),
                        )
                        cprev = float(cdf_seg[-1]) if cdf_seg.size else cprev
                        pos += nb
                else:

                    y_step = np.concatenate([[0.0], cdf])
                    ax.step(
                        e_item,
                        y_step,
                        where="post",
                        label=_legend_label(h),
                        color=group_metric_edge_color_for_label(
                            h.group, g_idx, metric_palette_family
                        ),
                    )

        if not any_data:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue

        ax.set_title(plabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        if p_idx == 0:
            if ylabel:
                ax.set_ylabel(_wrapped_ylabel_text(ylabel))

        if ymax is not None:
            ax.set_ylim(bottom=0, top=float(ymax))

        # ---- bin-range x tick labels for grouped PDF bars ----
        if mode == "pdf" and layout == "grouped":
            # centers/edges may have been truncated above
            ax.set_xticks(centers_x)

            # label bins as ranges: "0-10", "10-20", ...
            labels_xt = []
            for a, b in bin_ranges:
                a = float(a)
                b = float(b)

                # If edges look like integer bins (including half-integer edges), show integer ranges.
                # Examples:
                #  -0.5..10.5  -> 0-10
                #  10.5..20.5  -> 11-20
                #  80.5..110.5 -> 81-110
                a_is_intish = np.isclose(a, round(a)) or np.isclose(a % 1.0, 0.5)
                b_is_intish = np.isclose(b, round(b)) or np.isclose(b % 1.0, 0.5)

                if a_is_intish and b_is_intish:
                    lo = int(math.ceil(a))  # -0.5 -> 0, 10.5 -> 11
                    hi = int(math.floor(b))  # 10.5 -> 10, 20.5 -> 20

                    # Guard: if something weird happens (e.g., very narrow bin), fall back
                    if hi < lo:
                        labels_xt.append(f"{a:g}-{b:g}")
                    else:
                        labels_xt.append(f"{lo}-{hi}")
                else:
                    # Non-integer-ish bins: keep compact float formatting
                    labels_xt.append(f"{a:g}-{b:g}")

            tick_rotation = 0
            if bins >= 6 and font_scale >= 1.15:
                tick_rotation = 20
            if bins >= 7 and font_scale >= 1.25:
                tick_rotation = 28
            if bins >= 8 and font_scale >= 1.35:
                tick_rotation = 35
            ax.set_xticklabels(
                labels_xt,
                rotation=tick_rotation,
                ha="right" if tick_rotation else "center",
            )

        if mode == "pdf" and layout == "grouped":
            if categorical_x:
                ax.set_xlim(-0.5, B - 0.5)
            else:
                ax.set_xlim(x0, x1)

        if stats and mode == "pdf" and layout == "grouped":
            # require per-fly PDF inputs
            if not all(h.per_fly for h in hists):
                raise ValueError(
                    "Stats require per_fly=True exports (one PDF per fly)."
                )
            if any(pu is None for pu in per_unit_by_group):
                raise ValueError(
                    "Stats requested but per_unit_panel missing in one or more inputs. "
                    "Re-export with per_unit_panel enabled."
                )

            if stats_paired and any(ids is None for ids in per_unit_ids_by_group):
                raise ValueError(
                    "Paired stats requested but per_unit_ids_panel missing in one or more inputs. "
                    "Re-export with per_unit_ids_panel enabled."
                )

            cfg_stats = StatAnnotConfig(
                alpha=float(stats_alpha),
                min_n_per_group=3,
                nlabel_off_frac=0.0,  # IMPORTANT: n is in legend, not above bars
                stack_gap_frac=0.030,
            )

            annotate_grouped_bars_per_bin(
                ax,
                x_centers=centers_x,
                xpos_by_group=xpos_by_group,
                per_unit_by_group=per_unit_by_group,
                per_unit_ids_by_group=per_unit_ids_by_group,
                hi_by_group=hi_by_group,
                group_names=group_names,
                cfg=cfg_stats,
                paired=bool(stats_paired),
                panel_label=plabel,
                debug=debug,
            )
        # Optional per-bin n labels in paired mode (only if n varies by bin)
        if (
            paired_plot_mode
            and paired_n_per_bin is not None
            and paired_n_constant is None
        ):
            # Put n labels just above the tallest bar/CI in that bin
            ylim0, ylim1 = ax.get_ylim()
            y_rng = float(ylim1 - ylim0) if np.isfinite(ylim1 - ylim0) else 1.0
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax_bbox = ax.get_window_extent(renderer=renderer)
            ax_h_px = max(float(ax_bbox.height), 1.0)
            data_per_px = y_rng / ax_h_px
            font_px = float(annotation_font_size) * float(fig.dpi) / 72.0
            y_pad = 0.020 * y_rng + (0.45 * font_px + 2.0) * data_per_px
            for j in range(int(centers_x.size)):
                nbin = int(paired_n_per_bin[j])
                if nbin <= 0:
                    continue
                # derive a reasonable baseline from hi_by_group if available
                y_top = np.nan
                for gg in range(len(hi_by_group)):
                    if j < hi_by_group[gg].shape[0] and np.isfinite(hi_by_group[gg][j]):
                        y_top = (
                            hi_by_group[gg][j]
                            if not np.isfinite(y_top)
                            else max(y_top, hi_by_group[gg][j])
                        )
                if not np.isfinite(y_top):
                    continue
                ax.text(
                    float(centers_x[j]),
                    float(y_top + y_pad),
                    f"n={nbin}",
                    ha="center",
                    va="bottom",
                    fontsize=annotation_font_size,
                    color="0.2",
                    clip_on=False,
                    zorder=9,
                )

        if not show_overflow:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                prop={"style": "italic", "size": legend_font_size},
            )

    if show_overflow and overflow_ax is not None:
        p_idx = 0
        threshold = overflow_threshold
        if threshold is None or not np.isfinite(float(threshold)):
            panel_edges = normalize_panel_edges(hists[0].bin_edges[p_idx])
            if is_grouped_edges(panel_edges):
                threshold = float(panel_edges[-1][-1])
            else:
                threshold = float(panel_edges[-1])
        threshold = float(threshold)
        overflow_values = np.asarray(
            [_panel_overflow_y(h, p_idx) for h in hists],
            dtype=float,
        )
        G_over = max(1, len(hists))
        center = np.asarray([0.0], dtype=float)
        bar_w_over = 0.70 / G_over
        offsets_over = (np.arange(G_over, dtype=float) - (G_over - 1) / 2.0) * bar_w_over
        xpos_by_group_over: list[np.ndarray] = []
        per_unit_by_group_over: list[np.ndarray | None] = []
        per_unit_ids_by_group_over: list[np.ndarray | None] = []
        hi_by_group_over: list[np.ndarray] = []

        for g_idx, h in enumerate(hists):
            y = float(overflow_values[g_idx])
            x = center + offsets_over[g_idx]
            xpos_by_group_over.append(np.asarray(x, dtype=float))
            pu_over = _panel_overflow_per_unit(h, p_idx)
            per_unit_by_group_over.append(pu_over)

            ids = None
            ids_obj = getattr(h, "per_unit_ids_panel", None)
            if ids_obj is not None and np.asarray(ids_obj).shape[0] > p_idx:
                ids = np.asarray(ids_obj[p_idx], dtype=object).ravel()
            per_unit_ids_by_group_over.append(ids)

            ci = _panel_overflow_ci(h, p_idx)
            hi_val = y
            if ci is not None:
                _lo, hi_val = ci
            hi_by_group_over.append(np.asarray([hi_val], dtype=float))

            bar_color = group_metric_fill_color_for_label(
                h.group, g_idx, metric_palette_family
            )
            edge_color = group_metric_edge_color_for_label(
                h.group, g_idx, metric_palette_family
            )
            overflow_ax.bar(
                x,
                0.0 if not np.isfinite(y) else y,
                width=bar_w_over,
                align="center",
                label=f"{h.group} (n={_panel_n_label(h, p_idx)})",
                color=bar_color,
                edgecolor=edge_color,
                alpha=0.90,
                linewidth=0.9,
            )
            if ci is not None and np.isfinite(y):
                lo, hi = ci
                if np.isfinite(lo) and np.isfinite(hi):
                    overflow_ax.errorbar(
                        x,
                        [y],
                        yerr=np.asarray([[y - lo], [hi - y]], dtype=float),
                        fmt="none",
                        ecolor=NEUTRAL_DARK,
                        capsize=capsize,
                        capthick=1.0,
                        elinewidth=1.0,
                        alpha=0.9,
                        zorder=3,
                    )

        overflow_ax.set_xticks(center)
        overflow_ax.set_xticklabels([f"{threshold:g}+"])
        overflow_ax.set_xlim(-0.5, 0.5)
        overflow_ax.set_ylim(bottom=0)
        if overflow_ymax is not None:
            overflow_ax.set_ylim(top=float(overflow_ymax))
        overflow_ax.yaxis.tick_right()
        overflow_ax.yaxis.set_label_position("right")
        _restore_overflow_y_ticks(overflow_ax)
        overflow_ax.set_ylabel("")

        if stats:
            if any(pu is None for pu in per_unit_by_group_over):
                print(
                    "[overlay_training_metric_hist] NOTE: overflow stats skipped; "
                    "re-export bundles to include overflow_per_unit_panel."
                )
            else:
                cfg_stats = StatAnnotConfig(
                    alpha=float(stats_alpha),
                    min_n_per_group=3,
                    nlabel_off_frac=0.0,
                )
                bracket_tops = annotate_grouped_bars_per_bin(
                    overflow_ax,
                    x_centers=center,
                    xpos_by_group=xpos_by_group_over,
                    per_unit_by_group=per_unit_by_group_over,
                    per_unit_ids_by_group=per_unit_ids_by_group_over,
                    hi_by_group=hi_by_group_over,
                    group_names=group_names,
                    cfg=cfg_stats,
                    paired=bool(stats_paired),
                    panel_label=f"{panel_labels[p_idx]} overflow",
                    debug=debug,
                )
                if overflow_ymax is None:
                    finite_tops = np.asarray(bracket_tops, dtype=float)
                    finite_tops = finite_tops[np.isfinite(finite_tops)]
                    if finite_tops.size:
                        ylim0, ylim1 = overflow_ax.get_ylim()
                        y_rng = (
                            float(ylim1 - ylim0)
                            if np.isfinite(ylim1 - ylim0) and ylim1 > ylim0
                            else 1.0
                        )
                        target_top = float(np.nanmax(finite_tops) + 0.04 * y_rng)
                        if target_top > ylim1:
                            overflow_ax.set_ylim(top=target_top)

        handles, labels = overflow_ax.get_legend_handles_labels()
        if legend_ax is not None and handles:
            legend_ax.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.0),
                borderaxespad=0.0,
                frameon=True,
                prop={"style": "italic", "size": legend_font_size},
            )

    if title:
        fig.suptitle(title)
    if customizer.customized:
        customizer.adjust_padding_proportionally(
            wrap_legend_labels=not show_overflow,
        )
        xlabel_text = next(
            (
                ax.xaxis.get_label().get_text()
                for ax in axes
                if ax.xaxis.get_label().get_text()
            ),
            "",
        )
        xlabel_lines = max(1, str(xlabel_text).count("\n") + 1)
        long_xlabel = len(str(xlabel_text).replace("\n", " ")) >= 36
        dense_grouped_layout = mode == "pdf" and layout == "grouped" and bins >= 6
        if dense_grouped_layout or (long_xlabel and font_scale >= 1.15):
            bottom = 0.105 + 0.040 * max(font_scale - 1.0, 0.0)
            if dense_grouped_layout:
                bottom += 0.014
            if xlabel_lines == 1 and long_xlabel and font_scale >= 1.15:
                bottom += 0.018
            if xlabel_lines >= 2:
                bottom += 0.034
            fig.subplots_adjust(bottom=min(bottom, 0.30), right=0.97, top=0.96)
    else:
        fig.tight_layout()
    if show_overflow:
        fig.subplots_adjust(right=min(max(fig.subplotpars.right, 0.92), 0.97))
    else:
        fig.subplots_adjust(right=min(fig.subplotpars.right, 0.74))
    _ensure_ylabel_visible(fig, list(axes))
    _ensure_xlabel_visible(fig, list(axes))
    if mode == "pdf":
        _restore_fraction_y_tick_precision(list(axes))
    if show_overflow and overflow_ax is not None:
        _restore_overflow_y_ticks(overflow_ax)
    return fig
