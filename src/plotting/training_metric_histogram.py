from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from typing import Any, Sequence, Union
import json
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.bin_edges import is_grouped_edges, normalize_panel_edges
from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import writeImage
from src.utils.util import meanConfInt


@dataclass
class TrainingMetricHistogramConfig:
    out_file: str
    bins: int = 30
    xmax: float | None = None
    bin_edges: Sequence[float] | None = None
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
    # Minimum number of segments (values) required for a fly/unit to be included
    # when per_fly=True. Ignored otherwise.
    min_segs_per_fly: int = 10
    # Sync-bucket windowing knobs for this plotter (typically derived from global opts).
    skip_first_sync_buckets: int = 0
    keep_first_sync_buckets: int = 0  # 0 = no cap
    # Omit figure suptitle by default
    show_suptitle: bool = False


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

    @staticmethod
    def _unit_id(va, *, f: int) -> str:
        # best-effort stable ID
        video_fn = getattr(va, "fn", None)
        base = os.path.basename(str(video_fn)) if video_fn else "unknown_video"
        va_id = int(getattr(va, "f", 0) or 0)  # if that's your VA identifier
        return f"{base}|va_tag={va_id}|trx_idx={int(f)}"

    @staticmethod
    def _split_unit(item):
        """
        Accept either:
        - ndarray-like values
        - (unit_id, values) pairs
        - None
        Return (unit_id_or_None, values_or_None)
        """
        if item is None:
            return None, None
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return item[0], item[1]
        return None, item

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

    def _validated_bin_edges(self) -> Union[np.ndarray, list[np.ndarray], None]:
        be = getattr(self.cfg, "bin_edges", None)
        if be is None:
            return None

        # Detect "groups": first element is itself a sequence (and not a scaler)
        def _is_seq(x):
            return isinstance(x, (list, tuple, np.ndarray))

        is_grouped = (
            _is_seq(be) and len(be) > 0 and _is_seq(be[0]) and not np.isscalar(be[0])
        )

        def _validate_1d(edges: np.ndarray, ctx: str) -> np.ndarray:
            edges = np.asarray(edges, dtype=float).ravel()
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError(
                    f"{ctx}: need at least 2 edges; got shape={edges.shape}"
                )
            if np.any(~np.isfinite(edges)):
                bad = np.where(~np.isfinite(edges))[0].tolist()
                raise ValueError(f"{ctx}: non-finite edges at indices {bad}")
            diffs = np.diff(edges)
            if np.any(diffs <= 0):
                i = int(np.where(diffs <= 0)[0][0])
                raise ValueError(
                    f"{ctx}: edges must be strictly increasing. "
                    f"Found {edges[i]:.6g} then {edges[i+1]:.6g}."
                )
            return edges.astype(float, copy=False)

        if not is_grouped:
            edges = _validate_1d(be, "bin_edges")
            if getattr(self.cfg, "xmax", None) is not None:
                print(
                    f"[{self.log_tag}] NOTE: cfg.bin_edges is set; cfg.xmax will be ignored for binning."
                )
            return edges

        # ---- normalize grouped inputs before building `groups`
        # make be_iter an iterable of group edge sequences, regardless of how cfg.bin_edges was provided.
        if isinstance(be, np.ndarray) and be.dtype != object and be.ndim == 2:
            # interpret rows as groups, e.g., [[0,30],[800,1600]]
            be_iter = [be[i, :] for i in range(be.shape[0])]
        else:
            be_iter = be

        # Grouped mode
        groups: list[np.ndarray] = []
        for gi, g in enumerate(be_iter):
            edges_g = _validate_1d(g, f"bin_edges group {gi}")
            groups.append(edges_g)

        # Enforce groups in ascending non-overlapping order
        for gi in range(len(groups) - 1):
            if groups[gi][-1] >= groups[gi + 1][0]:
                raise ValueError(
                    "bin_edges groups must be strictly increasing and non-overlapping. "
                    f"Group {gi} ends at {groups[gi][-1]:.6g} but group {gi+1} starts at {groups[gi+1][0]:.6g}."
                )

        if getattr(self.cfg, "xmax", None) is not None:
            print(
                f"[{self.log_tag}] NOTE: cfg.bin_edges is set; cfg.xmax will be ignored for binning."
            )
        return groups

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
        user_edges = self._validated_bin_edges()
        user_edge_groups = isinstance(user_edges, list)

        if user_edges is None:
            bins_eff = int(self.cfg.bins)
            binning_mode = "bins"
            bin_edges_user = None
        elif user_edge_groups:
            bins_eff = int(sum(g.size - 1 for g in user_edges))
            binning_mode = "edge_groups"
            bin_edges_user = [[float(x) for x in g] for g in user_edges]
        else:
            bins_eff = int(user_edges.size - 1)
            binning_mode = "edges"
            bin_edges_user = [float(x) for x in user_edges]

        if user_edge_groups:
            empty_edges = np.empty((0,), dtype=object)
        else:
            empty_edges = np.zeros((0, bins_eff + 1), dtype=float)

        if user_edges is None:
            xmin_effective = 0.0
        elif user_edge_groups:
            xmin_effective = float(user_edges[0][0])
        else:
            xmin_effective = float(user_edges[0])
        if self.cfg.per_fly:
            vals_by_trn_by_fly = self._collect_values_by_training_per_fly()
            if not any(len(vlist) for vlist in vals_by_trn_by_fly):
                return {
                    "panel_labels": [],
                    "bin_edges": empty_edges,
                    "mean": np.zeros((0, bins_eff), dtype=float),
                    "ci_lo": np.zeros((0, bins_eff), dtype=float),
                    "ci_hi": np.zeros((0, bins_eff), dtype=float),
                    "n_units": np.zeros((0, bins_eff), dtype=int),
                    "n_units_panel": np.zeros((0,), dtype=int),
                    "n_raw": np.zeros((0,), dtype=int),
                    "n_used": np.zeros((0,), dtype=int),
                    "n_dropped": np.zeros((0,), dtype=int),
                    "meta": {
                        "log_tag": self.log_tag,
                        "x_label": self.x_label,
                        "base_title": self.base_title,
                        "binning": binning_mode,
                        "bins_user": int(self.cfg.bins),
                        "bins": int(bins_eff),
                        "bin_edges_user": bin_edges_user,
                        "normalize": bool(self.cfg.normalize),
                        "xmin_effective": xmin_effective,
                        "xmax_user": self.cfg.xmax,
                        "xmax_effective": None,
                        "pool_trainings": bool(self.cfg.pool_trainings),
                        "skip_first_sync_buckets": int(
                            getattr(self.cfg, "skip_first_sync_buckets", 0)
                        ),
                        "skip_first_sync_buckets_global": int(
                            getattr(self.opts, "skip_first_sync_buckets", 0) or 0
                        ),
                        "skip_first_sync_buckets_effective": int(
                            self._effective_skip_first_sync_buckets()
                        ),
                        "subset_label": self.cfg.subset_label,
                        "per_fly": True,
                        "min_segs_per_fly": int(
                            getattr(self.cfg, "min_segs_per_fly", 10)
                        ),
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
                "counts": np.zeros((0, bins_eff), dtype=int),
                "bin_edges": empty_edges,
                "n_raw": np.zeros((0,), dtype=int),
                "n_used": np.zeros((0,), dtype=int),
                "n_dropped": np.zeros((0,), dtype=int),
                "meta": {
                    "log_tag": self.log_tag,
                    "x_label": self.x_label,
                    "base_title": self.base_title,
                    "binning": binning_mode,
                    "bins_user": int(self.cfg.bins),
                    "bins": int(bins_eff),
                    "bin_edges_user": bin_edges_user,
                    "normalize": bool(self.cfg.normalize),
                    "xmin_effective": xmin_effective,
                    "xmax_user": self.cfg.xmax,
                    "xmax_effective": None,
                    "pool_trainings": bool(self.cfg.pool_trainings),
                    "skip_first_sync_buckets": int(
                        getattr(self.cfg, "skip_first_sync_buckets", 0)
                    ),
                    "skip_first_sync_buckets_global": int(
                        getattr(self.opts, "skip_first_sync_buckets", 0) or 0
                    ),
                    "skip_first_sync_buckets_effective": int(
                        self._effective_skip_first_sync_buckets()
                    ),
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
                    pooled_units: list[object] = []
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
                            "bin_edges": empty_edges,
                            "mean": np.zeros((0, bins_eff), dtype=float),
                            "ci_lo": np.zeros((0, bins_eff), dtype=float),
                            "ci_hi": np.zeros((0, bins_eff), dtype=float),
                            "n_units": np.zeros((0, bins_eff), dtype=int),
                            "n_units_panel": np.zeros((0,), dtype=int),
                            "n_raw": np.zeros((0,), dtype=int),
                            "n_used": np.zeros((0,), dtype=int),
                            "n_dropped": np.zeros((0,), dtype=int),
                            "meta": {
                                "log_tag": self.log_tag,
                                "x_label": self.x_label,
                                "base_title": self.base_title,
                                "binning": binning_mode,
                                "bins_user": int(self.cfg.bins),
                                "bins": int(bins_eff),
                                "normalize": bool(self.cfg.normalize),
                                "bin_edges_user": bin_edges_user,
                                "xmin_effective": xmin_effective,
                                "xmax_user": self.cfg.xmax,
                                "xmax_effective": None,
                                "pool_trainings": bool(self.cfg.pool_trainings),
                                "skip_first_sync_buckets": int(
                                    getattr(self.cfg, "skip_first_sync_buckets", 0)
                                ),
                                "skip_first_sync_buckets_global": int(
                                    getattr(self.opts, "skip_first_sync_buckets", 0)
                                    or 0
                                ),
                                "skip_first_sync_buckets_effective": int(
                                    self._effective_skip_first_sync_buckets()
                                ),
                                **sel_info,
                                "subset_label": self.cfg.subset_label,
                                "per_fly": True,
                                "min_segs_per_fly": int(
                                    getattr(self.cfg, "min_segs_per_fly", 10)
                                ),
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
                for item in vlist:
                    _, v = self._split_unit(item)
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
                            "counts": np.zeros((0, bins_eff), dtype=int),
                            "bin_edges": empty_edges,
                            "n_raw": np.zeros((0,), dtype=int),
                            "n_used": np.zeros((0,), dtype=int),
                            "n_dropped": np.zeros((0,), dtype=int),
                            "meta": {
                                "log_tag": self.log_tag,
                                "x_label": self.x_label,
                                "base_title": self.base_title,
                                "binning": binning_mode,
                                "bins_user": int(self.cfg.bins),
                                "bins": int(bins_eff),
                                "normalize": bool(self.cfg.normalize),
                                "bin_edges_user": bin_edges_user,
                                "xmin_effective": xmin_effective,
                                "xmax_user": self.cfg.xmax,
                                "xmax_effective": None,
                                "pool_trainings": bool(self.cfg.pool_trainings),
                                "skip_first_sync_buckets": int(
                                    getattr(self.cfg, "skip_first_sync_buckets", 0)
                                ),
                                "skip_first_sync_buckets_global": int(
                                    getattr(self.opts, "skip_first_sync_buckets", 0)
                                    or 0
                                ),
                                "skip_first_sync_buckets_effective": int(
                                    self._effective_skip_first_sync_buckets()
                                ),
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

        # If explicit edges are provided, define eff_xmax by the last edge so exports are deterministic.
        if user_edges is not None:
            if user_edge_groups:
                lo_env = float(user_edges[0][0])
                hi_env = float(user_edges[-1][-1])
            else:
                lo_env = float(user_edges[0])
                hi_env = float(user_edges[-1])
            eff_xmax = hi_env
        else:
            lo_env = 0.0
            hi_env = float(eff_xmax) if eff_xmax is not None else None

        # Guard against degenerate histogram ranges
        if eff_xmax is not None and eff_xmax <= 0:
            eff_xmax = None
            # If we had no explicit edges, hi_env should follow eff_xmax
            if user_edges is None:
                hi_env = None

        # For downstream code that expects lo_edge/hi_edge scalars in the *non-grouped* case:
        if user_edges is not None and not user_edge_groups:
            lo_edge = lo_env
            hi_edge = hi_env
        elif user_edges is None:
            lo_edge = 0.0
            hi_edge = hi_env
        else:
            # grouped mode: we'll use per-group ranges in _clip(), not scalar lo/hi
            lo_edge = lo_env
            hi_edge = hi_env
            ranges = [(float(g[0]), float(g[-1])) for g in user_edges]

        def _clip(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return v

            if user_edge_groups:
                # keep values in the union of segment ranges
                keep = np.zeros(v.shape, dtype=bool)
                for lo, hi in ranges:
                    keep |= (v >= lo) & (v <= hi)
                return v[keep]

            # non-grouped: continuous interval
            if hi_edge is not None:
                return v[(v >= lo_edge) & (v <= hi_edge)]
            return v[v >= lo_edge]

        counts_list: list[np.ndarray] = []
        edges_list: list[Any] = []
        n_raw_list: list[int] = []
        n_used_list: list[int] = []
        n_dropped_list: list[int] = []

        mean_list: list[np.ndarray] = []
        lo_list: list[np.ndarray] = []
        hi_list: list[np.ndarray] = []
        n_units_list: list[np.ndarray] = []
        n_units_panel_list: list[int] = []
        per_unit_panel_list: list[np.ndarray | None] = []
        per_unit_ids_panel_list: list[np.ndarray | None] = []

        # for vals, label in zip(vals_by_panel, panel_labels):
        if self.cfg.per_fly:
            for vlist, label in zip(vals_by_panel_by_fly, panel_labels):
                min_n = int(getattr(self.cfg, "min_segs_per_fly", 10) or 0)
                if min_n < 0:
                    min_n = 0

                # Flatten for raw segment counting / xmax filtering diagnostics
                raw_all = []
                used_all = []
                used_ids: list[object] = []
                n_units_small = 0

                for item in vlist:
                    unit_id, v = self._split_unit(item)
                    if v is None:
                        continue

                    vv = np.asarray(v, dtype=float)
                    vv = vv[np.isfinite(vv)]
                    if vv.size == 0:
                        continue

                    vv_used = _clip(vv)
                    if vv_used.size == 0:
                        continue

                    # Enforce min_n consistently for diagnostics too
                    if min_n > 0 and vv_used.size < min_n:
                        n_units_small += 1
                        continue

                    raw_all.append(vv)
                    used_all.append(vv_used)

                    # used_ids.append(str(unit_id) if unit_id is not None else None)
                    if unit_id is None:
                        unit_id = f"anon_{len(used_ids)}"
                        print(
                            f"[{self.log_tag}] WARNING: missing unit_id; using {unit_id} in {label}"
                        )
                    used_ids.append(str(unit_id))

                ids_clean = [x for x in used_ids if x is not None]
                if len(set(ids_clean)) != len(ids_clean):
                    print(f"[{self.log_tag}] WARNING: duplicate unit_id(s) in {label}")

                if min_n > 0 and n_units_small > 0:
                    print(
                        f"[{self.log_tag}] {label}: skipped {n_units_small} fly-units with <{min_n} segments (per_fly)"
                    )

                if not used_all:
                    # No data: still emit deterministic edges
                    hi = float(eff_xmax) if eff_xmax is not None else 1.0
                    if user_edges is not None:
                        edges = user_edges
                    else:
                        edges = np.linspace(0, hi, bins_eff + 1, dtype=float)
                    edges_list.append(edges)
                    mean_list.append(np.full((bins_eff,), np.nan, dtype=float))
                    lo_list.append(np.full((bins_eff,), np.nan, dtype=float))
                    hi_list.append(np.full((bins_eff,), np.nan, dtype=float))
                    n_units_list.append(np.zeros((bins_eff,), dtype=int))
                    n_raw_list.append(0)
                    n_used_list.append(0)
                    n_dropped_list.append(0)
                    n_units_panel_list.append(0)
                    per_unit_panel_list.append(None)
                    per_unit_ids_panel_list.append(None)
                    continue

                flat_raw = np.concatenate(raw_all, axis=0)  # for n_raw
                flat_used = np.concatenate(used_all, axis=0)  # for edges / n_used
                n_raw = int(flat_raw.size)
                n_used = int(flat_used.size)
                n_dropped = int(n_raw - n_used)

                # Shared edges per panel
                if user_edges is not None:
                    edges = user_edges
                elif eff_xmax is not None:
                    edges = np.linspace(0.0, float(eff_xmax), bins_eff + 1, dtype=float)
                else:
                    # Data-driven edges per panel (may not align across groups)
                    edges = np.histogram_bin_edges(flat_used, bins=bins_eff)
                if user_edge_groups:
                    edges_list.append(edges)
                else:
                    edges_list.append(
                        np.asarray(edges, dtype=float).astype(float, copy=False)
                    )

                # Per-fly histograms
                fly_hists: list[np.ndarray] = []

                for vv_used in used_all:
                    # vv_used is already finite, clipped, and min_n-filtered
                    if user_edge_groups:
                        parts = []
                        for g in user_edges:
                            c_g, _ = np.histogram(vv_used, bins=g)
                            parts.append(c_g.astype(float, copy=False))
                        c = np.concatenate(parts, axis=0)
                    else:
                        c, _ = np.histogram(vv_used, bins=edges)

                    c = c.astype(float, copy=False)
                    if self.cfg.normalize:
                        tot = float(np.sum(c))
                        if tot > 0:
                            c = c / tot
                        else:
                            c[:] = np.nan

                    fly_hists.append(c)

                n_units_panel_list.append(len(fly_hists))
                if not fly_hists:
                    mean = np.full((bins_eff,), np.nan, dtype=float)
                    lo = np.full((bins_eff,), np.nan, dtype=float)
                    hi = np.full((bins_eff,), np.nan, dtype=float)
                    n_units = np.zeros((bins_eff,), dtype=int)
                    per_unit_panel_list.append(None)
                    per_unit_ids_panel_list.append(None)
                else:
                    M = np.stack(fly_hists, axis=0)  # (n_units, bins)
                    mean = np.full((bins_eff,), np.nan, dtype=float)
                    lo = np.full((bins_eff,), np.nan, dtype=float)
                    hi = np.full((bins_eff,), np.nan, dtype=float)
                    n_units = np.zeros((bins_eff,), dtype=int)
                    per_unit_panel_list.append(M)
                    per_unit_ids_panel_list.append(np.asarray(used_ids, dtype=object))
                    for j in range(bins_eff):
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
                    counts_list.append(np.zeros((bins_eff,), dtype=int))
                    hi = float(eff_xmax) if eff_xmax is not None else 1.0
                    if user_edges is not None:
                        edges_list.append(user_edges)
                    else:
                        edges_list.append(np.linspace(0, hi, bins_eff + 1, dtype=float))
                    n_raw_list.append(0)
                    n_used_list.append(0)
                    n_dropped_list.append(0)
                    continue
                vals = np.asarray(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                n_raw = int(vals.size)

                vals = _clip(vals)
                n_used = int(vals.size)
                n_dropped = int(n_raw - n_used)

                if n_used == 0:
                    counts = np.zeros((bins_eff,), dtype=int)
                    hi = float(eff_xmax) if eff_xmax is not None else 1.0
                    if user_edges is not None:
                        edges = user_edges
                    else:
                        edges = np.linspace(0, hi, bins_eff + 1, dtype=float)
                else:
                    if user_edge_groups:
                        counts_parts = []
                        # note: don't recompute edges; use user_edges groups verbatim
                        for g in user_edges:
                            c, _ = np.histogram(vals, bins=g)
                            counts_parts.append(c.astype(int, copy=False))
                        counts = np.concatenate(counts_parts, axis=0)
                        edges = user_edges
                    elif user_edges is not None:
                        counts, edges = np.histogram(vals, bins=user_edges)
                    elif eff_xmax is not None:
                        # Enforce shared edges via an explicit range.
                        counts, edges = np.histogram(
                            vals, bins=bins_eff, range=(0.0, float(eff_xmax))
                        )
                    else:
                        # Fallback: data-driven edges (will likely not align across groups).
                        counts, edges = np.histogram(vals, bins=bins_eff)

                counts_list.append(counts.astype(int, copy=False))
                if user_edge_groups:
                    edges_list.append(user_edges)
                else:
                    edges_list.append(
                        np.asarray(edges, dtype=float).astype(float, copy=False)
                    )
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
            else np.zeros((0, bins_eff), dtype=int)
        )
        if user_edge_groups:
            edges_arr = np.empty((len(edges_list),), dtype=object)
            for i, item in enumerate(edges_list):
                edges_arr[i] = item
        else:
            edges_arr = (
                np.stack(edges_list, axis=0)
                if edges_list
                else np.zeros((0, bins_eff + 1), dtype=float)
            )

        if self.cfg.per_fly:
            mean_arr = (
                np.stack(mean_list, axis=0)
                if mean_list
                else np.zeros((0, bins_eff), dtype=float)
            )
            lo_arr = (
                np.stack(lo_list, axis=0)
                if lo_list
                else np.zeros((0, bins_eff), dtype=float)
            )
            hi_arr = (
                np.stack(hi_list, axis=0)
                if hi_list
                else np.zeros((0, bins_eff), dtype=float)
            )
            n_units_arr = (
                np.stack(n_units_list, axis=0)
                if n_units_list
                else np.zeros((0, bins_eff), dtype=int)
            )

        meta = {
            "log_tag": self.log_tag,
            "x_label": self.x_label,
            "base_title": self.base_title,
            # binning bookkeeping
            "binning": binning_mode,  # "edges" or "bins"
            "bins_user": int(
                self.cfg.bins
            ),  # user-supplied bins parameter (used only when bin_edges is None)
            "bins": int(bins_eff),  # effective payload bins
            "bin_edges_user": bin_edges_user,  # explicit edges (if provided)
            "normalize": bool(self.cfg.normalize),
            "xmax_user": self.cfg.xmax,
            "xmax_effective": eff_xmax,
            "xmin_effective": float(lo_edge),
            "pool_trainings": bool(self.cfg.pool_trainings),
            "skip_first_sync_buckets": int(
                getattr(self.cfg, "skip_first_sync_buckets", 0)
            ),
            "skip_first_sync_buckets_global": int(
                getattr(self.opts, "skip_first_sync_buckets", 0) or 0
            ),
            "skip_first_sync_buckets_effective": int(
                self._effective_skip_first_sync_buckets()
            ),
            **sel_info,
            "subset_label": self.cfg.subset_label,
            "per_fly": bool(self.cfg.per_fly),
            "min_segs_per_fly": int(getattr(self.cfg, "min_segs_per_fly", 10)),
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
                    "n_units_panel": np.asarray(n_units_panel_list, dtype=int),
                    "per_unit_panel": np.asarray(per_unit_panel_list, dtype=object),
                    "per_unit_ids_panel": np.asarray(
                        per_unit_ids_panel_list, dtype=object
                    ),
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
            counts=None if self.cfg.per_fly else data.get("counts", None),
            # per-fly-mode payload
            mean=data.get("mean", None),
            ci_lo=data.get("ci_lo", None),
            ci_hi=data.get("ci_hi", None),
            n_units=data.get("n_units", None),
            n_units_panel=data.get("n_units_panel", None),
            per_unit_panel=data.get("per_unit_panel", None),
            per_unit_ids_panel=data.get("per_unit_ids_panel", None),
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
            edges_item = edges_arr[idx]

            # ---- choose y payload (per-fly vs pooled)
            if self.cfg.per_fly:
                y = np.asarray(data["mean"][idx], dtype=float)
                if not np.any(np.isfinite(y)):
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    continue
                lo_ci = (
                    np.asarray(data["ci_lo"][idx], dtype=float) if self.cfg.ci else None
                )
                hi_ci = (
                    np.asarray(data["ci_hi"][idx], dtype=float) if self.cfg.ci else None
                )
            else:
                counts = np.asarray(data["counts"][idx], dtype=float)
                total = float(np.sum(counts))
                if not np.isfinite(total) or total <= 0:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    continue
                if self.cfg.normalize:
                    y = counts / total
                else:
                    y = counts
                lo_ci = hi_ci = None  # not used in pooled mode

            # ---- draw bars (grouped edges vs flat edges)
            edges_item = edges_arr[idx]
            edges_norm = normalize_panel_edges(edges_item)
            if isinstance(edges_norm, list):
                # Grouped mode: edges_item is a list of 1D edge arrays
                pos = 0
                groups_iter = edges_norm
                x0 = float(groups_iter[0][0])
                x1 = float(groups_iter[-1][-1])

                for g in groups_iter:
                    g = np.asarray(g, dtype=float).ravel()
                    nb = int(g.size - 1)
                    if nb <= 0:
                        continue

                    y_seg = np.asarray(y[pos : pos + nb], dtype=float).ravel()
                    lefts = g[:-1]
                    widths = np.diff(g)
                    centers = lefts + 0.5 * widths

                    ax.bar(lefts, y_seg, width=widths, align="edge")

                    if (
                        self.cfg.per_fly
                        and self.cfg.ci
                        and lo_ci is not None
                        and hi_ci is not None
                    ):
                        lo_seg = lo_ci[pos : pos + nb]
                        hi_seg = hi_ci[pos : pos + nb]
                        yerr = np.vstack([y_seg - lo_seg, hi_seg - y_seg])
                        yerr = np.where(np.isfinite(yerr), yerr, 0)
                        ax.errorbar(
                            centers,
                            y_seg,
                            yerr=yerr,
                            fmt="none",
                            capsize=2,
                            ecolor="0.2",
                            elinewidth=1.0,
                            zorder=3,
                        )
                    pos += nb

                ax.set_xlim(x0, x1)
            else:
                # Flat mode: edges_item is a 1D numeric array
                edges = edges_norm
                bin_widths = np.diff(edges)
                lefts = edges[:-1]
                centers = lefts + 0.5 * bin_widths

                ax.bar(lefts, y, width=bin_widths, align="edge")

                if (
                    self.cfg.per_fly
                    and self.cfg.ci
                    and lo_ci is not None
                    and hi_ci is not None
                ):
                    yerr = np.vstack([y - lo_ci, hi_ci - y])
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
                if edges.size >= 2:
                    ax.set_xlim(float(edges[0]), float(edges[-1]))

            # ---- shared axis formatting
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
        if getattr(self.cfg, "show_suptitle", False):
            title = self.base_title
            if self.cfg.subset_label:
                title = f"{title}\n{self.cfg.subset_label}"
            fig.suptitle(title)
        fig.tight_layout()

        writeImage(self.cfg.out_file, format=self.opts.imageFormat)
        plt.close(fig)
        print(f"[{self.log_tag}] wrote {self.cfg.out_file}")
