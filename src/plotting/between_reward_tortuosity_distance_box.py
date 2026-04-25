from __future__ import annotations

from dataclasses import dataclass
import os
import json
from typing import Sequence

import numpy as np

from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.between_reward_segment_metrics import (
    path_length_and_max_radius_mm_masked,
)
from src.plotting.event_chain_plotter import EventChainPlotter
from src.plotting.plot_customizer import PlotCustomizer
from src.plotting.wall_contact_utils import build_wall_contact_mask_for_window
from src.utils import util


@dataclass
class BetweenRewardTortuosityDistanceBoxConfig:
    out_file: str = "imgs/btw_rwd_tortuosity_by_max_radius_box.png"
    export_npz: str | None = None
    training_index: int = 0
    skip_first_sync_buckets: int = 0
    keep_first_sync_buckets: int = 0
    x_bin_edges_mm: Sequence[float] | None = None
    x_min_mm: float = 3.0
    x_max_mm: float = 50.0
    x_bin_width_mm: float = 3.0
    exclude_wall_contact: bool = False
    exclude_nonwalking_frames: bool = False
    exclude_reward_endpoints: bool = False
    min_walk_frames: int = 2
    min_radius_mm: float = 0.0
    min_segments_per_bin: int = 1
    unit_stat: str = "median"  # median, mean
    segment_level: bool = False
    showfliers: bool = False
    ymax: float | None = None
    show_plot: bool = True
    segments_tsv: str | None = None
    segments_plot_dir: str | None = None
    segments_top_k: int = 3
    segments_median_k: int = 3
    segments_random_k: int = 3
    segments_seed: int = 0
    segments_plot_zoom: bool = False
    segments_plot_zoom_radius_mm: float | None = None
    segments_plot_pad: int = 5


@dataclass(frozen=True)
class BetweenRewardTortuosityDistanceBoxResult:
    x_edges_mm: np.ndarray
    bin_labels: np.ndarray
    values_by_bin: np.ndarray
    n_segments: np.ndarray
    n_units: np.ndarray
    unit_stat: str
    segment_level: bool
    q1: np.ndarray
    median: np.ndarray
    q3: np.ndarray
    whisker_low: np.ndarray
    whisker_high: np.ndarray
    meta: dict

    def save_npz(self, path: str) -> None:
        util.ensureDir(path)
        np.savez_compressed(
            path,
            x_edges_mm=np.asarray(self.x_edges_mm, dtype=float),
            bin_labels=np.asarray(self.bin_labels, dtype=object),
            values_by_bin=np.asarray(self.values_by_bin, dtype=object),
            n_segments=np.asarray(self.n_segments, dtype=int),
            n_units=np.asarray(self.n_units, dtype=int),
            unit_stat=np.asarray(str(self.unit_stat), dtype=object),
            segment_level=np.asarray(bool(self.segment_level), dtype=bool),
            q1=np.asarray(self.q1, dtype=float),
            median=np.asarray(self.median, dtype=float),
            q3=np.asarray(self.q3, dtype=float),
            whisker_low=np.asarray(self.whisker_low, dtype=float),
            whisker_high=np.asarray(self.whisker_high, dtype=float),
            meta_json=json.dumps(self.meta, sort_keys=True),
        )

    @staticmethod
    def load_npz(path: str) -> "BetweenRewardTortuosityDistanceBoxResult":
        z = np.load(path, allow_pickle=True)
        meta_json = z["meta_json"].item() if "meta_json" in z.files else "{}"
        if isinstance(meta_json, (bytes, bytearray)):
            meta_json = meta_json.decode("utf-8")
        return BetweenRewardTortuosityDistanceBoxResult(
            x_edges_mm=np.asarray(z["x_edges_mm"], dtype=float),
            bin_labels=np.asarray(z["bin_labels"], dtype=object),
            values_by_bin=np.asarray(z["values_by_bin"], dtype=object),
            n_segments=np.asarray(z["n_segments"], dtype=int),
            n_units=np.asarray(z["n_units"], dtype=int),
            unit_stat=str(z["unit_stat"].item()) if "unit_stat" in z.files else "raw_segment",
            segment_level=bool(z["segment_level"].item()) if "segment_level" in z.files else True,
            q1=np.asarray(z["q1"], dtype=float),
            median=np.asarray(z["median"], dtype=float),
            q3=np.asarray(z["q3"], dtype=float),
            whisker_low=np.asarray(z["whisker_low"], dtype=float),
            whisker_high=np.asarray(z["whisker_high"], dtype=float),
            meta=json.loads(str(meta_json)),
        )


class BetweenRewardTortuosityDistanceBoxPlotter:
    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardTortuosityDistanceBoxConfig,
    ):
        self.vas = vas
        self.opts = opts
        self.gls = gls
        self.customizer = customizer
        self.cfg = cfg
        self.log_tag = "btw_rwd_tortuosity_box"

    def _x_edges(self) -> np.ndarray:
        if self.cfg.x_bin_edges_mm is not None:
            edges = np.asarray(self.cfg.x_bin_edges_mm, dtype=float).ravel()
        else:
            w = float(self.cfg.x_bin_width_mm)
            if not np.isfinite(w) or w <= 0:
                w = 3.0
            x0 = float(self.cfg.x_min_mm)
            x1 = float(self.cfg.x_max_mm)
            edges = np.arange(x0, x1 + 0.5 * w, w, dtype=float)
        if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("tortuosity box distance bin edges must be finite and increasing")
        return edges

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

    @staticmethod
    def _unit_id(va, *, f: int) -> str:
        return f"{getattr(va, 'fn', 'unknown_video')}|trx_idx={int(f)}"

    @staticmethod
    def _fly_id(va, *, role_idx: int, trx_idx: int) -> int:
        """
        Return a stable fly/chamber ID for exported diagnostics.

        `va.f` is often scalar, but some code paths treat it as sequence-like.
        Prefer role_idx, then fall back to trx_idx.
        """
        f = getattr(va, "f", None)
        if f is None:
            return -1
        try:
            if isinstance(f, (list, tuple, np.ndarray)):
                if len(f) > role_idx:
                    return int(f[role_idx])
                if len(f) > trx_idx:
                    return int(f[trx_idx])
        except Exception:
            pass
        try:
            return int(f)
        except Exception:
            return -1

    @staticmethod
    def _unit_summary(vals: Sequence[float], mode: str) -> float:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        mode = str(mode or "median").strip().lower()
        if mode == "mean":
            return float(np.nanmean(arr))
        return float(np.nanmedian(arr))

    @staticmethod
    def _box_stats(vals: np.ndarray) -> tuple[float, float, float, float, float]:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        q1, med, q3 = np.nanpercentile(vals, [25, 50, 75])
        iqr = q3 - q1
        lo_fence = q1 - 1.5 * iqr
        hi_fence = q3 + 1.5 * iqr
        lo_vals = vals[vals >= lo_fence]
        hi_vals = vals[vals <= hi_fence]
        wlo = float(np.nanmin(lo_vals)) if lo_vals.size else float(np.nanmin(vals))
        whi = float(np.nanmax(hi_vals)) if hi_vals.size else float(np.nanmax(vals))
        return float(q1), float(med), float(q3), wlo, whi

    @staticmethod
    def _video_base(va) -> str:
        fn = getattr(va, "fn", None)
        if fn:
            try:
                return os.path.splitext(os.path.basename(str(fn)))[0]
            except Exception:
                pass
        return f"va_{id(va)}"

    @staticmethod
    def _bin_dirname(lo: float, hi: float) -> str:
        def fmt(x):
            return f"{float(x):g}".replace(".", "p")

        return f"dmax_{fmt(lo)}_{fmt(hi)}mm"

    @staticmethod
    def _bin_label(lo: float, hi: float) -> str:
        return f"[{float(lo):g}, {float(hi):g})"

    def _iter_segment_records(self, edges: np.ndarray):
        B = int(edges.size - 1)
        warned_missing_wc = [False]
        t_idx = int(self.cfg.training_index)

        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if va.trx[0].bad() or t_idx >= len(getattr(va, "trns", [])):
                continue

            trn = va.trns[t_idx]
            video_id = self._video_base(va)
            for role_idx, f in enumerate(va.flies):
                if not va.noyc and f != 0:
                    continue

                traj = va.trx[f]
                unit_id = self._unit_id(va, f=f)
                skip_first = int(max(0, self.cfg.skip_first_sync_buckets or 0))
                keep_first = int(max(0, self.cfg.keep_first_sync_buckets or 0))
                fi, df, n_buckets, complete = sync_bucket_window(
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

                n_frames = int(max(1, n_buckets * df))
                px_per_mm = float(traj.pxPerMmFloor * va.xf.fctr)
                if not np.isfinite(px_per_mm) or px_per_mm <= 0:
                    continue

                wc = build_wall_contact_mask_for_window(
                    va,
                    f,
                    fi=fi,
                    n_frames=n_frames,
                    enabled=bool(self.cfg.exclude_wall_contact),
                    warned_missing_wc=warned_missing_wc,
                    log_tag=self.log_tag,
                )
                nonwalk_mask = self._build_nonwalk_mask(
                    va,
                    f,
                    fi=fi,
                    n_frames=n_frames,
                    enabled=bool(self.cfg.exclude_nonwalking_frames),
                )
                min_walk_frames = int(max(2, self.cfg.min_walk_frames or 2))
                try:
                    cx, cy, _ = trn.circles(f)[0]
                    center_xy = (float(cx), float(cy))
                except Exception:
                    continue

                for seg in va._iter_between_reward_segment_com(
                    trn,
                    f,
                    fi=fi,
                    df=df,
                    n_buckets=n_buckets,
                    complete=complete,
                    relative_to_reward=True,
                    per_segment_min_meddist_mm=0.0,
                    exclude_wall=bool(self.cfg.exclude_wall_contact),
                    wc=wc,
                    exclude_nonwalk=bool(self.cfg.exclude_nonwalking_frames),
                    nonwalk_mask=nonwalk_mask,
                    min_walk_frames=min_walk_frames,
                    exclude_reward_endpoints=bool(self.cfg.exclude_reward_endpoints),
                    debug=False,
                    yield_skips=False,
                ):
                    endpoint_offset = 1 if self.cfg.exclude_reward_endpoints else 0
                    s = int(seg.s) + endpoint_offset
                    e = int(seg.e) - endpoint_offset
                    if e <= s:
                        continue

                    path_mm, radius_mm = path_length_and_max_radius_mm_masked(
                        traj=traj,
                        s=s,
                        e=e,
                        fi=fi,
                        nonwalk_mask=nonwalk_mask,
                        exclude_nonwalk=bool(self.cfg.exclude_nonwalking_frames),
                        px_per_mm=px_per_mm,
                        center_xy=center_xy,
                        min_keep_frames=min_walk_frames,
                    )
                    if not (np.isfinite(path_mm) and np.isfinite(radius_mm)):
                        continue
                    if radius_mm <= max(0.0, float(self.cfg.min_radius_mm or 0.0)):
                        continue

                    j = int(np.searchsorted(edges, radius_mm, side="right") - 1)
                    if j < 0 or j >= B:
                        continue
                    tort = float(path_mm / radius_mm)
                    yield {
                        "bin_idx": int(j),
                        "bin_lo_mm": float(edges[j]),
                        "bin_hi_mm": float(edges[j + 1]),
                        "path_length_mm": float(path_mm),
                        "max_radius_mm": float(radius_mm),
                        "tortuosity": tort,
                        "metric_s": int(s),
                        "metric_e": int(e),
                        "s": int(seg.s),
                        "e": int(seg.e),
                        "b_idx": int(getattr(seg, "b_idx", -1)),
                        "video_id": str(video_id),
                        "unit_id": str(unit_id),
                        "fly_id": int(self._fly_id(va, role_idx=role_idx, trx_idx=f)),
                        "trx_idx": int(f),
                        "role_idx": int(role_idx),
                        "fly_role": "exp" if int(role_idx) == 0 else "yok",
                        "training_index": int(t_idx),
                        "va": va,
                    }

    def compute_result(self) -> BetweenRewardTortuosityDistanceBoxResult:
        edges = self._x_edges()
        B = int(edges.size - 1)
        segment_values: list[list[float]] = [[] for _ in range(B)]
        by_unit_by_bin: list[dict[str, list[float]]] = [dict() for _ in range(B)]
        t_idx = int(self.cfg.training_index)

        for row in self._iter_segment_records(edges):
            j = int(row["bin_idx"])
            val = float(row["tortuosity"])
            unit_id = str(row["unit_id"])
            segment_values[j].append(val)
            by_unit_by_bin[j].setdefault(unit_id, []).append(val)

        min_n = int(max(1, self.cfg.min_segments_per_bin or 1))
        values_arr = np.empty((B,), dtype=object)
        q1 = np.full((B,), np.nan, dtype=float)
        med = np.full((B,), np.nan, dtype=float)
        q3 = np.full((B,), np.nan, dtype=float)
        wlo = np.full((B,), np.nan, dtype=float)
        whi = np.full((B,), np.nan, dtype=float)
        n_segments = np.zeros((B,), dtype=int)
        n_units = np.zeros((B,), dtype=int)

        for j in range(B):
            n_segments[j] = int(len(segment_values[j]))
            n_units[j] = int(len(by_unit_by_bin[j]))
            if self.cfg.segment_level:
                arr = np.asarray(segment_values[j], dtype=float)
            else:
                unit_vals = [
                    self._unit_summary(vals, self.cfg.unit_stat)
                    for vals in by_unit_by_bin[j].values()
                ]
                arr = np.asarray(unit_vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if n_segments[j] < min_n:
                arr = np.asarray([], dtype=float)
            values_arr[j] = arr
            q1[j], med[j], q3[j], wlo[j], whi[j] = self._box_stats(arr)

        labels = np.asarray(
            [f"[{edges[j]:g}, {edges[j + 1]:g})" for j in range(B)],
            dtype=object,
        )
        meta = {
            "log_tag": self.log_tag,
            "training_index": int(t_idx),
            "training_label": f"training {t_idx + 1}",
            "metric": "between_reward_tortuosity_by_max_radius",
            "x_label": "Maximum distance from reward circle center [mm]",
            "y_label": "Loop tortuosity",
            "tortuosity_definition": "path_length_mm / max_radial_distance_from_reward_center_mm",
            "skip_first_sync_buckets": int(self.cfg.skip_first_sync_buckets),
            "keep_first_sync_buckets": int(self.cfg.keep_first_sync_buckets),
            "exclude_wall_contact": bool(self.cfg.exclude_wall_contact),
            "exclude_nonwalking_frames": bool(self.cfg.exclude_nonwalking_frames),
            "exclude_reward_endpoints": bool(self.cfg.exclude_reward_endpoints),
            "min_walk_frames": int(self.cfg.min_walk_frames),
            "min_radius_mm": float(self.cfg.min_radius_mm),
            "min_segments_per_bin": int(min_n),
            "unit_stat": str(self.cfg.unit_stat),
            "segment_level": bool(self.cfg.segment_level),
            "box_values": (
                "raw_segments" if self.cfg.segment_level else f"per_fly_{self.cfg.unit_stat}"
            ),
        }
        return BetweenRewardTortuosityDistanceBoxResult(
            x_edges_mm=edges,
            bin_labels=labels,
            values_by_bin=values_arr,
            n_segments=n_segments,
            n_units=n_units,
            unit_stat=str(self.cfg.unit_stat),
            segment_level=bool(self.cfg.segment_level),
            q1=q1,
            median=med,
            q3=q3,
            whisker_low=wlo,
            whisker_high=whi,
            meta=meta,
        )

    def export_npz(self, out_npz: str) -> None:
        res = self.compute_result()
        res.save_npz(out_npz)
        print(f"[{self.log_tag}] wrote boxplot export {out_npz}")

    def _collect_sampled_segments(self) -> list[dict]:
        edges = self._x_edges()
        B = int(edges.size - 1)
        by_bin: list[list[dict]] = [[] for _ in range(B)]
        for row in self._iter_segment_records(edges):
            by_bin[int(row["bin_idx"])].append(row)

        rng = np.random.default_rng(int(self.cfg.segments_seed or 0))
        top_k = int(max(0, self.cfg.segments_top_k or 0))
        median_k = int(max(0, self.cfg.segments_median_k or 0))
        random_k = int(max(0, self.cfg.segments_random_k or 0))
        rows: list[dict] = []

        for j, bin_rows in enumerate(by_bin):
            if not bin_rows:
                continue
            sorted_rows = sorted(bin_rows, key=lambda r: float(r["tortuosity"]))

            for rank, row in enumerate(reversed(sorted_rows[-top_k:]), start=1):
                out = dict(row)
                out["sample_type"] = "top_tortuosity"
                out["rank"] = int(rank)
                rows.append(out)

            if median_k > 0:
                med = float(np.nanmedian([float(r["tortuosity"]) for r in sorted_rows]))
                med_rows = sorted(
                    sorted_rows, key=lambda r: abs(float(r["tortuosity"]) - med)
                )
                for rank, row in enumerate(med_rows[:median_k], start=1):
                    out = dict(row)
                    out["sample_type"] = "median_tortuosity"
                    out["rank"] = int(rank)
                    rows.append(out)

            if random_k > 0:
                k = min(random_k, len(bin_rows))
                idxs = rng.choice(len(bin_rows), size=k, replace=False)
                for rank, idx in enumerate(idxs, start=1):
                    out = dict(bin_rows[int(idx)])
                    out["sample_type"] = "random"
                    out["rank"] = int(rank)
                    rows.append(out)

        return rows

    def write_sampled_segments_tsv(self, path: str) -> None:
        rows = self._collect_sampled_segments()
        util.ensureDir(path)
        cols = [
            "sample_type",
            "rank",
            "bin_lo_mm",
            "bin_hi_mm",
            "path_length_mm",
            "max_radius_mm",
            "tortuosity",
            "metric_s",
            "metric_e",
            "s",
            "e",
            "b_idx",
            "video_id",
            "unit_id",
            "fly_id",
            "trx_idx",
            "role_idx",
            "fly_role",
            "training_index",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for row in rows:
                f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
        print(f"[{self.log_tag}] wrote sampled segment TSV {path}")

    def plot_sampled_segments(self, out_dir: str) -> None:
        rows = self._collect_sampled_segments()
        if not rows:
            print(f"[{self.log_tag}] no sampled tortuosity segments to plot")
            return

        image_format = str(getattr(self.opts, "imageFormat", "png")).lstrip(".")
        os.makedirs(out_dir, exist_ok=True)
        wrote = 0
        for row in rows:
            va = row.get("va")
            trx_idx = row.get("trx_idx")
            role_idx = row.get("role_idx")
            if va is None or trx_idx is None:
                continue
            trj = va.trx[int(trx_idx)]
            bin_dir = os.path.join(
                out_dir,
                self._bin_dirname(float(row["bin_lo_mm"]), float(row["bin_hi_mm"])),
            )
            os.makedirs(bin_dir, exist_ok=True)
            out_path = os.path.join(
                bin_dir,
                (
                    f"{row['sample_type']}_rank{int(row['rank']):02d}_"
                    f"{row['video_id']}_trx{int(trx_idx)}_"
                    f"trn{int(row['training_index']) + 1}_"
                    f"rw{int(row['metric_s'])}-{int(row['metric_e'])}.{image_format}"
                ),
            )
            title_suffix = (
                f"| {row['sample_type']} #{int(row['rank'])} | "
                f"bin {self._bin_label(float(row['bin_lo_mm']), float(row['bin_hi_mm']))}"
            )
            annotation = (
                f"path length = {float(row['path_length_mm']):.2f} mm\n"
                f"max radius = {float(row['max_radius_mm']):.2f} mm\n"
                f"tortuosity = {float(row['tortuosity']):.2f}"
            )
            plotter = EventChainPlotter(trj=trj, va=va, image_format=image_format)
            plotter.plot_between_reward_interval(
                trn_index=int(row["training_index"]),
                start_reward=int(row["s"]),
                end_reward=int(row["e"]),
                image_format=image_format,
                role_idx=int(role_idx),
                pad=int(max(0, self.cfg.segments_plot_pad or 0)),
                zoom=bool(self.cfg.segments_plot_zoom),
                zoom_radius_mm=self.cfg.segments_plot_zoom_radius_mm,
                out_path=out_path,
                title_suffix=title_suffix,
                annotation_text=annotation,
            )
            wrote += 1
        print(f"[{self.log_tag}] wrote {wrote} sampled segment image(s) to {out_dir}")

    def plot_boxplot(self) -> None:
        from src.plotting.between_reward_tortuosity_distance_box_plotter import (
            plot_single_box_result,
        )

        res = self.compute_result()
        plot_single_box_result(
            res,
            out_file=self.cfg.out_file,
            customizer=self.customizer,
            showfliers=bool(self.cfg.showfliers),
            ymax=self.cfg.ymax,
        )
