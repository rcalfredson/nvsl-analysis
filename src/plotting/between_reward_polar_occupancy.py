# src/plotting/between_reward_polar_occupancy.py

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import os
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.plotting.plot_customizer import PlotCustomizer
from src.utils.common import writeImage


@dataclass
class BetweenRewardPolarOccupancyConfig:
    out_file: str
    bins: int = 36  # angular sectors
    normalize: bool = True
    pool_trainings: bool = False
    subset_label: str | None = None

    # --- plot mode ---
    # "theta": 1D angular histogram
    # "theta_r": 2D heatmap (theta x radius)
    mode: str = "theta"

    # --- 2D (theta x r) heatmap params ---
    r_bins: int = 12
    # Max radius (mm) for binning/axis in 2D mode; if None, auto-derived from data.
    r_max: float | None = None
    r_tick_step: float = 5.0
    # Normalization for 2D mode:
    # - "global": divide by total samples
    # - "per_theta": normalize within each theta sector
    # - "none": raw counts
    theta_r_normalize: str = "global"

    # Coordinate convention: for image coords (y down), flip_y=True makes "up" positive.
    flip_y: bool = True

    # --- geometry normalization ---
    # If True: mirror coordinate frame using VideoAnalysis.mirror() so that
    # right-column (va_f==1) large/large2 data overlays with left-column data.
    mirror: bool = True

    # Optional: stabilize radial scale across plots if desired
    rmax: float | None = None

    # --- behavior filtering ---
    only_walking: bool = False
    exclude_wall_contact: bool = False

    # --- granularity ---
    per_fly: bool = False
    per_fly_out_dir: str = "imgs/btw_rwd_polar_per_fly"
    max_per_fly_plots: int | None = None  # optional cap
    # How to aggregate across flies:
    # - "frames": current behavior (each frame equal weight)
    # - "flies": average per-fly normalized distributions (each fly equal weight)
    aggregate: str = "frames"
    min_frames_per_fly: int = 0

    # --- debug ---
    debug: bool = False
    debug_out_tsv: str | None = None  # e.g. "imgs/btw_rwd_polar_debug.tsv"
    debug_max_rows: int = 20000  # cap so you don’t dump millions
    debug_stride: int = 1  # 1=every frame, 5=every 5th frame

    # --- debug per-fly TSVs + sampling
    debug_per_fly: bool = False
    debug_max_rows_per_fly: int = 20000
    debug_per_fly_out_dir: str = "imgs/btw_rwd_polar_debug_per_fly"
    debug_sample_mode: str = "first"  # "first" | "random_windows"
    debug_num_windows: int = 50
    debug_window_len: int = 50
    debug_seed: int = 0

    # --- segment-level bin attribution (for chasing peaks) ---
    seg_majority_out_tsv: str | None = (
        None  # e.g., "imgs/btw_rwd_polar_seg_majority.tsv"
    )
    seg_majority_target_degs: tuple[float, ...] = (45.0, 315.0)
    seg_majority_thresh: float = 0.5  # "majority of frames"
    seg_majority_min_frames: int = 20  # ignore tiny segments
    seg_majority_max_rows: int = 50000  # safety cap


class BetweenRewardPolarOccupancyPlotter:
    """
    Aggregate between-reward positions across VideoAnalysis instances and
    plot a normalized polar angular occupancy distribution around the
    reward-circle center.

    Uses only experimental flies (f == 0 in multi-fly recordings).
    """

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        customizer: PlotCustomizer,
        cfg: BetweenRewardPolarOccupancyConfig,
    ):
        self.vas = vas
        self.opts = opts
        self.gls = gls
        self.customizer = customizer
        self.cfg = cfg

        # to be used only for label naming, not for geometry
        self.trns = vas[0].trns
        self.n_trn = len(self.trns)

    # ---------- debug helpers ----------

    def _stable_seed_int(self, *parts: object) -> int:
        """
        Turn a tuple of identifiers into a stable 32-bit seed.
        (Python's built-in hash() is randomized between processes.)
        """
        s = "|".join("" if p is None else str(p) for p in parts)
        h = hashlib.md5(s.encode("utf-8")).hexdigest()  # stable
        return int(h[:8], 16)

    def _select_random_windows(
        self,
        on: Sequence[int],
        win_len: int,
        num_windows: int,
        seed: int,
    ) -> list[tuple[int, int, int]]:
        """
        Returns list of (seg_i, start_frame, stop_frame) windows sampled from
        between-reward segments defined by consecutive indices in `on`.

        Sampling is deterministic given the seed.
        """
        segs: list[tuple[int, int, int]] = []
        for seg_i in range(len(on) - 1):
            s = int(on[seg_i])
            e = int(on[seg_i + 1])
            if e - s >= win_len:
                segs.append((seg_i, s, e))

        segs.sort(key=lambda t: (t[1], t[2], t[0]))  # (start, end, seg_i)
        if not segs:
            return []

        rng = np.random.default_rng(seed)
        n = min(num_windows, len(segs))
        chosen = rng.choice(len(segs), size=n, replace=False)

        out: list[tuple[int, int, int]] = []
        for idx in chosen:
            seg_i, s, e = segs[int(idx)]
            start = int(rng.integers(s, e - win_len + 1))
            out.append((seg_i, start, start + win_len))
        return out

    def _iter_debug_windows(
        self,
        *,
        on: Sequence[int],
        seed: int,
    ) -> list[tuple[int, int, int]]:
        """
        Returns windows in the form: (seg_i, start_frame, stop_frame).

        - "first": returns full between-reward segments in encounter order.
        - "random_windows": returns fixed-length windows sampled across segments.
        """
        mode = str(self.cfg.debug_sample_mode or "first")

        if mode == "random_windows":
            return self._select_random_windows(
                on=on,
                win_len=int(self.cfg.debug_window_len),
                num_windows=int(self.cfg.debug_num_windows),
                seed=int(seed),
            )

        # default: "first"
        out: list[tuple[int, int, int]] = []
        for seg_i in range(len(on) - 1):
            s = int(on[seg_i])
            e = int(on[seg_i + 1])
            if e > s + 1:
                out.append((seg_i, s, e))
        return out

    def _bin_centers_deg_0_360(self, bin_edges: np.ndarray) -> np.ndarray:
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        deg = (np.degrees(centers) + 360.0) % 360.0
        return deg

    def _closest_bin_idx_for_deg(
        self, bin_center_degs: np.ndarray, target_deg: float
    ) -> int:
        # circular distance on [0, 360)
        d = np.abs(((bin_center_degs - target_deg + 180.0) % 360.0) - 180.0)
        return int(np.argmin(d))

    # ---------- data collection ----------
    def _fly_keys(self, per_fly_by_trn: dict[tuple[str, int], np.ndarray]) -> list[str]:
        return sorted({fk for (fk, _t) in per_fly_by_trn.keys()})

    def _is_fly_aggregate(self) -> bool:
        return str(getattr(self.cfg, "aggregate", "frames") or "frames") == "flies"

    def _pool_per_fly_by_trn(
        self, per_fly_by_trn: dict[tuple[str, int], np.ndarray]
    ) -> dict[tuple[str, int], np.ndarray]:
        """
        If cfg.pool_trainings: collapse all trainings into t_idx=0 per fly.
        Else: return as-is.
        """
        if not bool(getattr(self.cfg, "pool_trainings", False)):
            return per_fly_by_trn
        out: dict[tuple[str, int], list[float]] = defaultdict(list)
        for (fk, t_idx), arr in per_fly_by_trn.items():
            if arr.size:
                out[(fk, 0)].extend(arr.tolist())
        return {k: np.asarray(v, dtype=float) for k, v in out.items()}

    def _mm_per_unit(self, va, used_template: bool) -> float:
        px_per_mm_floor = float(va.ct.pxPerMmFloor())
        if used_template:
            return 1.0 / px_per_mm_floor if px_per_mm_floor > 0 else 1.0
        fctr = float(getattr(va.xf, "fctr", 1.0) or 1.0)
        px_per_mm = px_per_mm_floor * fctr
        return 1.0 / px_per_mm if px_per_mm > 0 else 1.0

    def _collect_theta_r(
        self,
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        dict[tuple[str, int], np.ndarray],
        dict[tuple[str, int], np.ndarray],
    ]:
        """
        Collect (theta, r) samples for between-reward segments.

        Returns:
            pooled_theta_by_trn:
                list length n_trainings, each an array of theta (radians)
            pooled_r_by_trn:
                list length n_trainings, each an array of radius (mm)
            per_fly_theta_by_trn:
                dict keyed by (fly_key, t_idx) -> array of theta
            per_fly_r_by_trn:
                dict keyed by (fly_key, t_idx) -> array of radius
        """
        n_trn = len(self.trns)
        pooled_theta_by_trn: list[list[float]] = [[] for _ in range(n_trn)]
        pooled_r_by_trn: list[list[float]] = [[] for _ in range(n_trn)]

        per_fly_theta_by_trn: dict[tuple[str, int], list[float]] = defaultdict(list)
        per_fly_r_by_trn: dict[tuple[str, int], list[float]] = defaultdict(list)

        # segment-majority output (optional)
        want_seg_majority = bool(self.cfg.seg_majority_out_tsv)
        seg_rows: list[list[object]] = []
        seg_written = 0

        # bin geometry must match plotting
        bin_edges = np.linspace(-np.pi, np.pi, int(self.cfg.bins) + 1)
        bin_center_degs = self._bin_centers_deg_0_360(bin_edges)
        target_degs = tuple(float(x) for x in (self.cfg.seg_majority_target_degs or ()))
        target_bin_idxs = {
            deg: self._closest_bin_idx_for_deg(bin_center_degs, deg)
            for deg in target_degs
        }

        debug_rows_global: list[list[object]] = []
        debug_written_global = 0
        want_debug_global = bool(self.cfg.debug and self.cfg.debug_out_tsv)

        want_debug_per_fly = bool(self.cfg.debug_per_fly)
        per_fly_debug_rows: dict[str, list[list[object]]] = defaultdict(list)
        per_fly_written: dict[str, int] = defaultdict(int)

        missing_wall_regions = 0
        checked_wall_regions = 0

        def emit_debug_row(
            *,
            row: list[object],
            fly_key: str | None = None,
        ) -> None:
            nonlocal debug_written_global

            # global TSV
            if want_debug_global and debug_written_global < self.cfg.debug_max_rows:
                debug_rows_global.append(row)
                debug_written_global += 1

            # per-fly TSV
            if (
                want_debug_per_fly
                and fly_key is not None
                and per_fly_written[fly_key] < self.cfg.debug_max_rows_per_fly
            ):
                per_fly_debug_rows[fly_key].append(row)
                per_fly_written[fly_key] += 1

        for va in self.vas:
            # Skip VAs that were skipped or have bad main trajectory
            if getattr(va, "_skipped", False):
                continue
            if va.trx[0].bad():
                continue

            # Choose a stable-ish ID for this VA for per-fly naming.
            va_fn = getattr(va, "fn", "")
            va_f = getattr(va, "f", "")
            va_id = f"{va_fn}__{va_f}".strip("_") or str(id(va))

            try:
                px_per_mm = float(va.ct.pxPerMmFloor() * va.xf.fctr)
            except Exception:
                px_per_mm = 0.0

            mm_per_px = 1.0 / px_per_mm if px_per_mm > 0 else 1.0

            if px_per_mm <= 0 and self.cfg.debug:
                print(
                    f"[btw_rwd_polar] WARNING: non-positive px_per_mm for {va_fn}; leaving r in pixels"
                )

            if len(va.trns) != n_trn:
                if self.cfg.debug:
                    print(
                        f"[btw_rwd_polar] skipping {va_fn} (trainings {len(va.trns)} != {n_trn})"
                    )
                continue

            for t_idx, trn in enumerate(va.trns):
                for f in va.flies:
                    # multi-fly (exp + yoked); keep only experimental fly
                    if not va.noyc and f != 0:
                        continue

                    on = va._getOn(trn, False, f=f)
                    if on is None or len(on) < 2:
                        continue

                    trj = va.trx[f]
                    if getattr(trj, "_bad", False):
                        continue

                    wall_regions = None
                    if self.cfg.exclude_wall_contact:
                        try:
                            wall_regions = trj.boundary_event_stats["wall"]["all"][
                                "edge"
                            ]["boundary_contact_regions"]
                        except (KeyError, TypeError, AttributeError):
                            wall_regions = None
                        checked_wall_regions += 1
                        if wall_regions is None:
                            missing_wall_regions += 1

                    cx, cy, _ = trn.circles(f)[0]

                    # Per-fly key: VA identity + actual fly index
                    fly_key = f"{va_id}__fly{f}"

                    # Precompute debug windows for this fly/training once (if needed)
                    debug_windows: list[tuple[int, int, int]] = []
                    need_any_debug = want_debug_global or want_debug_per_fly
                    if need_any_debug and (
                        (
                            want_debug_global
                            and debug_written_global < self.cfg.debug_max_rows
                        )
                        or (
                            want_debug_per_fly
                            and per_fly_written[fly_key]
                            < self.cfg.debug_max_rows_per_fly
                        )
                    ):
                        # deterministically vary by base seed + VA + fly + training
                        seed_i = self._stable_seed_int(
                            int(self.cfg.debug_seed),
                            va_fn,
                            va_f,
                            f,
                            t_idx,
                        )
                        debug_windows = self._iter_debug_windows(on=on, seed=seed_i)

                    for i in range(len(on) - 1):
                        s = int(on[i])
                        e = int(on[i + 1])
                        if e <= s + 1:
                            continue

                        xs = trj.x[s:e]
                        ys = trj.y[s:e]
                        good = np.isfinite(xs) & np.isfinite(ys)

                        if self.cfg.only_walking:
                            # walking should be aligned to frames; treat as boolean mask
                            w = np.asarray(trj.walking[s:e], dtype=bool)
                            if w.shape != good.shape:
                                continue
                            good &= w

                        if self.cfg.exclude_wall_contact and wall_regions:
                            # frames s..e-1; build a boolean "in_contact" then exclude them
                            in_contact = np.zeros(e - s, dtype=bool)

                            # wall_regions is a list of slices in absolute frame coords
                            for sl in wall_regions:
                                # robustly get start/stop for slice
                                a = 0 if sl.start is None else int(sl.start)
                                b = 0 if sl.stop is None else int(sl.stop)

                                # intersect with [s, e)
                                aa = max(a, s)
                                bb = min(b, e)
                                if bb > aa:
                                    in_contact[(aa - s) : (bb - s)] = True

                            good &= ~in_contact

                        if not np.any(good):
                            continue

                        # absolute frame indices for the good points
                        idx = np.arange(s, e)[good]

                        # optional stride downsampling for debug file size control
                        if self.cfg.debug_stride > 1:
                            keep = (idx - idx[0]) % self.cfg.debug_stride == 0
                            idx = idx[keep]
                            xg = xs[good][keep]
                            yg = ys[good][keep]
                        else:
                            xg = xs[good]
                            yg = ys[good]

                        xg2, yg2, cx2, cy2, used_template = (
                            self._maybe_mirror_xy_center(va, x=xg, y=yg, cx=cx, cy=cy)
                        )
                        dx = xg2 - cx2
                        dy_raw = yg2 - cy2
                        dy = -dy_raw if self.cfg.flip_y else dy_raw

                        theta_up = np.arctan2(dx, dy)  # 0 = up, +90 = right, -90 = left
                        theta_up = (theta_up + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi]

                        # choose scale based on coordinate space
                        mm_per_unit = self._mm_per_unit(va, used_template)

                        r = np.sqrt(dx * dx + dy * dy) * mm_per_unit

                        # --- segment-majority attribution (optional) ---
                        if (
                            want_seg_majority
                            and seg_written < int(self.cfg.seg_majority_max_rows)
                            and theta_up.size >= int(self.cfg.seg_majority_min_frames)
                            and target_bin_idxs
                        ):
                            # Use same binning convention as the plot
                            b = (
                                np.digitize(theta_up, bin_edges, right=False) - 1
                            )  # [0..bins-1]
                            b = np.clip(b, 0, len(bin_edges) - 2)
                            counts = np.bincount(b, minlength=len(bin_edges) - 1)
                            total = int(theta_up.size)

                            for target_deg, bidx in target_bin_idxs.items():
                                c = int(counts[bidx])
                                frac = (c / total) if total else 0.0
                                if frac >= float(self.cfg.seg_majority_thresh):
                                    seg_rows.append(
                                        [
                                            va_fn,
                                            va_f,
                                            t_idx,
                                            trn.name(),
                                            f,  # fly_role
                                            int(i),  # seg_i
                                            int(s),  # start_frame
                                            int(e),  # end_frame
                                            total,  # n_frames_used
                                            float(target_deg),
                                            float(bin_center_degs[bidx]),
                                            c,
                                            float(frac),
                                        ]
                                    )
                                    seg_written += 1
                                    if seg_written >= int(
                                        self.cfg.seg_majority_max_rows
                                    ):
                                        break

                        # pooled
                        pooled_theta_by_trn[t_idx].extend(theta_up.tolist())
                        pooled_r_by_trn[t_idx].extend(r.tolist())
                        # per-fly
                        per_fly_theta_by_trn[(fly_key, t_idx)].extend(theta_up.tolist())
                        per_fly_r_by_trn[(fly_key, t_idx)].extend(r.tolist())

                    # ---- debug emission (separate from plot data collection) ----
                    if (want_debug_global or want_debug_per_fly) and debug_windows:
                        for seg_i, s_win, e_win in debug_windows:
                            s_win = int(s_win)
                            e_win = int(e_win)
                            if e_win <= s_win + 1:
                                continue

                            xs = trj.x[s_win:e_win]
                            ys = trj.y[s_win:e_win]
                            good = np.isfinite(xs) & np.isfinite(ys)

                            if self.cfg.only_walking:
                                w = np.asarray(trj.walking[s_win:e_win], dtype=bool)
                                if w.shape != good.shape:
                                    continue
                                good &= w

                            if self.cfg.exclude_wall_contact and wall_regions:
                                # frames s_win..e_win-1; build a boolean "in_contact" then exclude them
                                in_contact = np.zeros(e_win - s_win, dtype=bool)

                                # wall_regions is a list of slices in absolute frame coords
                                for sl in wall_regions:
                                    # robustly get start/stop for slice
                                    a = 0 if sl.start is None else int(sl.start)
                                    b = 0 if sl.stop is None else int(sl.stop)

                                    # intersect with [s_win, e_win)
                                    aa = max(a, s_win)
                                    bb = min(b, e_win)
                                    if bb > aa:
                                        in_contact[(aa - s_win) : (bb - s_win)] = True
                                good &= ~in_contact

                            if not np.any(good):
                                continue

                            idx = np.arange(s_win, e_win)[good]

                            if self.cfg.debug_stride > 1:
                                keep = (idx - idx[0]) % self.cfg.debug_stride == 0
                                idx = idx[keep]
                                xg = xs[good][keep]
                                yg = ys[good][keep]
                            else:
                                xg = xs[good]
                                yg = ys[good]

                            xg2, yg2, cx2, cy2, used_template = (
                                self._maybe_mirror_xy_center(
                                    va, x=xg, y=yg, cx=cx, cy=cy
                                )
                            )
                            dx = xg2 - cx2
                            dy_raw = yg2 - cy2
                            dy = -dy_raw if self.cfg.flip_y else dy_raw

                            theta_up = np.arctan2(dx, dy)
                            theta_up = (theta_up + np.pi) % (2 * np.pi) - np.pi

                            # choose scale based on coordinate space
                            mm_per_unit = self._mm_per_unit(va, used_template)

                            for fm, x, y, ddx, ddy_raw_i, ddy_i, th in zip(
                                idx, xg, yg, dx, dy_raw, dy, theta_up
                            ):
                                th_0_2pi = float((th + 2 * np.pi) % (2 * np.pi))
                                th_deg = float(np.degrees(th))
                                th_deg_i = int(np.rint(th_deg))

                                emit_debug_row(
                                    row=[
                                        va_fn,
                                        va_f,
                                        t_idx,
                                        trn.name(),
                                        f,  # fly role
                                        int(seg_i),  # seg_i (sampled segment id)
                                        int(fm),
                                        float(cx),
                                        float(cy),
                                        float(x),
                                        float(y),
                                        float(ddx),
                                        float(ddy_raw_i),
                                        float(ddy_i),
                                        float(
                                            np.sqrt(ddx * ddx + ddy_i * ddy_i)
                                            * mm_per_unit
                                        ),
                                        float(th),
                                        th_0_2pi,
                                        th_deg,
                                        th_deg_i,
                                    ],
                                    fly_key=fly_key,
                                )

        # write debug TSV once
        if want_debug_global and debug_rows_global:
            with open(self.cfg.debug_out_tsv, "w", newline="") as fp:
                w = csv.writer(fp, delimiter="\t")
                w.writerow(
                    [
                        "video_id",
                        "fly_num",
                        "t_idx",
                        "training",
                        "fly_role",
                        "seg_i",
                        "frame",
                        "cx",
                        "cy",
                        "x",
                        "y",
                        "dx",
                        "dy_raw",
                        "dy_used",
                        "r_used",
                        "theta_rad",
                        "theta_0_2pi",
                        "theta_deg",
                        "theta_deg_i",
                    ]
                )
                w.writerows(debug_rows_global)
            print(
                f"[btw_rwd_polar][debug] wrote {len(debug_rows_global)} rows to {self.cfg.debug_out_tsv}"
            )

        if want_debug_per_fly and per_fly_debug_rows:
            os.makedirs(self.cfg.debug_per_fly_out_dir, exist_ok=True)

            for fk, rows in per_fly_debug_rows.items():
                if not rows:
                    continue

                out_fn = f"btw_rwd_polar_debug__{self._safe_filename(fk)}.tsv"
                out_path = os.path.join(self.cfg.debug_per_fly_out_dir, out_fn)

                with open(out_path, "w", newline="") as fp:
                    w = csv.writer(fp, delimiter="\t")
                    w.writerow(
                        [
                            "video_id",
                            "fly_num",
                            "t_idx",
                            "training",
                            "fly_role",
                            "seg_i",
                            "frame",
                            "cx",
                            "cy",
                            "x",
                            "y",
                            "dx",
                            "dy_raw",
                            "dy_used",
                            "r_used",
                            "theta_rad",
                            "theta_0_2pi",
                            "theta_deg",
                            "theta_deg_i",
                        ]
                    )
                    w.writerows(rows)

            print(
                f"[btw_rwd_polar][debug] wrote {len(per_fly_debug_rows)} per-fly TSV(s) to "
                f"{self.cfg.debug_per_fly_out_dir}"
            )

        if want_seg_majority and seg_rows:
            out_path = str(self.cfg.seg_majority_out_tsv)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", newline="") as fp:
                w = csv.writer(fp, delimiter="\t")
                w.writerow(
                    [
                        "video_id",
                        "fly_num",
                        "t_idx",
                        "training",
                        "fly_role",
                        "seg_i",
                        "start_frame",
                        "end_frame",
                        "n_frames_used",
                        "target_deg",
                        "bin_center_deg",
                        "count_in_bin",
                        "frac_in_bin",
                    ]
                )
                w.writerows(seg_rows)
            print(
                f"[btw_rwd_polar][seg_majority] wrote {len(seg_rows)} rows to {out_path}"
            )

        pooled_theta_arrays = [
            np.asarray(vals, dtype=float) for vals in pooled_theta_by_trn
        ]
        pooled_r_arrays = [np.asarray(vals, dtype=float) for vals in pooled_r_by_trn]

        per_fly_theta_arrays = {
            k: np.asarray(v, dtype=float) for k, v in per_fly_theta_by_trn.items()
        }
        per_fly_r_arrays = {
            k: np.asarray(v, dtype=float) for k, v in per_fly_r_by_trn.items()
        }

        if self.cfg.per_fly:
            sizes = sorted(
                (
                    (
                        fk,
                        sum(
                            len(per_fly_theta_by_trn.get((fk, t), []))
                            for t in range(n_trn)
                        ),
                    )
                    for fk in sorted({k[0] for k in per_fly_theta_by_trn.keys()})
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            print("[btw_rwd_polar] per-fly sample counts (top 10):")
            for fk, n in sizes[:10]:
                print(f"  {fk}: {n}")

        if self.cfg.exclude_wall_contact and missing_wall_regions:
            print(
                f"[btw_rwd_polar] WARNING: requested wall-contact exclusion, but wall-contact "
                f"regions were missing for {missing_wall_regions}/{checked_wall_regions} "
                f"(va,training,fly) combinations. Those cases were plotted without exclusion."
            )
        return (
            pooled_theta_arrays,
            pooled_r_arrays,
            per_fly_theta_arrays,
            per_fly_r_arrays,
        )

    def _maybe_mirror_xy_center(
        self, va: "VideoAnalysis", *, x: np.ndarray, y: np.ndarray, cx: float, cy: float
    ) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
        """
        If cfg.mirror:
          - convert frame coords -> template coords (shift to TL using f=va.f, noMirr=True)
          - mirror in template coords using va.mirror()
          - return (x2,y2,cx2,cy2, used_template=True)

        Else: return inputs and used_template=False
        """
        if not bool(getattr(self.cfg, "mirror", False)):
            return x, y, cx, cy, False

        # Need Xformer to map frame->template
        xf = getattr(va, "xf", None)
        if xf is None or not getattr(xf, "initialized", lambda: False)():
            # Fall back safely: no conversion, no mirroring
            if getattr(self.cfg, "debug", False):
                print(
                    "[btw_rwd_polar] WARNING: cfg.mirror=True but va.xf not initialized; skipping mirroring"
                )
            return x, y, cx, cy, False

        # Chamber index for shifting into TL template coords
        va_f = getattr(va, "f", None)

        if getattr(self.cfg, "debug", False) and va.ct in (CT.large, CT.large2):
            ef = getattr(va, "ef", None)
            if va_f is not None and ef is not None:
                if int(va_f) != int(ef):
                    print(
                        f"[btw_rwd_polar] NOTE: va.f ({va_f}) != va.ef ({ef}) for {getattr(va, 'fn', '')}; "
                        "mirroring uses va.ef"
                    )

        # Frame -> template (shift to TL), but do NOT mirror here (va.mirror owns that)
        xt, yt = xf.f2t(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), f=va_f, noMirr=True
        )
        cxt, cyt = xf.f2t(
            np.asarray([cx], dtype=float),
            np.asarray([cy], dtype=float),
            f=va_f,
            noMirr=True,
        )

        xy_m = va.mirror([np.asarray(xt, dtype=float), np.asarray(yt, dtype=float)])
        cxcy_m = va.mirror([np.asarray(cxt, dtype=float), np.asarray(cyt, dtype=float)])

        x2 = np.asarray(xy_m[0], dtype=float)
        y2 = np.asarray(xy_m[1], dtype=float)
        cx2 = float(np.asarray(cxcy_m[0], dtype=float)[0])
        cy2 = float(np.asarray(cxcy_m[1], dtype=float)[0])
        return x2, y2, cx2, cy2, True

    # ---------- plotting helpers ----------

    def _plot_theta_hist_multi_training(
        self,
        thetas_by_trn: list[np.ndarray],
        per_fly_thetas_by_trn: dict[tuple[str, int], np.ndarray] | None,
        trn_labels: list[str],
        out_file: str,
        title: str,
        subtitle: str | None = None,
    ) -> bool:
        """
        Plot one polar histogram per training (subplots), writing to out_file.
        Returns True if something was plotted (i.e., any data existed).
        """
        if not any(th.size for th in thetas_by_trn):
            return False

        # Optional pooling: for frames-aggregate we can just concatenate.
        # For fly-aggregate, we need to pool per-fly arrays too.
        per_fly_mode = self._is_fly_aggregate()

        # Optionally pool all trainings into a single distribution (for this call)
        if self.cfg.pool_trainings:
            pooled = np.concatenate([th for th in thetas_by_trn if th.size > 0])
            thetas_by_trn = [pooled]
            trn_labels = ["all trainings combined"]
            if per_fly_mode and per_fly_thetas_by_trn is not None:
                per_fly_thetas_by_trn = self._pool_per_fly_by_trn(per_fly_thetas_by_trn)

        n_trn = len(thetas_by_trn)
        fig, axes = plt.subplots(
            1,
            n_trn,
            figsize=(4.2 * n_trn if n_trn > 1 else 6.5, 4.8),
            squeeze=False,
            subplot_kw={"projection": "polar"},
        )
        axes = axes[0]

        bin_edges = np.linspace(-np.pi, np.pi, self.cfg.bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        widths = np.diff(bin_edges)

        any_plotted = False
        for t_idx, (ax, th, label) in enumerate(zip(axes, thetas_by_trn, trn_labels)):
            if th.size == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue

            if not per_fly_mode:
                counts, _ = np.histogram(th, bins=bin_edges)
                total = counts.sum()
                if total == 0:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    continue
                vals = counts / total if self.cfg.normalize else counts
            else:
                # Each fly contributes one normalized histogram (then mean across flies).
                if per_fly_thetas_by_trn is None:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no per-fly data", ha="center", va="center")
                    continue

                fly_keys = self._fly_keys(per_fly_thetas_by_trn)
                fly_hists: list[np.ndarray] = []

                for fk in fly_keys:
                    th_f = per_fly_thetas_by_trn.get((fk, t_idx), np.asarray([], float))
                    if th_f.size == 0:
                        continue
                    if th_f.size < int(getattr(self.cfg, "min_frames_per_fly", 0) or 0):
                        continue

                    c_f, _ = np.histogram(th_f, bins=bin_edges)
                    s_f = float(c_f.sum())
                    if s_f <= 0:
                        continue

                    if bool(getattr(self.cfg, "normalize", True)):
                        fly_hists.append(c_f / s_f)
                    else:
                        # If normalize=False, "flies" mode becomes "mean counts per fly"
                        fly_hists.append(c_f.astype(float))

                if not fly_hists:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    continue

                vals = np.mean(np.stack(fly_hists, axis=0), axis=0)
            ax.bar(bin_centers, vals, width=widths, align="center")

            # display convention (does not change histogram bins)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)

            if self.cfg.rmax is not None:
                ax.set_rmax(self.cfg.rmax)

            ax.set_title(label)
            any_plotted = True

        if subtitle:
            fig.suptitle(f"{title}\n{subtitle}")
        else:
            fig.suptitle(title)

        fig.tight_layout()
        writeImage(out_file, format=self.opts.imageFormat)
        plt.close(fig)
        return any_plotted

    def _plot_theta_r_heatmap_multi_training(
        self,
        thetas_by_trn: list[np.ndarray],
        rs_by_trn: list[np.ndarray],
        per_fly_theta_by_trn: dict[tuple[str, int], np.ndarray] | None,
        per_fly_r_by_trn: dict[tuple[str, int], np.ndarray] | None,
        trn_labels: list[str],
        out_file: str,
        title: str,
        subtitle: str | None = None,
    ) -> bool:
        """
        Plot one polar heatmap per training (subplots), where color encodes
        occupancy in (theta, r) bins.
        """
        if not any(th.size for th in thetas_by_trn):
            return False

        per_fly_mode = self._is_fly_aggregate()
        norm_mode = str(self.cfg.theta_r_normalize or "global")

        # Optionally pool all trainings into a single distribution (for this call)
        if self.cfg.pool_trainings:
            pooled_th = np.concatenate([th for th in thetas_by_trn if th.size > 0])
            pooled_r = np.concatenate([rr for rr in rs_by_trn if rr.size > 0])
            thetas_by_trn = [pooled_th]
            rs_by_trn = [pooled_r]
            trn_labels = ["all trainings combined"]
            if (
                per_fly_mode
                and per_fly_theta_by_trn is not None
                and per_fly_r_by_trn is not None
            ):
                per_fly_theta_by_trn = self._pool_per_fly_by_trn(per_fly_theta_by_trn)
                per_fly_r_by_trn = self._pool_per_fly_by_trn(per_fly_r_by_trn)

        n_trn = len(thetas_by_trn)
        fig, axes = plt.subplots(
            1,
            n_trn,
            figsize=(4.6 * n_trn if n_trn > 1 else 7.2, 5.2),
            squeeze=False,
            subplot_kw={"projection": "polar"},
        )
        axes = axes[0]

        theta_edges = np.linspace(-np.pi, np.pi, int(self.cfg.bins) + 1)
        n_r_bins = int(self.cfg.r_bins)

        # If user didn't fix r_max, choose a robust max across all data in this call.
        r_max = self.cfg.r_max
        if r_max is None:
            all_r = (
                np.concatenate([rr for rr in rs_by_trn if rr.size > 0])
                if any(rr.size for rr in rs_by_trn)
                else np.asarray([], dtype=float)
            )
            if all_r.size > 0:
                # robust to outliers
                r_max = float(np.nanpercentile(all_r, 99.0))
            else:
                r_max = 1.0
        # avoid degenerate
        r_max = max(float(r_max), 1e-9)

        r_edges = np.linspace(0.0, r_max, n_r_bins + 1)

        # Share color scale across subplots for comparability:
        # compute all H first to get a global vmax, then render with common vmin/vmax.
        any_plotted = False
        H_by_ax: list[tuple[plt.Axes, np.ndarray, str]] = []
        vmax = 0.0

        for t_idx, (ax, th, rr, label) in enumerate(
            zip(axes, thetas_by_trn, rs_by_trn, trn_labels)
        ):
            if th.size == 0 or rr.size == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue

            if not per_fly_mode:
                # Defensive: ensure aligned lengths
                n = min(th.size, rr.size)
                th2 = th[:n]
                rr2 = rr[:n]

                H, _te, _re = np.histogram2d(th2, rr2, bins=[theta_edges, r_edges])

                if norm_mode == "global":
                    s = float(H.sum())
                    if s > 0:
                        H = H / s
                elif norm_mode == "per_theta":
                    denom = H.sum(axis=1, keepdims=True)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        H = np.where(denom > 0, H / denom, 0.0)
                else:
                    # "none": raw counts
                    pass
            else:
                if per_fly_theta_by_trn is None or per_fly_r_by_trn is None:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no per-fly data", ha="center", va="center")
                    continue

                fly_keys = self._fly_keys(per_fly_theta_by_trn)
                fly_Hs: list[np.ndarray] = []

                for fk in fly_keys:
                    th_f = per_fly_theta_by_trn.get((fk, t_idx), np.asarray([], float))
                    rr_f = per_fly_r_by_trn.get((fk, t_idx), np.asarray([], float))
                    n = min(th_f.size, rr_f.size)
                    if n == 0:
                        continue
                    if n < int(getattr(self.cfg, "min_frames_per_fly", 0) or 0):
                        continue
                    th_f = th_f[:n]
                    rr_f = rr_f[:n]

                    Hf, _tw, _re = np.histogram2d(
                        th_f, rr_f, bins=[theta_edges, r_edges]
                    )

                    # Apply same normalization mode, but within-fly.
                    if norm_mode == "global":
                        s = float(Hf.sum())
                        if s > 0:
                            Hf = Hf / s
                    elif norm_mode == "per_theta":
                        denom = Hf.sum(axis=1, keepdims=True)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            Hf = np.where(denom > 0, Hf / denom, 0.0)
                    else:
                        # "none": interpret as mean counts per fly
                        pass
                    fly_Hs.append(Hf.astype(float))

                if not fly_Hs:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    continue
                H = np.mean(np.stack(fly_Hs, axis=0), axis=0)

            H_by_ax.append((ax, H, label))
            if H.size:
                vmax = max(vmax, float(np.nanmax(H)))
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            return False

        # Avoid degenerate color scales (e.g., all zeros)
        vmax = max(float(vmax), 1e-12)

        TH, RR = np.meshgrid(theta_edges, r_edges, indexing="ij")
        meshes = []
        for ax, H, label in H_by_ax:
            m = ax.pcolormesh(TH, RR, H, shading="auto", vmin=0.0, vmax=vmax)
            meshes.append(m)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_rmax(r_max)
            step = float(self.cfg.r_tick_step)  # mm
            if step > 0:
                ticks = np.arange(step, r_max + 1e-9, step)
                if ticks.size:
                    ax.set_rticks(ticks)
            ax.set_title(label)

            yticklabels = ax.get_yticklabels()
            if yticklabels:
                for t in yticklabels:
                    t.set_color("white")

        # One shared colorbar, in its own axes so it never overlaps the polar plots.
        if meshes:
            # Reserve space on the right for the colorbar
            fig.subplots_adjust(right=0.86)
            # [left, bottom, width height] in figure coordinates
            cax = fig.add_axes([0.88, 0.17, 0.03, 0.66])
            cb = fig.colorbar(meshes[-1], cax=cax)
            if str(self.cfg.theta_r_normalize or "global") == "none":
                cb.set_label("frame count")
            elif str(self.cfg.theta_r_normalize or "global") == "per_theta":
                cb.set_label("fraction within theta sector")
            else:
                cb.set_label("fraction of frames")

        fig.subplots_adjust(top=0.84)

        fig.text(0.5, 0.97, title, ha="center", va="top")
        if subtitle:
            fig.text(0.5, 0.935, subtitle, ha="center", va="top")

        writeImage(out_file, format=self.opts.imageFormat)
        plt.close(fig)
        return True

    def _safe_filename(self, s: str) -> str:
        """Make a string safe-ish for filenames."""
        return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)

    def _plot_per_fly(self, per_fly_by_trn: dict[tuple[str, int], np.ndarray]) -> None:
        if not per_fly_by_trn:
            print("[btw_rwd_polar] per-fly requested but no data found; skipping.")
            return

        os.makedirs(self.cfg.per_fly_out_dir, exist_ok=True)

        fly_keys = sorted({fk for (fk, _t) in per_fly_by_trn.keys()})
        written = 0

        prev_agg = self.cfg.aggregate
        self.cfg.aggregate = "frames"

        try:
            for fk in fly_keys:
                if (
                    self.cfg.max_per_fly_plots is not None
                    and written >= self.cfg.max_per_fly_plots
                ):
                    break

                thetas_by_trn = [
                    per_fly_by_trn.get((fk, t_idx), np.asarray([], dtype=float))
                    for t_idx in range(self.n_trn)
                ]
                trn_labels = [t.name() for t in self.trns]

                out_fn = f"btw_rwd_polar__{self._safe_filename(fk)}.png"
                out_path = os.path.join(self.cfg.per_fly_out_dir, out_fn)

                title = "Between-reward angular occupancy (around reward center)"
                subtitle = fk
                if self.cfg.subset_label:
                    subtitle = f"{fk} | {self.cfg.subset_label}"

                ok = self._plot_theta_hist_multi_training(
                    thetas_by_trn=thetas_by_trn,
                    per_fly_thetas_by_trn=None,
                    trn_labels=trn_labels,
                    out_file=out_path,
                    title=title,
                    subtitle=subtitle,
                )
                if ok:
                    written += 1
        finally:
            self.cfg.aggregate = prev_agg

        print(
            f"[btw_rwd_polar] wrote {written} per-fly plot(s) to {self.cfg.per_fly_out_dir}"
        )

    def _outfile_with_suffix(self, suffix: str) -> str:
        base, ext = os.path.splitext(self.cfg.out_file)
        return f"{base}{suffix}{ext}"

    # ---------- main entry ----------

    def plot(self) -> None:
        base_title_1d = "Between-reward angular occupancy (around reward center)"
        base_title_2d = "Between-reward occupancy heatmap (theta x radius around reward center; radius in mm)"
        subtitle = self.cfg.subset_label
        if self._is_fly_aggregate():
            subtitle = (
                subtitle + " | " if subtitle else ""
            ) + "fly-averaged (per-fly normalized)"

        pooled_theta_by_trn, pooled_r_by_trn, per_fly_theta_by_trn, per_fly_r_by_trn = (
            self._collect_theta_r()
        )

        trn_labels = [t.name() for t in self.trns]

        mode = str(self.cfg.mode or "theta")

        wrote_any = False

        if mode in ("theta", "both"):
            out_1d = (
                self.cfg.out_file
                if mode == "theta"
                else self._outfile_with_suffix("__theta")
            )
            ok_1d = self._plot_theta_hist_multi_training(
                thetas_by_trn=pooled_theta_by_trn,
                per_fly_thetas_by_trn=per_fly_theta_by_trn,
                trn_labels=trn_labels,
                out_file=out_1d,
                title=base_title_1d,
                subtitle=subtitle,
            )
            if ok_1d:
                print(f"[btw_rwd_polar] wrote {out_1d}")
                wrote_any = True

        if mode in ("theta_r", "both"):
            out_2d = (
                self.cfg.out_file
                if mode == "theta_r"
                else self._outfile_with_suffix("__theta_r")
            )
            ok_2d = self._plot_theta_r_heatmap_multi_training(
                thetas_by_trn=pooled_theta_by_trn,
                rs_by_trn=pooled_r_by_trn,
                per_fly_theta_by_trn=per_fly_theta_by_trn,
                per_fly_r_by_trn=per_fly_r_by_trn,
                trn_labels=trn_labels,
                out_file=out_2d,
                title=base_title_2d,
                subtitle=subtitle,
            )
            if ok_2d:
                print(f"[btw_rwd_polar] wrote {out_2d}")
                wrote_any = True

        if not wrote_any:
            print(
                "[btw_rwd_polar] no between-reward position data found; skipping plot."
            )
            return

        # optional per-fly plots
        if self.cfg.per_fly:
            # keep per-fly behavior in 1D mode only
            if mode == "theta_r":
                print(
                    "[btw_rwd_polar] NOTE: per-fly plots currently render 1D histograms only."
                )
            else:
                # mode == theta or both: output per-fly 1D
                if mode == "both":
                    print(
                        "[btw_rwd_polar] NOTE: per-fly plots render 1D histograms only (even when --both is used)."
                    )
                self._plot_per_fly(per_fly_theta_by_trn)
