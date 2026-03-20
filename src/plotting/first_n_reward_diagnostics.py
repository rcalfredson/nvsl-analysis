from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from src.plotting.between_reward_segment_binning import video_base
from src.plotting.reward_window_utils import (
    cumulative_window_seconds_for_frame,
    frames_in_windows,
    locate_window_for_frame,
    selected_training_indices,
    selected_windows_for_va,
    training_window_label,
)
import src.utils.util as util


@dataclass(frozen=True)
class FirstNRewardDiagnosticsConfig:
    csv_out: str
    npz_out: str | None = None
    plot_out: str | None = None
    trainings: Sequence[int] | None = None
    skip_first_sync_buckets: int = 0
    keep_first_sync_buckets: int = 0
    first_n_rewards: int = 10
    subset_label: str | None = None
    sli_values: Sequence[float] | None = None
    pi_threshold: int = 10
    max_span_s: float | None = None
    max_time_to_nth_s: float | None = None
    x_by: str = "first_n_reward_span_s"
    y_by: str = "sli"
    color_by: str = "control_circle_entry_count_by_cutoff"
    label_low_sli_outliers: int = 0


@dataclass(frozen=True)
class FirstNRewardDiagnosticRow:
    video_basename: str
    video_path: str
    va_tag: int
    fly_idx: int
    selected_subset_label: str
    selected_trainings_label: str
    skip_first_sync_buckets: int
    keep_first_sync_buckets: int
    nth_reward_target: int
    has_selected_window: bool
    actual_reward_count_in_selected_window: int
    eligible_for_nth_reward_cutoff: bool
    sli: float
    cutoff_frame: float
    cutoff_training: float
    cutoff_time_since_selected_window_start_s: float
    cutoff_time_since_cutoff_training_start_s: float
    time_to_first_actual_reward_s: float
    time_to_nth_actual_reward_s: float
    first_n_reward_span_s: float
    actual_reward_count_by_cutoff: float
    control_reward_count_by_cutoff: float
    actual_circle_entry_count_by_cutoff: float
    control_circle_entry_count_by_cutoff: float
    reward_pi_by_cutoff: float
    actual_entry_minus_reward_count_by_cutoff: float
    control_entry_minus_reward_count_by_cutoff: float
    control_to_actual_entry_ratio_by_cutoff: float
    control_to_actual_reward_ratio_by_cutoff: float


class FirstNRewardDiagnosticsPlotter:
    def __init__(self, vas, opts, gls, cfg: FirstNRewardDiagnosticsConfig):
        self.vas = list(vas)
        self.opts = opts
        self.gls = gls
        self.cfg = cfg
        self.log_tag = "first_n_reward_diag"

    def _sli_value_for_index(self, i: int) -> float:
        sli_values = self.cfg.sli_values
        if sli_values is None or i >= len(sli_values):
            return np.nan
        try:
            value = float(sli_values[i])
        except Exception:
            return np.nan
        return value if np.isfinite(value) else np.nan

    def _selected_subset_label(self) -> str:
        return str(self.cfg.subset_label) if self.cfg.subset_label else "All flies"

    @staticmethod
    def _metric_field_names() -> list[str]:
        return list(FirstNRewardDiagnosticRow.__dataclass_fields__.keys())

    @staticmethod
    def _metric_label(name: str) -> str:
        return str(name).replace("_", " ")

    def _resolve_metric_name(self, name: str | None, *, fallback: str) -> str:
        candidate = str(name or fallback)
        if candidate in self._metric_field_names():
            return candidate
        print(
            f"[{self.log_tag}] WARNING: unknown metric '{candidate}'; "
            f"falling back to '{fallback}'."
        )
        return fallback

    def _passes_final_time_filters(self, row: FirstNRewardDiagnosticRow) -> bool:
        if not row.eligible_for_nth_reward_cutoff:
            return False

        max_span_s = self.cfg.max_span_s
        if max_span_s is not None:
            try:
                max_span_s = float(max_span_s)
            except Exception:
                max_span_s = None
            if max_span_s is not None:
                if not np.isfinite(row.first_n_reward_span_s):
                    return False
                if row.first_n_reward_span_s > max_span_s:
                    return False

        max_time_to_nth_s = self.cfg.max_time_to_nth_s
        if max_time_to_nth_s is not None:
            try:
                max_time_to_nth_s = float(max_time_to_nth_s)
            except Exception:
                max_time_to_nth_s = None
            if max_time_to_nth_s is not None:
                if not np.isfinite(row.time_to_nth_actual_reward_s):
                    return False
                if row.time_to_nth_actual_reward_s > max_time_to_nth_s:
                    return False

        return True

    def _eligible_rows(self, rows: list[FirstNRewardDiagnosticRow]) -> list[FirstNRewardDiagnosticRow]:
        return [
            row
            for row in rows
            if self._passes_final_time_filters(row)
        ]

    def compute_all_rows(self) -> list[FirstNRewardDiagnosticRow]:
        if not self.vas:
            return []

        ref_va = self.vas[0]
        selected_trainings = selected_training_indices(
            ref_va,
            self.cfg.trainings,
            log_tag=self.log_tag,
        )
        training_label = training_window_label(selected_trainings)
        subset_label = self._selected_subset_label()
        n_target = max(1, int(self.cfg.first_n_rewards or 1))
        rows: list[FirstNRewardDiagnosticRow] = []

        for i, va in enumerate(self.vas):
            if getattr(va, "_skipped", False):
                continue
            try:
                if va.trx[0].bad():
                    continue
            except Exception:
                pass

            windows = selected_windows_for_va(
                va,
                selected_trainings,
                skip_first_sync_buckets=self.cfg.skip_first_sync_buckets,
                keep_first_sync_buckets=self.cfg.keep_first_sync_buckets,
                f=0,
            )
            fps = float(getattr(va, "fps", 1.0) or 1.0)
            if not np.isfinite(fps) or fps <= 0:
                fps = 1.0

            actual_rewards = frames_in_windows(va, windows, calc=False, ctrl=False, f=0)
            actual_entries = frames_in_windows(va, windows, calc=True, ctrl=False, f=0)
            control_entries = frames_in_windows(va, windows, calc=True, ctrl=True, f=0)

            has_window = bool(windows)
            n_actual = int(actual_rewards.size)
            eligible = has_window and n_actual >= n_target
            sli = self._sli_value_for_index(i)

            cutoff_frame = np.nan
            cutoff_training = np.nan
            cutoff_time_s = np.nan
            cutoff_time_training_s = np.nan
            time_to_first_s = np.nan
            time_to_nth_s = np.nan
            span_s = np.nan
            actual_reward_count_by_cutoff = np.nan
            control_reward_count_by_cutoff = np.nan
            actual_entry_count_by_cutoff = np.nan
            control_entry_count_by_cutoff = np.nan
            reward_pi = np.nan
            actual_entry_minus_reward = np.nan
            control_entry_minus_reward = np.nan
            control_to_actual_entry_ratio = np.nan
            control_to_actual_reward_ratio = np.nan

            if n_actual > 0:
                time_to_first_s = cumulative_window_seconds_for_frame(
                    windows, int(actual_rewards[0]), fps=fps
                )

            if eligible:
                cutoff_frame = float(actual_rewards[n_target - 1])
                cutoff_time_s = cumulative_window_seconds_for_frame(
                    windows, int(actual_rewards[n_target - 1]), fps=fps
                )
                time_to_nth_s = cutoff_time_s
                if np.isfinite(time_to_first_s) and np.isfinite(time_to_nth_s):
                    span_s = float(time_to_nth_s - time_to_first_s)

                cutoff_window = locate_window_for_frame(windows, int(actual_rewards[n_target - 1]))
                if cutoff_window is not None:
                    cutoff_training = float(cutoff_window.training_idx + 1)
                    cutoff_time_training_s = max(
                        0.0,
                        float(actual_rewards[n_target - 1] - cutoff_window.trn.start) / fps,
                    )

                actual_reward_count_by_cutoff = float(
                    np.searchsorted(actual_rewards, int(actual_rewards[n_target - 1]), side="right")
                )
                actual_entry_count_by_cutoff = float(
                    np.searchsorted(actual_entries, int(actual_rewards[n_target - 1]), side="right")
                )
                control_entry_count_by_cutoff = float(
                    np.searchsorted(control_entries, int(actual_rewards[n_target - 1]), side="right")
                )
                control_reward_count_by_cutoff = control_entry_count_by_cutoff
                reward_pi = float(
                    np.asarray(
                        util.prefIdx(
                            actual_entry_count_by_cutoff,
                            control_entry_count_by_cutoff,
                            n=max(0, int(self.cfg.pi_threshold or 0)),
                        ),
                        dtype=float,
                    )
                )
                actual_entry_minus_reward = (
                    actual_entry_count_by_cutoff - actual_reward_count_by_cutoff
                )
                control_entry_minus_reward = (
                    control_entry_count_by_cutoff - actual_reward_count_by_cutoff
                )
                if actual_entry_count_by_cutoff > 0:
                    control_to_actual_entry_ratio = (
                        control_entry_count_by_cutoff / actual_entry_count_by_cutoff
                    )
                if actual_reward_count_by_cutoff > 0:
                    control_to_actual_reward_ratio = (
                        control_entry_count_by_cutoff / actual_reward_count_by_cutoff
                    )

            rows.append(
                FirstNRewardDiagnosticRow(
                    video_basename=video_base(va),
                    video_path=str(getattr(va, "fn", "") or ""),
                    va_tag=int(getattr(va, "f", 0) or 0),
                    fly_idx=0,
                    selected_subset_label=subset_label,
                    selected_trainings_label=training_label,
                    skip_first_sync_buckets=int(self.cfg.skip_first_sync_buckets or 0),
                    keep_first_sync_buckets=int(self.cfg.keep_first_sync_buckets or 0),
                    nth_reward_target=n_target,
                    has_selected_window=has_window,
                    actual_reward_count_in_selected_window=n_actual,
                    eligible_for_nth_reward_cutoff=eligible,
                    sli=sli,
                    cutoff_frame=cutoff_frame,
                    cutoff_training=cutoff_training,
                    cutoff_time_since_selected_window_start_s=cutoff_time_s,
                    cutoff_time_since_cutoff_training_start_s=cutoff_time_training_s,
                    time_to_first_actual_reward_s=time_to_first_s,
                    time_to_nth_actual_reward_s=time_to_nth_s,
                    first_n_reward_span_s=span_s,
                    actual_reward_count_by_cutoff=actual_reward_count_by_cutoff,
                    control_reward_count_by_cutoff=control_reward_count_by_cutoff,
                    actual_circle_entry_count_by_cutoff=actual_entry_count_by_cutoff,
                    control_circle_entry_count_by_cutoff=control_entry_count_by_cutoff,
                    reward_pi_by_cutoff=reward_pi,
                    actual_entry_minus_reward_count_by_cutoff=actual_entry_minus_reward,
                    control_entry_minus_reward_count_by_cutoff=control_entry_minus_reward,
                    control_to_actual_entry_ratio_by_cutoff=control_to_actual_entry_ratio,
                    control_to_actual_reward_ratio_by_cutoff=control_to_actual_reward_ratio,
                )
            )

        return rows

    def compute_rows(self) -> list[FirstNRewardDiagnosticRow]:
        return self._eligible_rows(self.compute_all_rows())

    def _write_csv(self, rows: list[FirstNRewardDiagnosticRow]) -> None:
        path = self.cfg.csv_out
        util.ensureDir(path)
        fieldnames = list(asdict(rows[0]).keys()) if rows else list(
            FirstNRewardDiagnosticRow.__dataclass_fields__.keys()
        )
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        print(f"[{self.log_tag}] wrote CSV: {path}")

    def _write_npz(self, rows: list[FirstNRewardDiagnosticRow]) -> None:
        path = self.cfg.npz_out
        if not path:
            return
        util.ensureDir(path)
        cols = {}
        fieldnames = list(FirstNRewardDiagnosticRow.__dataclass_fields__.keys())
        row_dicts = [asdict(row) for row in rows]
        for key in fieldnames:
            vals = [rd[key] for rd in row_dicts]
            if vals and all(isinstance(v, (bool, np.bool_)) for v in vals):
                cols[key] = np.asarray(vals, dtype=bool)
            elif vals and all(isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) for v in vals):
                cols[key] = np.asarray(vals, dtype=float)
            else:
                cols[key] = np.asarray(vals, dtype=object)
        meta = {
            "subset_label": self._selected_subset_label(),
            "trainings": list(self.cfg.trainings) if self.cfg.trainings else None,
            "first_n_rewards": int(self.cfg.first_n_rewards or 0),
            "pi_threshold": int(self.cfg.pi_threshold or 0),
            "max_span_s": self.cfg.max_span_s,
            "max_time_to_nth_s": self.cfg.max_time_to_nth_s,
            "x_by": self.cfg.x_by,
            "y_by": self.cfg.y_by,
            "color_by": self.cfg.color_by,
            "skip_first_sync_buckets": int(self.cfg.skip_first_sync_buckets or 0),
            "keep_first_sync_buckets": int(self.cfg.keep_first_sync_buckets or 0),
        }
        np.savez(path, **cols, metadata=np.asarray(meta, dtype=object))
        print(f"[{self.log_tag}] wrote NPZ: {path}")

    def _label_outliers(self, ax, rows: list[FirstNRewardDiagnosticRow], x, y) -> None:
        k = max(0, int(self.cfg.label_low_sli_outliers or 0))
        if k <= 0 or len(rows) < 3:
            return
        try:
            coeffs = np.polyfit(x, y, deg=1)
            residuals = y - np.polyval(coeffs, x)
        except Exception:
            residuals = y - np.nanmedian(y)
        order = np.argsort(residuals)
        for idx in order[: min(k, len(order))]:
            row = rows[int(idx)]
            ax.annotate(
                row.video_basename,
                (x[idx], y[idx]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    @staticmethod
    def _correlation_text(x: np.ndarray, y: np.ndarray) -> str:
        n = int(min(len(x), len(y)))
        if n < 2:
            return "Pearson r: n/a\np: n/a\nn: 0"
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            return f"Pearson r: n/a\np: n/a\nn: {n}"
        try:
            r, p = pearsonr(x, y)
        except Exception:
            return f"Pearson r: n/a\np: n/a\nn: {n}"
        return f"Pearson r = {r:.3f}\np = {p:.3g}\nn = {n}"

    def _write_plot(self, rows: list[FirstNRewardDiagnosticRow]) -> None:
        path = self.cfg.plot_out
        if not path:
            return

        x_key = self._resolve_metric_name(
            getattr(self.cfg, "x_by", None), fallback="first_n_reward_span_s"
        )
        y_key = self._resolve_metric_name(getattr(self.cfg, "y_by", None), fallback="sli")
        color_key = self._resolve_metric_name(
            getattr(self.cfg, "color_by", None),
            fallback="control_circle_entry_count_by_cutoff",
        )
        eligible_rows = self._eligible_rows(rows)
        util.ensureDir(path)

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
        if not eligible_rows:
            ax.text(0.5, 0.5, "no eligible flies", ha="center", va="center")
            ax.set_axis_off()
        else:
            x_all = np.asarray([getattr(row, x_key, np.nan) for row in eligible_rows], dtype=float)
            y_all = np.asarray([getattr(row, y_key, np.nan) for row in eligible_rows], dtype=float)
            c_all = np.asarray([getattr(row, color_key, np.nan) for row in eligible_rows], dtype=float)
            finite_xy = np.isfinite(x_all) & np.isfinite(y_all)
            x = x_all[finite_xy]
            y = y_all[finite_xy]
            c = c_all[finite_xy]
            plot_rows = [row for row, keep in zip(eligible_rows, finite_xy) if keep]

            if x.size == 0:
                ax.text(0.5, 0.5, "no finite plot values", ha="center", va="center")
                ax.set_axis_off()
            else:
                finite_c = np.isfinite(c)
                if finite_c.any():
                    sc = ax.scatter(x, y, c=c, cmap="viridis", s=34, alpha=0.9)
                    cb = fig.colorbar(sc, ax=ax)
                    cb.set_label(self._metric_label(color_key))
                else:
                    ax.scatter(x, y, s=34, alpha=0.9)

                self._label_outliers(ax, plot_rows, x, y)
                ax.text(
                    0.02,
                    0.98,
                    self._correlation_text(x, y),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )
                ax.set_xlabel(self._metric_label(x_key))
                ax.set_ylabel(self._metric_label(y_key))
                ax.grid(True, alpha=0.2)

        title = (
            f"First-{int(self.cfg.first_n_rewards)} reward diagnostics"
            f" ({self._selected_subset_label()})"
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[{self.log_tag}] wrote plot: {path}")

    def run(self) -> list[FirstNRewardDiagnosticRow]:
        rows = self.compute_rows()
        self._write_csv(rows)
        self._write_npz(rows)
        self._write_plot(rows)
        if rows:
            eligible = sum(bool(r.eligible_for_nth_reward_cutoff) for r in rows)
            print(
                f"[{self.log_tag}] exported {len(rows)} flies; "
                f"{eligible} reached reward {int(self.cfg.first_n_rewards)}"
            )
        else:
            print(f"[{self.log_tag}] no rows to export")
        return rows
