from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from src.analysis.sli_tools import default_single_bucket_idx
from src.plotting.palettes import correlation_plot_color_for_metrics
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


BBOX_STYLE = dict(
    facecolor="white", alpha=0.80, edgecolor="none", boxstyle="round,pad=0.25"
)
STATS_BOX_MIN_FONTSIZE = 12.0
TREND_LINE_P_THRESHOLD = 0.05


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
    color_by: str | None = "control_circle_entry_count_by_cutoff"
    xlabel: str | None = None
    ylabel: str | None = None
    label_low_sli_outliers: int = 0
    reward_event_type: str = "actual"
    include_reward_event_type_in_labels: bool = False
    sli_training_idx: int | None = None
    sli_average_over_buckets: bool = False
    sli_skip_first_sync_buckets: int | None = None
    sli_keep_first_sync_buckets: int | None = None
    sli_explicit_bucket_idx: int | None = None
    sli_total_sync_buckets: int | None = None


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
    reward_event_type: str
    selected_reward_count_in_selected_window: int
    time_to_first_selected_reward_s: float
    time_to_nth_selected_reward_s: float
    first_n_selected_reward_span_s: float
    selected_reward_rate_to_nth_per_min: float


FIRST_N_REWARD_DIAGNOSTIC_FIELDS = tuple(
    FirstNRewardDiagnosticRow.__dataclass_fields__.keys()
)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _as_int(value, *, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _as_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _row_value(row, key: str):
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key)


def reward_rate_between_first_and_nth_per_min(n_target: int, span_s: float) -> float:
    try:
        n_intervals = int(n_target) - 1
    except Exception:
        return np.nan
    if n_intervals <= 0:
        return np.nan
    span_s = _as_float(span_s)
    if not np.isfinite(span_s) or span_s <= 0:
        return np.nan
    return float(n_intervals * 60.0 / span_s)


def validate_first_n_reward_diagnostic_rows(rows: Sequence, *, path: str | None = None) -> None:
    where = path or "<rows>"
    for idx, row in enumerate(rows):
        missing = [key for key in FIRST_N_REWARD_DIAGNOSTIC_FIELDS if _row_value(row, key) is None]
        if missing:
            raise ValueError(
                f"First-N reward diagnostics {where} row {idx} is missing columns: {missing}"
            )

        n_target = _as_int(_row_value(row, "nth_reward_target"), default=0)
        if n_target < 1:
            raise ValueError(
                f"First-N reward diagnostics {where} row {idx} has nth_reward_target={n_target}"
            )

        has_window = _as_bool(_row_value(row, "has_selected_window"))
        eligible = _as_bool(_row_value(row, "eligible_for_nth_reward_cutoff"))
        selected_count = _as_int(
            _row_value(row, "selected_reward_count_in_selected_window"),
            default=0,
        )
        actual_count = _as_int(
            _row_value(row, "actual_reward_count_in_selected_window"),
            default=0,
        )
        if selected_count < 0 or actual_count < 0:
            raise ValueError(
                f"First-N reward diagnostics {where} row {idx} has negative reward counts"
            )
        if eligible and not has_window:
            raise ValueError(
                f"First-N reward diagnostics {where} row {idx} is eligible without a selected window"
            )
        if eligible and selected_count < n_target:
            raise ValueError(
                f"First-N reward diagnostics {where} row {idx} is eligible with "
                f"{selected_count} selected rewards for target {n_target}"
            )

        rate = _as_float(_row_value(row, "selected_reward_rate_to_nth_per_min"))
        span_s = _as_float(_row_value(row, "first_n_selected_reward_span_s"))
        expected_rate = reward_rate_between_first_and_nth_per_min(n_target, span_s)

        if not eligible:
            if np.isfinite(rate):
                raise ValueError(
                    f"First-N reward diagnostics {where} row {idx} has finite rate while ineligible"
                )
            continue

        for key in (
            "cutoff_frame",
            "cutoff_time_since_selected_window_start_s",
            "time_to_first_selected_reward_s",
            "time_to_nth_selected_reward_s",
        ):
            value = _as_float(_row_value(row, key))
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"First-N reward diagnostics {where} row {idx} has invalid {key}={value}"
                )

        if np.isfinite(expected_rate):
            if not np.isfinite(rate):
                raise ValueError(
                    f"First-N reward diagnostics {where} row {idx} has non-finite selected reward rate"
                )
            if abs(rate - expected_rate) > 1e-9:
                raise ValueError(
                    f"First-N reward diagnostics {where} row {idx} has inconsistent "
                    f"selected reward rate {rate}; expected {expected_rate}"
                )
        elif np.isfinite(rate):
            raise ValueError(
                f"First-N reward diagnostics {where} row {idx} has finite selected "
                "reward rate without a positive first-to-nth interval"
            )


def validate_first_n_reward_diagnostics_csv(path: str) -> None:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [
            key for key in FIRST_N_REWARD_DIAGNOSTIC_FIELDS if key not in fieldnames
        ]
        if missing:
            raise ValueError(
                f"First-N reward diagnostics {path} is missing columns: {missing}"
            )
        validate_first_n_reward_diagnostic_rows(list(reader), path=path)


def load_first_n_reward_diagnostics_npz(path: str) -> dict:
    with np.load(path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}
    validate_first_n_reward_diagnostics_bundle(out, path=path)
    return out


def validate_first_n_reward_diagnostics_bundle(
    bundle: dict, *, path: str | None = None
) -> None:
    where = path or "<bundle>"
    missing = [key for key in FIRST_N_REWARD_DIAGNOSTIC_FIELDS if key not in bundle]
    if missing:
        raise ValueError(
            f"First-N reward diagnostics {where} is missing keys: {missing}"
        )
    lengths = {
        key: int(np.asarray(bundle[key]).reshape(-1).shape[0])
        for key in FIRST_N_REWARD_DIAGNOSTIC_FIELDS
    }
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            f"First-N reward diagnostics {where} has inconsistent column lengths: {lengths}"
        )
    n_rows = unique_lengths.pop() if unique_lengths else 0
    rows = [
        {
            key: np.asarray(bundle[key], dtype=object).reshape(-1)[idx]
            for key in FIRST_N_REWARD_DIAGNOSTIC_FIELDS
        }
        for idx in range(n_rows)
    ]
    validate_first_n_reward_diagnostic_rows(rows, path=where)


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
        return list(FIRST_N_REWARD_DIAGNOSTIC_FIELDS)

    def _nth_reward_target(self) -> int:
        return max(1, int(self.cfg.first_n_rewards or 1))

    @staticmethod
    def _reward_rate_between_first_and_nth_per_min(
        n_target: int,
        span_s: float,
    ) -> float:
        return reward_rate_between_first_and_nth_per_min(n_target, span_s)

    def _selected_trainings(self) -> list[int]:
        if not self.vas:
            return []
        return selected_training_indices(
            self.vas[0],
            self.cfg.trainings,
            log_tag=self.log_tag,
        )

    def _sli_axis_label(self) -> str:
        sli_training_idx = getattr(self.cfg, "sli_training_idx", None)
        if sli_training_idx is None:
            trainings = self._selected_trainings()
            training_txt = training_window_label(trainings)
        else:
            training_txt = f"T{int(sli_training_idx) + 1}"

        explicit_bucket_idx = getattr(self.cfg, "sli_explicit_bucket_idx", None)
        if explicit_bucket_idx is not None:
            return f"SLI at {training_txt} SB{int(explicit_bucket_idx) + 1}"

        skip_first_raw = getattr(self.cfg, "sli_skip_first_sync_buckets", None)
        keep_first_raw = getattr(self.cfg, "sli_keep_first_sync_buckets", None)
        skip_first = max(
            0,
            int(
                self.cfg.skip_first_sync_buckets
                if skip_first_raw is None
                else skip_first_raw
            ),
        )
        keep_first = max(
            0,
            int(
                self.cfg.keep_first_sync_buckets
                if keep_first_raw is None
                else keep_first_raw
            ),
        )
        start_sb = skip_first + 1
        if keep_first > 0:
            end_sb = start_sb + keep_first - 1
        else:
            total_sync_buckets = getattr(self.cfg, "sli_total_sync_buckets", None)
            if total_sync_buckets is None:
                total_sync_buckets = self._infer_sli_total_sync_buckets()
            start_idx = start_sb - 1
            end_sb = (
                None
                if total_sync_buckets is None or int(total_sync_buckets) <= start_idx
                else default_single_bucket_idx(start_idx, int(total_sync_buckets)) + 1
            )

        if end_sb is None:
            window_txt = f"{training_txt} SB{start_sb}–end"
        elif end_sb == start_sb:
            window_txt = f"{training_txt} SB{start_sb}"
        else:
            window_txt = f"{training_txt} SB{start_sb}–{end_sb}"
        if bool(getattr(self.cfg, "sli_average_over_buckets", False)):
            return f"Mean SLI over {window_txt}"
        if end_sb is None and start_sb == 1:
            return f"SLI at final SB of {training_txt}"
        if end_sb != start_sb:
            return f"SLI at final SB in {window_txt}"
        return f"SLI at {window_txt}"

    def _infer_sli_total_sync_buckets(self) -> int | None:
        if not self.vas:
            return None
        sli_training_idx = getattr(self.cfg, "sli_training_idx", None)
        if sli_training_idx is None:
            return None
        try:
            ref_va = self.vas[0]
            vals = np.asarray(getattr(ref_va, "rewardPI", []))
            trns = getattr(ref_va, "trns", None) or []
            n_trainings = len(trns)
            flies = getattr(ref_va, "flies", None) or []
            n_flies = len(flies)
            if vals.size <= 0 or n_trainings <= 0 or n_flies <= 0:
                return None
            per_training = vals.size // n_trainings
            if per_training <= 0 or per_training % n_flies != 0:
                return None
            return max(1, int(per_training // n_flies))
        except Exception:
            return None

    def _metric_label(self, name: str) -> str:
        n_target = self._nth_reward_target()
        reward_event_type = str(
            getattr(self.cfg, "reward_event_type", "actual") or "actual"
        )
        include_reward_type = bool(
            getattr(self.cfg, "include_reward_event_type_in_labels", False)
        )
        reward_type_txt = {
            "actual": "actual",
            "calc": "calculated",
        }.get(reward_event_type, reward_event_type)
        reward_phrase = "rewards"
        reward_phrase_singular = "reward"
        if include_reward_type:
            reward_phrase = f"{reward_type_txt} rewards"
            reward_phrase_singular = f"{reward_type_txt} reward"
        labels = {
            "sli": self._sli_axis_label(),
            "time_to_first_actual_reward_s": "Time to 1st actual reward (s)",
            "time_to_nth_actual_reward_s": f"Time to {n_target}th actual reward (s)",
            "first_n_reward_span_s": f"Time from 1st to {n_target}th actual reward (s)",
            "time_to_first_selected_reward_s": (
                f"Time to 1st {reward_phrase_singular} (s)"
            ),
            "time_to_nth_selected_reward_s": (
                f"Time to first {n_target} {reward_phrase} (s)"
            ),
            "first_n_selected_reward_span_s": (
                f"Time from 1st to {n_target}th {reward_phrase_singular} (s)"
            ),
            "selected_reward_rate_to_nth_per_min": (
                f"Reward rate during first {n_target} {reward_phrase} "
                r"($min^{-1}$)"
            ),
        }
        return labels.get(str(name), str(name).replace("_", " "))

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

        reward_event_type = str(
            getattr(self.cfg, "reward_event_type", "actual") or "actual"
        )
        span_value = (
            row.first_n_selected_reward_span_s
            if reward_event_type == "calc"
            else row.first_n_reward_span_s
        )
        time_to_nth_value = (
            row.time_to_nth_selected_reward_s
            if reward_event_type == "calc"
            else row.time_to_nth_actual_reward_s
        )

        max_span_s = self.cfg.max_span_s
        if max_span_s is not None:
            try:
                max_span_s = float(max_span_s)
            except Exception:
                max_span_s = None
            if max_span_s is not None:
                if not np.isfinite(span_value):
                    return False
                if span_value > max_span_s:
                    return False

        max_time_to_nth_s = self.cfg.max_time_to_nth_s
        if max_time_to_nth_s is not None:
            try:
                max_time_to_nth_s = float(max_time_to_nth_s)
            except Exception:
                max_time_to_nth_s = None
            if max_time_to_nth_s is not None:
                if not np.isfinite(time_to_nth_value):
                    return False
                if time_to_nth_value > max_time_to_nth_s:
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

            reward_event_type = str(getattr(self.cfg, "reward_event_type", "actual") or "actual")
            use_calc_rewards = reward_event_type == "calc"
            actual_rewards = frames_in_windows(va, windows, calc=False, ctrl=False, f=0)
            selected_rewards = frames_in_windows(
                va, windows, calc=use_calc_rewards, ctrl=False, f=0
            )
            actual_entries = frames_in_windows(va, windows, calc=True, ctrl=False, f=0)
            control_entries = frames_in_windows(va, windows, calc=True, ctrl=True, f=0)

            has_window = bool(windows)
            n_actual = int(actual_rewards.size)
            n_selected = int(selected_rewards.size)
            eligible = has_window and n_selected >= n_target
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
            time_to_first_selected_s = np.nan
            time_to_nth_selected_s = np.nan
            selected_span_s = np.nan
            selected_reward_rate_to_nth_per_min = np.nan

            if n_actual > 0:
                time_to_first_s = cumulative_window_seconds_for_frame(
                    windows, int(actual_rewards[0]), fps=fps
                )
            if n_selected > 0:
                time_to_first_selected_s = cumulative_window_seconds_for_frame(
                    windows, int(selected_rewards[0]), fps=fps
                )

            if eligible:
                cutoff_frame = float(selected_rewards[n_target - 1])
                cutoff_time_s = cumulative_window_seconds_for_frame(
                    windows, int(selected_rewards[n_target - 1]), fps=fps
                )
                time_to_nth_s = cutoff_time_s
                time_to_nth_selected_s = cutoff_time_s
                if np.isfinite(time_to_first_s) and np.isfinite(time_to_nth_s):
                    span_s = float(time_to_nth_s - time_to_first_s)
                if (
                    np.isfinite(time_to_first_selected_s)
                    and np.isfinite(time_to_nth_selected_s)
                ):
                    selected_span_s = float(
                        time_to_nth_selected_s - time_to_first_selected_s
                    )
                selected_reward_rate_to_nth_per_min = (
                    self._reward_rate_between_first_and_nth_per_min(
                        n_target,
                        selected_span_s,
                    )
                )

                cutoff_window = locate_window_for_frame(
                    windows, int(selected_rewards[n_target - 1])
                )
                if cutoff_window is not None:
                    cutoff_training = float(cutoff_window.training_idx + 1)
                    cutoff_time_training_s = max(
                        0.0,
                        float(selected_rewards[n_target - 1] - cutoff_window.trn.start)
                        / fps,
                    )

                actual_reward_count_by_cutoff = float(
                    np.searchsorted(
                        actual_rewards,
                        int(selected_rewards[n_target - 1]),
                        side="right",
                    )
                )
                actual_entry_count_by_cutoff = float(
                    np.searchsorted(
                        actual_entries,
                        int(selected_rewards[n_target - 1]),
                        side="right",
                    )
                )
                control_entry_count_by_cutoff = float(
                    np.searchsorted(
                        control_entries,
                        int(selected_rewards[n_target - 1]),
                        side="right",
                    )
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
                    reward_event_type=reward_event_type,
                    selected_reward_count_in_selected_window=n_selected,
                    time_to_first_selected_reward_s=time_to_first_selected_s,
                    time_to_nth_selected_reward_s=time_to_nth_selected_s,
                    first_n_selected_reward_span_s=selected_span_s,
                    selected_reward_rate_to_nth_per_min=selected_reward_rate_to_nth_per_min,
                )
            )

        return rows

    def compute_rows(self) -> list[FirstNRewardDiagnosticRow]:
        return self._eligible_rows(self.compute_all_rows())

    def _write_csv(self, rows: list[FirstNRewardDiagnosticRow]) -> None:
        path = self.cfg.csv_out
        validate_first_n_reward_diagnostic_rows(rows, path=path)
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
        validate_first_n_reward_diagnostic_rows(rows, path=path)
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
            "reward_event_type": str(getattr(self.cfg, "reward_event_type", "actual") or "actual"),
        }
        payload = {**cols, "metadata": np.asarray(meta, dtype=object)}
        validate_first_n_reward_diagnostics_bundle(payload, path=path)
        np.savez(path, **payload)
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
    def _correlation_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int] | None:
        n = int(min(len(x), len(y)))
        if n < 3:
            return None
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            return None
        try:
            r, p = pearsonr(x, y)
        except Exception:
            return None
        return float(r), float(p), n

    @staticmethod
    def _correlation_text(stats: tuple[float, float, int] | None, n: int) -> str:
        if stats is None:
            return f"n = {int(n)}, r = n/a, p = n/a"
        r, p, n = stats
        return f"n = {int(n)}, r = {r:.3f}, p = {p:.3g}"

    @staticmethod
    def _add_significant_trend_line(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        stats: tuple[float, float, int] | None,
        *,
        color: str,
    ) -> bool:
        if stats is None:
            return False
        _, p, _ = stats
        if not np.isfinite(p) or float(p) > TREND_LINE_P_THRESHOLD:
            return False

        x = np.asarray(x, float)
        y = np.asarray(y, float)
        finite = np.isfinite(x) & np.isfinite(y)
        x_f = x[finite]
        y_f = y[finite]
        if x_f.size < 3 or np.unique(x_f).size < 2:
            return False

        try:
            slope, intercept = np.polyfit(x_f, y_f, 1)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return False
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            return False

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_line = np.asarray(xlim, float)
        if x_line.size != 2 or not np.all(np.isfinite(x_line)):
            return False
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            color=color,
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            zorder=2,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return True

    @staticmethod
    def _stats_box_fontsize(ax) -> float:
        reference_sizes = [
            ax.xaxis.label.get_size(),
            ax.yaxis.label.get_size(),
            *(tick.get_size() for tick in ax.get_xticklabels()),
            *(tick.get_size() for tick in ax.get_yticklabels()),
        ]
        finite_sizes = [
            float(size)
            for size in reference_sizes
            if size is not None and np.isfinite(float(size))
        ]
        reference_size = max(finite_sizes) if finite_sizes else STATS_BOX_MIN_FONTSIZE
        return max(STATS_BOX_MIN_FONTSIZE, 0.90 * reference_size)

    @staticmethod
    def _split_axis_label_evenly(text: str) -> str:
        words = text.split()
        if len(words) < 4:
            return text
        mid = len(words) // 2
        left, right = words[:mid], words[mid:]
        if len(left) < 2 or len(right) < 2:
            return text
        return " ".join(left) + "\n" + " ".join(right)

    @classmethod
    def _wrap_clipped_axis_labels(cls, fig, *, pad_px: float = 2.0) -> bool:
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            fig_bbox = fig.get_window_extent(renderer=renderer)
        except Exception:
            return False

        changed = False
        for ax in fig.get_axes():
            x_label = ax.xaxis.get_label()
            x_text = x_label.get_text()
            if x_label.get_visible() and x_text and "\n" not in x_text:
                bbox = x_label.get_window_extent(renderer=renderer)
                clipped = (
                    float(bbox.x0) < float(fig_bbox.x0) + pad_px
                    or float(bbox.x1) > float(fig_bbox.x1) - pad_px
                )
                if clipped:
                    wrapped = cls._split_axis_label_evenly(x_text)
                    if wrapped != x_text:
                        x_label.set_text(wrapped)
                        changed = True

            y_label = ax.yaxis.get_label()
            y_text = y_label.get_text()
            if y_label.get_visible() and y_text and "\n" not in y_text:
                bbox = y_label.get_window_extent(renderer=renderer)
                clipped = (
                    float(bbox.y0) < float(fig_bbox.y0) + pad_px
                    or float(bbox.y1) > float(fig_bbox.y1) - pad_px
                )
                if clipped:
                    wrapped = cls._split_axis_label_evenly(y_text)
                    if wrapped != y_text:
                        y_label.set_text(wrapped)
                        changed = True

        return changed

    @staticmethod
    def _title_for_metrics(x_key: str, y_key: str) -> str:
        if x_key == "selected_reward_rate_to_nth_per_min" and y_key == "sli":
            return "Initial reward rate and SLI"
        return ""

    def _write_plot(self, rows: list[FirstNRewardDiagnosticRow]) -> None:
        path = self.cfg.plot_out
        if not path:
            return

        x_key = self._resolve_metric_name(
            getattr(self.cfg, "x_by", None), fallback="first_n_reward_span_s"
        )
        y_key = self._resolve_metric_name(getattr(self.cfg, "y_by", None), fallback="sli")
        color_cfg = getattr(self.cfg, "color_by", None)
        color_key = (
            None
            if color_cfg is None
            else self._resolve_metric_name(
                color_cfg,
                fallback="control_circle_entry_count_by_cutoff",
            )
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
            finite_xy = np.isfinite(x_all) & np.isfinite(y_all)
            x = x_all[finite_xy]
            y = y_all[finite_xy]
            plot_rows = [row for row, keep in zip(eligible_rows, finite_xy) if keep]

            if x.size == 0:
                ax.text(0.5, 0.5, "no finite plot values", ha="center", va="center")
                ax.set_axis_off()
            else:
                plain_scatter_color = correlation_plot_color_for_metrics(x_key, y_key)
                if color_key is not None:
                    c_all = np.asarray(
                        [getattr(row, color_key, np.nan) for row in eligible_rows],
                        dtype=float,
                    )
                    c = c_all[finite_xy]
                    finite_c = np.isfinite(c)
                else:
                    c = np.asarray([], dtype=float)
                    finite_c = np.asarray([], dtype=bool)

                if color_key is not None and finite_c.any():
                    sc = ax.scatter(x, y, c=c, cmap="viridis", s=34, alpha=0.9)
                    cb = fig.colorbar(sc, ax=ax)
                    cb.set_label(self._metric_label(color_key))
                else:
                    scatter_kwargs = {"s": 34, "alpha": 0.9}
                    if color_key is None:
                        scatter_kwargs["color"] = plain_scatter_color
                    ax.scatter(x, y, **scatter_kwargs)

                corr_stats = self._correlation_stats(x, y)
                self._add_significant_trend_line(
                    ax,
                    x,
                    y,
                    corr_stats,
                    color=plain_scatter_color,
                )
                self._label_outliers(ax, plot_rows, x, y)
                ax.set_xlabel(str(self.cfg.xlabel or self._metric_label(x_key)))
                ax.set_ylabel(str(self.cfg.ylabel or self._metric_label(y_key)))
                ax.grid(True, alpha=0.2)
                ax.text(
                    0.02,
                    0.98,
                    self._correlation_text(corr_stats, len(x)),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=self._stats_box_fontsize(ax),
                    bbox=BBOX_STYLE,
                )

        title = self._title_for_metrics(
            self._resolve_metric_name(
                getattr(self.cfg, "x_by", None), fallback="first_n_reward_span_s"
            ),
            self._resolve_metric_name(getattr(self.cfg, "y_by", None), fallback="sli"),
        )
        if not title:
            title = (
                f"First-{int(self.cfg.first_n_rewards)} reward diagnostics"
                f" ({self._selected_subset_label()})"
            )
        ax.set_title(title)
        fig.tight_layout()
        if self._wrap_clipped_axis_labels(fig):
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
