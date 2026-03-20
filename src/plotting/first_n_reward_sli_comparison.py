from __future__ import annotations

import csv
from dataclasses import asdict, dataclass

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

from src.plotting.first_n_reward_diagnostics import (
    FirstNRewardDiagnosticRow,
    FirstNRewardDiagnosticsConfig,
    FirstNRewardDiagnosticsPlotter,
)
import src.utils.util as util


@dataclass(frozen=True)
class FirstNRewardSLIComparisonConfig:
    plot_out: str
    csv_out: str | None = None
    trainings: tuple[int, ...] | None = None
    skip_first_sync_buckets: int = 0
    keep_first_sync_buckets: int = 0
    first_n_rewards: int = 10
    pi_threshold: int = 10
    max_span_s: float | None = None
    max_time_to_nth_s: float | None = None
    metric: str = "time_to_nth_actual_reward_s"
    top_label: str = "Top SLI-selected"
    bottom_label: str = "Bottom SLI-selected"
    title: str | None = None


@dataclass(frozen=True)
class _GroupComputation:
    all_rows: list[FirstNRewardDiagnosticRow]
    filtered_rows: list[FirstNRewardDiagnosticRow]
    subset_label: str


class FirstNRewardSLIComparisonPlotter:
    def __init__(
        self,
        *,
        vas_top,
        vas_bottom,
        opts,
        gls,
        top_sli_values,
        bottom_sli_values,
        cfg: FirstNRewardSLIComparisonConfig,
    ):
        self.vas_top = list(vas_top)
        self.vas_bottom = list(vas_bottom)
        self.opts = opts
        self.gls = gls
        self.top_sli_values = list(top_sli_values) if top_sli_values is not None else None
        self.bottom_sli_values = (
            list(bottom_sli_values) if bottom_sli_values is not None else None
        )
        self.cfg = cfg
        self.log_tag = "first_n_reward_sli_cmp"

    @staticmethod
    def _metric_field_names() -> list[str]:
        return list(FirstNRewardDiagnosticRow.__dataclass_fields__.keys())

    def _resolve_metric_name(self, name: str | None) -> str:
        candidate = str(name or "time_to_nth_actual_reward_s")
        if candidate in self._metric_field_names():
            return candidate
        print(
            f"[{self.log_tag}] WARNING: unknown metric '{candidate}'; "
            "falling back to 'time_to_nth_actual_reward_s'."
        )
        return "time_to_nth_actual_reward_s"

    @staticmethod
    def _pretty_phrase(name: str) -> str:
        raw = str(name or "")
        special = {
            "time_to_first_actual_reward_s": "Time to first actual reward",
            "time_to_nth_actual_reward_s": "Time to nth actual reward",
            "first_n_reward_span_s": "Span from first to nth actual reward",
            "cutoff_time_since_selected_window_start_s": "Nth reward time from selected window start",
            "cutoff_time_since_cutoff_training_start_s": "Nth reward time from cutoff training start",
            "reward_pi_by_cutoff": "Reward PI at nth reward cutoff",
            "actual_reward_count_by_cutoff": "Actual reward count at nth reward cutoff",
            "control_reward_count_by_cutoff": "Control reward count at nth reward cutoff",
            "actual_circle_entry_count_by_cutoff": "Actual circle entry count at nth reward cutoff",
            "control_circle_entry_count_by_cutoff": "Control circle entry count at nth reward cutoff",
            "actual_reward_count_in_selected_window": "Actual reward count in selected window",
            "sli": "SLI",
        }
        return special.get(raw, raw.replace("_", " "))

    def _metric_label(self, name: str) -> str:
        phrase = self._pretty_phrase(name)
        if self._looks_like_seconds_metric(name) and not phrase.endswith("(s)"):
            return f"{phrase} (s)"
        return phrase

    @staticmethod
    def _looks_like_seconds_metric(name: str) -> bool:
        metric = str(name)
        return metric.endswith("_s") or "_time_" in metric or "span" in metric

    def _format_mean(self, value: float, metric: str) -> str:
        suffix = "s" if self._looks_like_seconds_metric(metric) else ""
        return f"{float(value):.2f}{suffix}"

    def _diag_cfg(self, *, subset_label: str, sli_values) -> FirstNRewardDiagnosticsConfig:
        return FirstNRewardDiagnosticsConfig(
            csv_out="",
            npz_out=None,
            plot_out=None,
            trainings=self.cfg.trainings,
            skip_first_sync_buckets=int(self.cfg.skip_first_sync_buckets or 0),
            keep_first_sync_buckets=int(self.cfg.keep_first_sync_buckets or 0),
            first_n_rewards=max(1, int(self.cfg.first_n_rewards or 1)),
            subset_label=subset_label,
            sli_values=sli_values,
            pi_threshold=int(self.cfg.pi_threshold or 0),
            max_span_s=self.cfg.max_span_s,
            max_time_to_nth_s=self.cfg.max_time_to_nth_s,
        )

    def _compute_rows_for_group(self, vas, *, sli_values, subset_label: str) -> _GroupComputation:
        plotter = FirstNRewardDiagnosticsPlotter(
            vas=vas,
            opts=self.opts,
            gls=self.gls,
            cfg=self._diag_cfg(subset_label=subset_label, sli_values=sli_values),
        )
        all_rows = plotter.compute_all_rows()
        return _GroupComputation(
            all_rows=all_rows,
            filtered_rows=plotter._eligible_rows(all_rows),
            subset_label=subset_label,
        )

    def _comparison_df(self) -> tuple[pd.DataFrame, dict[str, _GroupComputation]]:
        metric = self._resolve_metric_name(self.cfg.metric)
        rows_top = self._compute_rows_for_group(
            self.vas_top,
            sli_values=self.top_sli_values,
            subset_label=self.cfg.top_label,
        )
        rows_bottom = self._compute_rows_for_group(
            self.vas_bottom,
            sli_values=self.bottom_sli_values,
            subset_label=self.cfg.bottom_label,
        )

        records = []
        groups = {
            self.cfg.bottom_label: rows_bottom,
            self.cfg.top_label: rows_top,
        }
        for group_label, group in (
            (self.cfg.bottom_label, rows_bottom),
            (self.cfg.top_label, rows_top),
        ):
            for row in group.filtered_rows:
                value = getattr(row, metric, np.nan)
                if np.isfinite(value):
                    row_dict = asdict(row)
                    row_dict["sli_group"] = group_label
                    row_dict["comparison_metric"] = metric
                    row_dict["comparison_value"] = float(value)
                    records.append(row_dict)

        if not records:
            return pd.DataFrame(
                columns=["sli_group", "comparison_metric", "comparison_value"]
            ), groups
        return pd.DataFrame.from_records(records), groups

    def _welch_ttest(self, df: pd.DataFrame) -> tuple[float, float] | None:
        vals_bottom = (
            df.loc[df["sli_group"] == self.cfg.bottom_label, "comparison_value"]
            .to_numpy(dtype=float)
        )
        vals_top = (
            df.loc[df["sli_group"] == self.cfg.top_label, "comparison_value"]
            .to_numpy(dtype=float)
        )
        vals_bottom = vals_bottom[np.isfinite(vals_bottom)]
        vals_top = vals_top[np.isfinite(vals_top)]
        if len(vals_bottom) < 2 or len(vals_top) < 2:
            return None
        t_stat, p_value = ttest_ind(
            vals_bottom,
            vals_top,
            equal_var=False,
            nan_policy="omit",
        )
        return float(t_stat), float(p_value)

    def _ttest_text(self, df: pd.DataFrame, groups: dict[str, _GroupComputation]) -> str:
        metric = self._resolve_metric_name(self.cfg.metric)
        vals_bottom = (
            df.loc[df["sli_group"] == self.cfg.bottom_label, "comparison_value"]
            .to_numpy(dtype=float)
        )
        vals_top = (
            df.loc[df["sli_group"] == self.cfg.top_label, "comparison_value"]
            .to_numpy(dtype=float)
        )
        vals_bottom = vals_bottom[np.isfinite(vals_bottom)]
        vals_top = vals_top[np.isfinite(vals_top)]

        lines = [
            f"{self.cfg.bottom_label}: n={len(vals_bottom)}, mean={self._format_mean(np.nanmean(vals_bottom), metric)}"
            if len(vals_bottom)
            else f"{self.cfg.bottom_label}: n=0",
            f"{self.cfg.top_label}: n={len(vals_top)}, mean={self._format_mean(np.nanmean(vals_top), metric)}"
            if len(vals_top)
            else f"{self.cfg.top_label}: n=0",
        ]
        ttest = self._welch_ttest(df)
        if ttest is not None:
            t_stat, p_value = ttest
            lines.append(f"Welch t-test: t={t_stat:.3f}, p={p_value:.3g}")
        else:
            lines.append("Welch t-test: n/a")
        cutoff_note = self._cutoff_note(groups)
        if cutoff_note:
            lines.extend(cutoff_note.split("\n"))
        return "\n".join(lines)

    def _cutoff_note(self, groups: dict[str, _GroupComputation]) -> str:
        notes = []
        max_time = self.cfg.max_time_to_nth_s
        if max_time is not None:
            limited = []
            for label, group in groups.items():
                pre = sum(bool(r.eligible_for_nth_reward_cutoff) for r in group.all_rows)
                post = len(group.filtered_rows)
                if pre > post:
                    limited.append(f"{label}: {post}/{pre} kept")
            if limited:
                notes.append(
                    f"Max nth-reward time filter: <= {float(max_time):.0f} s"
                )
                notes.extend(limited)

        max_span = self.cfg.max_span_s
        if max_span is not None:
            limited = []
            for label, group in groups.items():
                pre = sum(bool(r.eligible_for_nth_reward_cutoff) for r in group.all_rows)
                post = len(group.filtered_rows)
                if pre > post:
                    limited.append(f"{label}: {post}/{pre} kept")
            if limited:
                notes.append(f"Max first-to-nth span filter: <= {float(max_span):.0f} s")
                notes.extend(limited)

        return "\n".join(notes)

    @staticmethod
    def _add_smart_stats_box(
        ax,
        text: str,
        x: np.ndarray,
        y: np.ndarray,
        *,
        fontsize: int = 9,
        reserved_y_min: float | None = None,
    ):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        finite = np.isfinite(x) & np.isfinite(y)
        x_f = x[finite]
        y_f = y[finite]
        bbox_style = {
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
        }

        if x_f.size == 0:
            return ax.text(
                0.05,
                0.94,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=fontsize,
                zorder=5,
                bbox=bbox_style,
            )

        fig = ax.figure
        probe = ax.text(
            0.05,
            0.94,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fontsize,
            zorder=5,
            alpha=0.0,
            bbox=bbox_style,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pts_display = ax.transData.transform(np.column_stack([x_f, y_f]))
        reserved_patch_bbox = None
        if reserved_y_min is not None and np.isfinite(reserved_y_min):
            x0, x1 = ax.get_xlim()
            y1 = ax.get_ylim()[1]
            reserved_pts = ax.transData.transform(
                np.array([[x0, reserved_y_min], [x1, y1]], dtype=float)
            )
            rx0 = min(reserved_pts[0, 0], reserved_pts[1, 0])
            rx1 = max(reserved_pts[0, 0], reserved_pts[1, 0])
            ry0 = min(reserved_pts[0, 1], reserved_pts[1, 1])
            ry1 = max(reserved_pts[0, 1], reserved_pts[1, 1])
            reserved_patch_bbox = (rx0, rx1, ry0, ry1)

        candidates = [
            dict(x=0.04, y=0.94, ha="left", va="top"),
            dict(x=0.96, y=0.94, ha="right", va="top"),
            dict(x=0.04, y=0.05, ha="left", va="bottom"),
            dict(x=0.96, y=0.05, ha="right", va="bottom"),
        ]
        best_candidate = None
        best_overlap = None
        best_patch_bbox = None

        for candidate in candidates:
            probe.set_position((candidate["x"], candidate["y"]))
            probe.set_ha(candidate["ha"])
            probe.set_va(candidate["va"])
            fig.canvas.draw()
            patch_bbox = probe.get_bbox_patch().get_window_extent(renderer=renderer)
            inside = (
                (pts_display[:, 0] >= patch_bbox.x0)
                & (pts_display[:, 0] <= patch_bbox.x1)
                & (pts_display[:, 1] >= patch_bbox.y0)
                & (pts_display[:, 1] <= patch_bbox.y1)
            )
            overlap = float(np.mean(inside))
            if reserved_patch_bbox is not None:
                rx0, rx1, ry0, ry1 = reserved_patch_bbox
                overlaps_reserved = not (
                    patch_bbox.x1 < rx0
                    or patch_bbox.x0 > rx1
                    or patch_bbox.y1 < ry0
                    or patch_bbox.y0 > ry1
                )
                if overlaps_reserved:
                    overlap += 1.0
            if best_overlap is None or overlap < best_overlap:
                best_overlap = overlap
                best_candidate = candidate
                best_patch_bbox = patch_bbox

        probe.remove()

        if best_candidate is not None and best_overlap is not None and best_overlap <= 0.05:
            return ax.text(
                best_candidate["x"],
                best_candidate["y"],
                text,
                transform=ax.transAxes,
                va=best_candidate["va"],
                ha=best_candidate["ha"],
                fontsize=fontsize,
                zorder=5,
                bbox=bbox_style,
            )

        y0, y1 = ax.get_ylim()
        y_span = y1 - y0 if np.isfinite(y1 - y0) and y1 > y0 else 1.0
        box_height_frac = 0.22
        if best_patch_bbox is not None and ax.bbox.height > 0:
            box_height_frac = best_patch_bbox.height / ax.bbox.height
        extra_top = max((box_height_frac + 0.08) * y_span, 0.18 * y_span)
        original_top = y1
        ax.set_ylim(y0, y1 + extra_top)

        x0, x1 = ax.get_xlim()
        x_span = x1 - x0 if np.isfinite(x1 - x0) and x1 > x0 else 1.0
        return ax.text(
            x0 + 0.02 * x_span,
            original_top + 0.90 * extra_top,
            text,
            va="top",
            ha="left",
            fontsize=fontsize,
            zorder=5,
            bbox=bbox_style,
        )

    def _add_significance_bracket(self, ax, df: pd.DataFrame) -> float | None:
        ttest = self._welch_ttest(df)
        if ttest is None:
            return None
        _t_stat, p_value = ttest
        stars = util.p2stars(p_value, nanR="")
        if not stars:
            stars = "n.s."

        vals = df["comparison_value"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None

        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min if np.isfinite(y_max - y_min) and y_max > y_min else 1.0
        bracket_y = float(np.nanmax(vals)) + 0.07 * y_span
        leg_h = 0.03 * y_span
        text_y = bracket_y + 0.015 * y_span

        ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + leg_h, bracket_y + leg_h, bracket_y],
                color="black", linewidth=1.2, clip_on=False)
        ax.text(
            0.5,
            text_y + leg_h,
            stars,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
        bracket_top = text_y + leg_h + 0.04 * y_span
        ax.set_ylim(y_min, max(y_max, bracket_top + 0.02 * y_span))
        return bracket_y

    @staticmethod
    def _force_swarm_marker_fill(ax, order: list[str], palette: dict[str, str]) -> None:
        if not getattr(ax, "collections", None):
            return
        recent = ax.collections[-len(order) :] if len(ax.collections) >= len(order) else ax.collections
        for label, collection in zip(order, recent):
            try:
                offsets = collection.get_offsets()
            except Exception:
                continue
            n_pts = len(offsets)
            if n_pts == 0:
                continue
            face_rgba = np.tile(mcolors.to_rgba(palette[label], alpha=0.95), (n_pts, 1))
            edge_rgba = np.tile(mcolors.to_rgba("black", alpha=1.0), (n_pts, 1))
            try:
                collection.set_facecolors(face_rgba)
                collection.set_edgecolors(edge_rgba)
                collection.set_linewidth(0.9)
                collection.set_alpha(0.95)
            except Exception:
                pass

    def _write_csv(self, df: pd.DataFrame) -> None:
        path = self.cfg.csv_out
        if not path:
            return
        util.ensureDir(path)
        fieldnames = list(df.columns)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in df.to_dict(orient="records"):
                writer.writerow(row)
        print(f"[{self.log_tag}] wrote CSV: {path}")

    def _write_plot(self, df: pd.DataFrame, groups: dict[str, _GroupComputation]) -> None:
        path = self.cfg.plot_out
        util.ensureDir(path)
        metric = self._resolve_metric_name(self.cfg.metric)

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.6))
        order = [self.cfg.bottom_label, self.cfg.top_label]
        palette = {
            self.cfg.bottom_label: "#d95f02",
            self.cfg.top_label: "#1b9e77",
        }

        if df.empty:
            ax.text(0.5, 0.5, "no eligible flies", ha="center", va="center")
            ax.set_axis_off()
        else:
            sns.boxplot(
                data=df,
                x="sli_group",
                y="comparison_value",
                order=order,
                palette=palette,
                width=0.55,
                showcaps=True,
                showfliers=False,
                boxprops={"alpha": 0.45},
                whiskerprops={"linewidth": 1.2},
                medianprops={"color": "black", "linewidth": 1.4},
                ax=ax,
            )
            sns.swarmplot(
                data=df,
                x="sli_group",
                y="comparison_value",
                order=order,
                palette=palette,
                size=5.5,
                linewidth=0.9,
                edgecolor="black",
                ax=ax,
            )
            self._force_swarm_marker_fill(ax, order, palette)
            bracket_y = self._add_significance_bracket(ax, df)
            x_positions = np.where(
                df["sli_group"].to_numpy() == self.cfg.bottom_label, 0.0, 1.0
            )
            self._add_smart_stats_box(
                ax,
                self._ttest_text(df, groups),
                x_positions,
                df["comparison_value"].to_numpy(dtype=float),
                fontsize=9,
                reserved_y_min=bracket_y,
            )
            ax.set_xlabel("")
            ax.set_ylabel(self._metric_label(metric))
            ax.grid(True, axis="y", alpha=0.2)

        title = self.cfg.title
        if not title:
            title = (
                f"{self._metric_label(metric)} for first "
                f"{int(self.cfg.first_n_rewards)} rewards by SLI group"
            )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[{self.log_tag}] wrote plot: {path}")

    def run(self) -> pd.DataFrame:
        df, groups = self._comparison_df()
        self._write_csv(df)
        self._write_plot(df, groups)
        print(
            f"[{self.log_tag}] exported comparison rows: {len(df)} "
            f"({self.cfg.bottom_label} vs {self.cfg.top_label})"
        )
        return df
