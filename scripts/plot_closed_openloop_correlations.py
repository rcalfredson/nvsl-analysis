#!/usr/bin/env python3
"""Join closed-loop SLI and open-loop PI exports, then plot correlations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.exporting.cross_experiment_correlation import (  # noqa: E402
    AUDIT_FIELDS,
    CLOSED_FIELDS,
    OPEN_FIELDS,
    join_closed_to_experimental_open_loop,
    read_rows_csv,
    write_rows_csv,
)
from src.plotting.cross_fly_correlations import (  # noqa: E402
    CorrelationPlotConfig,
    plot_correlation_scatter,
)
from src.plotting.palettes import correlation_plot_color  # noqa: E402
from src.plotting.plot_customizer import PlotCustomizer  # noqa: E402


def _axis_limits(value: str | None, option: str) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = str(value).split(",")
    if len(parts) != 2:
        raise ValueError(f"{option} must be MIN,MAX")
    lo, hi = (float(item) for item in parts)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"{option} must have finite MIN < MAX")
    return lo, hi


def _write_stats(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ("metric", "x_label", "y_label", "r", "p", "n")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _debug_value(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}" if np.isfinite(number) else "nan"


def _print_match_debug(audit: list[dict[str, object]]) -> None:
    print("\n=== closed/open-loop match audit ===")
    for row in audit:
        status = str(row.get("status", ""))
        if status == "matched":
            print(
                f"MATCH {row.get('closed_exp_subject_key', '')}\n"
                f"  closed: {Path(str(row.get('closed_video', ''))).name}, "
                f"exp={row.get('exp_fly_id', '')}, "
                f"yoked={row.get('yoked_fly_id', '')}\n"
                f"  open:   {Path(str(row.get('open_loop_video', ''))).name}, "
                f"fly={row.get('open_loop_fly_id', '')}\n"
                f"  SLI={_debug_value(row.get('sli_t2_sb5'))}, "
                f"PI-off={_debug_value(row.get('preference_pi_led_off'))}, "
                f"PI-on={_debug_value(row.get('preference_pi_led_on'))}"
            )
        elif status == "unmatched_closed":
            print(
                f"UNMATCHED {row.get('closed_exp_subject_key', '')}\n"
                f"  closed: {Path(str(row.get('closed_video', ''))).name}, "
                f"exp={row.get('exp_fly_id', '')}, "
                f"yoked={row.get('yoked_fly_id', '')}"
            )


def _require_expected_count(
    *, actual: int, expected: int | None, label: str, audit_path: Path
) -> None:
    if expected is None or actual == expected:
        return
    raise ValueError(
        f"expected {expected} {label}, found {actual}; see {audit_path}"
    )


def run(args: argparse.Namespace) -> None:
    closed = read_rows_csv(args.closed, CLOSED_FIELDS)
    opened = read_rows_csv(args.open_loop, OPEN_FIELDS)
    matched, audit = join_closed_to_experimental_open_loop(closed, opened)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = (
        Path(args.audit_csv) if args.audit_csv else out_dir / "matching_audit.csv"
    )
    write_rows_csv(audit, audit_path, AUDIT_FIELDS)
    print(
        f"[closed-openloop] wrote {audit_path} "
        f"({len(matched)} matched pairs; {len(audit) - len(matched)} audit-only rows)"
    )
    if args.debug_matches:
        _print_match_debug(audit)

    _require_expected_count(
        actual=len(closed),
        expected=args.expect_closed_rows,
        label="closed-loop export row(s)",
        audit_path=audit_path,
    )
    _require_expected_count(
        actual=len(opened),
        expected=args.expect_open_loop_rows,
        label="open-loop export row(s)",
        audit_path=audit_path,
    )
    _require_expected_count(
        actual=len(matched),
        expected=args.expect_matched_rows,
        label="matched experimental-fly row(s)",
        audit_path=audit_path,
    )
    _require_expected_count(
        actual=sum(
            row.get("status") == "available_yoked_open_loop" for row in audit
        ),
        expected=args.expect_yoked_open_loop_rows,
        label="retained former-yoked open-loop row(s)",
        audit_path=audit_path,
    )

    invalid_matches = [
        row
        for row in audit
        if row.get("status") == "matched" and not row.get("match_validated")
    ]
    if invalid_matches:
        raise ValueError(
            f"{len(invalid_matches)} matched row(s) failed independent identity validation; "
            f"see {audit_path}"
        )
    unmatched_closed = [
        row for row in audit if row.get("status") == "unmatched_closed"
    ]
    if unmatched_closed and not args.allow_unmatched:
        raise ValueError(
            f"{len(unmatched_closed)} closed-loop pair(s) have no experimental-fly "
            f"open-loop match; see {audit_path}. Review the mapping, or rerun with "
            "--allow-unmatched to plot the matched subset explicitly."
        )
    if not matched:
        raise ValueError("no closed-loop pairs matched experimental open-loop flies")

    x = np.asarray([float(row["sli_t2_sb5"]) for row in matched], dtype=float)
    customizer = PlotCustomizer()
    if args.font_size is not None:
        customizer.update_font_size(args.font_size)
    customizer.update_font_family(args.font_family)
    cfg_base = dict(
        out_dir=out_dir,
        image_format=args.image_format,
        xlim=_axis_limits(args.xlim, "--xlim"),
        ylim=_axis_limits(args.ylim, "--ylim"),
        export_npz_dir=out_dir / "npz",
        export_group_label=args.group_label,
    )

    plot_specs = (
        (
            "led_off",
            "preference_pi_led_off",
            "Final SLI and open-loop LED-off positional PI",
            "Open-loop positional PI,\nLED off (exp)",
        ),
        (
            "led_on",
            "preference_pi_led_on",
            "Final SLI and open-loop LED-on positional PI",
            "Open-loop positional PI,\nLED on (exp)",
        ),
    )
    stats_rows: list[dict[str, object]] = []
    x_label = "Final SLI at T2 SB5 (exp - yoked)"
    for plot_key, column, title, y_label in plot_specs:
        y = np.asarray([float(row[column]) for row in matched], dtype=float)
        filename = f"corr_final_sli_vs_openloop_pi_{plot_key}"
        cfg = CorrelationPlotConfig(
            **cfg_base,
            dot_color=correlation_plot_color(filename),
        )
        summary = plot_correlation_scatter(
            x=x,
            y=y,
            title=title,
            x_label=x_label,
            y_label=y_label,
            cfg=cfg,
            filename=filename,
            customizer=customizer,
        )
        stats_rows.append(
            {
                "metric": plot_key,
                "x_label": x_label,
                "y_label": y_label,
                "r": summary.r,
                "p": summary.p,
                "n": summary.n,
            }
        )

    stats_path = out_dir / "correlation_stats.csv"
    _write_stats(stats_path, stats_rows)
    print(f"[closed-openloop] wrote {stats_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match closed-loop T2/SB5 SLI pairs to the same experimental flies' "
            "open-loop LED-on/off positional PI values."
        )
    )
    parser.add_argument("--closed", required=True, help="Closed-loop SLI CSV export.")
    parser.add_argument(
        "--open-loop", required=True, help="Open-loop positional-PI CSV export."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--audit-csv", default=None)
    parser.add_argument(
        "--debug-matches",
        action="store_true",
        help="Print each matched or unmatched closed-loop pair and its source records.",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help=(
            "Allow plotting the matched subset when closed-loop pairs are unmatched. "
            "By default, any unmatched closed-loop pair stops plotting after the audit "
            "CSV is written."
        ),
    )
    parser.add_argument("--expect-closed-rows", type=int, default=None)
    parser.add_argument("--expect-open-loop-rows", type=int, default=None)
    parser.add_argument("--expect-matched-rows", type=int, default=None)
    parser.add_argument("--expect-yoked-open-loop-rows", type=int, default=None)
    parser.add_argument("--group-label", default=None)
    parser.add_argument(
        "--image-format",
        "--imgFormat",
        dest="image_format",
        default="png",
    )
    parser.add_argument("--xlim", default=None, metavar="MIN,MAX")
    parser.add_argument("--ylim", default=None, metavar="MIN,MAX")
    parser.add_argument("--fontFamily", dest="font_family", default=None)
    parser.add_argument("--fs", dest="font_size", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
