#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.agarose_time_summary import (  # noqa: E402
    DEFAULT_POST_COL,
    DEFAULT_PRE_COL,
    DEFAULT_SECTION,
)
from src.exporting.graphpad_csv import (  # noqa: E402
    write_agarose_time_graphpad_csv,
    write_scalar_exports_graphpad_csv,
)
from src.utils.parsers import parse_labeled_path  # noqa: E402


def _parse_panel(value: str) -> int | str:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export GraphPad Prism-friendly column CSV files from existing "
            "nvsl-analysis scalar exports or agarose learning_stats.csv files."
        )
    )
    sub = p.add_subparsers(dest="command", required=True)

    scalar = sub.add_parser(
        "scalar-npz",
        help=(
            "Convert scalar-bar NPZ exports into a wide numeric CSV. Use this for "
            "mean wall contacts per reward interval and mean between-reward "
            "tortuosity per fly."
        ),
    )
    scalar.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=EXPORT.NPZ",
        help="Repeatable scalar export input. LABEL=PATH and LABEL:PATH are accepted.",
    )
    scalar.add_argument("--out", required=True, help="Output GraphPad CSV path.")
    scalar.add_argument(
        "--scalar-panel",
        "--panel",
        dest="scalar_panel",
        type=_parse_panel,
        default=None,
        help=(
            "Optional internal scalar-export panel/training to export. Use a "
            "1-based index such as 1, or an exact panel label. By default, all "
            "scalar panels are exported as separate columns."
        ),
    )

    agarose = sub.add_parser(
        "agarose-time",
        help=(
            "Compute pre-minus-post percent-time-on-agarose deltas from "
            "learning_stats.csv files and write a wide numeric CSV."
        ),
    )
    agarose.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="LABEL=LEARNING_STATS.CSV",
        help="Repeatable group input. LABEL=PATH and LABEL:PATH are accepted.",
    )
    agarose.add_argument("--out", required=True, help="Output GraphPad CSV path.")
    agarose.add_argument(
        "--section",
        default=DEFAULT_SECTION,
        help="Exact learning_stats.csv section title to parse.",
    )
    agarose.add_argument("--pre-col", default=DEFAULT_PRE_COL, help="Pre column.")
    agarose.add_argument("--post-col", default=DEFAULT_POST_COL, help="Post column.")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "scalar-npz":
        from src.plotting.overlay_training_metric_scalar_bars import load_export_npz

        exports = []
        for spec in args.input:
            label, path = parse_labeled_path(spec, separators=("=", ":"))
            exports.append(load_export_npz(label, path))
        write_scalar_exports_graphpad_csv(exports, args.out, panel=args.scalar_panel)
        print(f"[graphpad_csv] wrote {args.out}")
        return 0

    if args.command == "agarose-time":
        groups = [
            parse_labeled_path(spec, separators=("=", ":")) for spec in args.group
        ]
        write_agarose_time_graphpad_csv(
            groups,
            args.out,
            section=args.section,
            pre_col=args.pre_col,
            post_col=args.post_col,
        )
        print(f"[graphpad_csv] wrote {args.out}")
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
