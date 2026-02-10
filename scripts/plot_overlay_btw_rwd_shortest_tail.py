#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

from src.plotting.btw_rwd_shortest_tail_bundle_plotter import (
    plot_btw_rwd_shortest_tail_bundles,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay per-training shortest-tail between-reward bundle exports."
    )
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable: 'GroupLabel=/path/to/export.npz'",
    )
    p.add_argument("--out", required=True, help="Output image path (png/pdf/etc).")
    p.add_argument(
        "--title",
        action="store_true",
        help="Opt-in title (default: no title).",
    )
    p.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable per-training stats annotations (t-test for 2 groups, ANOVA for 3+).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    labels = []
    paths = []
    for item in args.input:
        if "=" not in item:
            raise SystemExit(
                f"Bad --input {item!r}. Expected 'Label=/path/to/file.npz'"
            )
        label, path = item.split("=", 1)
        labels.append(label)
        paths.append(path)

    out_fn = args.out
    os.makedirs(os.path.dirname(out_fn) or ".", exist_ok=True)

    plot_btw_rwd_shortest_tail_bundles(
        paths,
        out_fn,
        labels=labels,
        show_title=bool(args.title),
        stats=(not bool(args.no_stats)),
    )

    print(f"[overlay_shortest_tail] wrote {out_fn}")


if __name__ == "__main__":
    main()
