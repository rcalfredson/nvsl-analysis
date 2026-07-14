#!/usr/bin/env python3
"""Inspect or safely edit the group-label metadata in an NPZ data bundle."""

from __future__ import annotations

import argparse

from src.exporting.bundle_metadata import edit_group_labels, read_group_labels


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Edit group labels without loading or modifying bundle data arrays."
    )
    parser.add_argument("bundle", help="Input .npz data bundle")
    parser.add_argument(
        "--output", help="Write a new bundle instead of atomically updating the input"
    )
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument("--group-label", help="New scalar group_label value")
    labels.add_argument(
        "--group-labels",
        nargs="+",
        metavar="LABEL",
        help="Replacement group_labels, in their existing order",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    before = read_group_labels(args.bundle)
    if args.group_label is None and args.group_labels is None:
        if not before:
            raise SystemExit("Bundle has no editable group-label metadata")
        for field, values in before.items():
            print(f"{field}: {values}")
        return

    field = "group_label" if args.group_label is not None else "group_labels"
    values = [args.group_label] if args.group_label is not None else args.group_labels
    destination = edit_group_labels(
        args.bundle, values, field=field, output=args.output
    )
    after = read_group_labels(destination)[field]
    print(f"[metadata] {field}: {before[field]} -> {after}")
    print(f"[metadata] wrote {destination}")


if __name__ == "__main__":
    main()
