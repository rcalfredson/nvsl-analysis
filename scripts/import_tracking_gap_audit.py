#!/usr/bin/env python3
"""Reconcile selected cases from a cumulative ODS with immutable review bundles."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.tracking_gap_review import compare_annotations
from src.analysis.tracking_gap_review_import import (
    match_observations,
    read_ods_observations,
    reconcile,
    select_observations,
)
from src.exporting.tracking_gap_review_report import render_reconciliation_report
from src.utils.tracking_gap_review_io import load_episodes, read_csv, write_csv


def import_audit(args):
    """Validate first, then write a new reconciliation without altering inputs."""
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new/empty output directory; existing review artifacts are preserved")

    # Parse exactly the bytes we later freeze and hash, even if the live ODS changes.
    source_bytes = args.spreadsheet.read_bytes()
    rows = read_ods_observations(BytesIO(source_bytes), args.sheet)
    episodes, gaps, bundle_sources = [], [], []
    for bundle in args.bundle:
        episodes.extend(dict(e, case_number=int(e.get("case_number") or i))
                        for i, e in enumerate(load_episodes(bundle), 1))
        gaps.extend(read_csv(bundle / "candidates.csv"))
        bundle_sources.append(dict(
            path=str(bundle.resolve()),
            episodes_sha256=hashlib.sha256((bundle / "episodes.csv").read_bytes()).hexdigest(),
            candidates_sha256=hashlib.sha256((bundle / "candidates.csv").read_bytes()).hexdigest(),
        ))
    selected_rows, selected_episodes, selection = select_observations(
        rows, episodes, case_start=args.case_start, case_end=args.case_end,
        allow_partial=args.allow_partial,
    )
    matched, annotations = match_observations(selected_rows, selected_episodes)
    comparisons = compare_annotations(selected_episodes, annotations, require_confidence=False)
    audit = reconcile(matched, selected_episodes, comparisons, gaps)
    recheck = [r for r in audit if r["category"] in (
        "count_disagreement_requires_recheck", "matching_count_temporal_mismatch"
    ) or r["annotation_flags"]]
    markdown, html = render_reconciliation_report(audit, selection)

    out.mkdir(parents=True, exist_ok=True)
    (out / "source_audit.ods").write_bytes(source_bytes)
    write_csv(out / "spreadsheet_observations.csv", matched)
    write_csv(out / "annotations_imported.csv", annotations)
    write_csv(out / "reconciliation.csv", audit)
    write_csv(out / "recheck_cases.csv", recheck, fields=list(audit[0]))
    (out / "report.md").write_text(markdown)
    (out / "report.html").write_text(html)
    (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    counts = Counter(r["category"] for r in audit)
    provenance = dict(
        source=str(args.spreadsheet.resolve()),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        sheet=args.sheet, bundles=bundle_sources, selection=selection,
        options=dict(case_start=args.case_start, case_end=args.case_end,
                     allow_partial=args.allow_partial, tolerance_frames=2),
        manual_confidence="not supplied; left blank", counts=dict(counts),
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Reconciled {len(audit)} cases: {counts['count_disagreement_requires_recheck']} "
          f"count discrepancies; {len(recheck)} cases flagged for recheck. "
          f"Excluded {len(selection['excluded_spreadsheet_cases'])} spreadsheet cases; "
          f"missing {len(selection['missing_cases'])} selected cases. Report: {out / 'report.html'}")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spreadsheet", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True,
                        help="Repeat for bundles with distinct global case numbers")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sheet")
    parser.add_argument("--case-start", type=int, help="Inclusive first case number")
    parser.add_argument("--case-end", type=int, help="Inclusive last case number")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Explicitly skip selected bundle cases absent from the spreadsheet")
    args = parser.parse_args()
    import_audit(args)


if __name__ == "__main__":
    main()
