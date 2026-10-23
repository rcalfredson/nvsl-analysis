"""Render reconciliation results without performing annotation or bundle I/O."""
from __future__ import annotations

from collections import Counter
import html


def render_reconciliation_report(audit, selection):
    """Return Markdown and HTML; missing observations never count as agreements."""
    counts = Counter(r["category"] for r in audit)
    types = Counter(r["interval_type"] for r in audit)
    reviewed = sum(r["review_status"] == "reviewed" for r in audit)
    differences = counts["count_disagreement_requires_recheck"]
    fraction = f"{100 * differences / reviewed:.1f}%" if reviewed else "not available: no reviewed cases"
    trajectories = len({(r["video_path"], r["fly"], r["role"]) for r in audit})
    recheck = [r for r in audit if r["category"] in (
        "count_disagreement_requires_recheck", "matching_count_temporal_mismatch"
    ) or r["annotation_flags"]]
    lines = [
        "# Tracking-gap spreadsheet reconciliation", "",
        f"Validated {len(audit)} cases from {trajectories} fly trajectories by case number, "
        "basename, fly ID, role, target circle, and scored interval.", "",
        f"Reviewed: {reviewed}; unscorable: {len(audit) - reviewed}. "
        "Blank or unknown entry observations are unscorable, not observed zeros.",
        f"Intervals: {types['gap_in_scored_interval']} scored-gap; "
        f"{types['gap_in_context_only']} context-only gap; "
        f"{types['fully_tracked_comparison']} fully tracked comparison.", "",
        f"Literal entry-count disagreements: {differences}/{reviewed} ({fraction}). "
        "This is an annotation/calculated-count disagreement fraction, "
        "**not an established tracking-error rate**.",
        f"Matching counts and times: {counts['matching_count_and_time']}; "
        f"small frame offsets only: {counts['matching_count_small_timing_offset']}; "
        f"matching counts with unmatched events: {counts['matching_count_temporal_mismatch']}.", "",
        "## Selection", "",
    ]
    for key, label in (
        ("included_cases", "Included cases"),
        ("excluded_spreadsheet_cases", "Spreadsheet cases outside this selection"),
        ("excluded_bundle_cases", "Bundle cases outside the requested range"),
        ("missing_cases", "Selected bundle cases absent from spreadsheet (explicitly skipped)"),
    ):
        lines.append(f"{label}: {selection[key]}")
    lines += [
        "", "## Cases requiring reconciliation", "",
        "| Case | Circle | Observed frames | Calculated frames | Notes / flags |",
        "|---|---|---|---|---|",
    ]
    for row in recheck:
        notes = (row["annotation_flags"] + "; " + row["notes"]).strip("; ")
        values = (row["case"], row["circle"], row["observed_entry_text"],
                  row["calculated_entry_frames"], notes)
        lines.append("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ")
                                        for v in values) + " |")
    lines += [
        "", "## Interpretation", "",
        "A stratified diagnostic sample cannot establish a cohort-wide error rate. "
        "Count differences, temporal mismatches, and prose flags are separate. "
        "The source videos have not been independently rescored here.",
        "Prose flags are heuristic prompts for review, not automatic corrections. "
        "Counts come from the observed-entry column. Confidence and reviewer "
        "remain blank because the spreadsheet supplies neither.",
        "Equal entry counts may hide timing changes; check synchronized-bucket "
        "boundaries before inferring PI/SLI impact. Exit delays do not directly "
        "change entry counts and require separate interpretation.", "",
        "## Outputs", "",
        "annotations_imported.csv contains only included cases. The standard "
        "summarizer requires every case of one bundle: use that file with "
        "--allow-missing-confidence only when importing a complete single bundle. "
        "For partial or multiple-bundle imports, use this reconciliation report.",
        "The source ODS and input bundles were not modified. The copied ODS and "
        "provenance.json preserve the source bytes, bundle hashes, and selection.", "",
    ]
    markdown = "\n".join(lines)
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<title>Tracking-gap reconciliation</title>',
        '<style>body{font:15px system-ui;margin:2rem;line-height:1.5}'
        'pre{white-space:pre-wrap;max-width:110ch}table{border-collapse:collapse}'
        'td,th{border:1px solid #ccc;padding:.5rem;text-align:left;vertical-align:top}</style>',
        '<h1>Tracking-gap spreadsheet reconciliation</h1>',
        '<p><a href="report.md">Markdown report</a> · '
        '<a href="reconciliation.csv">All comparisons</a> · '
        '<a href="selection.json">Selection</a></p>',
        '<pre>' + html.escape(markdown) + '</pre>',
        '<table><tr><th>Case</th><th>Circle</th><th>Observed frames</th>'
        '<th>Calculated frames</th><th>Category / flags</th><th>Notes</th></tr>',
    ]
    for row in audit:
        values = (row["case"], row["circle"], row["observed_entry_text"],
                  row["calculated_entry_frames"], row["category"] + "; " + row["annotation_flags"],
                  row["notes"])
        parts.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>'
                                     for v in values) + '</tr>')
    parts.append('</table></html>')
    return markdown, "\n".join(parts)
