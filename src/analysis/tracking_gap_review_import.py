"""Read the user's ODS observation table as data and match immutable review IDs."""
from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
NS = {"t": TABLE, "x": TEXT}


def paragraph_text(element):
    result = element.text or ""
    for child in element:
        if child.tag == f"{{{TEXT}}}s":
            result += " " * int(child.get(f"{{{TEXT}}}c", "1"))
        elif child.tag == f"{{{TEXT}}}tab":
            result += "\t"
        elif child.tag == f"{{{TEXT}}}line-break":
            result += "\n"
        else:
            result += paragraph_text(child)
        result += child.tail or ""
    return result


def read_ods_observations(path, sheet=None):
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("content.xml"))
    tables = root.findall(".//t:table", NS)
    if sheet is not None:
        tables = [t for t in tables if t.get(f"{{{TABLE}}}name") == sheet]
    if not tables:
        raise ValueError("Requested worksheet was not found")
    table = tables[0]
    rows = []
    for source in table.findall("t:table-row", NS):
        cells = []
        for cell in source:
            if cell.tag not in (f"{{{TABLE}}}table-cell", f"{{{TABLE}}}covered-table-cell"):
                continue
            text = "\n".join(paragraph_text(p) for p in cell.findall("x:p", NS))
            repeat = int(cell.get(f"{{{TABLE}}}number-columns-repeated", "1"))
            cells.extend([text.replace("\xa0", " ").strip()] * min(repeat, max(0, 9-len(cells))))
        if not cells or not cells[0].isdigit():
            continue
        if len(cells) != 9 or int(source.get(f"{{{TABLE}}}number-rows-repeated", "1")) != 1:
            raise ValueError("Expected nine observation columns and unique numbered rows")
        rows.append(dict(case=int(cells[0]), video_filename=cells[1], fly=cells[2], role=cells[3],
                         circle=cells[4], scored_frames=cells[5], interval_code=cells[6],
                         observed_entry_text=cells[7], notes=cells[8]))
    if not rows or len({r["case"] for r in rows}) != len(rows):
        raise ValueError("No observation rows, or duplicate case numbers")
    return rows


def parse_observed_frames(text):
    value = text.strip()
    if value.lower() in ("none", "[]"):
        return []
    if not value or value.lower() in ("n/a", "unknown", "unscorable"):
        return None
    inner = value[1:-1] if value.startswith("[") and value.endswith("]") else value
    pieces = re.split(r"[,;\s]+", inner)
    if not pieces or any(not re.fullmatch(r"\d+", p) for p in pieces):
        raise ValueError(f"Unrecognized observed-entry notation: {text!r}")
    frames = [int(p) for p in pieces]
    if len(set(frames)) != len(frames):
        raise ValueError("Duplicate observed entry frames")
    return frames


def match_observations(rows, episodes):
    """Match an exact selected set of cases without guessing missing observations.

    Returns rows carrying stable review IDs and annotation records. ``none`` is
    an observed zero; blank/unknown observations are unscorable. Confidence and
    reviewer fields remain blank because the nine-column ODS supplies neither.
    """
    by_case = {int(e.get("case_number") or i): e for i, e in enumerate(episodes, 1)}
    if (
        len(by_case) != len(episodes)
        or len({r["case"] for r in rows}) != len(rows)
        or {r["case"] for r in rows} != set(by_case)
    ):
        raise ValueError("Spreadsheet must cover the bundle's case numbers exactly once")
    matched, annotations = [], []
    for row in sorted(rows, key=lambda r: r["case"]):
        e = by_case[row["case"]]
        interval = f"[{e['review_start_frame']}, {e['review_stop_frame_exclusive']})"
        if (row["video_filename"], row["fly"], row["role"], row["circle"], row["scored_frames"]) != (
            Path(e["video_path"]).name, str(e["fly"]), e["role"], e["circle"], interval
        ):
            raise ValueError(f"Spreadsheet identity or interval mismatch at case {row['case']}")
        expected_code = "T" if e["source"] == "observed_control" else "G"
        if e.get("interval_type") == "gap_in_context_only":
            expected_code = "C"
        if row["interval_code"] != expected_code:
            raise ValueError(f"Interval source-code mismatch at case {row['case']}")
        frames = parse_observed_frames(row["observed_entry_text"])
        start = int(e["review_start_frame"])
        stop = int(e["review_stop_frame_exclusive"])
        if frames is not None and any(not start <= f < stop for f in frames):
            raise ValueError(f"Observed entry outside scored interval at case {row['case']}")
        annotations.append(dict(
            review_id=e["review_id"],
            review_status="reviewed" if frames is not None else "unscorable",
            observed_entry_frames=json.dumps(frames) if frames is not None else "",
            confidence="", reviewer="", notes=row["notes"],
        ))
        matched.append(dict(**row, review_id=e["review_id"]))
    return matched, annotations


def select_observations(rows, episodes, *, case_start=None, case_end=None, allow_partial=False):
    """Select cumulative-sheet rows explicitly; report omissions, never fill them.

    Bundle case numbers must be globally unique, including before range filtering.
    Missing selected cases fail unless allow_partial is explicitly requested.
    Identity/frame validation is performed separately by match_observations.
    """
    if (case_start is not None and case_start < 1) or (case_end is not None and case_end < 1):
        raise ValueError("Case bounds must be positive")
    if case_start is not None and case_end is not None and case_start > case_end:
        raise ValueError("Case start must not exceed case end")
    by_case = {int(e["case_number"]): e for e in episodes}
    if len(by_case) != len(episodes):
        raise ValueError("Bundles contain duplicate global case numbers")
    if len({r["case"] for r in rows}) != len(rows):
        raise ValueError("Spreadsheet contains duplicate case numbers")
    requested = {
        n for n in by_case
        if (case_start is None or n >= case_start)
        and (case_end is None or n <= case_end)
    }
    if not requested:
        raise ValueError("No bundle cases in the requested range")
    available = {r["case"] for r in rows}
    missing = requested - available
    if missing and not allow_partial:
        raise ValueError(
            f"Spreadsheet is missing selected cases {sorted(missing)}; "
            "use --allow-partial to explicitly skip them"
        )
    included = requested & available
    if not included:
        raise ValueError("No spreadsheet observations match the selected cases")
    selection = dict(
        included_cases=sorted(included),
        missing_cases=sorted(missing),
        excluded_spreadsheet_cases=sorted(available - included),
        excluded_bundle_cases=sorted(set(by_case) - requested),
    )
    return (
        [r for r in rows if r["case"] in included],
        [by_case[n] for n in sorted(included)],
        selection,
    )


def reconcile(rows, episodes, comparisons, gaps):
    """Classify count/timing disagreements and flag prose for human review.

    Episode missingness is authoritative for new bundles; candidate ranges provide
    a fallback for legacy bundles. Flags never override the observed-entry column.
    """
    by_id = {e["review_id"]: e for e in episodes}
    manual = {r["review_id"]: r for r in rows}
    ranges = {}
    for gap in gaps:
        key = (gap["video_path"], gap["fly"], gap["role"])
        ranges.setdefault(key, set()).add(
            (int(gap["gap_start_frame"]), int(gap["gap_stop_frame_exclusive"]))
        )
    result = []
    for comparison in comparisons:
        e = by_id[comparison["review_id"]]
        row = manual[e["review_id"]]
        start = int(e["review_start_frame"])
        stop = int(e["review_stop_frame_exclusive"])
        if e.get("scored_missing_frames") not in (None, ""):
            missing = int(e["scored_missing_frames"])
        else:
            key = (e["video_path"], e["fly"], e["role"])
            missing = sum(
                max(0, min(stop, gap_stop) - max(start, gap_start))
                for gap_start, gap_stop in ranges.get(key, ())
            )
        if e["source"] == "observed_control":
            status = "fully_tracked_comparison"
        else:
            status = "gap_in_scored_interval" if missing else "gap_in_context_only"
        notes = row["notes"].lower()
        flags = []
        other = "control" if e["circle"] == "reward" else "reward"
        if f"{other} circle" in notes and f"{e['circle']} circle" not in notes:
            flags.append("notes name the other circle")
        if comparison["observed_count"] == 0 and "enters" in notes:
            flags.append("notes describe an entry but entry list is empty")
        no_interaction_phrases = (
            "no circle interactions during scored frames",
            f"no {e['circle']} circle interactions during scored frames",
        )
        if comparison["observed_count"] and any(p in notes for p in no_interaction_phrases):
            flags.append("notes say no interactions but entry list is nonempty")
        difference = comparison["observed_minus_calculated"]
        if difference is None:
            category = "unscorable"
        elif difference:
            category = "count_disagreement_requires_recheck"
        elif (
            json.loads(comparison["unmatched_calculated_frames"])
            or json.loads(comparison["unmatched_observed_frames"])
        ):
            category = "matching_count_temporal_mismatch"
        elif any(json.loads(comparison["calculated_minus_observed_frame_offsets"])):
            category = "matching_count_small_timing_offset"
        else:
            category = "matching_count_and_time"
        result.append(dict(
            case=row["case"], **comparison, interval_type=status,
            scored_missing_frames=missing, category=category,
            annotation_flags="; ".join(flags), scored_frames=row["scored_frames"],
            observed_entry_text=row["observed_entry_text"],
            calculated_entry_frames=e["calculated_entry_frames"],
        ))
    return sorted(result, key=lambda r: r["case"])
