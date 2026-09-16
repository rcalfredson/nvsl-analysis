#!/usr/bin/env python3
"""Audit representative videos from wiki lists previously validated as protocol-uniform.

Run full-list analyses without --aem first; this script does not independently
establish within-list uniformity. It checks the first matching video in each list.

Usage: python scripts/protocol_audit.py tmp/methods_review_wiki.html
The wiki source must contain its 'video lists referenced' section. HTL lists are
checked with --rmCC 5, and Large lists with --rCC 15, matching the paper audit.
The subprocess is intentionally stopped after reporting T1-T3; this is a protocol
check, not a completed metric analysis. Full analyze.py runs may write their usual
logs before that point.
"""

import glob
from collections import deque
import html
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZE = str(REPO_ROOT / "analyze.py")
EXPECTED_TRAININGS = {1, 2, 3}

def parse_audit_record(line):
    if not line.startswith("PROTOCOL_AUDIT "):
        return None
    record = json.loads(line[len("PROTOCOL_AUDIT "):])
    if (type(record.get("training")) is not int
            or not isinstance(record.get("type"), str)
            or type(record.get("hasSymCtrl")) is not bool):
        raise ValueError("Invalid protocol audit record")
    return record


def strip_html(text):
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).replace("\xa0", " ").strip()


def extract_video_lists(page_source):
    marker = re.search(
        r"<p><em>video lists referenced</em></p>",
        page_source,
        flags=re.IGNORECASE,
    )
    if marker is None:
        raise RuntimeError("Could not find 'video lists referenced' section")

    remainder = page_source[marker.end() :]

    ul_end = remainder.find("</ul>")
    if ul_end < 0:
        raise RuntimeError("Could not find end of video-list section")

    section = remainder[:ul_end]

    entries = []
    for raw_li in re.findall(
        r"<li>(.*?)</li>",
        section,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        line = strip_html(raw_li)

        match = re.match(r"^(.*):\s*(/media/.*)$", line)

        if match is None:
            raise RuntimeError(
                f"Could not separate description from video list:\n{line}"
            )

        description, video_list = match.groups()
        fields = [field.strip() for field in description.split("|")]

        if len(fields) < 6:
            raise RuntimeError(f"Unexpected video-list entry:\n{line}")

        chamber = fields[-2]

        if chamber not in {"HTL", "Large"}:
            raise RuntimeError(f"Unexpected chamber {chamber!r} in:\n{line}")

        patterns = [
            re.sub(r"Yang\s+Chen", "Yang Chen", pattern.strip())
            for pattern in video_list.split(",")
            if pattern.strip()
        ]

        entries.append(
            {
                "label": description.strip(),
                "chamber": chamber,
                "patterns": patterns,
            }
        )

    return entries


def first_video(patterns):
    """
    Return the first AVI matched by the first pattern that yields AVI files.
    """
    for pattern in patterns:
        matches = sorted(
            path for path in glob.glob(pattern) if path.lower().endswith(".avi")
        )
        if matches:
            return matches[0]

    return None


def command_for(video, chamber):
    cmd = [
        sys.executable,
        ANALYZE,
        "-v",
        video,
        "--protocol-audit-report",
    ]

    if chamber == "HTL":
        cmd += ["-f", "0-9", "--rmCC", "5"]
    elif chamber == "Large":
        cmd += ["-f", "0-1", "--rCC", "15"]
    else:
        raise ValueError(chamber)

    return cmd


def audit_entry(entry):
    video = first_video(entry["patterns"])

    if video is None:
        return {
            "status": "NO_VIDEO",
            "video": None,
            "trainings": {},
            "returncode": None,
        }

    cmd = command_for(video, entry["chamber"])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=REPO_ROOT,
    )

    trainings = {}
    diagnostics = deque(maxlen=30)

    try:
        for line in proc.stdout:
            diagnostics.append(line.rstrip())
            record = parse_audit_record(line)
            if record is None:
                continue

            training = record["training"]
            training_type = record["type"]
            has_sym = record["hasSymCtrl"]

            previous = trainings.get(training)
            current = (training_type, has_sym)

            if previous is not None and previous != current:
                proc.terminate()
                proc.wait()

                return {
                    "status": "INCONSISTENT",
                    "video": video,
                    "trainings": trainings,
                    "returncode": proc.returncode,
                }

            trainings[training] = current

            # We have all the information needed for this video list.
            if EXPECTED_TRAININGS.issubset(trainings):
                proc.terminate()
                proc.wait()

                status = (
                    "SYMMETRIC"
                    if any(trainings[n][1] for n in EXPECTED_TRAININGS)
                    else "OK"
                )

                return {
                    "status": status,
                    "video": video,
                    "trainings": trainings,
                    "returncode": proc.returncode,
                }

        # analyze.py exited before all three trainings were observed.
        proc.wait()

    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()

    if not EXPECTED_TRAININGS.issubset(trainings):
        status = "ANALYSIS_FAILED" if proc.returncode != 0 else "INCOMPLETE"
    elif any(trainings[n][1] for n in EXPECTED_TRAININGS):
        status = "SYMMETRIC"
    else:
        status = "OK"

    return {
        "status": status,
        "video": video,
        "trainings": trainings,
        "returncode": proc.returncode,
        "diagnostics": list(diagnostics),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} WIKI_SOURCE.txt")

    page_source = Path(sys.argv[1]).read_text(
        encoding="utf-8",
        errors="replace",
    )

    entries = extract_video_lists(page_source)

    print(f"Found {len(entries)} video lists.\n")

    counts = {}

    for idx, entry in enumerate(entries, start=1):
        print(f"[{idx:02d}/{len(entries):02d}] {entry['label']}")

        result = audit_entry(entry)
        status = result["status"]

        counts[status] = counts.get(status, 0) + 1

        print(f"  chamber: {entry['chamber']}")
        print(f"  video:   {result['video']}")
        print(f"  result:  {status}")
        if status in {"ANALYSIS_FAILED", "INCOMPLETE"}:
            for line in result.get("diagnostics", []):
                print(f"    {line}")

        for training in sorted(result["trainings"]):
            training_type, has_sym = result["trainings"][training]
            print(
                f"    T{training}: " f"type={training_type}, " f"hasSymCtrl={has_sym}"
            )

        print()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(f"Video lists checked: {len(entries)}")
    for status in (
        "OK",
        "SYMMETRIC",
        "INCOMPLETE",
        "INCONSISTENT",
        "ANALYSIS_FAILED",
        "NO_VIDEO",
    ):
        print(f"{status:16s}: {counts.get(status, 0)}")

    success = counts.get("OK", 0) == len(entries) and len(entries) > 0

    if success:
        print(
            "\nPASS: the representative video from every previously validated "
            "protocol-uniform list had T1-T3 checked and "
            "hasSymCtrl=False for all three trainings."
        )
        return 0

    print("\nFAIL/REVIEW: at least one video list requires inspection.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
