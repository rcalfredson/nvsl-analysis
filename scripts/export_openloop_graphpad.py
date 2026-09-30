"""Extract the experimental LED-on positional PI from learning_stats.csv."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def export_preference(source, out, audit, label="273-Gal4>CsChrimson"):
    lines = Path(source).read_text().splitlines()
    section = "positional preference for LED side:"
    starts = [i + 1 for i, line in enumerate(lines) if line.strip().strip('"') == section]
    if len(starts) != 1:
        raise ValueError(f"Expected exactly one {section!r} section in {source}")
    table = []
    for line in lines[starts[0]:]:
        if not line.strip():
            break
        table.append(line)
    reader = csv.DictReader(table)
    # analyze.py inserts extra spaces in these headers; normalize whitespace.
    headers = {" ".join(h.split()): h for h in reader.fieldnames or []}
    required = {"video", "fly", "exp fly on"}
    if not required.issubset(headers):
        raise ValueError(f"Missing columns: {sorted(required - headers.keys())}")
    rows, values, seen = [], [], set()
    for row in reader:
        video, fly = row[headers['video']], row[headers['fly']]
        key = (video, fly)
        if key in seen:
            raise ValueError(f"Duplicate video/fly: {key}")
        seen.add(key)
        raw = row[headers['exp fly on']].strip()
        value = float(raw) if raw else float('nan')
        included = math.isfinite(value)
        if included and not -1 <= value <= 1:
            raise ValueError(f"PI outside [-1, 1] for {key}: {value}")
        rows.append([video, fly, raw, included, '' if included else 'nonfinite LED-on PI'])
        if included:
            values.append(raw)
    if not rows or not values:
        raise ValueError("No finite experimental LED-on preferences found")
    for path in (out, audit):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([label])
        writer.writerows([value] for value in values)
    with Path(audit).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'fly', 'preference_pi_led_on', 'included', 'exclusion_reason'])
        writer.writerows(rows)
    print(f"LED-on PI: {len(values)} included, {len(rows) - len(values)} nonfinite; {out}; audit: {audit}")
    return len(values), len(rows) - len(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--label', default='273-Gal4>CsChrimson')
    args = parser.parse_args()
    export_preference(args.source, args.out, args.audit, args.label)
