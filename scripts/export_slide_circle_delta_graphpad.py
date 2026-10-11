"""Export Figures 4g/4h: post-slide T1 SB1 minus pre-slide T2 SB5 % time."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from itertools import zip_longest
from pathlib import Path
import re

SECTION = '% time in slide circle (concentric), experimental fly only, by sync bucket:'
SECTIONS = {
    'concentric': SECTION,
    'shifted': '% time in slide circle (shifted (2 cm down)), experimental fly only, by sync bucket:',
}
WINDOWS = {'before': (2, 5), 'after': (1, 1)}


def bucket_column(headers, training, bucket):
    """Resolve abbreviated, repeated bucket headers within their training block."""
    matches, current_training = [], None
    for index, header in enumerate(headers):
        start = re.fullmatch(r'training (\d+) b(\d+)', header.strip())
        continuation = re.fullmatch(r'b(\d+)', header.strip())
        if start:
            current_training, current_bucket = map(int, start.groups())
        elif continuation and current_training is not None:
            current_bucket = int(continuation.group(1))
        else:
            current_training = None
            continue
        if (current_training, current_bucket) == (training, bucket):
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one training {training} b{bucket} column')
    return matches[0]


def read_percentages(path, side, *, circle='concentric'):
    section = SECTIONS[circle]
    lines = Path(path).read_text().splitlines()
    starts = [i + 1 for i, line in enumerate(lines)
              if line.strip().strip('"') == section]
    if len(starts) != 1:
        raise ValueError(f'Expected exactly one {section!r} section in {path}')
    table = []
    for line in lines[starts[0]:]:
        if not line.strip():
            break
        table.append(line)
    reader = csv.reader(table)
    headers = next(reader, [])
    if headers.count('video') != 1 or headers.count('fly') != 1:
        raise ValueError(f'Missing or duplicate video/fly columns in {path}')
    column = bucket_column(headers, *WINDOWS[side])
    records = {}
    for row in reader:
        if len(row) != len(headers):
            raise ValueError(f'Row/header length mismatch in {path}')
        video, fly = row[headers.index('video')], row[headers.index('fly')]
        name = Path(video).name
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', name)
        camera = re.match(r'(c\d+)_', name)
        if len(dates) != 1 or camera is None:
            raise ValueError(f'Cannot identify recording date/camera in {path}: {name}')
        key = dates[0], camera.group(1), str(int(fly))
        if key in records:
            raise ValueError(f'Ambiguous {side} date/camera/fly match in {path}: {key}')
        raw = row[column].strip()
        value = Decimal(raw or 'NaN')
        if value.is_finite() and not 0 <= value <= 100:
            raise ValueError(f'Percentage outside [0, 100] in {path}: {key}')
        records[key] = {'video': video, 'raw': raw,
                        'value': value if value.is_finite() else None}
    return records


def export_deltas(groups, out, audit, *, circle='concentric'):
    """Export independent group columns and audit every matched/unmatched identity."""
    labels = [label for label, _, _ in groups]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Provide at least one group with unique labels')
    columns, audit_rows, counts = [], [], []
    for label, before_path, after_path in groups:
        before = read_percentages(before_path, 'before', circle=circle)
        after = read_percentages(after_path, 'after', circle=circle)
        keys = list(before) + [key for key in after if key not in before]
        values = []
        for key in keys:
            left, right = before.get(key), after.get(key)
            reasons = []
            for side, record in (('before', left), ('after', right)):
                if record is None:
                    reasons.append(f'missing_{side}_identity')
                elif record['value'] is None:
                    reasons.append(f'nonfinite_{side}_percentage')
            included = not reasons
            delta = str(right['value'] - left['value']) if included else ''
            if included:
                values.append(delta)
            audit_rows.append([
                label, circle, str(before_path), str(after_path), *key,
                left['video'] if left else '', right['video'] if right else '',
                left['raw'] if left else '', right['raw'] if right else '',
                delta, included, ';'.join(reasons),
            ])
        columns.append(values)
        counts.append((label, len(values), len(keys) - len(values)))
    for path in (out, audit):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(audit).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'group', 'circle', 'before_source', 'after_source', 'recording_date', 'camera', 'fly',
            'before_video', 'after_video', 'before_t2_sb5_pct', 'after_local_t1_sb1_pct',
            'delta_percentage_points', 'included', 'exclusion_reason',
        ])
        writer.writerows(audit_rows)
    if any(included == 0 for _, included, _ in counts):
        raise ValueError(f'No finite matched percentages for one or more groups; see {audit}')
    with Path(out).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(labels)
        writer.writerows(zip_longest(*columns, fillvalue=''))
    for label, included, excluded in counts:
        print(f'{label}: {included} included, {excluded} unmatched/nonfinite excluded')
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', nargs=3, action='append', required=True,
                        metavar=('LABEL', 'BEFORE_CSV', 'AFTER_CSV'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--circle', choices=tuple(SECTIONS), default='concentric',
                        help='Concentric reward-centered circle (4g, default) or circle shifted 2 cm down (4h).')
    args = parser.parse_args()
    export_deltas(args.group, args.out, args.audit, circle=args.circle)
