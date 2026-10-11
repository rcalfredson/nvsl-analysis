"""Export paired post-slide T1 SB1 minus pre-slide T2 SB5 SLI for Figure 4f."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from itertools import zip_longest
from pathlib import Path
import re

SECTION = '__calculated__ reward PI by bucket:'
WINDOWS = {'before': 'training 2 last SB', 'after': 'training 1 first SB'}


def read_statistics(path, side):
    """Index by the film-slide helper's date/camera/experimental-fly rule."""
    lines = Path(path).read_text().splitlines()
    starts = [i + 1 for i, line in enumerate(lines)
              if line.strip().strip('"') == SECTION]
    if len(starts) != 1:
        raise ValueError(f'Expected exactly one {SECTION!r} section in {path}')
    table = []
    for line in lines[starts[0]:]:
        if not line.strip():
            break
        table.append(line)
    reader = csv.DictReader(table)
    fields = [f'{WINDOWS[side]} ({fly})' for fly in ('exp', 'yok')]
    required = {'video', 'fly', *fields}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f'Missing required columns in {path}: {required - set(reader.fieldnames or [])}')
    records = {}
    for row in reader:
        # learning_stats stores camera__date__time.avi rather than full paths.
        name = Path(row['video']).name
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', name)
        camera = re.match(r'(c\d+)_', name)
        if len(dates) != 1 or camera is None:
            raise ValueError(f'Cannot identify recording date/camera in {path}: {name}')
        key = dates[0], camera.group(1), str(int(row['fly']))
        if key in records:
            raise ValueError(f'Ambiguous {side} date/camera/fly match in {path}: {key}')
        raw = [row[field].strip() for field in fields]
        values = [Decimal(value or 'NaN') for value in raw]
        if any(value.is_finite() and not -1 <= value <= 1 for value in values):
            raise ValueError(f'PI outside [-1, 1] in {path}: {key}')
        sli = values[0] - values[1] if all(v.is_finite() for v in values) else None
        records[key] = {'video': row['video'], 'raw': raw, 'sli': sli}
    return records


def export_deltas(groups, out, audit):
    """Groups are (label, before CSV, after CSV); all unmatched rows are audited."""
    labels = [label for label, _, _ in groups]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Provide at least one group with unique labels')
    columns, audit_rows, counts = [], [], []
    for label, before_path, after_path in groups:
        before = read_statistics(before_path, 'before')
        after = read_statistics(after_path, 'after')
        values = []
        # Retain pre-slide source order, followed by any post-slide-only flies.
        keys = list(before) + [key for key in after if key not in before]
        for key in keys:
            left, right = before.get(key), after.get(key)
            reasons = []
            for side, record in (('before', left), ('after', right)):
                if record is None:
                    reasons.append(f'missing_{side}_identity')
                elif record['sli'] is None:
                    reasons.append(f'nonfinite_{side}_experimental_or_yoked_PI')
            included = not reasons
            delta = str(right['sli'] - left['sli']) if included else ''
            if included:
                values.append(delta)
            audit_rows.append([
                label, str(before_path), str(after_path), *key,
                left['video'] if left else '', right['video'] if right else '',
                *(left['raw'] if left else ['', '']),
                *(right['raw'] if right else ['', '']),
                str(left['sli']) if left and left['sli'] is not None else '',
                str(right['sli']) if right and right['sli'] is not None else '',
                delta, included, ';'.join(reasons),
            ])
        columns.append(values)
        counts.append((label, len(values), len(keys) - len(values)))
    for path in (out, audit):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(audit).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'group', 'before_source', 'after_source', 'recording_date', 'camera',
            'fly', 'before_video', 'after_video', 'before_t2_sb5_exp_pi',
            'before_t2_sb5_yoked_pi', 'after_local_t1_sb1_exp_pi',
            'after_local_t1_sb1_yoked_pi', 'before_t2_sb5_sli',
            'after_t3_sb1_sli', 'delta_sli', 'included', 'exclusion_reason',
        ])
        writer.writerows(audit_rows)
    if any(included == 0 for _, included, _ in counts):
        raise ValueError(f'No finite matched SLI values for one or more groups; see {audit}')
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
    args = parser.parse_args()
    export_deltas(args.group, args.out, args.audit)
