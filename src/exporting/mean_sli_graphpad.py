"""Export T2 SB2–5 mean SLI from keyed closed-loop CSVs to GraphPad."""
from __future__ import annotations

import csv
from decimal import Decimal
from itertools import zip_longest
from pathlib import Path

MEAN = 'sli_t2_sb2_sb5_mean'
COUNT = 'sli_t2_sb2_sb5_valid_bucket_count'
MINIMUM = 'sli_min_valid_sync_buckets'
FIELDS = ['closed_video', 'exp_fly_id', 'yoked_fly_id', 'sli_training',
          MEAN, COUNT, MINIMUM, 'exp_reward_pi_t2_sb2_sb5_mean',
          'yoked_reward_pi_t2_sb2_sb5_mean']


def write_mean_sli_graphpad_csv(groups, out, audit):
    """Preserve exported numeric strings; require at least three paired buckets."""
    labels = [label for label, _ in groups]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Provide at least one group with unique labels')
    columns, audit_rows, counts = [], [], []
    for label, path in groups:
        values, seen = [], set()
        total = 0
        with Path(path).open(newline='') as f:
            reader = csv.DictReader(f)
            missing = set(FIELDS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f'Missing required columns in {path}: {missing}')
            for row in reader:
                total += 1
                key = row['closed_video'], row['exp_fly_id']
                if key in seen:
                    raise ValueError(f'Duplicate video/fly in {path}: {key}')
                seen.add(key)
                if int(row['sli_training']) != 2 or int(row[MINIMUM]) != 3:
                    raise ValueError(f'Expected T2 with minimum 3 valid buckets in {path}')
                count = int(row[COUNT])
                if not 0 <= count <= 4:
                    raise ValueError(f'Invalid SB2–5 valid bucket count in {path}: {count}')
                raw = row[MEAN].strip()
                mean = Decimal(raw or 'NaN')
                if mean.is_finite() and not Decimal('-2') <= mean <= Decimal('2'):
                    raise ValueError(f'SLI outside [-2, 2] in {path}: {key}')
                included = count >= 3 and mean.is_finite()
                reason = ('fewer than 3 valid paired buckets' if count < 3 else
                          'nonfinite mean SLI' if not mean.is_finite() else '')
                if included:
                    values.append(raw)
                audit_rows.append([label, str(path), *[row[field] for field in FIELDS],
                                   included, reason])
        if not values:
            raise ValueError(f'No eligible mean SLI values for {label!r}')
        columns.append(values)
        counts.append((label, len(values), total - len(values)))
    for path in (out, audit):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(labels)
        writer.writerows(zip_longest(*columns, fillvalue=''))
    with Path(audit).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'source', *FIELDS, 'included', 'exclusion_reason'])
        writer.writerows(audit_rows)
    for label, included, excluded in counts:
        print(f'{label}: {included} included, {excluded} excluded')
    return counts
