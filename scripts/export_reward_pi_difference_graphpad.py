"""Export paired T2 last-SB calculated reward PI differences for GraphPad."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from itertools import zip_longest
from pathlib import Path

SECTION = '__calculated__ reward PI by bucket:'
EXP = 'training 2 last SB (exp)'
YOK = 'training 2 last SB (yok)'


def export_differences(groups, out, audit, *, keyed_sli_csv=False):
    """Export ordered groups from learning_stats or keyed closed-loop SLI CSVs."""
    labels = [label for label, _ in groups]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Provide at least one group with unique labels')
    columns, audit_rows, counts = [], [], []
    for label, path in groups:
        lines = Path(path).read_text().splitlines()
        if keyed_sli_csv:
            table = lines
            video_col, fly_col = 'closed_video', 'exp_fly_id'
            exp_col, yok_col = 'exp_reward_pi_t2_sb5', 'yoked_reward_pi_t2_sb5'
        else:
            starts = [i + 1 for i, line in enumerate(lines) if line.strip().strip('"') == SECTION]
            if len(starts) != 1:
                raise ValueError(f'Expected exactly one {SECTION!r} section in {path}')
            table = []
            for line in lines[starts[0]:]:
                if not line.strip():
                    break
                table.append(line)
            video_col, fly_col, exp_col, yok_col = 'video', 'fly', EXP, YOK
        reader = csv.DictReader(table)
        required = {video_col, fly_col, exp_col, yok_col}
        if keyed_sli_csv:
            required.update({'yoked_fly_id', 'sli_training', 'sli_sync_bucket'})
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing required columns in {path}: {required - set(reader.fieldnames or [])}')
        values, seen = [], set()
        total = 0
        for row in reader:
            total += 1
            if keyed_sli_csv and (int(row['sli_training']) != 2 or int(row['sli_sync_bucket']) != 5):
                raise ValueError(f'Expected T2 SB5 in {path}')
            key = row[video_col], row[fly_col]
            if key in seen:
                raise ValueError(f'Duplicate video/fly in {path}: {key}')
            seen.add(key)
            exp_raw, yok_raw = row[exp_col].strip(), row[yok_col].strip()
            exp, yok = Decimal(exp_raw or 'NaN'), Decimal(yok_raw or 'NaN')
            included = exp.is_finite() and yok.is_finite()
            for value in (exp, yok):
                if value.is_finite() and not Decimal('-1') <= value <= Decimal('1'):
                    raise ValueError(f'PI outside [-1, 1] in {path}: {key}')
            difference = str(exp - yok) if included else ''
            if included:
                values.append(difference)
            fly_ids = [row['yoked_fly_id']] if keyed_sli_csv else []
            audit_rows.append([label, str(path), *key, *fly_ids, exp_raw, yok_raw, difference, included,
                               '' if included else 'nonfinite experimental or yoked PI'])
        if not values:
            raise ValueError(f'No finite paired values for {label!r}')
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
        fly_headers = ['yoked_fly_id'] if keyed_sli_csv else []
        writer.writerow(['group', 'source', 'video', 'fly', *fly_headers, EXP, YOK, 'exp_minus_yoked',
                         'included', 'exclusion_reason'])
        writer.writerows(audit_rows)
    for label, included, excluded in counts:
        print(f'{label}: {included} included, {excluded} nonfinite pairs excluded')
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', action='append', required=True, metavar='LABEL=CSV')
    parser.add_argument('--out', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--keyed-sli-csv', action='store_true',
                        help='Read T2 SB5 paired PIs from keyed closed-loop SLI CSVs; no mean-window filter.')
    args = parser.parse_args()
    groups = []
    for item in args.group:
        label, separator, path = item.partition('=')
        if not separator or not label or not path:
            parser.error('--group must be LABEL=CSV')
        groups.append((label, path))
    export_differences(groups, args.out, args.audit, keyed_sli_csv=args.keyed_sli_csv)
