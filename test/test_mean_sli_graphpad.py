import csv
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import export_graphpad_csv
from src.exporting.mean_sli_graphpad import COUNT, MINIMUM, write_mean_sli_graphpad_csv
from src.exporting.cross_experiment_correlation import export_closed_loop_sli_csv


def test_paired_bucket_mean_eligibility_and_graphpad_audit(tmp_path):
    # The last bucket is missing for one eligible fly; two other flies have
    # fewer than three paired buckets despite finite values in both roles.
    pi = np.full((4, 2, 2, 6), np.nan)
    pi[:, 0, :, :] = 1  # T1 must not contribute.
    pi[:, 1, :, 0] = 1  # SB1 must not contribute.
    pi[:, 1, :, 5] = 1  # SB6 must not contribute.
    pi[0, 1, 0, 1:5] = [0.8, 0.4, 0.6, np.nan]
    pi[0, 1, 1, 1:5] = [0.1, 0.2, 0.3, 0.9]
    pi[1, 1, 0, 1:5] = [0.8, 0.4, np.nan, np.nan]
    pi[1, 1, 1, 1:5] = [0.1, 0.2, 0.3, 0.4]
    pi[2, 1, 0, 1:5] = [0.8, 0.4, 0.6, np.nan]
    pi[2, 1, 1, 1:5] = [np.nan, 0.2, 0.3, 0.4]
    pi[3, 1, 0, 1:5] = [0.8, 0.4, 0.6, 0.2]
    pi[3, 1, 1, 1:5] = [0.1, 0.2, 0.3, 0.4]
    vas = [SimpleNamespace(fn=f'/2025-07-14/c{i+1}1_test.avi', trxf=(0, 2))
           for i in range(4)]
    first, second = tmp_path / 'first.csv', tmp_path / 'second.csv'
    export_closed_loop_sli_csv(vas, pi, first, min_valid_sync_buckets=3)
    export_closed_loop_sli_csv(vas[:1], pi[:1], second, min_valid_sync_buckets=3)
    out, audit = tmp_path / 'out.csv', tmp_path / 'audit.csv'
    assert write_mean_sli_graphpad_csv([('Ctrl', first), ('PFNd', second)], out, audit) == [
        ('Ctrl', 2, 2), ('PFNd', 1, 0),
    ]
    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['Ctrl', 'PFNd']
    assert float(rows[1][0]) == pytest.approx(0.4)
    assert float(rows[2][0]) == pytest.approx(0.25)
    assert rows[2][1] == ''
    with audit.open() as f:
        rows = list(csv.DictReader(f))
    assert [r['included'] for r in rows] == ['True', 'False', 'False', 'True', 'True']
    assert [int(r[COUNT]) for r in rows] == [3, 2, 2, 4, 3]
    assert rows[0]['exp_fly_id'] == '0'
    assert rows[0]['yoked_fly_id'] == '2'
    assert rows[1]['exclusion_reason'] == 'fewer than 3 valid paired buckets'


@pytest.mark.parametrize('change', ['duplicate', 'minimum', 'training', 'missing'])
def test_rejects_invalid_source(tmp_path, change):
    pi = np.zeros((1, 2, 2, 6))
    source = tmp_path / 'source.csv'
    export_closed_loop_sli_csv(
        [SimpleNamespace(fn='/2025-07-14/c11_test.avi', trxf=(0, 2))],
        pi, source, min_valid_sync_buckets=3,
    )
    with source.open() as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    if change == 'duplicate':
        rows *= 2
    elif change == 'minimum':
        rows[0][MINIMUM] = '1'
    elif change == 'training':
        rows[0]['sli_training'] = '1'
    else:
        fields.remove(COUNT)
        rows[0].pop(COUNT)
    with source.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        write_mean_sli_graphpad_csv([('Ctrl', source)], tmp_path / 'out.csv', tmp_path / 'audit.csv')


def test_mean_sli_cli_preserves_library_outputs(tmp_path, monkeypatch):
    source = tmp_path / 'keyed sli.csv'
    pi = np.zeros((1, 2, 2, 6))
    pi[0, 1, 0, 1:5] = [0.8, 0.4, 0.6, np.nan]
    pi[0, 1, 1, 1:5] = [0.1, 0.2, 0.3, 0.9]
    export_closed_loop_sli_csv(
        [SimpleNamespace(fn='/2025-07-14/c11_test.avi', trxf=(0, 2))],
        pi, source, min_valid_sync_buckets=3,
    )
    expected, expected_audit = tmp_path / 'expected.csv', tmp_path / 'expected_audit.csv'
    groups = [('Ctrl', str(source)), ('PFNd', str(source))]
    write_mean_sli_graphpad_csv(groups, expected, expected_audit)
    out, audit = tmp_path / 'out.csv', tmp_path / 'audit.csv'
    monkeypatch.setattr('sys.argv', [
        'export_graphpad_csv.py', 'mean-sli-csv',
        '--group', f'Ctrl={source}', '--group', f'PFNd:{source}',
        '--out', str(out), '--audit', str(audit),
    ])
    assert export_graphpad_csv.main() == 0
    assert out.read_bytes() == expected.read_bytes()
    assert audit.read_bytes() == expected_audit.read_bytes()
