import csv

import pytest

from scripts.export_slide_circle_delta_graphpad import (
    SECTION, bucket_column, export_deltas, read_percentages,
)


def source(tmp_path, side, rows, name=None):
    path = tmp_path / (name or f'{side}.csv')
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([SECTION])
        # Match learning_stats: repeated short names, six columns per training.
        writer.writerow(['video', 'fly'] + [
            f'training {training} b{bucket}' if bucket == 1 else f'b{bucket}'
            for training in (1, 2, 3) for bucket in range(1, 7)])
        for day, fly, value in rows:
            values = ['99.999'] * 18
            values[10 if side == 'before' else 0] = value
            writer.writerow([f'c31__2025-08-{day:02d}__{12 if side == "before" else 15}-00-00.avi',
                             fly, *values])
        stream.write('\n"% time in slide circle (shifted (2 cm down)), experimental fly only, by sync bucket:"\n')
    return path


def test_repeated_bucket_headers_are_resolved_by_training():
    headers = ['video', 'fly', 'training 1 b1', 'b2', 'b3', 'b4', 'b5', 'b6',
               'training 2 b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    assert bucket_column(headers, 2, 5) == 12
    assert bucket_column(headers, 1, 1) == 2
    with pytest.raises(ValueError, match='exactly one'):
        bucket_column(headers, 3, 1)
    with pytest.raises(ValueError, match='exactly one'):
        bucket_column(headers + ['training 2 b5'], 2, 5)


def test_percentage_point_delta_matches_flies_and_preserves_precision(tmp_path):
    before = source(tmp_path, 'before', [(5, 0, '95.123'), (6, 1, '20.000'),
                                        (7, 0, 'nan'), (8, 0, '10'), (9, 0, '20')])
    after = source(tmp_path, 'after', [(6, 1, '30.001'), (5, 0, '40.456'),
                                      (7, 0, '10'), (9, 0, 'inf'), (10, 0, '30')])
    second_before = source(tmp_path, 'before', [(5, 0, '100')], 'second_before.csv')
    second_after = source(tmp_path, 'after', [(5, 0, '0')], 'second_after.csv')
    out, audit = tmp_path / 'out.csv', tmp_path / 'audit.csv'
    assert export_deltas([('Slide', before, after), ('AR', second_before, second_after)], out, audit) == [
        ('Slide', 2, 4), ('AR', 1, 0)]
    with out.open() as stream:
        assert list(csv.reader(stream)) == [['Slide', 'AR'], ['-54.667', '-100'], ['10.001', '']]
    with audit.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['before_t2_sb5_pct'] == '95.123'
    assert rows[0]['after_local_t1_sb1_pct'] == '40.456'
    assert rows[2]['exclusion_reason'] == 'nonfinite_before_percentage'
    assert rows[3]['exclusion_reason'] == 'missing_after_identity'
    assert rows[4]['exclusion_reason'] == 'nonfinite_after_percentage'
    assert rows[5]['exclusion_reason'] == 'missing_before_identity'


@pytest.mark.parametrize('value', ['100.001', '-0.001'])
def test_invalid_percentage_fails(tmp_path, value):
    path = source(tmp_path, 'before', [(5, 0, value)])
    with pytest.raises(ValueError, match='outside'):
        read_percentages(path, 'before')


def test_ambiguous_identity_fails(tmp_path):
    path = source(tmp_path, 'before', [(5, 0, '20'), (5, 0, '30')])
    with pytest.raises(ValueError, match='Ambiguous before'):
        read_percentages(path, 'before')


def test_all_excluded_writes_audit_and_no_graphpad(tmp_path):
    before = source(tmp_path, 'before', [(5, 0, 'nan')])
    after = source(tmp_path, 'after', [(5, 0, '10')])
    with pytest.raises(ValueError, match='No finite matched'):
        export_deltas([('Slide', before, after)], tmp_path / 'out.csv', tmp_path / 'audit.csv')
    assert (tmp_path / 'audit.csv').is_file()
    assert not (tmp_path / 'out.csv').exists()


def test_wrong_section_and_incomplete_rows_fail(tmp_path):
    path = source(tmp_path, 'before', [(5, 0, '10')])
    original = path.read_text()
    path.write_text(original.replace('concentric', 'other'))
    with pytest.raises(ValueError, match='section'):
        read_percentages(path, 'before')
    path.write_text(original.replace(',10,', ','))
    with pytest.raises(ValueError, match='length mismatch'):
        read_percentages(path, 'before')
