import csv

import pytest

from scripts.export_sli_delta_graphpad import export_deltas, read_statistics
from src.exporting.film_slide import film_slide_video_lists


def source(tmp_path, side, rows, name=None):
    window = 'training 2 last SB' if side == 'before' else 'training 1 first SB'
    path = tmp_path / (name or f'{side}.csv')
    with path.open('w', newline='') as stream:
        stream.write('"# command: analyze.py"\n\n__calculated__ reward PI by bucket:\n')
        writer = csv.writer(stream)
        writer.writerow(['video', 'fly', f'{window} (exp)', f'{window} (yok)'])
        writer.writerows(rows)
        stream.write('\nother section:\n')
    return path


def video(day, side):
    return f'c31__2025-07-{day:02d}__{12 if side == "before" else 15}-00-00.avi'


def test_delta_matches_identities_not_row_order_and_audits_all_exclusions(tmp_path):
    before = source(tmp_path, 'before', [
        [video(19, 'before'), 0, '.535', '.172'],
        [video(20, 'before'), 1, '-.2', '.3'],
        [video(21, 'before'), 0, 'nan', '.2'],
        [video(22, 'before'), 0, '.2', '.1'],
        [video(24, 'before'), 0, '.2', '.1'],
    ])
    after = source(tmp_path, 'after', [
        [video(20, 'after'), 1, '.7', '-.1'],
        [video(19, 'after'), 0, '-.730', '-.857'],
        [video(21, 'after'), 0, '.2', '.1'],
        [video(23, 'after'), 0, '.2', '.1'],
        [video(24, 'after'), 0, '.2', 'inf'],
    ])
    other_before = source(tmp_path, 'before', [[video(19, 'before'), 0, '1', '-1']], 'other_before.csv')
    other_after = source(tmp_path, 'after', [[video(19, 'after'), 0, '-1', '1']], 'other_after.csv')
    out, audit = tmp_path / 'out.csv', tmp_path / 'audit.csv'
    counts = export_deltas([('Slide', before, after), ('AR', other_before, other_after)], out, audit)
    assert counts == [('Slide', 2, 4), ('AR', 1, 0)]
    with out.open() as stream:
        assert list(csv.reader(stream)) == [['Slide', 'AR'], ['-0.236', '-4'], ['1.3', '']]
    with audit.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['before_t2_sb5_sli'] == '0.363'
    assert rows[0]['after_t3_sb1_sli'] == '0.127'
    assert rows[2]['exclusion_reason'] == 'nonfinite_before_experimental_or_yoked_PI'
    assert rows[3]['exclusion_reason'] == 'missing_after_identity'
    assert rows[4]['exclusion_reason'] == 'nonfinite_after_experimental_or_yoked_PI'
    assert rows[5]['exclusion_reason'] == 'missing_before_identity'


def test_ambiguous_physical_fly_identity_fails(tmp_path):
    path = source(tmp_path, 'before', [
        [video(19, 'before'), 0, '.2', '.1'],
        ['c31__2025-07-19__13-00-00.avi', 0, '.3', '.1'],
    ])
    with pytest.raises(ValueError, match='Ambiguous before'):
        read_statistics(path, 'before')


@pytest.mark.parametrize('value', ['1.001', '-1.001'])
def test_out_of_range_component_pi_fails(tmp_path, value):
    path = source(tmp_path, 'before', [[video(19, 'before'), 0, value, '.1']])
    with pytest.raises(ValueError, match='PI outside'):
        read_statistics(path, 'before')


def test_no_finite_matches_still_writes_audit(tmp_path):
    before = source(tmp_path, 'before', [[video(19, 'before'), 0, '.2', '.1']])
    after = source(tmp_path, 'after', [[video(20, 'after'), 0, '.2', '.1']])
    with pytest.raises(ValueError, match='No finite matched'):
        export_deltas([('Slide', before, after)], tmp_path / 'out.csv', tmp_path / 'audit.csv')
    assert (tmp_path / 'audit.csv').is_file()
    assert not (tmp_path / 'out.csv').exists()


def test_video_lists_share_heatmap_cohorts_and_exception():
    for cohort, count in [('mock-slide', 16), ('film-slide', 15), ('antennae-removed', 8)]:
        videos = film_slide_video_lists(cohort)
        assert len(videos['before'].split(',')) == count
        assert videos['after'] == ','.join(
            pattern.rsplit('/', 1)[0] + '/2nd round/' + pattern.rsplit('/', 1)[1]
            for pattern in videos['before'].split(',')
        )
        if cohort == 'film-slide':
            assert '2025-07-31/c31_*' not in videos['before']
