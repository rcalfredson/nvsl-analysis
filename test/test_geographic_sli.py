import json

import pytest

from src.exporting.geographic_sli import (
    geographic_sli_video_inventory,
    load_geographic_sli_cohorts,
    select_geographic_sli_stages,
)


def _cohort(cohort_id='line_0_flat', **kwargs):
    return {'id': cohort_id, 'line': 0, 'chamber': 'flat',
            'label': 'Line 0 | Flat chamber', 'video_globs': ['recordings/c1_*'], **kwargs}


def test_manifest_preserves_user_order_and_patterns(tmp_path):
    cohorts = [
        _cohort('w1118_flat', line='w1118', label='w1118 | Flat'),
        _cohort('w1118_agarose', line='w1118', chamber='agarose', label='w1118 | Agarose'),
        _cohort(),
        _cohort('line_0_agarose', chamber='agarose', label='Line 0 | Agarose'),
    ]
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(cohorts))
    assert load_geographic_sli_cohorts(path) == cohorts


@pytest.mark.parametrize('change', [
    {'id': '../escape'}, {'line': True}, {'line': -1}, {'line': ''},
    {'line': '0'}, {'line': ' w1118'}, {'line': []}, {'chamber': 'unknown'},
    {'label': ''}, {'video_globs': []}, {'video_globs': ['a,b']},
    {'video_globs': ['a', 'a']},
])
def test_manifest_rejects_invalid_cohorts(tmp_path, change):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps([_cohort(**change)]))
    with pytest.raises(ValueError):
        load_geographic_sli_cohorts(path)


@pytest.mark.parametrize('change', [
    {'id': 'line_0_flat', 'line': 1, 'label': 'different'},
    {'id': 'other', 'line': 1},
    {'id': 'other', 'label': 'different'},
])
def test_manifest_rejects_duplicate_identity_label_or_line_chamber(tmp_path, change):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps([_cohort(), _cohort(**change)]))
    with pytest.raises(ValueError, match='Duplicate cohort'):
        load_geographic_sli_cohorts(path)


def test_manifest_rejects_duplicate_named_line_chamber(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps([
        _cohort('w1118_flat', line='w1118', label='w1118 | Flat'),
        _cohort('other', line='w1118', label='another label'),
    ]))
    with pytest.raises(ValueError, match='Duplicate cohort'):
        load_geographic_sli_cohorts(path)


def test_live_inventory_ignores_tracking_sidecars_and_rejects_overlaps(tmp_path):
    video = tmp_path / 'c1_test.avi'
    video.touch()
    (tmp_path / 'c1_test.trx').touch()
    cohort = _cohort(video_globs=[str(tmp_path / 'c1_*')])
    assert geographic_sli_video_inventory([cohort]) == {'line_0_flat': [str(video)]}
    with pytest.raises(ValueError, match='Overlapping video'):
        geographic_sli_video_inventory([cohort, _cohort('other', video_globs=[str(video)])])
    with pytest.raises(ValueError, match='No AVI videos'):
        geographic_sli_video_inventory([_cohort(video_globs=[str(tmp_path / 'missing*')])])


def test_analysis_selection_keeps_stage_order_and_rejects_unknown_ids():
    stages = [{'cohort_id': name} for name in ['a', 'b', 'c']]
    assert select_geographic_sli_stages(stages) == stages
    assert select_geographic_sli_stages(stages, []) == []
    assert select_geographic_sli_stages(stages, ['c', 'a']) == [stages[0], stages[2]]
    assert len(stages) == 3
    with pytest.raises(ValueError, match='Unknown analysis cohort'):
        select_geographic_sli_stages(stages, ['typo'])
    with pytest.raises(ValueError, match='collection'):
        select_geographic_sli_stages(stages, 'a')
