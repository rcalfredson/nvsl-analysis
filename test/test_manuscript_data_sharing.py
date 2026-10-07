import hashlib
import json
from pathlib import Path

import pytest

from src.exporting.manuscript_data import archive_data_stage, data_panel_copy_paths


def test_archive_keeps_each_cohorts_full_statistics_and_provenance(tmp_path):
    stage = {'command': 'analyze cohort A', 'archive_source': 'source.csv',
             'archive_output': 'exports/a_sli_archive.csv',
             'learning_stats_output': 'exports/a_learning_stats.csv'}
    (tmp_path / 'source.csv').write_text('keyed A\n')
    (tmp_path / 'learning_stats.csv').write_text('full A\n')
    archive_data_stage(stage, tmp_path)
    (tmp_path / 'learning_stats.csv').write_text('full B\n')
    assert (tmp_path / stage['learning_stats_output']).read_text() == 'full A\n'
    meta = json.loads((tmp_path / 'exports/a_sli_archive.provenance.json').read_text())
    assert meta['source_sha256'] == hashlib.sha256(b'keyed A\n').hexdigest()
    assert meta['learning_stats_sha256'] == hashlib.sha256(b'full A\n').hexdigest()
    assert meta['analysis_command'] == stage['command']


def test_copy_includes_sources_stats_audit_and_keeps_graphpad_destination(tmp_path):
    stage = {'archive_output': 'exports/a_sli_archive.csv',
             'learning_stats_output': 'exports/a_learning_stats.csv'}
    recipe = {'panels': {'ED16j': {'output': 'exports/graphpad.csv'}},
              'artifacts': ['exports/a_sli_archive.csv', 'exports/a_learning_stats.csv',
                            'exports/graphpad.csv', 'exports/audit.csv'],
              'analysis_stages': [stage]}
    (tmp_path / 'exports').mkdir()
    (tmp_path / 'exports/a_sli_archive.provenance.json').write_text('{}')
    copies = data_panel_copy_paths(recipe, 'ED16j', tmp_path, tmp_path / 'network')
    assert copies[0] == (tmp_path / 'exports/graphpad.csv', tmp_path / 'network/edfig16/16j.csv')
    assert [p.name for _, p in copies[1:]] == [
        'a_sli_archive.csv', 'a_learning_stats.csv', 'audit.csv', 'a_sli_archive.provenance.json']
    assert all(p.parent == tmp_path / 'network/edfig16/16j_data' for _, p in copies[1:])
    assert len({p for _, p in copies}) == len(copies)


def test_missing_stats_fails_before_creating_false_provenance(tmp_path):
    (tmp_path / 'source.csv').write_text('keyed\n')
    stage = {'command': 'analyze', 'archive_source': 'source.csv', 'archive_output': 'archive.csv',
             'learning_stats_output': 'stats.csv'}
    with pytest.raises(FileNotFoundError):
        archive_data_stage(stage, tmp_path)
    assert not (tmp_path / 'archive.provenance.json').exists()
