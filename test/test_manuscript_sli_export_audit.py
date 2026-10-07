import csv
import os
from datetime import datetime

from scripts.audit_manuscript_sli_exports import EASTERN, assess_source, inventory, load_sli_recipes
from pathlib import Path

CUTOFF = datetime(2026, 10, 2, 14, 37, 41, tzinfo=EASTERN)


def test_regenerated_graphpad_remains_stale_when_sources_are_old(tmp_path):
    source = tmp_path / 'source.csv'
    source.write_text('"# command: analyze.py [r1234567]"\n')
    for name in ['graphpad.csv', 'audit.csv']:
        (tmp_path / name).write_text('new conversion\n')
    recipe = {'title': 'Final SLI', 'panels': {'ED12g': {'output': 'graphpad.csv'}},
              'analysis_stages': [{'archive_output': 'source.csv', 'command': 'analyze old cohort'}],
              'export_command': 'convert --group Control=source.csv',
              'artifacts': ['source.csv', 'graphpad.csv', 'audit.csv']}
    rows, panels = inventory([('recipe', recipe)], tmp_path, CUTOFF, lambda r: 'before')
    assert [r['status'] for r in rows] == ['stale', 'stale_dependency', 'stale_dependency']
    assert panels[0]['stale_groups'] == ['Control']


def test_recent_unversioned_and_dirty_sources_are_not_declared_verified(tmp_path):
    p = tmp_path / 'source.csv'
    for header in ['closed_video,exp_fly_id\n', '"# command: analyze.py [r1234567-M]"\n']:
        p.write_text(header)
        os.utime(p, (CUTOFF.timestamp() + 100, CUTOFF.timestamp() + 100))
        assert assess_source(p, CUTOFF, lambda r: 'after')['status'] == 'recent_unverified'
    p.write_text('"# command: analyze.py --reward-pi-sync midline [r1234567]"\n')
    assert assess_source(p, CUTOFF, lambda r: 'after')['status'] == 'stale'


def test_recipe_inventory_is_read_only_and_covers_final_and_companion_exports():
    notebook = Path(__file__).resolve().parents[1] / 'notebooks/manuscript_figure_panels.ipynb'
    recipes = dict(load_sli_recipes(notebook))
    assert len(recipes) == 13
    assert 'figure_4i_data' in recipes and 'figure_4i_mean_data' in recipes
    assert len(recipes['extended_data_figure_16j']['analysis_stages']) == 8
    assert 'figure_1p' not in recipes
