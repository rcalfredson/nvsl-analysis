"""SLI selection for the Figure 4j–4l numeric scalar exports."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.exporting.graphpad_csv import write_scalar_exports_graphpad_csv


ROOT = Path(__file__).resolve().parents[1]


def _bundle(group='Ctrl'):
    series = np.full((4, 2, 5), np.nan)
    series[0, 1, 4] = -0.5  # Final only, negative, stored mean missing.
    series[1, 1, 1:4] = 0.2  # Mean only; final missing.
    series[2, 1, 1:5] = 0.0  # Both, zero SLI also qualifies.
    series[3, 0, :] = 0.9  # Neither: T1 doesn't qualify.
    return dict(
        sli=np.array([np.nan, 0.2, 0.0, np.nan]), sli_ts=series,
        video_ids=np.array([f'/data/video.part{i}.avi::f0' for i in range(4)]),
        group_label=np.array(group), bucket_len_min=np.array(10),
        training_names=np.array(['T1', 'T2']), sli_training_idx=np.array(1),
        sli_use_training_mean=np.array(True),
    )


def _export(group='Ctrl'):
    return SimpleNamespace(
        group=group, panel_labels=['training 2'],
        per_unit_ids_panel=np.array([['video.part2:fly0', 'video.part1:fly0',
                                      '/else/video.part0.avi::f0', 'video.part3:fly0']], dtype=object),
        per_unit_values_panel=np.array([[13.125, 22.25, 31.5, 40.75]], dtype=object),
    )


@pytest.mark.parametrize('mode,rows', [
    ('final', ['13.125', '31.5']), ('mean', ['13.125', '22.25']),
    ('stored', ['13.125', '22.25']),
])
def test_scalar_selection_matches_reordered_ids_and_preserves_measurements(tmp_path, mode, rows):
    out = tmp_path / 'scalar.csv'
    write_scalar_exports_graphpad_csv(
        [_export()], out, sli_eligibility_bundles=[('Ctrl', _bundle())],
        sli_eligibility_mode=mode,
    )
    assert out.read_text().splitlines() == ['Ctrl', *rows]


def test_scalar_selection_is_cohort_local_and_keeps_numeric_layout(tmp_path):
    pfnd = _bundle('PFNd>Kir')
    pfnd['sli_ts'][:, 1, 4] = np.nan
    pfnd['sli_ts'][1, 1, 4] = 0.2
    out = tmp_path / 'scalar.csv'
    write_scalar_exports_graphpad_csv(
        [_export(), _export('PFNd>Kir')], out,
        sli_eligibility_bundles=[('PFNd>Kir', pfnd), ('Ctrl', _bundle())],
        sli_eligibility_mode='final',
    )
    assert out.read_text().splitlines() == ['Ctrl,PFNd>Kir', '13.125,22.25', '31.5,']


def test_scalar_unfiltered_and_selected_panel_keep_existing_behavior(tmp_path):
    export = _export()
    export.panel_labels.append('training 1')
    export.per_unit_ids_panel = [export.per_unit_ids_panel[0], ['video.part0:fly0']]
    export.per_unit_values_panel = [export.per_unit_values_panel[0], [99.5]]
    out = tmp_path / 'out.csv'
    write_scalar_exports_graphpad_csv([export], out, panel='training 2')
    assert out.read_text().splitlines() == ['Ctrl', '13.125', '22.25', '31.5', '40.75']
    write_scalar_exports_graphpad_csv(
        [export], out, panel=2, sli_eligibility_bundles=[('Ctrl', _bundle())],
        sli_eligibility_mode='final',
    )
    assert out.read_text().splitlines() == ['Ctrl', '99.5']


@pytest.mark.parametrize('problem,message', [
    ('missing', 'subjects missing'), ('source_duplicate', 'duplicate subject IDs'),
    ('scalar_duplicate', 'duplicate subject IDs'), ('groups', 'groups must match'),
    ('bundle_label', 'mismatched bundle cohort label'),
    ('duplicate_group', 'duplicate SLI eligibility group'),
    ('counts', 'inconsistent ID/SLI counts'), ('value_counts', 'different ID and value counts'),
])
def test_scalar_rejects_missing_or_ambiguous_cohort_matches(tmp_path, problem, message):
    export, bundle = _export(), _bundle()
    sources = [('Ctrl', bundle)]
    if problem == 'missing':
        bundle['video_ids'][0] = '/data/unknown.avi::f0'
    elif problem == 'source_duplicate':
        bundle['video_ids'][1] = '/else/video.part0.avi::f0'
    elif problem == 'scalar_duplicate':
        export.per_unit_ids_panel[0, 1] = 'video.part0:fly0'
    elif problem == 'groups':
        sources = [('Other', _bundle('Other'))]
    elif problem == 'bundle_label':
        bundle['group_label'] = np.array('Other')
    elif problem == 'duplicate_group':
        sources *= 2
    elif problem == 'counts':
        bundle['video_ids'] = bundle['video_ids'][:3]
    elif problem == 'value_counts':
        export.per_unit_values_panel = [[1.0]]
    out = tmp_path / 'out.csv'
    with pytest.raises(ValueError, match=message):
        write_scalar_exports_graphpad_csv(
            [export], out, sli_eligibility_bundles=sources, sli_eligibility_mode='final',
        )
    assert not out.exists()


def test_scalar_rejects_unused_options_and_final_bucket_override(tmp_path):
    with pytest.raises(ValueError, match='require eligibility bundles'):
        write_scalar_exports_graphpad_csv([_export()], tmp_path / 'out.csv',
                                          sli_eligibility_mode='final')
    with pytest.raises(ValueError, match='minimum bucket count'):
        write_scalar_exports_graphpad_csv(
            [_export()], tmp_path / 'out.csv', sli_eligibility_bundles=[('Ctrl', _bundle())],
            sli_eligibility_mode='final', sli_min_valid_buckets=3,
        )


def test_scalar_cli_external_final_eligibility(tmp_path):
    bundle_path, scalar_path, out = (tmp_path / name for name in ['sli.npz', 'scalar.npz', 'out.csv'])
    np.savez(bundle_path, **_bundle())
    export = _export()
    np.savez(scalar_path, panel_labels=export.panel_labels,
             per_unit_ids_panel=export.per_unit_ids_panel,
             per_unit_values_panel=export.per_unit_values_panel,
             mean=[1.0], ci_lo=[np.nan], ci_hi=[np.nan], n_units_panel=[4], meta_json='{}')
    base = [sys.executable, 'scripts/export_graphpad_csv.py', 'scalar-npz',
            '--input', f'Ctrl={scalar_path}', '--out', str(out)]
    subprocess.run([*base, '--sli-eligible-only', '--sli-eligibility-mode', 'final',
                    '--sli-eligibility-input', f'Ctrl={bundle_path}'], cwd=ROOT,
                   check=True, capture_output=True, text=True)
    assert out.read_text().splitlines() == ['Ctrl', '13.125', '31.5']
    for args in [['--sli-eligibility-mode', 'final'], ['--sli-eligible-only']]:
        result = subprocess.run([*base, *args], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode != 0
        assert 'requires --sli-eligibility-input' in result.stderr or 'require --sli-eligible-only' in result.stderr


@pytest.mark.parametrize('mode', ['all', 'final', 'mean'])
def test_notebook_scalar_option_preserves_upstream_and_existing_exports(mode):
    notebook = json.loads((ROOT / 'notebooks/graphpad_exports.ipynb').read_text())
    stages = []
    scope = {'run_stage': lambda *args, **kwargs: stages.append((args, kwargs))}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('id', '').startswith('graphpad-exports-figure4-scalars-'):
            source = ''.join(cell['source']).replace(
                'FIGURE_4_SCALAR_ELIGIBLE_MODE = "all"', f'FIGURE_4_SCALAR_ELIGIBLE_MODE = "{mode}"')
            exec(source, scope)
    assert len(stages) == 2 and all(not kwargs['run'] for _, kwargs in stages)
    assert len(scope['figure_4_scalar_upstream_exports']) == 9
    for item in scope['figure_4_scalar_upstream_exports']:
        assert '--min-between-reward-trajectories 5' in item['command']
        assert '--skip-first-sync-buckets 1 --keep-first-sync-buckets 4' in item['command']
        assert '--require-exp-target-sync-bucket' in item['command']
    exports = scope['figure_4_scalar_graphpad_exports']
    assert len(exports) == (3 if mode == 'all' else 6)
    assert all('--sli-eligible-only' not in item['command'] for item in exports[:3])
    for item in exports[3:]:
        assert f'--sli-eligibility-mode {mode}' in item['command']
        assert item['command'].count('--sli-eligibility-input ') == 3
        assert 'ar_ctrlKir' in item['command']
        assert '2026-09-25.npz' in item['command']
        assert '2026-09-28.npz' in item['command']
        suffix = 'sliEligibleT2Sb5' if mode == 'final' else 'sliEligibleT2Sb2-5'
        assert f'{suffix}_2026-09-28.csv' in item['output_path']


@pytest.mark.parametrize('schema', ['va_tag', 'fly_id'])
def test_scalar_native_ids_use_subject_tag_rather_than_trajectory_index(tmp_path, schema):
    bundle, export = _bundle(), _export()
    bundle['video_ids'] = np.array([f'/data/video.part{i}.avi::f1' for i in range(4)])
    export.per_unit_ids_panel = [[
        f'/other/video.part{i}.avi|{schema}=1|trx_idx=0' for i in [2, 1, 0, 3]
    ]]
    out = tmp_path / 'out.csv'
    write_scalar_exports_graphpad_csv(
        [export], out, sli_eligibility_bundles=[('Ctrl', bundle)], sli_eligibility_mode='final',
    )
    assert out.read_text().splitlines() == ['Ctrl', '13.125', '31.5']


def test_scalar_rejects_yoked_trajectory_and_cross_format_duplicates(tmp_path):
    export = _export()
    export.per_unit_ids_panel[0, 0] = 'video.part2.avi|va_tag=0|trx_idx=1'
    with pytest.raises(ValueError, match='experimental trx_idx=0'):
        write_scalar_exports_graphpad_csv(
            [export], tmp_path / 'out.csv', sli_eligibility_bundles=[('Ctrl', _bundle())],
            sli_eligibility_mode='final',
        )
    export.per_unit_ids_panel[0, 0] = 'video.part0.avi|fly_id=0|trx_idx=0'
    with pytest.raises(ValueError, match='duplicate subject IDs after normalization'):
        write_scalar_exports_graphpad_csv(
            [export], tmp_path / 'out.csv', sli_eligibility_bundles=[('Ctrl', _bundle())],
            sli_eligibility_mode='final',
        )


def test_scalar_eligibility_does_not_change_finite_measurement_policy(tmp_path):
    export, out = _export(), tmp_path / 'out.csv'
    export.per_unit_values_panel[0, 0] = np.nan
    write_scalar_exports_graphpad_csv(
        [export], out, sli_eligibility_bundles=[('Ctrl', _bundle())], sli_eligibility_mode='final',
    )
    assert out.read_text().splitlines() == ['Ctrl', '31.5']
    bundle = _bundle()
    bundle['sli_ts'][:, 1, 4] = np.nan
    write_scalar_exports_graphpad_csv(
        [export], out, sli_eligibility_bundles=[('Ctrl', bundle)], sli_eligibility_mode='final',
    )
    assert out.read_text().splitlines() == ['Ctrl']
