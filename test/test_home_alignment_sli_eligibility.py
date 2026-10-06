import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.exporting.graphpad_csv import write_repeated_measures_scalar_exports_graphpad_csv


def _eligibility():
    series = np.full((4, 2, 5), np.nan)
    series[0, 1, 4] = 0.5  # Final only.
    series[1, 1, 1:4] = 0.2  # Mean only.
    series[2, 1, 1:5] = 0.3  # Both.
    return dict(
        sli=np.array([np.nan, 0.2, 0.3, np.nan]), sli_ts=series,
        video_ids=np.array([f'/data/video{i}.avi::f0' for i in range(4)]),
        group_label=np.array('Ctrl'), bucket_len_min=np.array(10),
        training_names=np.array(['T1', 'T2']), sli_training_idx=np.array(1),
        sli_use_training_mean=np.array(True),
    )


def _scalar(group, ids, values):
    return SimpleNamespace(
        group=group, panel_labels=['T2'],
        per_unit_ids_panel=np.array([ids], dtype=object),
        per_unit_values_panel=np.array([values], dtype=object),
    )


def _exports():
    return [
        _scalar('Ctrl|3/5 mm', ['video2:fly0', 'video1:fly0', 'video0:fly0', 'video3:fly0'],
                [0.3, 0.2, 0.1, 0.4]),
        # Reversed order, missing mean-only fly, and mixed ID formats.
        _scalar('Ctrl|8/10 mm', ['/data/video0.avi::f0', '/data/video2.avi::f0', 'video3:fly0'],
                [0.6, np.nan, 0.8]),
    ]


@pytest.mark.parametrize('mode,rows', [
    ('final', ['video2:fly0,0.3,', 'video0:fly0,0.1,0.6']),
    ('mean', ['video2:fly0,0.3,', 'video1:fly0,0.2,']),
])
def test_home_alignment_selects_by_sli_and_preserves_radius_alignment(tmp_path, mode, rows):
    out = tmp_path / 'home.csv'
    write_repeated_measures_scalar_exports_graphpad_csv(
        _exports(), out, sli_eligibility_bundles=[('Ctrl', _eligibility())],
        sli_eligibility_mode=mode,
    )
    assert out.read_text().splitlines() == [
        'Ctrl | Subject ID,Ctrl | 3/5 mm,Ctrl | 8/10 mm', *rows,
    ]


def test_home_alignment_keeps_independent_cohort_selection(tmp_path):
    bundle = _eligibility()
    bundle['sli_ts'][:, 1, 4] = np.nan
    bundle['sli_ts'][1, 1, 4] = 0.2
    out = tmp_path / 'home.csv'
    exports = [_exports()[0], _scalar('PFNd|3/5 mm', ['video0:fly0', 'video1:fly0'], [0.7, 0.8])]
    write_repeated_measures_scalar_exports_graphpad_csv(
        exports, out, sli_eligibility_bundles=[('Ctrl', _eligibility()), ('PFNd', bundle)],
        sli_eligibility_mode='final',
    )
    assert out.read_text().splitlines()[1:] == [
        'video2:fly0,0.3,video1:fly0,0.8', 'video0:fly0,0.1,,',
    ]


@pytest.mark.parametrize('problem,message', [
    ('missing_subject', 'subjects missing'), ('wrong_group', 'groups must match'),
    ('ambiguous_source', 'duplicate subject IDs after normalization'),
    ('duplicate_group', 'duplicate SLI eligibility group'),
])
def test_home_alignment_rejects_incomplete_or_ambiguous_identity_matches(tmp_path, problem, message):
    bundle = _eligibility()
    sources = [('Ctrl', bundle)]
    if problem == 'missing_subject':
        bundle['video_ids'][0] = '/data/unknown.avi::f0'
    elif problem == 'wrong_group':
        sources = [('Other', bundle)]
    elif problem == 'ambiguous_source':
        bundle['video_ids'][1] = '/else/video0.avi::f0'
    elif problem == 'duplicate_group':
        sources *= 2
    with pytest.raises(ValueError, match=message):
        write_repeated_measures_scalar_exports_graphpad_csv(
            _exports(), tmp_path / 'out.csv', sli_eligibility_bundles=sources,
            sli_eligibility_mode='final',
        )


def test_home_alignment_final_mode_rejects_minimum_bucket_override(tmp_path):
    with pytest.raises(ValueError, match='minimum bucket count'):
        write_repeated_measures_scalar_exports_graphpad_csv(
            _exports(), tmp_path / 'out.csv', sli_eligibility_bundles=[('Ctrl', _eligibility())],
            sli_eligibility_mode='final', sli_min_valid_buckets=3,
        )


def test_home_alignment_matching_preserves_dots_in_video_stems(tmp_path):
    bundle = _eligibility()
    bundle['video_ids'] = np.array(['/data/video.part.avi::f0', *bundle['video_ids'][1:]])
    out = tmp_path / 'out.csv'
    write_repeated_measures_scalar_exports_graphpad_csv(
        [_scalar('Ctrl|3/5 mm', ['video.part:fly0'], [0.6])], out,
        sli_eligibility_bundles=[('Ctrl', bundle)], sli_eligibility_mode='final',
    )
    assert out.read_text().splitlines()[1:] == ['video.part:fly0,0.6']


def test_home_alignment_cli_uses_external_final_sli_bundle(tmp_path):
    source = tmp_path / 'sli.npz'
    np.savez(source, **_eligibility())
    inputs = []
    for i, export in enumerate(_exports()):
        path = tmp_path / f'scalar{i}.npz'
        np.savez(
            path, panel_labels=export.panel_labels,
            per_unit_ids_panel=export.per_unit_ids_panel,
            per_unit_values_panel=export.per_unit_values_panel,
            mean=[np.nan], ci_lo=[np.nan], ci_hi=[np.nan], n_units_panel=[4], meta_json='{}',
        )
        inputs += ['--input', f'{export.group}={path}']
    out = tmp_path / 'out.csv'
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, 'scripts/export_graphpad_csv.py', 'repeated-measures-scalar-npz',
        *inputs, '--sli-eligible-only', '--sli-eligibility-mode', 'final',
        '--sli-eligibility-input', f'Ctrl={source}', '--out', str(out),
    ], cwd=root, check=True, capture_output=True, text=True)
    assert out.read_text().splitlines()[1:] == ['video2:fly0,0.3,', 'video0:fly0,0.1,0.6']
