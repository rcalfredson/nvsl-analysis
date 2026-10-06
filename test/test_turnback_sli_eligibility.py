import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from src.analysis.sli_tools import sli_eligible_indices_from_bundle
from src.exporting.graphpad_csv import write_turnback_ratio_bundles_graphpad_csv
import src.plotting.turnback_excursion_bin_sli_bundle_plotter as plotter


def _bundle():
    timeseries = np.full((4, 2, 6), np.nan)
    timeseries[0, 1, 4] = 0.8  # Final-only: one valid bucket.
    timeseries[1, 1, 1:4] = [0.1, 0.2, 0.3]  # Mean-only: SB5 missing.
    timeseries[2, 1, 1:5] = [0.4, 0.5, 0.6, 0.7]  # Both.
    timeseries[3, 0, :] = 0.9  # T1 and out-of-window buckets don't qualify.
    timeseries[3, 1, [0, 5]] = 0.9
    return {
        'sli': np.asarray([np.nan, 0.2, 0.55, np.nan]),
        'sli_ts': timeseries,
        # Explicit criteria must ignore this different stored window/training.
        'sli_training_idx': np.asarray(0),
        'sli_select_skip_first_sync_buckets': np.asarray(0),
        'sli_select_keep_first_sync_buckets': np.asarray(6),
        'group_label': np.asarray('Ctrl'),
        'video_ids': np.asarray(['final_only', 'mean_only', 'both', 'neither']),
        'turnback_excursion_bin_ratio_exp': np.asarray([[0.1], [0.2], [0.3], [0.4]]),
        'turnback_excursion_bin_ratio_ctrl': np.zeros((4, 1)),
        'turnback_excursion_bin_pair_inner_deltas_mm': np.asarray([3]),
        'turnback_excursion_bin_pair_outer_deltas_mm': np.asarray([5]),
    }


@pytest.mark.parametrize('mode,expected', [('mean', [1, 2]), ('final', [0, 2])])
def test_explicit_eligibility_controls_csv_and_plot(tmp_path, monkeypatch, mode, expected):
    bundle = _bundle()
    np.testing.assert_array_equal(sli_eligible_indices_from_bundle(bundle, mode=mode), expected)
    out = tmp_path / 'out.csv'
    write_turnback_ratio_bundles_graphpad_csv(
        [('Ctrl', bundle)], out, sli_eligible_only=True, sli_eligibility_mode=mode,
    )
    assert out.read_text().splitlines()[1:] == [
        f'{bundle["video_ids"][i]},{bundle["turnback_excursion_bin_ratio_exp"][i, 0]}'
        for i in expected
    ]
    captured = []
    monkeypatch.setattr(plotter, 'load_sli_bundle', lambda _: bundle)
    monkeypatch.setattr(plotter, 'plot_overlays', lambda exports, **kwargs: captured.extend(exports))
    monkeypatch.setattr(plotter, 'writeImage', lambda *args, **kwargs: None)
    plotter.plot_turnback_excursion_bin_sli_bundles(
        ['bundle.npz'], str(tmp_path / 'out.png'), sli_eligible_only=True,
        sli_eligibility_mode=mode,
    )
    np.testing.assert_allclose(
        np.asarray(captured[0].per_unit_values_panel[0], dtype=float),
        bundle['turnback_excursion_bin_ratio_exp'][expected, 0],
    )


def test_invalid_or_unavailable_eligibility_fails_clearly():
    bundle = _bundle()
    with pytest.raises(ValueError, match='minimum bucket count'):
        sli_eligible_indices_from_bundle(bundle, mode='final', min_valid_buckets=1)
    with pytest.raises(ValueError, match='cannot exceed 4'):
        sli_eligible_indices_from_bundle(bundle, mode='mean', min_valid_buckets=5)
    bundle['sli_ts'] = bundle['sli_ts'][:, :, :4]
    for mode in ['mean', 'final']:
        with pytest.raises(ValueError, match='requires T2 SB5'):
            sli_eligible_indices_from_bundle(bundle, mode=mode)
    bundle.pop('sli_ts')
    with pytest.raises(ValueError, match='requires bucket-level sli_ts'):
        sli_eligible_indices_from_bundle(bundle, mode='final')


@pytest.mark.parametrize('mode', ['all', 'mean', 'final'])
def test_matrix_routes_eligibility_only_to_all_flies_comparison(mode):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PRINT_ONLY='1', RUN_DUAL_CIRCLE_TURNBACK_ONLY='1',
               RUN_TURNBACK_HOME_VECTOR_ALIGNMENT_ONLY='0', RUN_FLAT_HTL_TURNBACK_PAIRS='0',
               DUAL_CIRCLE_TURNBACK_REUSE_EXISTING_BUNDLES='0',
               DUAL_CIRCLE_TURNBACK_SLI_ELIGIBILITY=mode,
               INTACT_CTRL='ctrl', INTACT_PFND='pfnd', AR_CTRL='ar', DATE_TAG='TEST')
    env.pop('TURNBACK_COMPARISON_GROUP', None)
    result = subprocess.run(['bash', 'scripts/run_analysis_matrix.sh'], cwd=root,
                            env=env, capture_output=True, text=True, check=True)
    commands = [line for line in result.stdout.splitlines()
                if 'scripts.plot_turnback_excursion_bin_sli_bundles' in line]
    assert len(commands) == 5
    assert 'ar_ctrlKir' in commands[-1]
    assert 'mbkcKir' not in commands[-1]
    assert all('--sli-eligible-only' not in line for line in commands[:-1])
    if mode == 'all':
        assert '--sli-eligible-only' not in commands[-1]
        assert 'sliEligible' not in commands[-1]
    else:
        assert f'--sli-eligible-only --sli-eligibility-mode {mode}' in commands[-1]
        suffix = 'sliEligibleT2Sb2-5' if mode == 'mean' else 'sliEligibleT2Sb5'
        assert suffix in commands[-1]


@pytest.mark.parametrize('mode', ['mean', 'final'])
@pytest.mark.parametrize('comparison,third_slug', [
    ('ar_ctrl', 'ar_ctrlKir'), ('mbkc_kir', 'intact_mbkcKir'),
])
def test_notebook_shares_turnback_inputs_and_filters_ratio_and_home_alignment(
    mode, comparison, third_slug,
):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'notebooks/graphpad_exports.ipynb').read_text())
    stages = []
    scope = {'run_stage': lambda *args, **kwargs: stages.append((args, kwargs))}
    for cell in notebook['cells']:
        if (
            cell['cell_type'] != 'code'
            or not cell.get('id', '').startswith('graphpad-exports-turnback-')
        ):
            continue
        source = ''.join(cell['source'])
        if cell['id'] == 'graphpad-exports-turnback-01':
            source = source.replace('TURNBACK_DATE_TAG = "2026-09-25"',
                                    'TURNBACK_DATE_TAG = "2030-01-02"')
            source = source.replace('TURNBACK_COMPARISON_GROUP = "ar_ctrl"',
                                    f'TURNBACK_COMPARISON_GROUP = "{comparison}"')
            source = source.replace('TURNBACK_ELIGIBLE_MODE = "final"',
                                    f'TURNBACK_ELIGIBLE_MODE = "{mode}"')
        exec(source, scope)
    assert len(stages) == 2  # Upstream and CSV stages; no separate eligible stage.
    assert all(not kwargs['run'] for _, kwargs in stages)
    assert scope['turnback_groups'][-1][1] == third_slug
    assert scope['load_turnback_datasets'].startswith('eval "$(')
    assert scope['load_turnback_datasets'].endswith(')"')
    for item in scope['turnback_upstream_exports']:
        assert f'TURNBACK_COMPARISON_GROUP={comparison}' in item['command']
        assert 'DATE_TAG=2030-01-02' in item['command']
    exports = scope['turnback_graphpad_exports']
    assert len(exports) == 5
    for item in exports:
        assert third_slug in item['command']
        assert '2030-01-02.npz' in item['command']
    eligible = [item for item in exports if '--sli-eligible-only' in item['command']]
    assert len(eligible) == 2
    for item in eligible:
        assert f'--sli-eligibility-mode {mode}' in item['command']
        assert comparison in item['output_path']
        assert '2030-01-02' in item['output_path']
    assert eligible[1]['command'].count('--sli-eligibility-input ') == 3
    assert 'repeated-measures-scalar-npz' in eligible[1]['command']
    assert '--top-sli-fraction 0.9' in exports[0]['command']
    assert all('--sli-eligibility-mode' not in item['command']
               for item in exports if item not in eligible)
