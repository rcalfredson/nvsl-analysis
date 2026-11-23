import json
import shlex

import pytest

from src.exporting import manuscript_comparison as comparison


@pytest.fixture
def recordings(tmp_path):
    paths = [tmp_path / f'c41__2025-07-29__20-05-4{i}.avi' for i in range(3)]
    for path in paths:
        for suffix in ('.avi', '.data', '.trx', '.offline.trx'):
            path.with_suffix(suffix).touch()
    manifest = tmp_path / 'tracking.json'
    manifest.write_text(json.dumps(dict(schema_version=1, recordings={
        str(p): {comparison.VARIANT: str(p.with_suffix('.offline.trx'))}
        for p in paths[:2]
    })))
    config = dict(directory=tmp_path / 'runs', manifest=manifest)
    return paths, config


def _analysis(*groups):
    return shlex.join(['python', 'analyze.py', '-v', '|'.join(groups), '-f', '0-1'])


def test_comparison_preflights_both_sources_and_keeps_identical_groups(recordings, tmp_path):
    paths, config = recordings
    command = _analysis(f'{paths[0]}:0', f'{paths[1]},{paths[2]}')
    recipe = dict(title='Time plot', plot_type='time', panels={'demo': {}})

    with pytest.raises(ValueError, match='no entry'):
        comparison.run_comparison(tmp_path, recipe, command, 'pdf', config, {}, run=True)
    assert not config['directory'].exists()

    config['excluded_recordings'] = [paths[2]]
    plans = comparison.run_comparison(tmp_path, recipe, command, 'pdf', config, {}, run=False)
    original, offline = (plans[variant]['analysis'][0] for variant in ('original', comparison.VARIANT))
    assert original[3] == offline[3] == f'{paths[0]}:0|{paths[1]}'
    assert original[4:6] == offline[4:6] == ['-f', '0-1']
    assert original[6:8] == ['--tracking-variant', 'original']
    assert offline[6:11] == ['--tracking-variant', comparison.VARIANT, '--tracking-manifest',
                             str(config['manifest']), '--no-tracking-fallback']
    assert not config['directory'].exists()


def test_data_comparison_converts_each_variants_own_archived_export(recordings, tmp_path, monkeypatch):
    paths, config = recordings
    command = _analysis(str(paths[0]))
    recipe = dict(title='Table', output_type='data', panels={'demo': {}},
                  analysis_stages=[dict(command=command, archive_source='intermediate.csv',
                                        archive_output='archive.csv')],
                  export_command='python scripts/convert.py',
                  artifacts=['table.csv'])
    def launch(root, directory, argv, label):
        if label == 'analysis-1':
            (directory / 'intermediate.csv').write_text(argv[argv.index('--tracking-variant') + 1])
        else:
            assert label == 'conversion'
            (directory / 'table.csv').write_bytes((directory / 'archive.csv').read_bytes())

    monkeypatch.setattr(comparison, '_run_command', launch)
    results = comparison.run_comparison(tmp_path, recipe, None, None, config, {}, run=True)
    for variant, plan in results.items():
        assert (plan['directory'] / 'table.csv').read_text() == variant
