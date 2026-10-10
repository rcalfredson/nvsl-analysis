import json
from pathlib import Path
import subprocess

import pytest

from src.plotting.heatmap_workflow import ensure_heatmap_cache, heatmap_cache_provenance


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'analyze.py').write_text('# analysis source\n')
    (root / '.analyze.local.env').write_text('ENABLE_DEFAULT_BETWEEN_REWARD_SLI_PLOTS=false\n')
    return root


COMMAND = 'python analyze.py -v /unmounted/video.avi -f 0-1 --pltHm --hm-periods training --pltHmVmin 1e-6 --pltHmVmax 3e-4'


def test_analysis_then_reuse_then_invalidate(project, tmp_path, monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        Path(argv[argv.index('--hm-map-export') + 1]).write_bytes(b'probability cache')
        assert Path(kwargs['cwd'], 'imgs').is_dir()
    monkeypatch.setattr('src.plotting.heatmap_workflow.subprocess.run', run)
    target = tmp_path / 'cache/maps.npz'
    ensure_heatmap_cache(project, COMMAND, target)
    ensure_heatmap_cache(project, COMMAND.replace('1e-6', '3e-6') + ' --fs 25 --fontFamily Arial --imgFormat pdf', target)
    assert len(calls) == 1  # Bounds, fonts and output format only rerender.
    ensure_heatmap_cache(project, COMMAND.replace('training', 'post'), target)
    assert len(calls) == 2
    ensure_heatmap_cache(project, COMMAND.replace('training', 'post'), target, rebuild=True)
    assert len(calls) == 3
    (project / 'analyze.py').write_text('# changed source\n')
    ensure_heatmap_cache(project, COMMAND.replace('training', 'post'), target)
    assert len(calls) == 4
    (project / '.analyze.local.env').write_text('NEW_SETTING=true\n')
    ensure_heatmap_cache(project, COMMAND.replace('training', 'post'), target)
    assert len(calls) == 5


def test_adopts_compatible_experiment_maps_without_analysis(project, tmp_path, monkeypatch):
    candidate = tmp_path / 'experiment.npz'
    candidate.write_bytes(b'existing probability maps')
    candidate.with_suffix('.json').write_text(json.dumps(heatmap_cache_provenance(project, COMMAND)))
    # The manuscript toggles optional non-heatmap plots during Run All.
    (project / '.analyze.local.env').write_text('ENABLE_DEFAULT_BETWEEN_REWARD_SLI_PLOTS=true\n')
    def fail(*args, **kwargs):
        pytest.fail('A compatible existing cache should not run analysis')
    monkeypatch.setattr('src.plotting.heatmap_workflow.subprocess.run', fail)
    target = tmp_path / 'manuscript/maps.npz'
    ensure_heatmap_cache(project, COMMAND, target, reuse_paths=[candidate])
    assert target.read_bytes() == candidate.read_bytes()
    assert target.with_suffix('.json').is_file()


def test_failed_rebuild_keeps_previous_cache(project, tmp_path, monkeypatch):
    target = tmp_path / 'maps.npz'
    target.write_bytes(b'old valid maps')
    manifest = json.dumps(heatmap_cache_provenance(project, COMMAND))
    target.with_suffix('.json').write_text(manifest)
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr('src.plotting.heatmap_workflow.subprocess.run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        ensure_heatmap_cache(project, COMMAND, target, rebuild=True)
    assert target.read_bytes() == b'old valid maps'
    assert target.with_suffix('.json').read_text() == manifest
