import argparse
import copy
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.local_config import add_tracking_arguments, load_local_analyze_config
from src.utils.tracking_selection import (
    file_identity, load_tracking_data, select_tracking, validate_companion,
)
from src.caching.wall_contact_cache import build_wall_cache_spec


@pytest.fixture
def recording(tmp_path):
    video = tmp_path / 'session.avi'
    video.write_bytes(b'video')
    data = video.with_suffix('.data')
    data.write_bytes(pickle.dumps({'protocol': 'original'}))
    original = {key: np.ones((4, 8)) for key in ('x', 'y', 'w', 'h', 'theta')}
    original['ts'] = np.arange(8) * .125
    trx = video.with_suffix('.trx')
    trx.write_bytes(pickle.dumps(original))
    companion = copy.deepcopy(original)
    companion['x'] += 10
    companion['tracking_metadata'] = {'source_recording': {
        name: file_identity(path) for name, path in [('video', video), ('trx', trx), ('data', data)]}}
    offline = tmp_path / 'body-core.trx'
    offline.write_bytes(pickle.dumps(companion))
    manifest = tmp_path / 'tracking.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'recordings': {
        str(video): {'body-core': offline.name}}}))
    return SimpleNamespace(**locals())


def load(r, cache=None, **kwargs):
    def unpickle(path):
        with open(path, 'rb') as stream:
            return pickle.load(stream)
    return load_tracking_data(r.video, r.data, r.trx, SimpleNamespace(**kwargs),
                              {} if cache is None else cache, unpickle)


def test_default_original_and_strict_selection(recording):
    r = recording
    dt, original, selection = load(r)
    assert dt == {'protocol': 'original'}
    assert selection['requested_variant'] == selection['resolved_variant'] == 'original'
    np.testing.assert_array_equal(original['x'], r.original['x'])
    _, selected, selection = load(r, tracking_variant='body-core', tracking_manifest=r.manifest)
    np.testing.assert_array_equal(selected['x'], r.companion['x'])
    assert selection['trajectory']['path'] == str(r.offline)
    assert len(selection['trajectory']['sha256']) == 64


def test_strict_missing_entry_and_explicit_fallback(recording):
    r = recording
    with pytest.raises(ValueError, match='no entry.*other'):
        load(r, tracking_variant='other', tracking_manifest=r.manifest)
    _, original, selection = load(r, tracking_variant='other', tracking_manifest=r.manifest,
                                  tracking_fallback=True)
    assert selection['requested_variant'] == 'other'
    assert selection['resolved_variant'] == 'original'
    np.testing.assert_array_equal(original['x'], r.original['x'])
    with pytest.raises(ValueError, match='requires --tracking-manifest'):
        load(r, tracking_variant='body-core', tracking_fallback=True)


def test_missing_listed_file_never_falls_back(recording):
    r = recording
    r.offline.unlink()
    with pytest.raises(ValueError, match='repair the manifest/file'):
        load(r, tracking_variant='body-core', tracking_manifest=r.manifest, tracking_fallback=True)


@pytest.mark.parametrize('mismatch', ['source', 'shape', 'timestamps', 'metadata', 'hash', 'corrupt'])
def test_mismatched_companion_never_falls_back(recording, mismatch):
    r = recording
    other = copy.deepcopy(r.companion)
    if mismatch == 'source':
        other['tracking_metadata']['source_recording']['video']['path'] += '.wrong'
    elif mismatch == 'shape':
        other['theta'] = np.ones((3, 8))
    elif mismatch == 'timestamps':
        other['ts'][2] += 1e-12
    elif mismatch == 'metadata':
        del other['tracking_metadata']
    elif mismatch == 'hash':
        other['tracking_metadata']['source_recording']['trx']['sha256'] = 'bad'
    r.offline.write_bytes(b'broken' if mismatch == 'corrupt' else pickle.dumps(other))
    with pytest.raises(ValueError, match='Cannot use companion'):
        load(r, tracking_variant='body-core', tracking_manifest=r.manifest, tracking_fallback=True)


def test_recording_path_prevents_same_camera_day_collision(recording):
    r = recording
    other = r.tmp_path / 'another-session' / r.video.name
    other.parent.mkdir()
    other.write_bytes(b'video')
    with pytest.raises(ValueError, match='no entry'):
        select_tracking(other, r.trx, variant='body-core', manifest=r.manifest)


def test_memory_and_wall_cache_separation(recording):
    r, cache = recording, {}
    original = load(r, cache)
    offline = load(r, cache, tracking_variant='body-core', tracking_manifest=r.manifest)
    assert len(cache) == 2
    assert load(r, cache)[1] is original[1]
    assert original[1] is not offline[1]
    common = dict(video_fn=str(r.video), cache_dir=str(r.tmp_path / 'cache'), ct_name='large',
                  wall_orientation='all', thresholds_wall=[1, 2], offset_wall=0,
                  boundary_contact_binary_path='extension.so', boundary_contact_binary_sha256='hash', va_id=0)
    specs = [build_wall_cache_spec(**common, input_identities={'tracking': value[2]})
             for value in (original, offline)]
    assert specs[0].npz_path != specs[1].npz_path
    # Replacement in place also invalidates memory and disk identities.
    r.offline.write_bytes(pickle.dumps(dict(r.companion, extra='replacement')))
    replacement = load(r, cache, tracking_variant='body-core', tracking_manifest=r.manifest)
    assert len(cache) == 3
    assert replacement[2]['trajectory']['sha256'] != offline[2]['trajectory']['sha256']


def test_local_defaults_and_explicit_cli_precedence(tmp_path):
    config = tmp_path / '.analyze.local.env'
    config.write_text('TRACKING_VARIANT=body-core\nTRACKING_MANIFEST=/tmp/tracking.json\nTRACKING_FALLBACK=true\n')
    parser = argparse.ArgumentParser()
    add_tracking_arguments(parser, load_local_analyze_config(config))
    defaults = parser.parse_args([])
    assert defaults.tracking_variant == 'body-core'
    assert defaults.tracking_fallback is True
    explicit = parser.parse_args(['--tracking-variant', 'original', '--tracking-manifest', '/other.json',
                                  '--no-tracking-fallback'])
    assert explicit.tracking_variant == 'original'
    assert explicit.tracking_manifest == '/other.json'
    assert explicit.tracking_fallback is False


@pytest.mark.parametrize('payload', [
    {'schema_version': 2, 'recordings': {}},
    {'schema_version': 1, 'recordings': {'relative.avi': {}}},
    {'schema_version': 1, 'recordings': []},
])
def test_invalid_manifest_never_falls_back(recording, payload):
    r = recording
    r.manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Tracking manifest'):
        load(r, tracking_variant='body-core', tracking_manifest=r.manifest, tracking_fallback=True)


def test_acquisition_change_invalidates_cached_validation(recording):
    r, cache = recording, {}
    load(r, cache, tracking_variant='body-core', tracking_manifest=r.manifest)
    r.data.write_bytes(pickle.dumps({'protocol': 'changed acquisition'}))
    with pytest.raises(ValueError, match='source data .* mismatch'):
        load(r, cache, tracking_variant='body-core', tracking_manifest=r.manifest)
