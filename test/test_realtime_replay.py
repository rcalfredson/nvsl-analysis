"""Replay fidelity: tracker binning, rolling eviction, training windows, and MP4."""
import pickle

import cv2
import numpy as np
import pytest

from src.plotting.realtime_replay import Recorder, Replay, main, sidecar


def fixture_data():
    protocol = {
        'ct': 'regular', 'tm': {'x': 0, 'y': 0, 'fctr': 1},
        'pt': 'circle', 'circle': {'r': 4},
        'frameNums': {'startTrain': [2, 7], 'startPost': [5, 9]},
        'info': {'cPos': [(20, 20), (30, 20)], 'r': [4, 5]},
    }
    x = np.array([[1, 1, 4, 6, 8, np.nan, 12, 14, 16, 18],
                  [1, 1, 104, 106, 108, 110, 112, 114, 116, 118.]])
    tracks = {'x': x, 'y': np.full_like(x, 10), 'w': np.full_like(x, 4),
              'h': np.full_like(x, 8), 'theta': np.zeros_like(x)}
    return protocol, tracks


def test_rolling_counts_duplicates_rounding_and_seek():
    protocol, tracks = fixture_data()
    replay = Replay(protocol, tracks, (40, 200, 3), 1, minutes=3/60)
    replay.seek(0)
    # Two coincident flies, x/2 == 0.5 rounds up as in Python 2.
    assert replay.running[5, 1] == 3
    for i in range(1, 10):
        replay.seek(i)
        expected = Replay(protocol, tracks, (40, 200, 3), 1, minutes=3/60)
        expected.seek(i)
        np.testing.assert_array_equal(replay.running, expected.running)
        np.testing.assert_array_equal(replay.protocol_map, expected.protocol_map)
        assert replay.running.min() == 1
    replay.seek(3)
    assert replay.running.sum() - replay.running.size == 6


def test_protocol_freezes_after_training_and_resets():
    protocol, tracks = fixture_data()
    replay = Replay(protocol, tracks, (40, 200, 3), 1)
    replay.seek(1)
    assert np.all(replay.protocol_map == 1)
    replay.seek(4)
    frozen = replay.protocol_map.copy()
    replay.seek(5)
    replay.seek(6)
    np.testing.assert_array_equal(replay.protocol_map, frozen)
    replay.seek(7)
    assert replay.protocol_map.sum() - replay.protocol_map.size == 2


def test_chambers_normalize_independently_and_only_active_circles():
    protocol, tracks = fixture_data()
    replay = Replay(protocol, tracks, (40, 200, 3), 1)
    replay.seek(2)
    # Both chambers have their own color range despite unequal dwell counts.
    replay.running.fill(1)
    replay.running[5, 2] = 100
    replay.running[5, 52] = 2
    image = replay.heatmap('running')
    np.testing.assert_array_equal(image[10:12, 4:6], image[10:12, 104:106])
    frame = np.zeros((40, 200, 3), np.uint8)
    assert replay.render(frame)[52, 24].tolist() == [255, 255, 255]
    replay.seek(5)
    assert replay.render(frame)[52, 24].tolist() == [0, 0, 0]


@pytest.mark.parametrize('speed_args, multiplier', [
    ([], 1), (['--speed', '2'], 2), (['--speed', '0.5'], .5),
    (['--speed', '2', '--record-speed', '1'], 1),
    (['--speed', '0.5', '--record-speed', '4'], 4),
])
def test_headless_mp4_has_exact_interval_and_playback_speed(tmp_path, speed_args, multiplier):
    protocol, tracks = fixture_data()
    video = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 5, (200, 40))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((40, 200, 3), np.uint8))
    writer.release()
    for suffix, data in (('.data', {'protocol': protocol}), ('.trx', tracks)):
        video.with_suffix(suffix).write_bytes(pickle.dumps(data))
    output = tmp_path / 'clip.mp4'
    main([str(video), '--start', '2', '--end', '7', '--headless', '--output', str(output), *speed_args])
    cap = cv2.VideoCapture(str(output))
    try:
        output_fps = cap.get(cv2.CAP_PROP_FPS)
        assert output_fps == pytest.approx(5 * multiplier)
        frames = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            frames.append(image)
        assert len(frames) == 5
        assert len(frames) / output_fps == pytest.approx(1 / multiplier)
        assert frames[0].shape == (72, 400, 3)
    finally:
        cap.release()
    # Never replace an existing supplementary artifact.
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        main([str(video), '--headless', '--output', str(output)])
    assert output.read_bytes() == original
    assert sidecar(video, '.trx') == video.with_suffix('.trx')


@pytest.mark.parametrize('option', ['--speed', '--record-speed'])
@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_invalid_speed_is_rejected_before_opening_video(option, value):
    with pytest.raises(SystemExit) as exc:
        main(['missing.avi', option, value, '--headless'])
    assert exc.value.code == 2


def test_appended_sidecar_and_large_layout_requires_explicit_choice(tmp_path):
    video = tmp_path / 'source.avi'
    path = tmp_path / 'source.avi.trx'
    path.touch()
    assert sidecar(video, '.trx') == path
    protocol, tracks = fixture_data()
    protocol['ct'] = 'large'
    with pytest.raises(ValueError, match='large chambers require'):
        Replay(protocol, tracks, (720, 1280, 3), 5)


def test_each_experimental_fly_uses_its_own_training_interval():
    protocol, tracks = fixture_data()
    protocol['frameNums'] = [protocol['frameNums'], {'startTrain': [4], 'startPost': [6]}]
    protocol['info'] = [protocol['info'], {'cPos': [(120, 20)], 'r': [4]}]
    replay = Replay(protocol, tracks, (40, 200, 3), 1)
    frame = np.zeros((40, 200, 3), np.uint8)
    replay.seek(2)
    assert replay.render(frame)[52, 24].tolist() == [255, 255, 255]
    assert replay.render(frame)[52, 124].tolist() == [0, 0, 0]
    replay.seek(5)
    assert replay.render(frame)[52, 24].tolist() == [0, 0, 0]
    assert replay.render(frame)[52, 124].tolist() == [255, 255, 255]


def test_recording_pause_and_multiple_clips(tmp_path):
    image = np.zeros((40, 200, 3), np.uint8)
    recorder = Recorder(tmp_path / 'clip.mp4', 5)
    try:
        recorder.start(image)
        for index in [2, 2, 2, 3, 3, 4]:
            recorder.write(image, index)
        recorder.stop()
        recorder.start(image)
        recorder.write(image, 5)
    finally:
        recorder.stop()
    for name, count in [('clip.mp4', 3), ('clip_002.mp4', 1)]:
        cap = cv2.VideoCapture(str(tmp_path / name))
        try:
            assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == count
        finally:
            cap.release()
