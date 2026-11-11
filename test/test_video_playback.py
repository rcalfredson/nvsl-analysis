"""Shared command parsing and the annotated player's existing keyboard behavior."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.video_playback import FrameCommandInput, parse_frame_position


@pytest.mark.parametrize('value, expected', [('6750', 6750), ('00:15:00', 6750),
                                            ('15:00', 6750), ('00:00:01', 8),
                                            ('00:00:00.2', 2)])
def test_replay_frame_positions(value, expected):
    assert parse_frame_position(value, 7.5) == expected


def test_legacy_clock_and_truncation():
    entry = FrameCommandInput(7.5, strict_clock=True, nearest=False)
    for key in '00:00:01':
        entry.feed(key)
    assert entry.consume() == 7
    assert entry.consume() is None
    for text in ['15:00', '25:00:00', '00:00:00.2', 'garbage']:
        with pytest.raises(ValueError):
            parse_frame_position(text, 7.5, strict_clock=True, nearest=False)


def test_invalid_command_clears_buffer_and_unrelated_keys_are_ignored():
    entry = FrameCommandInput(7.5)
    for key in ':':
        entry.feed(key)
    assert entry.consume() is None
    for key in '1x2':
        entry.feed(key)
    assert entry.consume() == 12
    assert entry.text == ''
    with pytest.raises(ValueError):
        parse_frame_position('00:nan', 7.5)


def test_annotated_player_go_length_and_step(monkeypatch):
    from src.analysis import video_analysis as module

    # Exercise the real player method with display and analysis data stubbed.
    # waitKey must first report no event, matching its existing event-drain loop.
    commands = list('00:00:01g12l.q')
    keys = iter(code for key in commands for code in (-1, ord(key)))
    monkeypatch.setattr(module.cv2, 'waitKey', lambda delay: next(keys))
    monkeypatch.setattr(module.cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(module.Training, 'get', lambda *args: None)
    monkeypatch.setattr(module.util, 'readFrame', lambda cap, i: np.zeros((4, 4, 3), np.uint8))
    headers = []
    def combine(images):
        headers.append(images[0][1])
        return (images[0][0],)
    monkeypatch.setattr(module.util, 'combineImgs', combine)
    trajectory = SimpleNamespace(annotate=lambda *args: None, annotateTxt=lambda *args: None)
    player = SimpleNamespace(ct=None, trx=[trajectory], fps=7.5, cap=None, trns=[],
                             opts=SimpleNamespace(wall=False), fn='test.avi',
                             _DLT=100, _ARROW_KEY_MAP=module.VideoAnalysis._ARROW_KEY_MAP)
    module.VideoAnalysis._playAnnotatedVideo(player)
    assert headers[-1].endswith('(8) tlen=12')
    assert any(h.endswith('(7) tlen=100') for h in headers)
