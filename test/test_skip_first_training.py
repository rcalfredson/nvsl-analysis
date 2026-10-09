from types import SimpleNamespace

import numpy as np
import pytest

import src.analysis.video_analysis as video_analysis
from src.analysis.video_analysis import VideoAnalysis


def _load_analysis(monkeypatch, skip=False, starts=(100, 300, 500), pre=(0, 250)):
    va = object.__new__(VideoAnalysis)
    va.opts = SimpleNamespace(skipFT=skip, allowYC=False)
    va.f = 0
    proto = {
        "frameNums": {
            "startTrain": list(starts),
            "startPost": [s + 100 for s in starts],
            "startPre": list(pre),
        },
        "info": {"cPos": [(10, 20), (30, 40), (50, 60)], "r": [1, 2, 3]},
        "pt": "circle",
        "circle": {"r": 1},
    }
    va.fileCache = {"test.avi": [{"protocol": proto}, {"x": np.zeros((2, 800))}]}
    monkeypatch.setattr(video_analysis.util, "videoCapture", lambda fn: None)
    monkeypatch.setattr(video_analysis.util, "frameRate", lambda cap: 10)
    monkeypatch.setattr(video_analysis.util, "readFrame", lambda cap, fi: None)
    monkeypatch.setattr(video_analysis, "Xformer", lambda *args: None)

    class Training:
        TP = SimpleNamespace(circle="circle", choice="choice")

        def __init__(self, n, start, stop, va, opts, circle):
            self.n, self.start, self.stop = n, start, stop
            self.circle, self.tp = circle, self.TP.circle

        @staticmethod
        def processReport(*args):
            pass

    monkeypatch.setattr(video_analysis, "Training", Training)
    va._createExtractChamber = lambda: None
    va._loadData("test.avi")
    return va


@pytest.mark.parametrize("skip", [False, True])
def test_skip_training_preserves_session_geometry(monkeypatch, skip):
    va = _load_analysis(monkeypatch, skip=skip)
    first = int(skip)
    assert va.startPre == (250 if skip else 0)
    assert [t.n for t in va.trns] == list(range(1, 4 - first))
    assert [(t.start, t.stop) for t in va.trns] == [
        (100, 200), (300, 400), (500, 600)
    ][first:]
    assert [t.circle for t in va.trns] == [
        ((10, 20), 1), ((30, 40), 2), ((50, 60), 3)
    ][first:]


@pytest.mark.parametrize("starts,pre", [((100,), (0, 250)), ((100, 300, 500), (0,))])
def test_skip_training_requires_second_training_and_pre_start(monkeypatch, starts, pre):
    with pytest.raises(SystemExit):
        _load_analysis(monkeypatch, skip=True, starts=starts, pre=pre)


def test_speed_ignores_pulses_before_selected_pre_period():
    va = object.__new__(VideoAnalysis)
    va.startPre = 250
    va.on = np.array([110, 120, 270])
    va.trns = [SimpleNamespace(start=300, stop=400, name=lambda: "training 1")]
    va.flies = []
    va._min2f = lambda minutes: 100
    # The two pulses from the skipped training must not trigger the assertion
    # that each pre-period has at most one pulse.
    VideoAnalysis.speed(va)
