from types import SimpleNamespace

import numpy as np
import pytest

from scripts.survey_frame_loss import loss_metrics, parse_loss_log, validate_log
from scripts import survey_frame_loss
from src.analysis.trajectory import Trajectory
from src.utils.common import CT


@pytest.mark.parametrize("x", [
    [np.nan, np.nan, 1, np.nan, 2, np.nan, np.nan],
    [1, 2, 3],
    [1, np.nan, np.nan, np.nan, np.nan],
    [np.nan, 1, 2],
])
@pytest.mark.parametrize("chamber,cutoff", [(CT.large2, 60), (CT.htl, 10)])
def test_metrics_match_analyzer(x, capsys, chamber, cutoff):
    trajectory = Trajectory.__new__(Trajectory)
    trajectory.x = np.array(x, dtype=float)
    trajectory.y = trajectory.x.copy()
    trajectory.theta = None
    trajectory.va = SimpleNamespace(ct=chamber, openLoop=False, on=np.array([]))
    trajectory._setLostFrameThreshold()
    assert not trajectory._isEmpty()
    trajectory._interpolate()
    row = loss_metrics(x, cutoff)
    path_text = capsys.readouterr().out
    expected = f"lost: number frames: {row['lost_frames']} ({row['loss_pct']:.2f}%)"
    assert expected in path_text
    assert bool(trajectory._bad) == row["exceeds_cutoff"]
    assert trajectory.trxStartIdx == row["initial_missing_frames"]


def test_all_missing_is_flagged_without_division_by_zero():
    row = loss_metrics([np.nan] * 10)
    assert row["no_trajectory"]
    assert np.isnan(row["loss_pct"])
    assert row["missing_pct_all_frames"] == 100


def test_log_pair_role_and_rounding(tmp_path):
    path = tmp_path / "analysis.log"
    path.write_text('''=== analyzing example.avi, fly 1 ===
exp fly
  lost: number frames: 1 (33.33%), sequence length: avg: 1.0, max: 1
yok fly
  lost: number frames: 2 (66.67%) *** bad ***
''')
    assert parse_loss_log(path)[("example.avi", 1, "yok")] == (2, 66.67)
    row = dict(video="/somewhere/example.avi", fly=1, role="exp",
               **loss_metrics([1, np.nan, 2]))
    assert validate_log([row], path) == 1
    row["lost_frames"] = 2
    with pytest.raises(ValueError, match="disagrees"):
        validate_log([row], path)


@pytest.mark.parametrize("chamber,nef,cutoff", [("large2", 2, 60), ("htl", 10, 10)])
@pytest.mark.parametrize("explicit_mapping", [False, True])
def test_survey_selects_all_pairs_and_yoked_indices(monkeypatch, chamber, nef, cutoff, explicit_mapping):
    mapping = {i: 2 * nef - 1 - i for i in range(nef)} if explicit_mapping else {}
    proto = dict(frameNums=[{1: 1}] * nef, ef2yc=mapping)
    trx = {"x": [[1, np.nan, 2]] * (2 * nef)}
    monkeypatch.setattr(survey_frame_loss.glob, "glob", lambda _: ["/videos/example.avi"])
    monkeypatch.setattr(survey_frame_loss, "load_pickle",
                        lambda path: {"protocol": proto} if path.suffix == ".data" else trx)
    rows, manifest = survey_frame_loss.survey({"regular": "example*"}, chamber)
    assert len(manifest) == 1
    assert len(rows) == 2 * nef
    for fly in range(nef):
        exp, yok = rows[2 * fly:2 * fly + 2]
        assert exp["trajectory_index"] == fly
        assert yok["trajectory_index"] == (mapping[fly] if mapping else fly + nef)
        assert exp["cutoff_pct"] == cutoff
        assert exp["exceeds_cutoff"] == (chamber == "htl")


def test_exact_cutoff_passes():
    assert not loss_metrics([1] * 9 + [np.nan], 10)["exceeds_cutoff"]
    assert not loss_metrics([1] * 4 + [np.nan] * 6, 60)["exceeds_cutoff"]
