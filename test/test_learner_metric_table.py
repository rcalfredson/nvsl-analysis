import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.video_analysis import VideoAnalysis
from src.exporting import learner_metric_table as report


def _observation(metric, cohort, value, *, eligible=True, unit_id="fly"):
    return report.MetricObservation(
        unit_id=unit_id,
        sli=1.0 if cohort == "strong" else -1.0,
        cohort=cohort,
        metric=metric,
        value=float(value),
        n_events=8,
        eligible=eligible,
        exclusion_reason="" if eligible else "insufficient_episodes",
    )


def _complete_observations():
    observations = []
    for metric_idx, metric in enumerate(report.METRIC_ORDER):
        for fly_idx, value in enumerate((5.0, 6.0, 7.0)):
            observations.append(
                _observation(
                    metric,
                    "strong",
                    value + metric_idx,
                    unit_id=f"strong-{fly_idx}",
                )
            )
        for fly_idx, value in enumerate((1.0, 2.0, 3.0)):
            observations.append(
                _observation(
                    metric,
                    "weak",
                    value + metric_idx,
                    unit_id=f"weak-{fly_idx}",
                )
            )
    return observations


def test_com_is_pooled_from_raw_episodes_over_selected_window():
    class Trajectory:
        @staticmethod
        def bad():
            return False

    class Video:
        fn = "video.avi"
        f = 0
        trns = [SimpleNamespace(), SimpleNamespace()]
        trx = [Trajectory()]
        sync_bucket_ranges = [
            [(0, 10)] * 5,
            [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)],
        ]
        reward_turnback_dual_circle_counts = {
            "turnback": np.ones((2, 1, 5), dtype=int),
            "total": np.full((2, 1, 5), 2, dtype=int),
        }

        @staticmethod
        def _iter_between_reward_segment_com(trn, fly, **kwargs):
            assert kwargs["fi"] == 10
            assert kwargs["n_buckets"] == 4
            for value in (1.0, 2.0, 9.0):
                yield SimpleNamespace(mag_mm=value)

    opts = SimpleNamespace(
        learner_metric_table_training=2,
        learner_metric_table_skip_first_sync_buckets=1,
        learner_metric_table_keep_first_sync_buckets=4,
        require_exp_target_sync_bucket=False,
        min_between_reward_trajectories=1,
        min_turnback_episodes=1,
        com_exclude_wall_contact=False,
        com_per_segment_min_meddist_mm=0.0,
        btw_rwd_com_exclude_reward_endpoints=False,
    )
    observations = []

    report._collect_array_metrics(
        [(0, Video(), "strong")], np.asarray([0.75]), opts, observations
    )

    by_metric = {row.metric: row for row in observations}
    assert by_metric["dual_circle_turnback_ratio"].value == pytest.approx(0.5)
    assert by_metric["dual_circle_turnback_ratio"].n_events == 8
    assert by_metric["com_distance_to_reward_center"].value == pytest.approx(4.0)
    assert by_metric["com_distance_to_reward_center"].n_events == 3
    assert by_metric["com_distance_to_reward_center"].eligible is True


def test_summary_is_per_fly_and_holm_adjusted_across_five_metrics():
    observations = _complete_observations()
    observations.append(
        _observation(
            report.METRIC_ORDER[0],
            "strong",
            1000,
            eligible=False,
            unit_id="excluded",
        )
    )

    rows = report.summarize_observations(observations)

    assert [row["metric"] for row in rows] == list(report.METRIC_ORDER)
    assert len(rows) == 5
    for row in rows:
        assert row["strong_n"] == 3
        assert row["weak_n"] == 3
        assert row["difference_strong_minus_weak"] == pytest.approx(4.0)
        assert np.isfinite(row["p_value"])
        assert row["p_value_holm"] >= row["p_value"]


def test_export_writes_summary_audit_metadata_and_markdown(tmp_path, monkeypatch):
    observations = _complete_observations()
    metadata = {"cohort_definition": {"top_fraction": 0.2}}
    monkeypatch.setattr(
        report,
        "collect_learner_metric_observations",
        lambda vas, opts, gls: (observations, metadata),
    )

    paths = report.export_learner_metric_table(
        [], SimpleNamespace(), None, str(tmp_path / "learner_metrics.csv")
    )

    assert set(paths) == {
        "summary_csv",
        "per_fly_csv",
        "metadata_json",
        "summary_md",
    }
    with open(paths["summary_csv"], newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert len(summary_rows) == 5
    assert summary_rows[0]["metric"] == report.METRIC_ORDER[0]

    with open(paths["per_fly_csv"], newline="", encoding="utf-8") as fh:
        per_fly_rows = list(csv.DictReader(fh))
    assert len(per_fly_rows) == len(observations)
    assert {row["cohort"] for row in per_fly_rows} == {"strong", "weak"}

    with open(paths["metadata_json"], encoding="utf-8") as fh:
        written_metadata = json.load(fh)
    assert written_metadata["cohort_definition"]["top_fraction"] == 0.2
    assert written_metadata["outputs"] == paths

    with open(paths["summary_md"], encoding="utf-8") as fh:
        markdown = fh.read()
    assert "| Metric | Strong learners | Weak learners | Strong vs. weak |" in markdown
    assert markdown.count("\n|") == 6


def test_export_prevents_trajectory_slimming():
    va = SimpleNamespace(
        opts=SimpleNamespace(export_learner_metric_table="learner_metrics")
    )

    assert VideoAnalysis.memory_saver_can_slim_trajectories(va) is False
