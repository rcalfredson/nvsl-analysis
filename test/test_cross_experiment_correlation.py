from types import SimpleNamespace

import numpy as np
import pytest

from src.exporting.cross_experiment_correlation import (
    build_closed_loop_sli_rows,
    build_open_loop_preference_rows,
    join_closed_to_experimental_open_loop,
    recording_identity,
)


def _closed_va(fly_id=0, yoked_id=10):
    return SimpleNamespace(
        fn="/media/Synology4/Yang Chen/2024-08-08/afternoon/c41__closed.avi",
        f=fly_id,
        trxf=(fly_id, yoked_id),
        gidx=0,
    )


def _open_va(fly_id=0, preferences=(-0.2, 0.5, -0.4)):
    return SimpleNamespace(
        fn="/media/Synology4/Yang Chen/2024-08-08/afternoon/openloop/c41__open.avi",
        f=fly_id,
        trxf=(fly_id,),
        gidx=0,
        openLoop=True,
        alt=False,
        posPI=[preferences],
    )


def test_recording_identity_uses_date_and_full_camera_number():
    assert recording_identity(
        "/data/2024-08-08/openloop/c41__example.avi"
    ) == ("2024-08-08", "c41")


def test_closed_export_uses_exp_minus_yoked_at_t2_sb5():
    raw = np.zeros((1, 2, 2, 6), dtype=float)
    raw[0, 1, 0, 4] = 0.75
    raw[0, 1, 1, 4] = -0.15
    # Different values in adjacent buckets guard against final-bucket inference.
    raw[0, 1, 0, 3] = 0.1
    raw[0, 1, 1, 3] = 0.2

    row = build_closed_loop_sli_rows([_closed_va()], raw)[0]

    assert row["sli_training"] == 2
    assert row["sli_sync_bucket"] == 5
    assert row["exp_reward_pi_t2_sb5"] == pytest.approx(0.75)
    assert row["yoked_reward_pi_t2_sb5"] == pytest.approx(-0.15)
    assert row["sli_t2_sb5"] == pytest.approx(0.9)
    assert row["exp_subject_key"] == "2024-08-08::c41::f0"
    assert row["yoked_subject_key"] == "2024-08-08::c41::f10"


def test_closed_export_uses_mean_exp_minus_mean_yoked_over_t2_sb2_to_sb5():
    raw = np.zeros((1, 2, 2, 6), dtype=float)
    raw[0, 1, 0, 1:5] = [0.2, 0.4, np.nan, 0.8]
    raw[0, 1, 1, 1:5] = [-0.2, 0.0, 0.2, 0.4]

    row = build_closed_loop_sli_rows([_closed_va()], raw)[0]

    assert row["exp_reward_pi_t2_sb2_sb5_mean"] == pytest.approx(
        np.mean([0.2, 0.4, 0.8])
    )
    assert row["yoked_reward_pi_t2_sb2_sb5_mean"] == pytest.approx(0.1)
    assert row["sli_t2_sb2_sb5_mean"] == pytest.approx(
        np.mean([0.2, 0.4, 0.8]) - 0.1
    )


def test_open_export_preserves_positional_pi_scale_and_period_order():
    row = build_open_loop_preference_rows([_open_va()])[0]

    assert row["preference_pi_pre"] == pytest.approx(-0.2)
    assert row["preference_pi_led_on"] == pytest.approx(0.5)
    assert row["preference_pi_led_off"] == pytest.approx(-0.4)
    assert row["subject_key"] == "2024-08-08::c41::f0"


def test_join_uses_experimental_fly_and_audits_yoked_open_record():
    raw = np.zeros((1, 2, 2, 5), dtype=float)
    raw[0, 1, 0, 4] = 0.6
    raw[0, 1, 1, 4] = 0.1
    closed = build_closed_loop_sli_rows([_closed_va()], raw)
    opened = build_open_loop_preference_rows([_open_va(0), _open_va(10)])

    matched, audit = join_closed_to_experimental_open_loop(closed, opened)

    assert len(matched) == 1
    assert matched[0]["open_loop_video"].endswith("c41__open.avi")
    assert matched[0]["preference_pi_led_on"] == pytest.approx(0.5)
    assert [row["status"] for row in audit] == [
        "matched",
        "available_yoked_open_loop",
    ]
    assert audit[0]["included_led_on"] is True
    assert audit[0]["included_led_off"] is True
    assert audit[0]["included_mean_sli_led_on"] is True
    assert audit[0]["included_mean_sli_led_off"] is True
    assert audit[0]["closed_exp_subject_key"] == "2024-08-08::c41::f0"
    assert audit[0]["open_loop_subject_key"] == "2024-08-08::c41::f0"
    assert audit[0]["open_loop_fly_id"] == 0
    assert audit[0]["match_validated"] is True
    assert audit[0]["match_rule"] == "recording_date+camera_id+physical_fly_id"
    assert audit[1]["exp_subject_key"] == "2024-08-08::c41::f10"


def test_join_rejects_duplicate_open_subject_keys():
    raw = np.zeros((1, 2, 2, 5), dtype=float)
    closed = build_closed_loop_sli_rows([_closed_va()], raw)
    opened = build_open_loop_preference_rows([_open_va(), _open_va()])

    with pytest.raises(ValueError, match="duplicate open-loop subject_key"):
        join_closed_to_experimental_open_loop(closed, opened)


def test_join_rejects_duplicate_closed_pair_keys():
    raw = np.zeros((2, 2, 2, 5), dtype=float)
    closed = build_closed_loop_sli_rows([_closed_va(), _closed_va()], raw)
    opened = build_open_loop_preference_rows([_open_va()])

    with pytest.raises(ValueError, match="duplicate closed-loop pair_key"):
        join_closed_to_experimental_open_loop(closed, opened)


def test_open_export_rejects_alternating_side_protocol():
    va = _open_va()
    va.alt = True

    with pytest.raises(ValueError, match="alternating-side"):
        build_open_loop_preference_rows([va])


def test_join_audits_period_specific_nonfinite_values():
    raw = np.zeros((1, 2, 2, 5), dtype=float)
    closed = build_closed_loop_sli_rows([_closed_va()], raw)
    opened = build_open_loop_preference_rows(
        [_open_va(preferences=(0.0, np.nan, -0.2))]
    )

    _matched, audit = join_closed_to_experimental_open_loop(closed, opened)

    assert audit[0]["included_led_on"] is False
    assert audit[0]["included_led_off"] is True
    assert audit[0]["detail"] == "nonfinite LED-on PI"


def test_join_retains_unmatched_closed_pair_for_strict_audit():
    raw = np.zeros((1, 2, 2, 5), dtype=float)
    closed = build_closed_loop_sli_rows([_closed_va()], raw)

    matched, audit = join_closed_to_experimental_open_loop(closed, [])

    assert matched == []
    assert audit[0]["status"] == "unmatched_closed"
    assert audit[0]["closed_exp_subject_key"] == "2024-08-08::c41::f0"
    assert audit[0]["match_validated"] is False
