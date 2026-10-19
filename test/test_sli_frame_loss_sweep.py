import math

import numpy as np
import pytest

from scripts.sweep_sli_frame_loss import (
    GROUPS, ENDPOINTS, cluster_bootstrap, inclusion_audit, summarize, unique_index,
)


def subject(cohort="regular", endpoint="final_sli", video="a", fly=0,
            value=1., exp_loss=5., yok_loss=5., baseline=True):
    return dict(cohort=cohort, endpoint=endpoint, video=video, fly=fly,
                exp_loss_pct=exp_loss, yok_loss_pct=yok_loss,
                exp_no_trajectory=False, yok_no_trajectory=False,
                baseline_included=baseline, baseline_exclusion_reason="missing PI" if not baseline else "",
                value=value)


def test_paired_gate_includes_equality_and_keeps_baseline_exclusions():
    rows = [subject(exp_loss=10, yok_loss=10), subject(yok_loss=11),
            subject(exp_loss=11), subject(exp_loss=11, yok_loss=11),
            subject(baseline=False)]
    audit = inclusion_audit(rows, (10,))
    assert [r["included"] for r in audit] == [True, False, False, False, False]
    assert [r["exclusion_reason"] for r in audit] == [
        "", "yoked exceeds cutoff", "experimental exceeds cutoff",
        "both exceed cutoff", "baseline: missing PI"]


def test_empty_tracking_cannot_enter_even_with_small_post_prefix_loss():
    row = subject(exp_loss=0)
    row["exp_no_trajectory"] = True
    assert not inclusion_audit([row], (60,))[0]["included"]


def test_cluster_bootstrap_keeps_flies_together_and_weights_flies():
    # Video A has values 0,2; video B has value 10. Draw both or repeat one video.
    samples = cluster_bootstrap(np.array([0., 2., 10.]), np.array([0, 0, 1]),
                               np.array([[1, 1], [2, 0], [0, 2]]))
    np.testing.assert_allclose(samples, [4., 1., 10.])
    samples = cluster_bootstrap(np.array([10.]), np.array([1]), np.array([[2, 0], [0, 2]]))
    assert np.isnan(samples[0])
    assert samples[1] == 10


def test_sweep_distinct_endpoints_attrition_zero_sample_and_paired_shifts():
    rows = []
    for cohort in GROUPS:
        for endpoint in ENDPOINTS:
            # Strict group has zero retained flies; other groups retain one of two.
            loss = 20 if cohort == "antennae_removed" else 5
            offset = 0.2 if endpoint == "mean_sli" else 0.
            rows += [subject(cohort, endpoint, "a", value=1+offset, exp_loss=loss),
                     subject(cohort, endpoint, "b", value=0+offset, yok_loss=40),
                     subject(cohort, endpoint, "c", value=99, baseline=False)]
    summaries, contrasts = summarize(rows, draws=200, seed=42, cutoffs=(60, 30, 10))
    lookup = {(r["cohort"], r["endpoint"], r["cutoff_pct"]):r for r in summaries}
    base = lookup["regular", "final_sli", 60]
    strict = lookup["regular", "final_sli", 10]
    empty = lookup["antennae_removed", "final_sli", 10]
    assert base["n"] == 2 and base["mean"] == .5
    assert base["mean_shift_from_60"] == base["shift_ci_low"] == base["shift_ci_high"] == 0
    assert strict["n"] == 1 and strict["mean"] == 1
    assert strict["excluded_yok_only"] == 1
    assert strict["mean_shift_from_60"] == .5
    assert lookup["regular", "mean_sli", 10]["mean"] == 1.2
    assert empty["n"] == empty["n_videos"] == empty["bootstrap_valid_draws"] == 0
    assert math.isnan(empty["mean"]) and math.isnan(empty["ci_low"])
    assert all(math.isnan(c["difference"]) for c in contrasts
               if c["cutoff_pct"] == 10 and c["comparison"] == "antennae_removed_minus_regular")
    again = summarize(rows, draws=200, seed=42, cutoffs=(60, 30, 10))[0]
    assert again[0] == summaries[0]


def test_baseline_mismatch_fails_instead_of_silently_changing_baseline():
    rows = [subject(c, e, yok_loss=65) for c in GROUPS for e in ENDPOINTS]
    with pytest.raises(ValueError, match="60% strict filter differs"):
        summarize(rows, draws=100)


def test_duplicate_identity_fails():
    with pytest.raises(ValueError, match="Duplicate identity"):
        unique_index([dict(video="a"), dict(video="a")], lambda r:r["video"])
