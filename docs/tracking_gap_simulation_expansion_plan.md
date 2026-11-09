# Tracking-gap pilot review and expansion plan

Reviewed 9 November 2026. This document records a review and proposed design.
Step-one reporting repairs are implemented as described below; the expanded
support inventory and simulation experiments have not been implemented or run.

## The paper question and the attainable claims

The question is whether current metric values and cohort comparisons depend
materially on reconstructed trajectories during lost tracking. The current
single-gap pilot can establish susceptibility in supported observed contexts.
Increasing its sample size can estimate that conditional response more precisely.
It cannot, by itself, identify the error in genuinely unobserved trajectories.

The stronger recording-level claim needs long, trusted reference windows with
enough episodes, realistic repeated missingness, and the actual manuscript
aggregation and eligibility rules. Where such references do not exist, targeted
offline retracking/manual reconstruction is necessary, or the claim must remain
restricted to supported contexts. Matching endpoints does not make loss random
or establish the hidden path distribution.

## Agreed contrast targets and expanded agarose scope

Following discussion of the manuscript panels, the primary targets are changes
in group contrasts, assessed in terms of direction, magnitude and composition
of the contributing subject pool. Absolute metric changes remain supporting
diagnostics. Define each contrast's orientation explicitly and keep it fixed:
contrast change is the masked contrast minus the reference contrast, in native
metric units. Relative distortion can supplement this where the reference
contrast is sufficiently far from zero; it is not a universal acceptance rule.

- For between-reward tortuosity, wall contacts and wall-positive path length,
  evaluate experimental-cohort-minus-control contrasts, and the PFNd>Kir versus
  antennae-removed comparison where relevant to the manuscript claim.
- For turnback and home-vector alignment, retain all three radius pairs and
  assess both radius-specific group contrasts and changes in their pattern
  across radii. Do not substitute preservation of significance stars for
  preservation of the effect's magnitude.
- For agarose avoidance, include the actual flat-versus-agarose large-chamber
  comparisons in Extended Data Figure 17b–f: Ctrl, antennae removed, MBKC-1>Kir,
  PFNd>Kir and EPG>Kir, each in both chamber types. Define the contrast as
  agarose chamber minus flat chamber. Reproduce Pre, Training 1 and Training 2
  bucket settings and the displayed AUC calculation/eligibility from production
  code before comparing masked and reference summaries. The AUC endpoint's
  definition and missing-bucket handling must be verified, not inferred from
  integrating the plotted cohort means.

The notebook `notebooks/manuscript_figure_panels.ipynb` contains the ten source
video-list definitions and their export recipes. ED17b uses Figure 4b/4s lists;
ED17c uses Figure 4d/4t; ED17d uses ED14h/ED16h; ED17e uses Figure 4i/4u;
ED17f uses ED14i/ED16i. Recipes use experimental physical-site avoidance,
1 mm outer offset, Pre included, and two trainings. Derive admission and
per-cell eligibility from these recipes/production code; do not impose the
pooled metrics' final-SLI gate on the agarose cohorts. The original 196-subject
flat-chamber roster remains the starting population for the other metrics,
not the complete population for this expanded agarose analysis.

Profile missingness and reference support separately for each chamber/genotype;
do not transplant flat-chamber loss statistics into agarose chambers. Retain
the same-fly placement rule as the primary design. Source-list availability
does not yet establish that every video/raw tracking file is accessible or
that adequate reference trajectories exist.

No single acceptable-error cutoff is agreed yet. This does not block the
reporting repairs or support inventory. Specify scientific tolerances, or a
prespecified range of tolerance levels, before interpreting expanded effect
estimates. Any later equivalence statement remains conditional on reference
coverage and the tested loss mechanism.

## Review of the existing additions

Reviewed the profile/pilot runner, pure helpers, both report generators,
`test/test_tracking_gap_simulation.py`, and `docs/tracking_gap_simulation.md`.
Checked notebook references for the manuscript windows and selection settings.
The focused simulation, episode-loss and metric truth suite passed: 94 tests,
with two dependency deprecation warnings. These tests do not establish scientific
representativeness or replace an integration check of the expanded experiment.

Useful foundations to retain:

- Original missingness is distinguished from interpolated positions; artificial
  gaps hide all five tracker fields without mutating the source arrays.
- Duration is preserved exactly, endpoints and observed context are retained,
  and unsupported placements remain visible.
- Production metric functions are reused; baseline episode counts, measurements
  and frozen wall exports are checked before simulation.
- Fixed-boundary measurements and native redetection answer separate questions.
  Undefined values and changing episode membership are retained.
- Subject identity, source/code hashes, RNG settings and placement arms are saved.

Issues to address before a larger run:

1. **Denominators and filtering in reports.** The local-mean summary pools
   diagnostics below five episodes with better-supported measurements. The
   underlying episode CSV preserves outcomes, but numerator/denominator and
   eligibility are not visible alongside each trial's change. The 50-point
   agarose example is 1/2 to 0/1; it must not stand alone as a typical impact.
2. **Missing-cell handling.** The companion report's highlights select rows using
   `.iloc[0]` unconditionally. A spatial-only pilot or a metric with no matched
   trials can leave those selections empty and prevent report generation.
   Render explicit unsupported cells, including an entirely unsupported run.
3. **Coverage by actual exposure.** Turnback/alignment targeting uses a union of
   radius supports. Small-radius results can include trials with no directly
   exposed small-radius episode. Expose coverage per metric/radius and distinguish
   episode-support overlap from overlap with the sampled heading frames.
4. **Provenance completeness.** `track_code` does not include the wall-interval
   collector module whose region-assignment helper the runner calls. Include
   that dependency and audit other direct metric dependencies before expansion.
5. **Local state and boundaries.** Reference runs inherit preceding detector
   history; reference and simulation share it, but it is not independent ground
   truth. Validate hysteresis initialization, boundary handling and any effects
   extending beyond the evaluated run before interpreting full-window results.

No discrepancy in the existing pilot results was established by this review.
The changes above improve interpretation, robustness and the design of the next
experiment; historical results should remain immutable and labeled as a pilot.

## Proposed sequence and decision gates

### 1. Make the existing evidence auditable

Implemented 9 November: both report entry points now provide a denominator-aware
companion with validated per-attempt audit rows, separate all-defined and
minimum-five-in-both comparisons, explicit eligibility attrition, denominator
strata, and unsupported-cell handling. See the reporting section in
`docs/tracking_gap_simulation.md` for regeneration commands and output semantics.
Reports preserve historical pilot outputs; simulation behavior is unchanged.
Radius/heading-specific exposure inventory and detector-state validation remain
follow-on work rather than conclusions of this reporting repair.

Export one row per trial/metric/radius with fly/video identity, placement arm,
gap duration, evaluation duration, reference/simulated episode counts, ratio
numerators, values, signed change, and before/after minimum-count status.
Include original episode identities/support where available to detect reuse.

Report two views: all local diagnostics, and comparisons meeting the existing
five-episode rule in both conditions. Also retain baseline-eligible cases that
lose eligibility as a separate outcome; do not discard them from the analysis
of availability. Five episodes is the production rule, not a guarantee of a
precise ratio. Show denominators and denominator-stratified results explicitly.

### 2. Inventory reference support before choosing a larger sample size

Use the full 196-subject profile to estimate feasible support by cohort,
recording-loss stratum, duration, wall context, metric and radius. Start with
length-only feasibility, then spatial matching and complete episode support.
Enumerate/calculate candidates without repeatedly rerunning production detection.

Report support both as a fraction of donor gaps and as a fraction of donor lost
time, plus the number of independent videos/flies and unique reference episodes.
For a sampled candidate search, include selection probabilities and distinguish
sampled support from an exhaustive support calculation. Disclose boundary and
unbracketed loss excluded from the primary donor definition separately.

Decision: if important duration/context strata remain unsupported, add trusted
reference data there rather than substituting more short-gap trials. A larger
number of placements cannot repair missing long-gap reference trajectories.

### 3. Expand the conditional single-gap benchmark

Sample more independent videos/flies across the prespecified cohort/loss strata;
within fly, sample donors by duration and context and replicate placements.
Preserve strict same-fly matching as the primary design. Any cross-fly reference
library or tolerance change is a separately labeled sensitivity analysis with
its own matching diagnostics, never a silent fallback.

Keep two separate targets: duration/context-stratified stress tests, and an
empirical-frequency-weighted response over supported contexts. The latter needs
explicit selection probabilities, support denominators and matching assumptions.
Do not extrapolate that response into unsupported strata or call it an estimate
of all actual missing-path error.

Choose replication after the support inventory and a small expanded batch.
Assess precision at the video/fly level, not from thousands of reused episodes.
Set scientifically meaningful error tolerances for each metric and group
contrast before inspecting the final results. A failure to detect a difference
does not establish that error is smaller than those tolerances.

### 4. Evaluate whole windows with realistic repeated loss

Obtain trusted continuous reference windows matching the manuscript windows:
T2 SB2–5 for the pooled metrics, and the actual bucket/time-course cells for
agarose. Use offline retracking/manual reconstruction where original fully
observed runs are too short. Audit reference coordinates/body dimensions and
boundary-event behavior, especially near walls; replacement tracking is not
automatically ground truth.

Profile and transplant blocks containing multiple gaps and their intervening
observed intervals. Preserve temporal clustering, duration distribution and
burden; check endpoint/context agreement and report unsupported blocks.
Avoid independently scattering short gaps and calling that realistic loss.
Retain a separate controlled loss-dose experiment if useful, labeled as a
stress test rather than an empirical reproduction.

Rerun native detection and exact production aggregation on each reference and
masked window. Verify an unmasked expanded run reproduces the relevant exports.
Keep recorded rewards and the source admission/SLI gate fixed for the primary
measurement question; this cannot reconstruct counterfactual reward delivery.

### 5. Connect effects to the manuscript and quantify uncertainty

For each simulated recording/window retain the per-fly value, numerator/count,
eligibility, and paired change. Reproduce cohort means and the prespecified
between-cohort contrasts with the manuscript's fly weighting and selection.
Separate value changes in a fixed admitted pool from additional attrition caused
by measurement eligibility. Report how many subjects support each comparison.

Summarize repeated masks within fly while keeping variability across masks
visible. Construct uncertainty by resampling independent recording/video units
within cohort, keeping flies sharing a video together and preserving any
experimental/yoked pairing used by the manuscript. Synthetic masks are repeated
measurements, not new biological replicates. Report mask variability separately
from uncertainty across subjects and from unquantified unsupported-context error.

Possible final claim: bounded metric/contrast sensitivity under the tested
missingness mechanism in the supported reference population. A dataset-wide
robustness claim additionally needs adequate coverage of the high-loss and
long/clustered-gap contexts; otherwise state those limits explicitly.

## Wall-excluded results are a parallel sensitivity analysis

The log also proposes excluding wall-associated episodes. For tortuosity,
turnback, alignment and agarose, define the exclusion at the episode/support
level, use production wall detection, reapply minimum counts, and show both
within-common-fly changes and changes in retained subjects. Match the existing
manuscript settings before comparing cohort contrasts. Do not select a policy
because it preserves the preferred result.

Wall contacts per interval and wall-positive path length do not have directly
comparable wall-free analogs: removing wall-positive intervals makes the former
zero by selection and leaves no qualifying intervals for the latter. A wall-free
path-length metric would answer a different question and needs a new label.

Wall exclusion is not independent validation of missingness effects: wall
classification itself is tracking-sensitive, and unobserved wall contact can
remain. If useful, compare eligibility under reference-fixed wall labels and
redetected wall labels to distinguish numeric effects from selection effects.

## Suggested next implementation boundary

Implement reporting/denominator repairs and the all-subject support inventory
first, then review its coverage before committing to a larger simulation budget.
Keep the current pilot as a reproducible checkpoint. Larger single-gap runs,
clustered masking, reference reconstruction and wall-excluded analyses are
separate follow-on changes with explicit assumptions and validation.
