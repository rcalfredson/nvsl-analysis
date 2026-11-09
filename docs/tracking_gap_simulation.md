# Empirical tracking gaps and artificial-gap experiments

Component two starts from the committed fixed-episode audit and its reviewed
artifacts. It characterizes actual missing tracking, then hides observed data to
benchmark interpolation and episode detection against an observed reference.
This first implementation is a conditional single-gap pilot, not an estimate of
the bias in manuscript cohort means.

## Decisions for the first experiment

- Use experimental flies in the same three flat-large cohorts and preserve the
  component-one source roster. Profile all of T2; draw simulation donors only
  from bracketed gaps contained, with their endpoints, in T2 SB2–5. This is the
  shared starting window for the six metrics, rather than the full agarose
  Pre/T1/T2 time course.
- Preserve each donor gap's exact frame length and frame rate. Keep cohorts and
  flies separate: a donor is transplanted only within that same fly's recording.
  No synthetic missing interval contains originally missing reference data.
- Require finite original x, y, width, height, and orientation throughout the
  reference stretch. The full T2 audit found these five fields missing together
  at lost frames and finite together at tracked frames. The simulation masks
  all five, then calls the production interpolation. Width/height stay missing;
  production x/y/theta interpolation and missing-frame policies are preserved.
- Match the observed pre-gap and recovery endpoints. Wall-distance categories
  are <1 mm, 1–<3 mm, and >=3 mm. Both endpoint categories must match. Endpoint
  reward-center distances, nearest signed agarose-well-boundary distances, and
  the endpoint-to-endpoint chord must each agree within 2 mm. These coarse
  spatial constraints are prespecified pilot settings, not validated thresholds
  or proof that the hidden behavior matches. Interior path geometry is never
  used to characterize a real donor gap because it is unknown.
- Keep two seconds of observed context beyond each bracketing endpoint. Report
  unsupported donor contexts without shortening gaps, relaxing tolerances, or
  borrowing another cohort's trajectories.
- Sample subjects across recording-loss strata, choosing distinct videos within
  each cohort. Sample donor gaps separately within <=0.5 s, 0.5–2 s, 2–10 s,
  and >10 s bins. This deliberately tests a range of durations; it does not
  reproduce the population frequency of each bin.
- Use separate masking trials with one artificial gap each. Trials can reuse
  subjects and reference episodes and are not independent biological replicates.
  Real gaps are clustered;
  this isolates a local response before a later multi-gap/burden experiment.
  Recorded reward times remain fixed. The experiment does not simulate changes
  to reward delivery, animal behavior in response to different stimulation,
  or the SLI eligibility gate.

## Profile actual tracking loss

```bash
MPLCONFIGDIR=/tmp/gap_sim_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/simulate_tracking_gaps.py profile \
  --output-dir results/tracking_gap_simulation/profile_2026-11-08
```

The default source is
`results/episode_tracking_loss/flat_large_2026-11-08_reviewed`; override it with
`--source-dir`. Raw .trx/.data hashes must agree with component one's manifest.
Existing output directories cannot be overwritten.

Profile outputs:

- `gaps.csv`: every original gap intersecting T2, full original duration,
  T2-overlap frames, eligibility as a primary donor, and observed-endpoint
  geometry. A duration bin concerns the full gap, not its clipped T2 overlap.
- `subjects.csv`: explicit identities, primary-window availability, loss,
  number of usable donors, and longest fully observed reference stretch.
- `fully_observed_runs.csv`: original-data support for reference measurements.
- `missing_fields.csv`: missingness of all five tracker fields at lost and
  tracked frames.
- `summary.csv`: pooled gap-duration distributions, duration-weighted lost-time
  contributions, and an optimistic length-only support fraction. Spatial and
  episode constraints can reduce that support substantially. The lost-time
  weighting uses full donor gap durations; for T2-intersecting gaps that span a
  boundary, use `t2_overlap_frames` for a strictly T2-only frame total.

Initial full profile: 196 flies, 102 videos, 66,933 T2-intersecting gaps, and
39,409 primary donors. The pooled primary-donor medians are about 0.27, 0.27,
and 0.40 seconds for Control, PFNd>Kir, and AR; their 95th percentiles are about
2.67, 3.07, and 6.0 seconds. In AR flies, the 2.1% of primary gaps longer than
10 seconds account for 28.8% of primary lost time. Gap-weighted distributions
and the median of per-fly medians answer different questions; both are exported.

## Matched counterfactual pilot

```bash
MPLCONFIGDIR=/tmp/gap_sim_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/simulate_tracking_gaps.py pilot \
  --profile-dir results/tracking_gap_simulation/profile_2026-11-08 \
  --output-dir results/tracking_gap_simulation/pilot_targeted_2026-11-08 \
  --subjects-per-cohort 3 --gaps-per-bin 2 --episode-targeted
```

The command records the selected subjects, strata, exact donors, random seed,
candidate counts, matching differences, and every unsupported attempt. Source
and code snapshots, including the compiled boundary-contact module, are captured
before calculations and checked for changes before writing results.

There are two placement arms:

1. `spatial_matched`: choose uniformly among placements satisfying duration,
   observed-data context, and spatial constraints. Placement does not depend on
   whether it overlaps a complete metric episode.
2. `episode_targeted:<metric>`: additionally require overlap with an originally
   fully observed reference episode's support for that metric. This is a
   conditional susceptibility test. Turnback/alignment targeting currently
   uses support across the radius pairs together; each pair is evaluated and
   reported separately, and a match for 13/15 mm does not establish support at
   the smaller radii. Alignment targeting covers the bout/support, not only
   the sampled heading window.

Donor selection is shared across arms within a run. No unsupported donor is
silently replaced. Arms must be summarized separately; adding targeted trials
does not make the spatial arm more representative.

## Metric and detection comparison

Before injecting gaps, the script verifies wall/all/edge counts and full path
lengths interval-by-interval against the frozen Figure 4k CSV. It also verifies
the component-one scalar episode counts/measurements and pooled agarose counts
and numerators for each selected fly.

The wall detector calls `EllipseToBoundaryDistCalculator` with the production
wall/all/edge thresholds and lost-frame/censoring policy. It runs directly,
without reading or writing production caches. The other metrics reuse the
production functions already called by component one.

Realistic masking measures the combined effect of lost tracking, interpolation,
unavailable body dimensions, and the detector's missing-data/censoring rules.
A wall-count change cannot be attributed solely to the geometry of the x/y
chord. Fixed-boundary path-length/tortuosity comparisons isolate the
coordinate-dependent numeric response more directly; mechanistic ablations
would be needed to separate the detector-policy effects.

Each trial evaluates the same fully observed parent run before and after
injection. Complete episode **support**, including local transition/heading
frames, must lie within that run. Episodes are not clipped to make them complete.
Real gaps outside the run remain identical in both conditions; inherited
hysteretic state is not independently validated. The reference is the observed
tracking result, not manually established biological ground truth.

`redetection.csv` recomputes all six metrics after masking/interpolation. It
exports episode counts, local means, temporal-overlap associations, split/merge
diagnostics, and timing/value changes. Whole-run means without five episodes
are explicitly marked as unfiltered diagnostics; they are not manuscript
per-fly values. Episode counts and eligibility/availability changes remain
visible even if a scalar becomes undefined.

`fixed_boundaries.csv` isolates measurements on the original episode boundaries:

- Tortuosity is recomputed using the original full-path interval.
- Wall-event counts are reassigned to the original reward intervals after
  rerunning contact detection.
- Length is recomputed for originally wall-contact-positive intervals,
  preserving that original qualification for this comparison only.
- Alignment uses the original successful re-entry episode and existing heading
  quality rule; it can become unavailable.

Turnback and agarose ratios use the redetected episodes rather than invented
outcomes on frozen boundaries. For all metrics, the redetection comparison also
captures changes in which episodes qualify. Fixed-boundary errors should be
examined specifically where `support_intersects_injected_gap` is true, to avoid
dilution by unaffected bouts in the same run.

Temporal overlap is an association rule, not event identity ground truth.
Non-overlapping reference/simulated episodes are not automatically labeled
biological misses or false positives. Repeated trials can reuse episodes and
reference stretches; pooled episode/trial rows are not independent samples.

## Render the combined report

```bash
MPLCONFIGDIR=/tmp/gap_sim_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/report_tracking_gap_simulation.py \
  --profile-dir results/tracking_gap_simulation/profile_2026-11-08 \
  --pilot-dir results/tracking_gap_simulation/pilot_targeted_2026-11-08 \
  --output-dir results/tracking_gap_simulation/report_2026-11-08
```

The report exports standalone PNG/PDF plots, matching coverage by arm/bin/cohort,
gap-duration and recurrence summaries, and conditional measurement/detection
tables. Undefined cells remain explicit. The report records input CSV hashes
and its own source hash.

The initial expanded pilot uses nine subjects from nine videos and 55 unique
donor gaps. Of 385 donor/arm attempts, 79 matched (29 spatial-only and 50
targeted); 306 remained unsupported. No sampled >10-second donor matched in the
spatial arm. These numbers characterize the pilot's support, not the fraction
of all genuine gaps that are safe or problematic.

For a companion report with readable summary tables, fixed-boundary error
statistics, redetected local mean changes, and separate cohort/placement-arm
breakdowns, render the existing CSVs without rerunning simulations:

```bash
/home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/summarize_tracking_gap_pilot.py \
  --pilot-dir results/tracking_gap_simulation/pilot_targeted_2026-11-08 \
  --existing-report-dir results/tracking_gap_simulation/report_2026-11-08 \
  --output-dir tmp/tracking_gap_pilot_summary_2026-11-08_v2
```

The companion exports its summary tables as CSVs and records input/code hashes.
Fixed-boundary errors use only gap-exposed reference measurements; numeric
statistics exclude unavailable measurements and report their count separately.
Local mean changes require both means to be defined and use unfiltered parent-run
means. Ratio changes are expressed in percentage points. Trial means weight
trials equally, not flies. Repeated episode appearances are not independent
replicates. The numerical change tolerance is 1e-9 in the reported units.

### Denominator-aware reports (9 November 2026)

The combined renderer now creates an `audited_pilot/report.html` companion
automatically. Existing report directories remain immutable; use a new directory:

```bash
MPLCONFIGDIR=/tmp/gap_sim_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/report_tracking_gap_simulation.py \
  --profile-dir results/tracking_gap_simulation/profile_2026-11-08 \
  --pilot-dir results/tracking_gap_simulation/pilot_targeted_2026-11-08 \
  --output-dir results/tracking_gap_simulation/report_repaired_2026-11-09
```

The audited report adds:

- `trial_details.csv`: one row per attempt/metric/radius, including unsupported
  attempts. Records video/fly identity, gap and evaluation duration, episode
  counts, ratio numerators reconstructed from saved binary episode outcomes,
  local values, changes, eligibility transitions and saved episode intervals.
  Original counts and means are validated against `episodes.csv` before export.
  Numerators apply only to turnback and agarose ratios. Ratio values remain
  fractions; their displayed changes are percentage points. Frame rate is
  recovered from the saved donor frame count and duration.
- Separate all-defined-means and minimum-five-in-both mean-change summaries,
  also available by cohort for targeted placements. Eligibility transitions and
  availability counts retain the full evaluation pool, including cases removed
  from the filtered numeric comparison. These are local diagnostic minima, not
  recording-level admission decisions. Five episodes does not establish precision.
- Denominator strata (0, 1–4, 5–9, 10–19, 20+) grouped by original episode count,
  placement arm, cohort, metric and radius. These are descriptive bins, not new
  eligibility rules. Repeated episodes/trials remain repeated observations.
- Explicit statuses for arms not requested, no matched placements, missing
  evaluation rows, no complete episodes, and no exposed reference measurements.
  Unsupported counts/values in the trial audit remain undefined, distinct from
  an evaluated zero-episode denominator. An entirely unsupported pilot can have
  no redetection/fixed/episode files; both renderers still produce reports.
  Inconsistent evaluated episode records raise an error rather than inventing
  numerator/denominator values.

The HTML trial-level table shows the evaluated corresponding targeted-arm rows;
the downloadable audit retains other arms and unsupported rows. Filtering to
five episodes affects numeric local-mean comparisons, not fixed-boundary episode
errors, matching coverage, or counts of lost/gained eligibility. No simulations
are rerun and no population/cohort-bias estimates are added by these repairs.

For the existing pilot, agarose has 11 defined targeted local comparisons,
eight meeting the minimum in both conditions. The exploratory maximum is
50 percentage points (1/2 to 0/1); the filtered maximum is about 7.14 points.
No targeted wall-positive path-length local comparison meets the five-episode
minimum in both conditions. Its filtered numeric effects remain undefined.

## Limits and expansion

See [the review and staged expansion plan](tracking_gap_simulation_expansion_plan.md)
for reporting repairs, a reference-support inventory, and the conditions needed
to connect simulated missingness to manuscript-level results. This plan has not
yet been implemented.

Single-gap tests do not measure cumulative recording-level bias. Fully tracked
stretches can systematically differ from periods of tracking loss, even when
observable endpoint geometry matches. The long-gap tail and short observed
stretches are particularly important for wall-associated trajectories. Additional
observed/retracked data or manual review may be necessary where reference
episodes are unavailable.

Before interpreting a broader experiment as representative, expand subject and
donor replication; check support separately by metric, radius, cohort, and gap
duration; characterize clustered gaps; and add multi-gap trials with an explicit
loss-burden model. Never fill unsupported cells with zero error or extrapolate
from only the successfully matched short gaps to the entire recording.

## Tests

```bash
MPLCONFIGDIR=/tmp/gap_sim_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python -m pytest -q \
  test/test_tracking_gap_simulation.py test/test_episode_tracking_loss.py \
  test/paper_metrics/test_turnback_home_vector_alignment_truth.py \
  test/paper_metrics/test_turnback_ratio_truth.py \
  test/paper_metrics/test_agarose_avoidance_truth.py
```

These checks include observed-context protection, exact donor-duration handling,
matching constraints, explicit targeted placement, split/merge diagnostics,
masking all tracker fields without changing source arrays, and a synthetic
example in which interpolation hides an entire turnback.
