# Tracking-loss sensitivity of fixed episodes

Component one of the bout-metric investigation measures exposure to actual
tracking gaps and sensitivity to excluding detected episodes that touch them.
It does not reconstruct behavior during lost frames or estimate a correction.
The artificial-gap experiment is component two and remains a separate task.

## Run

From the repository root, using the analysis environment:

```bash
MPLCONFIGDIR=/tmp/episode_tracking_loss_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/audit_episode_tracking_loss.py \
  --output-dir results/episode_tracking_loss/flat_large_2026-11-08 \
  --bootstrap-draws 2000
```

Output must be a new or empty directory. Existing results are never overwritten.
For a pilot, add `--limit-per-cohort 1`; `--cohort ctrl` selects just the control
cohort. A pilot expands each selected video to all its archived subjects so the
agarose baseline can be validated. Pilot summaries must not be interpreted as
cohort estimates. `--seed` controls deterministic bootstrap resampling.

## Scope and frozen references

The first implementation covers experimental trajectories from three flat
large2 cohorts: Ctrl>Kir, PFNd>Kir, and antennae-removed control. Yoked tracking
loss is recorded as a diagnostic; yoked metric values are not analyzed here.
Existing paired SLI eligibility is frozen rather than recalculated after
excluding episodes. Empty tracking and previously excluded flies cannot be
admitted by this analysis. Agarose-chamber cohorts are outside this run.

| Metric | Reference | Window and episode minimum |
|---|---|---|
| Tortuosity | Figure 4j scalar archives | T2 SB2–5; 5 finite full-trajectory path/max-radius values |
| Wall contacts per interval | Figure 4k archives and episode CSVs | T2 SB2–5; 5 intervals, including zero-contact intervals |
| Wall-contact trajectory length | Figure 4l scalar archives; Figure 4k shared interval records | T2 SB2–5; 5 intervals with an assigned contact and finite full length |
| Turnback ratio | Figure 4n fixed-pair bundles | T2 SB2–5; 5 bouts separately at 3/5, 8/10, 13/15 mm |
| Home-vector alignment | Figure 4q scalar archives | T2 SB2–5; 5 successful bouts with valid alignment per radius pair |
| Agarose avoidance | Notebook's April-tagged flat-large NPZs | Pre last 10 min and individual 10-min buckets in T1–T2; 5 bouts per cell |

The first five metrics preserve experimental SB5 presence and finite paired
final SLI, including negative and zero SLI. Each panel uses its own eligibility
bundle. Agarose preserves its independent source pool without a final-SLI gate.
Absent agarose buckets are explicit zero-count/nonfinite cells, not zero ratios.

Production metadata supplies reward geometry and actual-reward synchronization.
Production interpolation, distance calculation, between-reward segment iterator,
turnback detector, agarose detector, and alignment estimator are called directly.
Contact detection is not rerun: the Figure 4k episode records freeze its output.
The reconstructed baseline must agree with Figure 4l values before its shared
interval records are used for length sensitivity.

Important conventions are preserved:

- Tortuosity uses maximum-overlap bucket assignment. Its path excludes the
  ending reward coordinate under the existing half-open metric convention.
- Wall events belong to the interval containing their **start**, and length
  uses the production `distTrav(start, stop)` forward-step convention, including
  the step to the ending reward coordinate. This intentionally differs from
  tortuosity's endpoint policy.
- Turnback successes have `stop` equal to the re-entry frame; the generic
  `event_frame` is the preceding outside frame. The actual re-entry is audited.
- Alignment uses mean heading, two samples on either side of the crossing,
  intersection home-vector anchor, and at most one originally missing sampled
  heading frame for its baseline. Exact sampled frame IDs enter the audit.
- Agarose bouts are assigned by entry frame; the rest of the bout can extend
  beyond that bucket. Entire bouts are examined, not clipped to the bucket.
- Original x-NaN masks define tracking loss, matching the production convention.
  Initial missing prefixes remain in window/episode exposure but are omitted
  from the whole-recording loss percentage, matching the existing survey.

## Baseline checks and identity

Scalar archives are joined by video/fly identity through explicitly recognized
historical ID formats. Ambiguous basenames, duplicate identities, changed
archive hashes, scalar mismatches, and turnback count/numerator mismatches abort
the run. Wall-episode inputs must have unique, consecutive reward intervals,
correct experimental identities, valid window bounds, and a row count matching
the exported pooled episode count. This catches duplicate or truncated exports
that could otherwise reproduce a mean by coincidence.
Frame calibration and SB5 synchronization are also checked indirectly
through scalar reproduction and explicitly through the archived SB5 stop frame.

Legacy agarose bundles store repeated video paths but omit fly IDs. The script
does **not** assign those rows to flies using order or numerical similarity.
Instead, it reconstructs explicitly keyed flies from raw tracking and production
geometry, then compares each video's unordered set of complete count/numerator
vectors across Pre and T1–T2 against the archive. Archived ratios are separately
checked against archived counts and the five-bout minimum. The individual-fly
agarose audit is newly computed; its validation level is video-wide multiset
agreement, not an independent archived fly-ID match. `baseline_validation`
records this distinction. A mismatch aborts before results are written.

The analysis reads already admitted subjects. Interpolation uses a permissive
admission threshold internally so it cannot silently remove archived subjects;
it does not expand the baseline pool. Every resulting metric is checked against
the source before any sensitivity result is written. Note files are hashed for
provenance; their admission effects remain frozen in the source pool.

## Policies

1. `baseline`: retain the existing finite episode measurements.
2. `complete_episode`: retain only episodes whose original half-open metric
   interval contains no lost frames.
3. `complete_support` (primary): also require the frame immediately preceding
   the episode and the frame at its exclusive stop to be tracked. Alignment
   additionally includes all frames sampled for heading. Support is clipped at
   recording boundaries.

The support rule includes the missing re-entry frame that could otherwise be
missed by a half-open successful-turnback interval, and the ending coordinate
used by wall-contact trajectory length. Exclusion always removes whole episodes;
points on opposite sides of a gap are never concatenated as a new trajectory.

These are **local support** rules. Hysteretic circle state and wall-contact state
can depend on earlier history. Neither rule guarantees independence from every
earlier gap, detects hidden bouts, or validates tracking identity. Artificial-gap
redetection and targeted video review address different parts of that problem.

The metric-specific episode minimum is reapplied after exclusion. Ratios retain
both numerator and denominator, and per-fly means retain equal fly weight.

Exposure percentages describe the episodes that qualify for each baseline
metric. In particular, alignment exposure includes only successful bouts with
a finite alignment under the existing heading-quality rule. It does not count
bouts already rejected for excessive interpolation in their heading samples.
Low exposure or a small exclusion effect is therefore not evidence that every
detected turnback has a reliable alignment measurement.

## Outputs and interpretation

- `report.html`, `report.md`, `sensitivity.png`, `sensitivity.pdf`: primary
  comparison and limits. The short table shows agarose T2 SB5; the CSV covers
  its whole Pre/T1/T2 time course.
- `episodes.csv`: source identity, original measurement, measurement/support
  frame bounds, lost fraction, intersecting gap count and duration, missing
  outcome frames, and exact heading samples.
- `subjects.csv`: all policies, baseline and retained episode counts, frozen
  eligibility, inclusion/exclusion reasons, baseline validation level, and
  experimental/yoked whole-recording loss diagnostics.
- `summary.csv`: fly and video n, attrition, episode exposure, mean, SD, median,
  95% confidence interval, and changes from baseline. `common_fly_shift`
  compares the same retained flies before and after episode exclusion;
  `mean_shift_from_full_baseline` also includes sample attrition.
- `contrasts.csv`: PFNd-minus-control and AR-minus-control differences, changes
  from baseline, confidence intervals, and numbers of valid bootstrap draws.
- `episode_strata.csv`: episode-weighted duration and value summaries for
  affected versus unaffected bouts among baseline-included flies. These are
  selection diagnostics, not fly-level effect estimates.
- `window_coverage.csv`: original tracking coverage in T1, T2, and the selected
  T2 SB2–5 window, including initial missing prefixes.
- `gaps.csv`: original gap bounds and durations plus observed endpoint
  positions and wall distances, suitable for planning artificial-gap experiments.
- `provenance.json`: invocation, settings, source content hashes, size/mtime,
  relevant code hashes, and Python/NumPy versions. Code is hashed before
  collection, and both code and data size/mtime are rechecked before writing.

Confidence intervals resample videos within cohorts, keeping flies together and
using the same draws for a metric's baseline and exclusion policies. Contrasts
use independent cohort draws. Empty draws remain nonfinite and are counted;
zero retained flies produce undefined statistics. Uncertainty is exploratory
when few videos remain. No cutoff is selected using significance.

A large change does not measure interpolation bias: affected bouts can be
systematically longer or more wall-associated. Report exposure, common-fly
change, attrition, and the behavioral composition diagnostics together. Component
two should test measurement changes and detection changes separately, using
realistic gap durations and placement rather than independently dropped frames.

## Review and commit boundary

Component one is an independent change suitable for committing before the
artificial-gap work. Generated results under `results/episode_tracking_loss/`
and the standalone HTML copies under `tmp/` are ignored; source, tests, and
documentation belong in the commit. Preserve the result directories separately
as analysis artifacts. Existing manuscript exports are read-only inputs.

Run the focused checks with:

```bash
MPLCONFIGDIR=/tmp/episode_tracking_loss_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python -m pytest -q \
  test/test_episode_tracking_loss.py \
  test/test_audit_tracking_gap_endpoints.py \
  test/paper_metrics/test_turnback_home_vector_alignment_truth.py \
  test/paper_metrics/test_turnback_ratio_truth.py \
  test/paper_metrics/test_agarose_avoidance_truth.py
```

The review additionally validates the full frozen wall-episode inputs and reruns
the three-cohort audit into a new directory ending in `_reviewed`. Compare the
numeric summaries with the original run before interpreting any apparent change.
The documented limitations of local support and unkeyed agarose references
remain intentional limits, rather than validations supplied by this component.

Review validation on 2026-11-08: 92 focused tests passed, all 22,915 frozen
wall-episode rows passed structural validation, and the full reviewed run
verified 3,920 subject/metric/window baselines. The summary, contrasts, subject
audit, window coverage, and episode-strata tables matched the original run
exactly; episode and gap CSV content hashes also matched. The reviewed run's
code hashes matched the working source. The review safeguards did not change
the reported numerical results.
