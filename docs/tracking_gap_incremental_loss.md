# Additional-loss sensitivity in manuscript measurement windows

This experiment compares each current reconstructed trajectory with the same
trajectory after one additional artificial tracking gap. Existing real loss
and its interpolation remain in both conditions. It measures **incremental
sensitivity conditional on the current reconstruction**. It does not estimate
the error already caused by real lost tracking, reconstruct hidden paths, or
provide corrected manuscript values.

The first batch covers the three pooled between-reward metrics in the flat
large chamber: tortuosity, wall contacts per reward interval, and path length
of wall-positive reward intervals. Turnback/alignment and the actual agarose
chamber comparisons remain separate follow-on stages.

## Population and production behavior

The reviewed episode-loss audit freezes admission, recorded reward times,
the final paired-SLI gate and source scalar values. All flies passing those
gates for any of these metrics are attempted, including flies below the
metric's current episode minimum. Baseline counts, sums and values are checked
against the reviewed audit; currently defined values must also reproduce the
manuscript archive. Wall interval counts and path lengths are independently
checked against the frozen per-interval export.

Every simulation masks original x, y, width, height and orientation at the
selected frames, then invokes production interpolation and wall detection.
Detection starts at recording start, preserving preceding state. Raw tracker
arrays and production analysis caches are not modified. The original missing
mask must remain unchanged outside the artificial interval.

Measurements use the complete T2 SB2–5 window and production episode assignment.
Tortuosity retains full reward segments assigned by maximum bucket overlap;
segments are not trimmed at the evaluation or observed-run boundaries. Wall
intervals follow the collector's reward-frame sanitization and region-start
assignment. The five-episode minimum is reapplied to the full pooled window.
There is **no local five-episode placement requirement**. Newly eligible and
newly undefined measurements are retained as composition outcomes.

Recorded rewards and source gates stay fixed: this is an offline measurement
experiment, not a simulation of altered real-time reward delivery or behavior.

## Matching and sampling

The exhaustive support census supplies same-fly real donors wholly bracketed
inside T2 SB2–5. A placement preserves its donor's exact frame duration and
lies within one originally fully observed run, with observed endpoints and
two seconds of observed padding beyond each endpoint. Endpoint nearest-wall
bins must match; before/after reward-center and signed nearest-well distances
and endpoint chord must agree within 2 mm. Missing interiors of real donors
are never used as known trajectories.

The sampled donor inventory and candidate counts are checked against the
census. Its raw sources must match the reviewed audit and current files.
Only the unchanged Ctrl, PFNd>Kir and antennae-removed pooled cohorts are reused
from that census; this does not endorse its historical EPG membership or
replace the corrected agarose source audit.

Two distinct arms are reported:

- **Empirical spatial:** uniformly sample a donor from that fly's spatially
  supported real gaps, then uniformly sample a matching start. This preserves
  empirical within-fly gap frequencies *conditional on support*. Report all
  three source-gated metrics for the same added gap, including currently
  undefined ones. Flies receive equal weight in cohort means; this is not a
  pooled-gap-frequency estimate across flies.
- **Episode targeted:** for each currently eligible metric, randomly permute
  its supported duration bins and cycle through them across repeats. Sample
  donors uniformly within the selected bin and placements uniformly among
  matches overlapping complete observed episode support for that metric.
  This intentionally balanced stress test is separate from empirical sampling.
  Donor probabilities are marginal over the random bin permutation; successive
  repeats are dependent by design.

Each repeat introduces exactly one gap, starting afresh from the original
tracking; gaps are not accumulated across repeats. Unsupported donors and
flies remain in coverage tables and are not assigned zero responses. Padding,
matching tolerance, donor duration and fly identity are never relaxed to find
a placement. Complete observed episode exposure is recorded per trial; a
false flag is not proof of no effect on the full measurement.

Four repeats are an initial bounded batch, not a guarantee of adequate Monte
Carlo precision or rare-event coverage. Inspect repeat variability before
choosing further replication. This dose does not reproduce total original
loss, temporal clustering or repeated-gap interactions.

## Contrasts and composition

Contrast orientation is PFNd>Kir − Ctrl, antennae removed − Ctrl, and antennae
removed − PFNd>Kir. Compute each cohort's mean with equal weight per eligible
fly in each repeat, then average contrasts across repeats. Repeated masks do
not increase the biological sample size. Undefined repeat contrasts remain
undefined rather than being silently removed.

The report separates:

1. **Support selection:** supported baseline contrast minus the original full
   manuscript contrast, before any artificial loss. Unsupported subjects are
   excluded from the simulation contrast, not assumed unchanged.
2. **Common-fly change:** after minus before contrast among exactly the flies
   eligible in both conditions in that repeat.
3. **Composition change:** the baseline shift when removing flies losing
   eligibility, plus the post-mask shift when adding newly eligible flies.
   Total contrast change equals common-fly change plus composition change.

Relative contrast change divides the signed change by the absolute supported
baseline contrast. It is a diagnostic that becomes unstable near zero, not an
acceptability threshold. Direction preservation does not establish magnitude
preservation or equivalence.

Percentile confidence intervals use recording-cluster bootstrap draws within
cohort. Each sampled video's flies and all their repeats share its bootstrap
weight. The repeat minimum/maximum shows a separate source of mask variability.
These intervals remain conditional on available support and the tested dose;
they do not quantify uncertainty about genuinely hidden paths. No scientific
equivalence threshold has been selected.

## Run and resume

```bash
MPLCONFIGDIR=/tmp/gap_sim_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/simulate_incremental_tracking_loss.py \
  --output-dir results/tracking_gap_simulation/incremental_pooled_2026-11-09 \
  --repeats 4 --bootstrap-draws 1000 --seed 20261109 --workers 4
```

Use an empty output directory. A deterministic smoke run can add
`--limit-per-cohort 2 --repeats 1 --bootstrap-draws 30` with a separate output.
For an interrupted run, repeat its original command with `--resume`.
Completed results cannot be overwritten. Source/code hashes, settings, raw
files, note presence and video size/mtime must agree with each checkpoint.
Worker count changes execution only; identity-based random streams preserve
the same donor/placement choices regardless of processing order.

For a compact readout with descriptive column labels and formatted numbers:

```bash
MPLCONFIGDIR=/tmp/gap_sim_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/report_incremental_tracking_loss.py \
  --run-dir results/tracking_gap_simulation/incremental_pooled_2026-11-09 \
  --output-dir tmp/incremental_tracking_loss_readout_2026-11-09
```

The derived readout adds no new estimates or filtering. It formats the source
coverage, dose, contrast and composition summaries and records their hashes.

Validated subject checkpoints are written atomically. Relevant raw inputs,
archives, reviewed subject table, support census and metric dependencies are
hashed; source size/mtime is checked during the run. Video containers are
tracked by size/mtime for metadata, without hashing the full video contents.

## Outputs

- `report.html`: cohort coverage, contrast direction/magnitude and composition.
- `coverage.csv`: one row per attempted fly/arm/metric; original loss, baseline
  eligibility, donor counts/time, supported donor counts/time and duration bins.
  Coverage fractions concern bracketed within-window donors, not all lost time.
- `trials.csv`: each supported fly/repeat/metric, donor and placement bounds,
  selection probabilities, observed parent, exact added dose, original loss,
  complete observed episode exposure, before/after counts, eligibility and
  values, the post-mask episode-value sum, and changes in exact-boundary
  episode membership. Boundary
  membership changes are not automatically missed behavioral events.
- `cohort_summary.csv`: full manuscript and supported means, gated and supported
  fly/video counts, donor coverage, undefined trials and unique flies gaining
  or losing eligibility. A fly may gain/lose eligibility in only some repeats.
- `contrasts.csv`: full and supported contrasts, changes and decomposition,
  conditional bootstrap intervals and number of defined bootstrap draws,
  relative changes, direction and repeat ranges.
- `checkpoints/`, `provenance.json`: replayable settings, identities, source
  snapshots and run metadata. Historical pilot outputs remain unchanged.

## Initial expanded batch, 9 November 2026

The completed run attempted 151 source-gated flies in 92 videos. It generated
2,216 artificial-gap trials in 149 supported flies/91 videos, yielding 3,408
per-metric comparisons across four repeats per supported arm. A serial replay
of a high-loss fly exactly reproduced its parallel checkpoint. Baseline
measurements and sampled candidate counts passed source checks throughout.

The largest absolute **average relative contrast change**, across the three
specified cohort contrasts, was:

| Metric | Empirical spatial | Episode-targeted stress test |
| --- | ---: | ---: |
| Tortuosity | 0.014% | 0.181% |
| Wall contacts per interval | 0.311% | 0.564% |
| Wall-positive path length | 0.286% | 32.327% |

These columns use different arm-specific supported populations; they are not
two estimates of the same missingness distribution. All 18 average contrast
directions, and the individual repeat directions, were preserved. This does
not establish equivalence or quantify existing interpolation error.

The targeted path-length result illustrates why composition must be retained.
One PFNd>Kir fly and one antennae-removed fly had exactly five wall-positive
intervals at baseline. Added loss reduced that count to four in 4/4 and 3/4
repeats respectively, rendering their measurements undefined. Independent
replays confirmed removal of one wall-positive interval in each case. The
PFNd>Kir-minus-control supported contrast changed from 68.82 to 46.57 mm
(-22.25 mm, -32.33%); -20.86 mm of that change was eligibility composition and
-1.38 mm was the common-fly metric change. Its conditional video-bootstrap
interval for total change was approximately -67.04 to +0.83 mm. This observed
case is not evidence for a precise population effect of that magnitude.

The empirical spatial arm had no eligibility gains or losses. It supported
69–79% of donor gaps but only 31–47% of donor lost time across cohorts.
Episode-targeted path-length support was still narrower: 49/50 original
eligible controls, 40/46 PFNd>Kir and 28/42 antennae-removed flies; only about
1.8% of antennae-removed donor lost time had such placements. The largest
injected gap in any arm was 14.67 seconds (0.61% of the full window); the
empirical arm's median was a single 0.133-second frame. Full-burden and
clustered-gap sensitivity remain untested.

Validation: 68 focused tests passed, including unsupported report cells,
eligibility gain/loss decomposition and recording-cluster resampling. Results
are in the run and derived-readout directories in the commands above.
