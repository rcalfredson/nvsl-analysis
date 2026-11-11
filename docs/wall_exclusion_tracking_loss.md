# Wall exclusion and tracking-loss exposure

This audit compares the existing measurements with measurements after removing
whole episodes associated with detected wall contact. It measures actual-gap
exposure, retained episodes/flies, per-fly values and cohort contrasts. It does
not introduce simulated loss, remove individual frames from trajectories, or
estimate the hidden paths in genuinely lost tracking.

The first implementation follows the existing wiki scope: Ctrl>Kir, PFNd>Kir
and antennae-removed experimental flies in the flat large chamber. It covers
between-reward tortuosity, dual-circle turnback ratio, turnback home-vector
alignment and physical-site agarose avoidance ratio. Wall-contact event count
and wall-positive path length are excluded: removing contact makes their
wall-free analogues trivial or undefined.

## Canonical readouts

The definitive settings are the recipes in
`notebooks/manuscript_figure_panels.ipynb` and the corresponding historical
recipes in `notebooks/graphpad_exports.ipynb`. The manuscript recipes migrate
and identify the current Figure 4j, 4n and 4q readouts; the graphpad notebook also
contains all-fly, top-learner and older alternatives that are not the primary
selection used here. Both notebooks are hashed in this run's provenance.

- **Figure 4j:** full between-reward path length / maximum radius from reward
  center, pooled T2 SB2–5, at least five trajectories, nonwalking frames retained.
- **Figure 4n:** turnback ratio for absolute 3/5, 8/10, 13/15 mm radii; fully
  contained bouts in T2 SB2–5; five episodes independently at each radius.
- **Figure 4q:** successful-turnback mean cos(heading − home vector), the same
  radius pairs and window; mean heading with two samples per side, intersection
  anchor, at most one interpolated heading sample; five valid successes per radius.
- **ED17b/c/e flat-chamber components:** physical-site agarose avoidance ratio,
  hysteretic detection, 1 mm outer offset, Pre and T1/T2 10-minute cells,
  five episodes per cell, independent source admission without a final-SLI gate.

The last item is not Figure 5g's percent-time reduction. Agarose chamber
comparisons, other genotypes and AUC effects remain outside this audit.

The reviewed episode-loss artifacts freeze source admission, target-bucket
presence and panel-specific finite paired final-SLI gates. Zero and negative
SLI remain admissible. The raw source files, source NPZs and wall-interval CSVs
must match the original audit's hashes. Original episodes are redetected with
production functions: identities, values, support bounds, heading missingness
and lost-frame counts must match the reviewed episode table. Subject-level
counts, sums and measured values are separately checked. Missing or changed
inputs stop the audit; production tracking caches are not written.

## Exclusion policies

`wall_allowed` reproduces the original calculation. The primary comparison is
`exclude_episode_contact`, removing any whole episode that overlaps production
wall/all/edge contact regions. Production wall detection excludes lost frames;
absence of detected contact is therefore not independent evidence of being
away from walls during a gap.

The existing options have important interval conventions:

- Tortuosity checks contact within the segment's overlap with the measurement
  window, matching its production windowed wall mask. A segment can extend
  outside the pooled window under maximum-overlap bucket assignment; no
  coordinates or metric values are trimmed by this audit.
- Turnback ratio and alignment use the shared production half-open episode
  overlap helper. Re-entry/adjacent heading frames outside that interval are
  not added to the primary wall criterion.
- Agarose does not have the same global wall-contact exclusion option in its
  canonical recipe. Here it receives an explicitly new audit filter using the
  shared half-open episode rule. Wall-facing-entry/arc filters are different
  behavioral restrictions and are not substituted for wall-contact exclusion.

The secondary `exclude_support_contact` policy rejects contact anywhere in the
full local measurement support, including adjacent boundary frames and sampled
heading frames. This tests sensitivity to the primary option's interval
convention; it is not silently substituted for the established options.

The baseline wall mask is required, never assumed absent because no contacts
were found. Retained episodes preserve their exact original values and bounds;
only episode membership and subsequent per-fly eligibility change. The
five-episode requirement is reapplied over the original production window,
not an isolated fully observed stretch. Source gates are fixed.

## Exposure and comparisons

Primary loss exposure means at least one original x-NaN frame in the complete
local support. Episode-interior exposure is retained separately. Exposure is a
fraction of qualifying episodes pooled across flies, not a fraction of missing
frames. Alignment exposure concerns already-valid successful bouts under its
existing maximum-interpolated-heading rule; it does not count rejected bouts.

The report distinguishes:

- Baseline exposure among originally included flies.
- Exposure after wall filtering but before the new episode-count attrition.
- Exposure among the final contributing flies.
- Exposure before/after among exactly the same contributing flies.

Cohort means weight eligible flies equally. Contrasts are PFNd − Ctrl,
AR − Ctrl and AR − PFNd, separately for each radius and agarose cell. Total
contrast change is split into common-fly metric change and composition change.
Relative change divides by absolute baseline contrast and is unstable near zero.
Whole videos are bootstrapped within cohort, preserving both flies together.
No repeated artificial-mask trials enter this analysis.

Observed body-center proximity within 1 mm of a floor wall is reported
separately, as are episodes containing gaps with at least one observed endpoint
near a wall. Missing positions never count as observed proximity evidence.
These indicators do not reconstruct hidden contact or define the primary
filter. Detector history can extend outside local support.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/gap_sim_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/audit_wall_exclusion_tracking_loss.py \
  --output-dir results/episode_tracking_loss/wall_exclusion_2026-11-10 \
  --workers 4 --bootstrap-draws 1000 --seed 20261110
```

Use an empty output directory. `--limit-per-cohort 1 --bootstrap-draws 30`
provides an integration smoke run in a separate directory. `--resume` reuses
validated per-subject checkpoints after an interrupted run, provided code,
settings and sources still match. Completed outputs cannot be overwritten.
Worker count changes execution only. Video containers are tracked by size and
mtime for metadata; raw data and code are content-hashed.

Outputs: `report.html`, `comparison_table.csv` (readable overview), `summary.csv`
(all cohort/window/policy results), `contrasts.csv`, `subjects.csv`, `episodes.csv`
(original episode rows augmented with wall/proximity diagnostics), subject
`checkpoints/` and `provenance.json`. The overview shows the pooled metrics and
the final T2 agarose cell; complete tables retain every Pre/T1/T2 cell. It must
not be interpreted as the paper's flat-versus-agarose or AUC comparison.

Wall-excluded results define a different behavioral population. Lower exposure,
retained sample size, contrast magnitude and direction all matter. Preserving
the preferred ordering alone is not a scientific criterion for choosing a policy.

## Initial findings, 10 November 2026

The complete run validated 335,491 native episodes from all 196 source flies,
producing 10,584 subject/policy cells and 108 policy/metric/variant/window
contrasts. All 1,000 bootstrap draws were defined for each reported contrast.

Primary tortuosity loss exposure fell from 11.65% to 7.09% in controls, 32.49%
to 14.31% in PFNd>Kir and 35.97% to 10.88% in AR. Means changed from 7.66,
12.50 and 14.49 respectively to 4.84, 4.34 and 3.72. All three cohort signs
reversed, including comparisons restricted to exactly the same retained flies.
Retention was 58→57 controls, 46→45 PFNd and 45→45 AR. Thus dropout alone does
not explain the changed ordering. The stricter support-wall rule gives the
same qualitative conclusion.

Turnback and alignment effects were negligible at smaller radii. At 13/15 mm,
loss-exposed fractions increased slightly after contact exclusion. Final-bucket
agarose exposure likewise increased: Ctrl 7.21%→7.44%, PFNd 11.64%→12.32%,
AR 24.78%→26.64%. Across all 33 agarose cohort/time cells, exposure increased in
32 and decreased in one; no primary agarose cohort contrast reversed sign.
All non-tortuosity contrast signs in the overview were preserved. These agarose
contrasts are descriptive flat-chamber cohort comparisons, not ED17 chamber
or AUC comparisons.

A rising fraction does not mean the filter created missing frames: removal can
change the denominator more than the numerator. Missing tracking can itself
hide contact. Detected-wall exclusion therefore did not generally achieve both
lower exposure and preserved cohort comparisons. It describes a different
behavioral subset and is not independent validation of either calculation.

Validation included 40 focused tests covering half-open boundaries, production
window clipping, whole-episode removal, minimum-count attrition, proximity
versus observed contact, undefined empty cells, and separation of exposure
changes from subject dropout. Every fly also passed production wall-option
cross-checks and baseline agreement before filtering.
