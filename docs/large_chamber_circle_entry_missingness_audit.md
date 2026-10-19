# Circle-entry reliability under large-chamber tracking loss

## Motivation and review observations

The reviewer examined three high-loss trajectories from final-SLI-eligible pairs.
These are visual observations supplied by the user, not automated classifications:

| Video | Player pair ID | Role / physical ID | Loss | Observation |
|---|---:|---|---:|---|
| c52__2025-07-04__13-09-27.avi | 1 | yoked / 3 | 48.81% | Active fly; most observed tracking loss associated with right and bottom sidewall walking. |
| c32__2025-07-20__21-08-49.avi | 1 | experimental / 1 | 48.03% | Sidewall walking, especially right and top; occasional top-to-right loss/recovery creates a diagonal interpolated path. Circle-region tracking appeared better preserved. |
| c52__2025-07-27__12-52-40.avi | 1 | experimental / 1 | 42.43% | Loss appeared concentrated at top and right sidewalls; reward/control entries appeared better preserved. |

Full paths and player commands are in `results/frame_loss_review/review_list.html`.
These observations motivate a hypothesis, rather than establishing that counts
are unaffected: large2 reward/control circle geometry may separate the relevant
entry boundaries from regions where tracking fails.

## What the current implementation establishes

In `Trajectory._calcRewards`, `calc_in_circle` initially classifies positions,
including interpolated coordinates. The method then replaces classifications
at originally missing frames with the last pre-gap classification (or outside
at the beginning of the analysis slice). `_calcEnEx` detects outside-to-inside
transitions after dropping border classifications.

Consequently, an interpolated diagonal crossing cannot by itself create an entry
inside a missing run. An outside-before / inside-after transition can still be
counted on recovery. A complete real entry and exit during a missing run can be
missed. Counts can also be displaced across sync-bucket boundaries. This is a
circle-entry safeguard, not a statement about all occupancy or trajectory metrics:
the code retains a separate, pre-replacement occupancy classification.

## Questions for a quantitative audit

1. **Are the circle boundaries separated from the walls?** Read actual circle
   geometry, the control-circle options, calibration, fly transforms, and floor
   bounds for each recording/training/role. Export minimum clearance from each
   reward and control boundary to each wall, in millimetres, including the entry
   border tolerance. Compare large2 and HTL under their actual notebook recipes;
   do not infer this from chamber names or nominal radii alone.
2. **Are gaps associated with walls rather than circle boundaries?** For every
   missing run, record duration and the last observed pre-gap and first observed
   post-gap positions. Measure both positions' wall distances and signed
   reward/control-boundary distances. Report same-wall, different-wall,
   near-circle, and unavailable-endpoint cases. Define wall-distance bands before
   comparing outcomes and show sensitivity to their widths. Weight summaries by
   both gap count and missing-frame count, then summarize per fly/cohort.
3. **How often are entries dependent on gap recovery?** Identify entries whose
   state transition spans missing data, preserving the production entry-border
   logic. Report their number and fraction separately for reward and control
   entries, gap duration, and whether a sync-bucket boundary was crossed. Inspect
   these cases on video. A small fraction does not bound visits missed entirely.
4. **How often do wall-associated gaps actually conceal visits?** Randomly sample
   gap episodes within duration/location strata across flies, including long
   wall-to-wall gaps; review the raw video rather than the interpolated overlay.
   Record whether a reward/control entry is visible, its time, whether it was
   counted, and any uncertainty. Include a random observed-frame comparison
   sample to check circle tracking outside obvious gaps. Highest-loss examples
   are useful pilots, but are not a representative validation sample by themselves.
5. **How much can a comparable failure pattern change PI and SLI?** On visually
   checked, well-tracked segments, mask known positions with gap lengths and
   onset locations resembling the observed failures. Recompute reward/control
   entries and both SLI endpoints with the production rules. Repeat over mask
   realizations and report reward/control count errors, experimental/yoked PI
   errors, SLI errors, and eligibility changes. Keep a fixed-subject analysis
   separate from attrition. Validate masked segments as observed ground truth;
   this experiment assesses the imposed missingness model, not unseen behavior
   during actual gaps.

Wall-associated endpoints do not reveal a fly's position during a missing run.
Do not classify missing positions using their interpolated coordinates and then
claim that this establishes where the loss occurred. Likewise, ordinary observed
circle visits cannot demonstrate that no unobserved visits were missed.

## Population, time windows, and summaries

Begin with all three surveyed large2 cohorts, highlighting final-SLI-eligible
pairs that actually contribute to the manuscript's shared gate. Also summarize
the broader measurable population to reveal selection effects. Preserve full
video paths and physical experimental/yoked IDs. Summarize T2 SB5 and T2 SB2–5
separately using the actual synchronized bucket boundaries; full-recording
statistics are supplementary. Include both experimental and yoked trajectories.

Recommended per-fly output: full-recording and analysis-window loss; longest gap;
fraction of missing time in wall-associated gaps; counts and duration of
near-circle gaps; reward/control entries spanning gaps; manually established
missed or false visits; and resulting PI/SLI errors where ground truth is known.
Distinguish unknown cases from zero errors. Summarize across flies and account
for common recording identity in uncertainty estimates.

Actual experimental reward commands can corroborate offline entry timing where
appropriate, but cannot validate virtual control/yoked entries or recover visits
missed by an online tracker driving those same commands. Avoid treating the
command record as independent ground truth without checking how it was produced.

## Interpretation

Support for retaining a high whole-recording loss tolerance would require evidence
that losses are spatially separated from entry boundaries **and** that relevant
count/PI/SLI errors remain small. Assess errors in reward and control counts
separately: the same number of missed entries can matter much more when a bucket
has few total visits. Define acceptable endpoint error before using the audit to
justify a particular cutoff. If errors are appreciable or cannot be resolved,
consider an endpoint-specific quality rule and disclose the resulting selection.

This audit concerns reward PI and SLI. It does not justify including interpolated
frames in path length, tortuosity, wall-contact, or occupancy measurements. Those
require their own validation. No across-fly spatial or entry-error statistics
have yet been calculated in this audit.
