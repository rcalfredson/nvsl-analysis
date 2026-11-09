# Reference-support audit for tracking-loss contrasts

This stage inventories what can be tested using the existing tracking. It does
not introduce artificial gaps, estimate metric errors, or correct cohort means.
It follows the reporting repairs described in `tracking_gap_simulation.md` and
the contrast targets in `tracking_gap_simulation_expansion_plan.md`.

## Reproduce the census

```bash
MPLCONFIGDIR=/tmp/gap_sim_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/audit_tracking_gap_support.py \
  --output-dir results/tracking_gap_simulation/support_audit_2026-11-09
```

Use a new or empty output directory. `--notebook` and `--source-dir` override
the default manuscript notebook and reviewed component-one source directory.
Notebook code, export commands and production analysis caches are not executed
or modified. Raw source, note-file, archive, notebook and relevant code hashes
are recorded, and tracked source files are checked for changes before outputs.
Video containers are recorded by size/mtime for their metadata dependency;
the large video files themselves are not content-hashed. An empty recording
absent from the manuscript archive can lack calibration; both excluded pairs
are recorded explicitly without inventing geometry.

Validated per-video results are written atomically under `checkpoints/`. After
an interrupted run, repeat the same command with `--resume`. Checkpoints must
match the code, frozen inputs, bundle, raw files, note file and video size/mtime;
incompatible checkpoints raise an error. Completed outputs cannot be overwritten.

## Populations and baseline validation

The notebook's literal video-list and bundle assignments define ten agarose
datasets: Ctrl, antennae removed, MBKC-1>Kir, PFNd>Kir and EPG>Kir, each in flat
and agarose large chambers (ED17b–f). The original 196-subject roster supplies
the other nine metric/radius cells: tortuosity, wall contacts, wall-positive
path length, and three radius pairs each for turnback and alignment.

Every matched source video is checked, including experimental pairs excluded
by production admission. Admission replays empty-trajectory, lost-frame and
suspicious-jump checks, followed by physical-fly note-file overrides. The source
agarose archives must agree with the admitted population at each video's level.
Legacy archives have duplicate video names and no fly IDs: reconstructed count
and numerator vectors are compared as unordered multisets within video. Values
are never used to assign an archived row to a physical fly. Independent source
ratio checks verify the minimum-five rule. The original pooled-metric source
roster and raw hashes must also agree with the reviewed first-stage artifacts.

Analysis uses original x-NaN intervals for gap donors. Fully observed references
require finite original x, y, body width, body height and orientation, which is
stricter than merely having interpolated x/y coordinates. Other-field failures
at nominally tracked frames remain explicit in `windows.csv`.

## Windows and exact support rules

The pooled metrics use a complete T2 SB2–5 window. Agarose uses its actual
10-minute Pre cell and synchronized Training 1/2 buckets, retaining absent
canonical buckets as explicit cells. Baseline agarose counts use episode entry
assignment, matching production; complete reference support may extend beyond
an entry's bucket and therefore not qualify for that bucket's reference pool.

For each metric/window, the census enumerates every donor and candidate start:

1. A donor is an entire original x-NaN run with observed x/y bracketing endpoints
   both inside the window. Boundary-crossing, initial, terminal or otherwise
   unbracketed loss remains in the all-loss inventory but outside donor coverage.
2. A length-compatible placement preserves the donor's exact frame duration,
   both observed endpoints and two seconds of observed padding beyond each
   endpoint. It lies wholly within one originally fully observed run in the
   same fly and window.
3. Spatial matching requires identical before/after nearest-wall bins (<1,
   1–<3, >=3 mm) and agreement within 2 mm for before/after reward-center
   distances, signed nearest-well-boundary distances, and endpoint chord length.
   The donor's unobserved interior is not used. Reward geometry comes from the
   appropriate training (T1 geometry for Pre).
4. Episode support additionally requires that the artificial missing interval
   would intersect complete original reference episode support for the exact
   metric/radius. Episodes are neither clipped into completeness nor pooled
   across radii. This is a feasibility count, not a claim the outcome changes.
5. Minimum-five support additionally requires at least five complete reference
   episodes for that metric/radius in the same observed parent run. This is a
   local diagnostic minimum, not a new manuscript inclusion rule or proof of
   adequate precision. Alignment records sampled-heading-frame overlap
   separately from overlap with the broader bout support.

All donor sampling probabilities are one; candidate searches are exhaustive.
Multiple matching starts are not independent biological replicates. The primary
matching rules remain fixed: no cross-fly/chamber substitution, tolerance
relaxation, gap shortening or selection based on preservation of a group effect.

## Outputs and interpretation

- `datasets.csv`: source variable/bundle, video and subject counts, validation.
- `source_availability.csv`: notebook glob expansions, including patterns with
  no matches. An archived video or required raw file missing from these sources
  stops validation rather than silently reducing the claimed population.
- `subjects.csv`: admitted and excluded experimental pairs, frame rate,
  production-style loss (excluding the initial missing run), raw loss and note
  overrides. Loss strata are 0–10%, >10–30%, >30–60%, and >60%/unavailable.
- `windows.csv`: original loss, full donor/excluded loss, observed runs,
  baseline eligibility/counts, complete observed episode counts and statuses.
  Adjacent-gap spacing measures recurrence within the cell; this census does
  not enumerate the feasibility of joint multi-gap block placement.
- `donor_support.csv`: one row per eligible donor and metric/radius/window,
  with matching counts, wall context, radius-specific exposure, local-five
  support and heading exposure. A donor is supported at a stage if its candidate
  count is positive. No candidate count is an estimate of a behavioral error.
- `reference_episodes.csv`: complete observed episode intervals, support bounds,
  parent-run IDs and sampled headings. A physical episode is identified by
  video/pair/metric/radius/start/stop, not by its row number across windows.
  Within a specific fly/metric/radius/window, `reference_episode_index` joins
  the donor table's `reference_episode_indices` list to this pool.
- `coverage.csv`: support by genotype, chamber, metric, radius, period and
  baseline eligibility. Gap fractions weight donor gaps equally; lost-time
  fractions weight their duration. These use *eligible donors*, not all loss,
  as denominators. Excluded lost time is reported separately. Subjects without
  donors remain in subject totals but cannot transplant their own gaps.
- `coverage_by_duration_context.csv`: duration, endpoint-wall context and
  recording-loss strata, without combining genotypes/chambers or eligibility.
- `contrast_coverage.csv`: counts supporting both sides of the intended group
  or chamber contrast. Positive support is necessary but does not establish
  representative coverage or sufficient statistical precision.
- `auc_windows.csv` / `auc_coverage.csv`: production AUC availability and strictly
  observed whole-window reference candidates, separate from local support.
- `report.html` / `provenance.json`: browsable overview and recorded inputs.

For a concise overview that also deduplicates actually matched reference
episodes (rather than reporting only the complete observed pool), run:

```bash
/home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/report_tracking_gap_support.py \
  --audit-dir results/tracking_gap_simulation/support_audit_2026-11-09 \
  --output-dir tmp/tracking_gap_support_summary_2026-11-09
```

This produces source-population, pooled-metric, agarose-local, gap-duration and
matched-reference coverage tables, with its own input hashes. Reference IDs are
deduplicated within fly/metric/radius/window even if multiple donors or starts
reuse them. Agarose's overview pools only baseline-eligible Pre/T1/T2 cells and
deduplicates subject counts across cells; use the census for timepoint-specific
coverage. These descriptive summaries are not per-fly effect estimates.

### Initial census findings (9 November 2026)

All ten manuscript datasets agree with the archived video-level count/numerator
multisets and finite production AUC samples. They contain 623 admitted
fly–dataset memberships, representing 619 distinct video/fly identities in 321
recordings. The other pooled metrics retain the complete original 196-subject
roster. The census has 670,898 donor/metric/window rows; these are repeated
support queries, not independent biological observations or completed simulations.

The source definitions share two recordings between antennae-removed and
EPG>Kir flat-chamber datasets:

- `/media/Synology4/Yang Chen/2025-07-21/night/c61__2025-07-21__20-41-33.avi`
- `/media/Synology4/Yang Chen/2025-07-21/night/c62__2025-07-21__20-41-36.avi`

Both experimental pairs are admitted in both datasets (four distinct flies,
eight memberships). The overlap is present in the notebook and source bundles;
this is agreement with current inputs, not verification of their cohort labels.
The overview exports `overlapping_source_memberships.csv`. Memberships have not
been changed. Clarify their intended assignment from experiment records before
interpreting affected genotype/manipulation contrasts, and preserve shared-video
dependence if reuse is intentional.

Local support is much broader than continuous whole-window support. For
wall-positive path length, only five eligible control flies, two PFNd>Kir flies,
and zero antennae-removed flies have an episode-overlapping matched placement
in a parent run containing at least five complete reference episodes. This is
reference-side availability only; after-masking eligibility cannot be known
without simulation. Of 1,246 AUC-window memberships (two per admitted dataset
membership), eight have all five fields continuously observed across the actual
span, and only one also meets the strict defined-AUC/inside-window complete-
episode-support criteria. These totals include reused memberships and are not
independent sample sizes. The strict boundary criterion is conservative: an
observed extension beyond the nominal window may recover additional reference
support without retracking and should be assessed before declaring it absent.

There are some supported >10-second donors in the full census, unlike the small
pilot; support remains dependent on metric/radius and covers only selected
contexts. Neither positive support nor failure to match estimates an error rate.
The expanded simulation design should address these coverage deficits rather
than simply increase trial counts.

## Agarose AUC is not an average of local simulations

AUC availability follows the production per-subject trapezoidal calculation,
with Pre included only in Training 1. The production helper removes one trailing
column only when it is NaN for the entire group's array. Other missing values
can make individual AUCs undefined. Reconstructed finite AUC samples must agree
with source samples at the group level after video-level count validation.
AUC uses production bucket-index spacing, not an independently chosen minute
scale or integration of cohort means.

`trusted_defined_auc_window` is a strict candidate-reference flag: the AUC is
defined, all five tracker fields are continuously observed across the relevant
actual span, and all contributing episodes have complete observed support inside
it. It does not mean an independent human audit established biological ground
truth. A recording can offer many local gap placements and no such continuous
AUC reference. Missing whole-window references are a reason to consider observed
boundary-context extensions, retracking, or restriction of the intended claim,
rather than infer AUC robustness from short local tests. Detector-state
initialization and inherited hysteresis history also remain to be validated;
agreement with archived baseline outcomes does not establish their independence
from earlier gaps.

The next design decision should use support by both gap count and lost time,
the independent-video coverage on each side of a contrast, and the remaining
long-gap/whole-window deficits. This stage supplies no contrast-effect estimates
and makes no equivalence or dataset-wide robustness claim.
