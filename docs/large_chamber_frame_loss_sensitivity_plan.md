# Large-chamber tracking-loss sensitivity analysis

## Question and scope

How much do the manuscript's large2 results depend on admitting trajectories
with tracking loss between 10% and the current 60% cutoff? Measure both sample
attrition and changes in group means and between-group differences.

Start with the same three flat-large cohorts as the survey: Ctrl>Kir,
PFNd>Kir, and antennae-removed control. The full four-group Figure 4i export
also includes antennae-removed PFNd>Kir; add that group as an explicit extension
after surveying its tracking loss. Do not silently treat the three-group analysis
as covering every Figure 4i column or other large-chamber cohorts.

## Endpoints

Preserve the notebook's existing definitions and inclusion rules:

| Endpoint | Notebook source | Definition |
|---|---|---|
| Training SLI time course | Figures 4d, 4i | Experimental minus yoked reward PI, each training sync bucket |
| Final SLI | Figure 4i final-SLI export | T2 SB5 difference; both component PIs finite |
| Mean SLI | Figure 4i mean-SLI companion | Mean paired finite differences in T2 SB2–5; at least 3 of 4 buckets valid |
| Tortuosity | Figure 4j | Per-fly mean full-path length / maximum reward-center distance, T2 SB2–5, at least 5 valid trajectories |
| Wall contacts | Figure 4k | Per-fly mean contact count per interval, including zero-contact intervals, T2 SB2–5, at least 5 intervals |
| Wall-contact trajectory length | Figure 4l | Per-fly mean full length of qualifying wall-contact trajectories, T2 SB2–5, at least 5 qualifying trajectories |

Figures 4j–l also preserve experimental SB5 presence and finite **paired final
SLI** eligibility. Do not substitute experimental-only eligibility or a
positive-SLI requirement. Means are over flies, not pooled episodes.

## Phase 1: Join existing measurements and sweep inclusion thresholds

1. Freeze notebook recipes, local configuration, source-export provenance,
   manual note files, and source tracking files. Reproduce the existing 60%
   baseline sample IDs, counts, and means before interpreting any differences.
   Existing archives must be checked for compatible cohorts and recipe versions.
2. Join loss records to keyed SLI audits and scalar/eligibility bundles using
   cohort, full video path, experimental fly ID, and yoked trajectory ID. Reject
   duplicate or missing identity matches; never join on CSV row order or values.
3. Use cutoffs **60, 50, 40, 30, 20, and 10%**. Retain equality, matching the
   analyzer's strictly-greater rejection rule. Use full-recording loss after the
   initial missing prefix, matching the current code. Keep empty-trajectory and
   pre-existing exclusions separate from new cutoff exclusions.
4. For paired SLI, require both members to pass: equivalently,
   `max(exp_loss_pct, yok_loss_pct) <= cutoff`. Apply the same paired gate to
   Figures 4j–l because those panels require paired final SLI. Preserve all
   endpoint-specific baseline eligibility rules.
5. Keep each retained fly's existing metric value fixed. This first analysis
   measures sensitivity to sample selection, without changing interpolation,
   reward geometry, episode definitions, or metric formulas.
6. Audit `__note.txt` overrides separately. `_readNoteFile` can override bad
   status after trajectory initialization. Label this phase as a strict
   post hoc loss filter; verify rather than assume equivalence to changing the
   analyzer's initialization cutoff.

For each cohort, endpoint, and cutoff, export:

- Eligible flies at baseline; retained flies; retention fraction; videos represented.
- Attrition by experimental loss, yoked loss, both, empty tracking, and other
  baseline eligibility failures. For time courses, report n separately per bucket.
- Mean, SD, median, and 95% confidence interval; absolute mean shift from 60%.
- PFNd-minus-control and antennae-removed-minus-control mean differences,
  confidence intervals, and changes in those differences from baseline.
- Numbers of contributing episodes for trajectory endpoints, alongside fly n.

Use video-cluster bootstrap intervals within cohorts, keeping experimental/yoked
identity and all flies from a recording together. Use the same bootstrap draws
across cutoffs when estimating shifts from baseline. Show the small number of
videos explicitly; describe uncertainty as exploratory when video counts are low.
Avoid relative percentage shifts for SLI, whose means can be zero or negative.
Do not select a cutoff based on which yields significance.

Primary figures: retained n versus cutoff; group mean versus cutoff;
between-group difference versus cutoff; and baseline versus 10% SLI time courses.
Also plot per-fly outcomes against experimental loss and yoked loss separately,
so selection related to behavior is visible. These associations are descriptive.

## Phase 2: Check equivalence with direct analyzer runs

Add an optional loss-cutoff argument while retaining existing chamber defaults.
Run representative baseline, intermediate (30%), and strict (10%) analyses into
isolated output directories, preserving notebook flags and note-file behavior.
Verify inclusion IDs and retained per-fly values against Phase 1; investigate
differences before expanding the sweep. Audit paired inclusion, manual overrides,
skipped experimental trajectories, bad yoked trajectories, and nonfinite exports.
Do not overwrite manuscript exports or change the default 60% policy during
this investigation. Phase 1 may be sufficient if these equivalence checks pass.

## Phase 3: Examine interpolation within retained trajectories

A fly-level cutoff can show dependence on admitting high-loss flies, but cannot
establish that interpolated paths or rewards are accurate. If a result shifts
materially, inspect when losses occur in the analysis window and the lengths of
contiguous gaps; whole-recording loss may hide loss concentrated in Training 2.
Initial missing prefixes are excluded from the current loss denominator, so
report those prefixes separately as a coverage diagnostic.

For Figures 4j–l, report episode loss fractions and maximum gap lengths. A
secondary analysis can exclude episodes intersecting missing tracking or long
gaps, then recompute per-fly means and the original minimum-episode rules.
Predefine gap criteria before examining resulting group differences, and compare
within the same eligible flies as well as after resulting attrition. Avoid
concatenating points across excluded gaps. This changes episode selection and
is a distinct analysis from tightening the fly-level cutoff.

If SLI changes need further explanation, inspect the experimental and yoked
reward-PI components and reward counts separately; calculated reward crossings
also depend on tracking. Retained low-loss flies may still contain problematic
gaps, so stability across cutoffs is evidence of selection robustness, not a
validation of tracking accuracy.

## Deliverables and first pass

Write a subject-level inclusion audit, a tidy endpoint/cohort/cutoff summary,
an HTML report, and standalone plots under a separate results directory.
Include source identities, provenance, metric units, and reproducible commands.

First implement final SLI and mean SLI for the three surveyed cohorts using
existing keyed exports. Then add Figures 4j–l, followed by the bucket time courses
and direct-run checks. This gives a quick assessment of sample loss and effect
stability before committing to episode-level reanalysis or additional cohorts.

The first-pass final- and mean-SLI sweep has now been implemented in
`scripts/sweep_sli_frame_loss.py`, with outputs under
`results/sli_frame_loss_sweep/`. The keyed joins reproduce the 60% baseline and
preserve each endpoint's eligibility rules. Direct analyzer reruns, trajectory
metric sweeps, and episode-level checks remain later phases.
