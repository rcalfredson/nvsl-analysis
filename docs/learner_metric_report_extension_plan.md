# Learner metric report extension plan

Status: implemented October 7, 2026; see `learner_metric_report.md` for usage.
The user approved the statistical proposal and clarified that current mean-SLI
rankings require at least three of four buckets. This supersedes the four-bucket
requirement described in the original wiki HTML.

## Scope and decisions

Extend the September 29 strong/weak learner report for the flat large-chamber
control cohort. Preserve the existing metric definitions. The user confirmed that the new
cohorts should be **ranked by T2 SB5 SLI**, rather than by a mean within an
SB5-eligible subset.

Both analyses retain top 20% strong learners and bottom 50% weak learners.
Behavior metrics remain pooled over T2 SB2–5, with experimental final-bucket
presence and at least five qualifying episodes per metric/per radius pair.
Missing metric values do not cause cohort reassignment or reranking.

## Current implementation

- Commit `207a0e6` added the report; `833431c` added the three absolute radius
  pairs and made repeated-radius results descriptive only.
- `analyze.py::_normalize_learner_metric_table_options` forces
  `sli_use_training_mean=True`, defaulting the SLI selection window to SB2–5.
- `src/exporting/learner_metric_table.py::_selected_vas_and_sli` uses the shared
  reward-PI SLI helper and `select_fractional_groups`, then fixes cohort membership
  across metrics.
- Shared SLI code defaults to **three** valid paired buckets. The current mean
  report uses this same >=3-of-4 policy. The user will remove the obsolete
  four-bucket requirement from the wiki; old numbers need verification on rerun.
- Collectors already retain one pooled observation per fly/metric/radius, episode
  counts, DCTR success counts, eligibility, and exclusion reasons.
- Current exports are summary CSV, per-fly CSV, Prism wide CSV, metadata JSON,
  and Markdown. All nine rows have descriptive statistics. Only tortuosity,
  COM distance, and return-leg distance have Welch tests, without adjustment.
- `statsmodels==0.14.5` is already declared. No mixed-model analysis was found
  in the repository.

## 1. Make cohort selection explicit

Add `--learner-metric-table-sli-mode {mean,final}`. Keep `mean` as the CLI default
to preserve existing invocation behavior; use explicit `final` in the new paper
recipe. Continue honoring existing mean-mode training/window/minimum overrides.

| Mode | Ranking score | Eligibility for ranking |
|---|---|---|
| `final` | Experimental minus yoked reward PI at T2 SB5 | Both SB5 component PIs finite; no mean-window minimum |
| `mean` current recipe | Mean paired bucket-level SLI at T2 SB2–5 | At least three of four bucket-level SLIs finite |

Implementation:

1. Update parser and normalization so final mode is no longer forced to mean.
2. Resolve the effective cohort-selection options on a report-local copy.
   Final mode explicitly selects training 2 and numeric bucket `5`, clearing
   inherited selection-window constraints. Do not use `last`: the shared helper
   has a historical padded-slot convention that can select a different bucket.
3. Reuse `_compute_sli_scalar_and_timeseries_from_rpid` and the existing
   `select_fractional_groups` rounding/tie behavior. Mask nonfinite scores
   before ranking; preserve stable input order for tied scores and audit it.
4. Keep metric-window options independent of cohort-selection options.
   Experimental bucket presence alone is not defined SLI: final SLI also needs
   the yoked PI.
5. Audit every non-skipped input fly, including unrankable and middle-ranked
   flies: unit ID, final SLI, mean SLI, valid paired-bucket count, active score,
   rank, selected cohort, and exclusion reason. Record skipped-input counts.
6. Include effective training, exact bucket/window, minimum, fractions,
   rankable/selected counts, and mode in metadata and report prose.

Check combined report/plot invocations while changing normalization so that
report-local choices do not unintentionally alter other outputs.

## 2. Add mixed models for the repeated-radius metrics

Create a separate analysis module, for example
`src/analysis/learner_metric_stats.py`, consuming existing `MetricObservation`
records. Fit turnback ratio and home-vector alignment separately.

Proposed starting model, in conventional formula notation:

```text
value ~ cohort * radius_pair + (1 | unit_id)
```

Use radius pair as a categorical factor, weak as the cohort reference, and
one fly random intercept. The repeats are the three radius pairs; SB2–5
episodes remain pooled as in the existing report. Do not treat episodes or
radius rows as independent flies, and do not weight flies by episode counts
in the primary analysis.

Build one long-form model-input table per metric using only eligible finite
observations. Retain partially observed flies, including flies with one usable
radius; discard only their missing/ineligible rows. Do not impute zero or
require complete measurements at every radius. Use identical eligible rows
for any nested model comparisons.

Proposed inference specification:

- Fit the full model with REML using Statsmodels `MixedLM`.
- Primary cohort contrast: strong minus weak, averaged equally across the
  configured radius pairs. Compute it explicitly from the fixed-effect design;
  the cohort coefficient alone is the effect at the reference radius.
- Report the cohort-by-radius interaction as a joint Wald test (two degrees
  of freedom for the default three pairs).
- Compute strong-minus-weak contrasts and 95% CIs separately at each radius
  from the same model and its fixed-effect covariance matrix.
- Label Wald p-values and confidence intervals as asymptotic. This proposal
  does not supply Prism-style ANOVA F tests or small-sample denominator-df
  corrections. If those are required, select and validate that inference
  backend before implementation.

Record row count, unique-fly count by cohort, available observations per radius,
missingness patterns, model formula, factor order, fitting method, optimizer,
convergence, variance estimates, and warnings. Validate unique fly/radius keys,
cohort consistency, design rank, estimable contrasts, and adequate repeated
observations. An empty cell or insufficient data should produce an explicit
unavailable result. Nonconvergence or invalid covariance must not yield an
ordinary successful p-value. Allow bounded optimizer retries with recorded
outcomes; never silently substitute independent-group tests.

A random intercept assumes a common within-fly covariance contribution and a
common residual variance. Inspect residuals and variance patterns. DCTR is a
bounded success ratio with varying denominators, and alignment is bounded too;
the Gaussian model is a proposed starting analysis subject to diagnostics.
If unsuitable, consider a separately specified binomial mixed-model sensitivity
analysis for DCTR using the retained counts. Do not switch models automatically.
Available-case mixed modeling depends on assumptions about the missingness
mechanism; it does not remove potential bias from too-few-episode exclusions.

## 3. Complete report statistics and exports

Preserve observed group means, fly-level t-based 95% CIs, per-row sample sizes,
and observed strong-minus-weak differences. Add separate fields for model-based
contrasts, their CIs, and p-values; these may differ from observed differences
when measurement availability differs across radii.

Retain existing Welch tests and difference CIs for the three standalone metrics.
Multiplicity policy, updated at the user's request to keep turnback and alignment
separate for both fitting and multiple-comparison adjustment:

- Holm across turnback's per-radius model contrasts (three by default).
- Holm across alignment's per-radius model contrasts (three by default),
  separately from turnback.
- Standalone Welch tests assessed individually without multiple-comparison
  adjustment (singleton families with unchanged p-values).
- Each metric's radius-averaged cohort and interaction tests is a separate
  singleton family, leaving its p-value unchanged.
- Export raw and adjusted p-values, family membership, and confidence-interval
  type. Pointwise 95% CIs are not simultaneous Holm-adjusted intervals.

Reuse `src/analysis/multiple_comparisons.py::holm_adjust`. Preserve unavailable
positions and report incomplete families explicitly, rather than interpreting
unavailable tests as negative results. Keep the historical descriptive/raw-Welch
statistics path selectable, for example with
`--learner-metric-table-stats {legacy,mixed}` (default `legacy` for compatibility).
Use `mixed` explicitly in both new analysis recipes.

Extend existing exports and add:

- `_cohort_audit.csv`: all non-skipped input flies and their selection decisions.
- `_model_input.csv`: exact long-form observations used by the two models.
- `_model_tests.csv`: primary cohort and interaction tests with sample counts.
- `_model_diagnostics.json`: fit status, warnings, and variance information.

Add per-radius model contrast columns to the nine-row summary. Render primary
and interaction tests in a separate Markdown section. Include methods text for
the wiki explaining cohort selection, metric pooling, missing values, test
families, and the distinction between observed and modeled differences.
Use separate output prefixes for final and historical mean selection.

## 4. Validation and reproducible recipes

Add meaningful tests for:

1. Final selection including a fly with finite SB5 but fewer than three valid
   window SLIs; rejecting a fly with finite experimental SB5 but missing yoked
   SB5; ignoring mean-window minimum in final mode.
2. Current mean selection requiring at least three of four buckets, and both modes
   independently ranking a fixture whose final and mean ranks differ.
3. Numeric SB5 selection despite padded trailing slots; unchanged metric
   pooling, episode minimum, final-bucket presence, and fixed cohort membership.
4. Balanced synthetic mixed-model data with known cohort/radius effects,
   recovered contrast direction, and a checked covariance-based contrast SE.
5. Partial-radius missingness retaining other observations; duplicate IDs,
   nonestimable designs, unavailable contrasts, and failed fits handled visibly.
6. Multiplicity families and raw/adjusted export values, preserving historical
   descriptive results and testing complete report generation.

Run the existing learner-report, SLI truth, bucket-presence, and collector truth
tests alongside the new model tests. Cross-check one fit and its planned
contrasts against an independent R/lme4 calculation with matching REML/Wald
settings, or a fixed reference fixture generated from that calculation.

Provide two documented recipes sharing the same original flat-chamber control
input list and preprocessing flags:

```text
# Additional report flags; append to the original data/preprocessing command.
# Shared by both recipes:
--top-sli-fraction 0.2 --bottom-sli-fraction 0.5
--min-turnback-episodes 5 --min-between-reward-trajectories 5
--learner-metric-table-training 2
--learner-metric-table-skip-first-sync-buckets 1
--learner-metric-table-keep-first-sync-buckets 4
--learner-metric-table-require-sb5
--learner-metric-table-circle-pairs-mm 3:5,8:10,13:15

# Current mean cohorts, with new statistics:
--export-learner-metric-table exports/learner_metrics_t2_mean
--learner-metric-table-sli-mode mean --sli-min-valid-sync-buckets 3
--best-worst-trn 2
--sli-select-skip-first-sync-buckets 1 --sli-select-keep-first-sync-buckets 4
--learner-metric-table-stats mixed

# New paper-aligned cohorts:
--export-learner-metric-table exports/learner_metrics_t2_sb5
--learner-metric-table-sli-mode final --learner-metric-table-stats mixed
```

Both recipes must explicitly set top/bottom fractions to 0.2/0.5 and the shared
episode-type minima to five, retain metric T2 SB2–5 and all three radius pairs,
and record input provenance and code revision. Compare rerun means, CI
half-widths, and sample sizes against the wiki, accounting for the newly
confirmed >=3-of-4 policy. Do not assume the prior numbers remain identical.
The supplied HTML alone cannot reconstruct new p-values: raw per-fly values
and their repeated-radius pairing are required.

## Implementation order and review points

1. Cohort mode, local option resolution, audit, selection tests, and recipes.
2. Mixed-model module, explicit contrasts, fit diagnostics, and reference tests.
3. Report integration, multiplicity, methods text, and export tests.
4. Historical reproduction, final-mode rerun, diagnostic review, and final
   methods specification for the manuscript.

Model covariance structure, asymptotic inference, and multiplicity families
above were approved for implementation. Review diagnostics before reporting
new inferential results. Retrieve the original input command/list for reruns; no
checked-in learner-report recipe or prior report export was found in this review.

The initial system-Python validation attempt lacked `cv2`. Implementation
validation uses `/home/tracking/miniconda3/envs/analysis3.13/bin/python`, which
provides the repository's dependencies. The original 23 report/SLI tests passed
before implementation. Mixed-model coefficients, covariance, and planned
contrasts are also checked against an independent R/nlme REML reference.

Reference: [Statsmodels mixed-model documentation](https://www.statsmodels.org/stable/mixed_linear.html)
describes the random-effects model and supported Wald inference;
[MixedLM fitting documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.fit.html)
specifies REML versus ML. The repository pins 0.14.5; verify against the installed
version during implementation because the live documentation tracks a newer
release.
