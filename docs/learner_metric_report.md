# Strong and weak learner metric reports

`analyze.py --export-learner-metric-table PREFIX` writes nine descriptive rows:
dual-circle turnback ratio and home-vector alignment at 3/5, 8/10, and 13/15 mm,
plus between-reward tortuosity, COM distance, and return-leg distance.

## Selection and calculation windows

`--learner-metric-table-sli-mode mean` (default) ranks by mean paired bucket-level
SLI at T2 SB2–5. At least **three of four** finite bucket-level SLIs are required
by default. Each bucket SLI is experimental reward PI minus yoked reward PI;
the components must both be defined in the same bucket. Mean-mode training,
window, and minimum can be overridden using the existing shared SLI flags.

`--learner-metric-table-sli-mode final` ranks by SLI at **T2 SB5**. It requires
finite experimental and yoked SB5 PIs, independently of the mean-window minimum.
Final mode always uses numeric SB5, even when arrays contain padded later slots;
it ignores shared training/window/bucket overrides for report selection.

Strong learners are the top 20%, weak learners the bottom 50% of rankable flies.
Existing `--top-sli-fraction` and `--bottom-sli-fraction` override those fractions.
Counts are floored with a minimum of one; tied scores follow input order. The
groups are disjoint and are assigned once, before metric exclusions.

Both modes retain the existing metric pooling: T2 SB2–5, experimental final
selected bucket presence, and at least five qualifying episodes per fly and
metric/radius. Metric settings use `--learner-metric-table-training`,
`--learner-metric-table-skip-first-sync-buckets`, and
`--learner-metric-table-keep-first-sync-buckets`. Cohort selection does not
override these settings or the selection options of other plots/exports.

## Running the report

Append these options to the original flat large-chamber control data command
(the same video list, chamber/genotype selection, and preprocessing options).

```bash
# Paper-aligned final-SLI cohorts, with mixed-model statistics:
--export-learner-metric-table exports/learner_metrics_t2_sb5 \
--learner-metric-table-sli-mode final \
--learner-metric-table-stats mixed
```

```bash
# Mean-SLI cohorts, >=3 of 4 buckets, with mixed-model statistics:
--export-learner-metric-table exports/learner_metrics_t2_mean \
--learner-metric-table-sli-mode mean \
--best-worst-trn 2 \
--sli-select-skip-first-sync-buckets 1 \
--sli-select-keep-first-sync-buckets 4 \
--sli-min-valid-sync-buckets 3 \
--learner-metric-table-stats mixed
```

To ensure the standard metric configuration when other options are present,
include these shared flags in either command:

```bash
--top-sli-fraction 0.2 --bottom-sli-fraction 0.5 \
--min-turnback-episodes 5 --min-between-reward-trajectories 5 \
--learner-metric-table-training 2 \
--learner-metric-table-skip-first-sync-buckets 1 \
--learner-metric-table-keep-first-sync-buckets 4 \
--learner-metric-table-require-sb5 \
--learner-metric-table-circle-pairs-mm 3:5,8:10,13:15
```

The examples are argument fragments, not standalone shell commands. Omit
`--learner-metric-table-stats mixed` or use `legacy` to retain descriptive
repeated-radius rows and raw standalone Welch tests.

## Mixed-model inference

Turnback and alignment each receive a separate Gaussian mixed model:

```text
value ~ cohort * radius_pair + (1 | unit_id)
```

The repeat is radius pair within fly; SB2–5 episodes remain pooled. Radius is
categorical. The model uses a fly random intercept, a common residual variance,
REML fitting, and asymptotic Wald inference. No episode weighting is used.
Partially observed flies retain their eligible radii without imputation;
all-unavailable flies remain in the cohort audit but cannot enter the model.

Patsy builds the fixed-effect design from `cohort * radius_pair`, with explicit
categorical levels preserving weak/first-radius reference coding. Contrast
vectors are differences of strong and weak prediction rows using that same
design metadata. Statsmodels supplies contrast estimates, standard errors,
normal-reference p-values, and CIs through `MixedLMResults.t_test(use_t=False)`;
the joint interaction uses `wald_test(use_f=False)`. Despite its name, `t_test`
uses a normal reference when `use_t=False`. Holm adjustment uses the existing
Statsmodels-backed `holm_adjust` helper. Report code retains eligibility,
fit validation, comparison definitions, correction families, and export logic.

Primary cohort contrasts average strong-minus-weak differences equally across
the configured radii. Separate radius contrasts and a joint cohort-by-radius
interaction test come from the same fit. The cohort coefficient itself is the
difference at the reference radius, not the across-radius average. The three
standalone metrics continue to use independent-group Welch tests.

Turnback and alignment remain separate for both fitting and multiple-comparison
adjustment. Holm adjustment is performed separately for:

- Turnback's per-radius strong-minus-weak contrasts (three with default radii).
- Alignment's per-radius strong-minus-weak contrasts (three with default radii).

Standalone Welch tests are assessed individually without multiple-comparison
adjustment. Each is represented as a singleton family in the exports, so its
`p_value_holm` field equals its raw p-value.

Each repeated metric's across-radius cohort and interaction tests is a separate
singleton family: its adjusted p-value equals its raw p-value. No adjustment
family combines turnback and alignment, or combines either with standalone
metrics.

Raw p-values are always retained. If any planned p-value in a family is
unavailable, adjusted p-values for that entire family are unavailable and the
family is marked incomplete. All 95% CIs are pointwise, not simultaneous.
Observed means/intervals remain fly-level t-based summaries. Model contrasts
can differ from observed differences with unequal radius availability.

Models require at least two observed flies per cohort, two flies with repeated
measurements, an estimable full design, and residual degrees of freedom. Fits
use bounded optimizer retries, record warnings, and reject nonconvergence,
invalid fixed-effect covariance, or a non-positive-definite Hessian. Unavailable
fits are reported explicitly without substituting independent-group tests.

Inspect fit warnings, residual summaries, and missingness patterns before using
results. Both outcomes are bounded and DCTR denominators vary. The Gaussian
model and asymptotic inference should be reviewed against those properties and
the sample size. Available-case mixed modeling assumes ignorable missingness;
episode-threshold exclusions may still bias results. This implementation does
not provide small-sample denominator-df corrections or automatic model changes.

## Artifacts

Every report writes:

- `_summary.csv`: observed means, fly counts, confidence intervals, differences,
  and statistics (separate model contrast columns in mixed mode).
- `_per_fly.csv`: pooled measurements, counts, eligibility, exclusion reasons.
- `_prism_wide.csv`: one fly per row with missing metric/radius values retained.
- `_cohort_audit.csv`: all non-skipped input flies, final/mean SLI, paired-bucket
  counts, active selection score, ascending rank, cohort, or exclusion reason.
- `_metadata.json`: effective selection/calculation policies, input identities,
  code revision, methods text, and output paths.
- `_summary.md`: descriptive table and methods prose suitable for the wiki.

Mixed mode also writes:

- `_model_input.csv`: exact eligible finite rows used by the models; radius pair
  is the zero-based position in the configured radius list.
- `_model_tests.csv`: across-radius cohort and interaction tests.
- `_model_diagnostics.json`: fit attempts, warnings, variance estimates,
  fixed-effect estimates/covariance, residual summaries, and missingness patterns.

Model-input CSVs may have only headers if no repeated measurements qualify.
Unavailable JSON numbers are `null`; CSV numbers are `nan` as in existing exports.

New report runs require the original input data and analysis options. Updating
the wiki's old four-bucket wording does not establish that its numeric results
will stay identical under the confirmed >=3-of-4 rule: verify that on rerun.

Implementation validation: 155 targeted tests passed in the `analysis3.13`
environment, covering selection, pooling, mixed models, missingness, exports,
and the existing metric truth/eligibility tests. Fixed-effect estimates,
covariance, and Wald contrasts match an independent R/nlme REML reference.
The original fly-data reports have not been regenerated as part of this change.

Package-based inference refactor validation: 25 focused tests passed, including
the independent R/nlme reference, partial missingness, missing cells, failed fits,
exports, and two/four configured radius pairs. Re-fitting the exported per-fly
measurements for both actual learner-selection reports reproduced all four
models, radius/average contrasts, interaction tests, and radius Holm p-values
within relative tolerance `1e-10`; the maximum absolute difference across 104
finite contrast/test fields was `7.11e-15`. Original report files were preserved.
Standalone raw p-values also matched; their historical adjusted fields predate
the decision to assess standalone metrics individually.
