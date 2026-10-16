# Walking-scale sensitivity audit

Use this explicitly invoked audit to compare the production walking classifier
with a template-scale-adjusted threshold. It does not change the classifier,
tracking rules, Methods, or manuscript figures. Generated reports describe
measurements; a scientific recommendation should be written separately after
reviewing them.

## Scope and assumptions

`scripts/audit_walking_scale.py` reads the 14 supported heatmap recipes for
Fig. 4b, ED13a, and ED15d/g/j/m from
`notebooks/manuscript_figure_panels.ipynb`. It evaluates only literal data
assignments and string concatenation/quoting, never notebook cells. The recipe
names are explicitly listed in `RECIPE_NAMES`; changes to the manuscript's
recipe structure should be reviewed rather than guessed automatically.

These recipes cover HTL and large-chamber recordings with experimental/yoked
pairing. Each recipe must have one panel, one period, one selected training
(except pre-training), and a supported `-f 0-9` or `-f 0-1` selector. Related
pre/training/post recipes share trajectory initialization within a cohort.
They must use the same recording lists and loading/calibration settings. This
is a manuscript audit adapter, not a general replacement for `analyze.py`.

For the same camera-pixel/second speed array, the masks are:

```text
current = speed > 2 * reference_pixels_per_mm
scaled  = speed > 2 * reference_pixels_per_mm * xf.fctr
```

Both comparisons are strict. The physical interpretation is conditional on
the chamber constants calibrating template pixels at the floor. Under that
interpretation, the current physical threshold is `2 / xf.fctr` mm/s.

## Running

Use the repository's analysis environment, with its existing compiled analysis
extensions, NumPy, OpenCV, matplotlib, and other `analyze.py` dependencies.
The original audit used the `analysis3.13` conda environment. Run from the
repository root with source recordings mounted at the notebook recipe paths.

Prefer a fresh output directory for each reviewed run. Generated outputs are
ignored by Git under `exports/`; do not commit recording-level data or binary
maps as part of the audit-code change.

```bash
python scripts/audit_walking_scale.py \
  --output exports/diagnostics/walking_scale_review
python scripts/summarize_walking_scale_audit.py \
  exports/diagnostics/walking_scale_review
```

To rerun selected recipes, pass their Python variable names:

```bash
python scripts/audit_walking_scale.py \
  --output exports/diagnostics/walking_scale_selected \
  --recipes figure_4b_training figure_4b_post
```

`--max-videos 1` provides a partial smoke run per selected cohort. It disables
aggregate historical-cache comparisons; the report generator rejects partial
runs. `--no-plots` on the report generator writes tables and text without
diagnostic charts.

## Validation and side effects

The collector uses the production loader, interpolation, distance and speed
calculations, note exclusions, window selection, coordinate transformations,
histogram bounds/binning, and numeric `calcHm()` baseline. It bypasses the full
`VideoAnalysis` constructor to avoid unrelated analyses and output generation.
The initialization adapter is isolated in `load_analysis()` and should be
reviewed when production initialization changes.

Every available full-trajectory current mask must equal `Trajectory.walking`
exactly before comparison. Every compared baseline histogram and denominator
must equal production exactly. The collector verifies that walking arrays
remain unchanged. Original missingness is descriptive: interpolated frames
remain eligible, and NaN-derived distances become zero as in production.

The report generator additionally requires one matching historical cache check
for every selected group map, with matching recipe/source/config provenance.
An input or cache disagreement is recorded by the collector and prevents a
validated report. It does not rebuild caches. Source tracking changes may
therefore require separate review; matching analysis-source hashes alone do
not establish identical input data.

The historical source signature covers more than walking and heatmap code.
An unrelated analysis change can therefore flag a source mismatch even when
every numerical map agrees. Preserve and review that flag rather than editing
provenance to force acceptance. A frozen prior audit can still be summarized
using its original, internally consistent records.

All generated files and incidental working files go beneath a dedicated
`exports/diagnostics/` subdirectory. No production arrays are replaced and no
manuscript output is rendered. Diagnostic difference charts have their own
explicitly labeled scales; they are not manuscript heatmap reproductions.

## Outputs and metric definitions

| File | Contents |
|---|---|
| `flies.csv` | One processed fly/window record, including rejected windows and frame accounting. |
| `window_totals.csv` | Pooled counts by panel/cohort/role/window; these are frame-weighted accounting totals. |
| `groups.json` | Equal-fly group-map comparisons and individual-value distributions. |
| `regions.csv` | Per-fly mass and absolute differences in spatial regions. |
| `regional_summary.json` | Regional baseline shares, absolute-change shares, and relative mass changes. |
| `scales.json` | Recording-level transformation metadata by cohort. |
| `skipped.csv` | Initialization skips, when present. |
| `maps/*.npz` | Per-fly maps, group maps, eligibility masks, and indices into `flies.csv` (zero-based data rows). |
| `provenance.json` | Exact commands, input lists, fingerprints, cache checks, exclusions context, and validation counts. |
| `initialization.log` | Captured production initialization messages. |
| `REPORT.md` | Generated factual tables and validation/scope information. |

In `flies.csv`, `fi` is inclusive and `la` is exclusive. `role=0` is
experimental and `role=1` is yoked; `absolute_fly` indexes the recording's
tracking arrays. `original_missing` follows the production original-x-NaN
mask. Finite position/speed counts describe processed arrays; a finite zero
speed is not proof of valid original tracking.

`walking_*` counts all classified window frames. `selected_positions_*` counts
positions remaining after walking selection and the production post/NaN policy.
`histogram_counts_*` counts positions actually inside the histogram bounds.
`oob_or_nonfinite_*` is the difference between the last two counts. The
`changed_*` fields report mask disagreement and its association with original
tracking gaps; `added` and `removed` give the direction. Speed summaries for
changed frames are in camera pixels/second. Fractions use all window frames;
`walking_delta_pp` is already in percentage points.

For each fly, the map is histogram counts divided by **all window frames**.
Each rule's group map is the arithmetic mean of that rule's nonempty maps.
Thus changes in eligibility are reported separately. `common_fly_metrics`
compares means restricted to flies eligible under both rules.

For current map `A` and scaled map `B`:

- `mass_current`, `mass_scaled`: sums of all bins.
- `relative_mass_delta`: `sum(B) / sum(A) - 1`.
- `l1`: `sum(abs(B - A))`; `relative_l1` divides this by `sum(A)`.
- `shape_tv`: `0.5 * sum(abs(B / sum(B) - A / sum(A)))`. This isolates spatial
  redistribution after removing the total-mass difference.
- `centroid_shift_mm`: displacement between unit-mass map centroids, using
  actual template bin widths and the chamber's reference pixels/mm.

Stored relative changes and TV distances are **fractions**, not percentages.
Reports multiply them by 100. Undefined zero-baseline ratios are `null`/`n/a`.
Distributions include finite individual values and report their sample count;
they are descriptive, not independent calibration replicates.

Floor interior, the inward 2 mm edge band, and outside-floor bins form a
partition based on bin centers. Reward-disk and outer-annulus regions overlap
that partition. Pre-training uses the upcoming T1 target. These are diagnostic
regions, not boundary-contact classifications. Absolute-change shares sum
absolute per-fly changes so opposing directions do not cancel.

## Maintenance and testing

The collector separates recipe preparation, production initialization,
full-mask validation, window comparison, and group aggregation. The report
generator separately validates inputs, derives accounting tables, writes text,
and plots differences. Neither embeds an automatic scientific recommendation.

```bash
python -m pytest test/test_walking_scale_audit.py -q
```

For a numerical refactor, collect into a new directory and compare its CSV/JSON
values and all NPZ arrays with a reviewed prior run. Timing and script hashes
will differ; masks, counts, eligibility, normalized maps, and map metrics should
not. Keep the original audit artifacts intact during that check.
