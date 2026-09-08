# Paired heatmaps across recordings

Use this workflow when the same physical flies occur in separate before/after
recordings. Pairing is opt-in and eligibility is based on the experimental fly.
Each retained recording contributes both its experimental and yoked-control
heatmaps. Existing unpaired heatmaps retain their behavior.

## Pair identities

Create a CSV with one explicit physical-fly pair per row:

```csv
pair_id,before_video,before_fly,after_video,after_fly
20250719_c31_f0,/data/2025-07-19/c31_before.avi,0,/data/2025-07-19/2nd round/c31_after.avi,0
20250719_c31_f1,/data/2025-07-19/c31_before.avi,1,/data/2025-07-19/2nd round/c31_after.avi,1
```

Use actual full video filenames, not globs, and the experimental fly IDs used
by `-f` / the CSV `fly` column. Absolute paths are recommended; relative paths
are resolved against the working directory of the analysis command. No automatic
matching by filename, timestamp, or list order is performed. Pair IDs and each
side's recording/fly identities must be unique. Recordings absent from an
eligibility report are excluded and identified in the audit. Analyzed flies
absent from the manifest are not included in the paired plots.

Eligibility reports retain attempted recording/fly identities even when analysis
returns early. A rejected experimental trajectory is recorded as
`rejected_trajectory`, with both eligibility flags false. Other early skips are
recorded as `analysis_skipped` (or `not_experimental_fly` for a yoked-only input).
These records can be matched normally, and their pairs are excluded with the
actual recorded reason rather than appearing unmatched. This does not add skipped
flies back to ordinary analysis outputs or heatmaps. If every attempted fly is
skipped, an eligibility export still writes their records; select one training
explicitly with `--num-trainings` when training metadata is unavailable.

For the July/August 2025 film-slide experiments, the helper supports three
cohorts and gives each a separate default output directory:

| Cohort argument | Default output directory |
|---|---|
| `film-slide` | `exports/film_slide_paired` |
| `mock-slide` | `exports/mock_slide_paired` |
| `antennae-removed` | `exports/antennae_removed_slide_paired` |

For example, `bash scripts/run_film_slide_paired_heatmaps.sh mock-slide manifest`
generates candidate pairs from that cohort's existing before/after JSON reports,
matching date, camera, and fly ID. It writes identities present on both sides to
`pairs.csv` and identities present on only one side to `unmatched.csv`. Review
this mapping to confirm physical-fly identity. Ambiguous matches still stop
generation. If there are no shared identities, the helper writes a header-only
`pairs.csv` and the unmatched report, then exits with an error. Matched identities
remain in the manifest even when their occupancy or heatmap is ineligible, so
the rendering audits can explain those exclusions separately.

The older one-argument form remains an alias for `film-slide`; for example,
`bash scripts/run_film_slide_paired_heatmaps.sh export` still runs the standard
film-slide cohort. `FILM_PAIR_DIR` can override any cohort's output directory.
All three helper cohorts render heatmaps with Arial at 20 pt by default.

## Run the two eligibility exports

Append `--hm-pair-export exports/film_before.json` to the existing before command,
retaining its video list and analysis options, including:

```text
--pltHm --hm-periods training --num-trainings 2
--hm-sync-bucket 5 --hm-sync-bucket-tail-minutes 5
--hm-pair-export exports/film_before.json
```

Append `--hm-pair-export exports/film_after.json` to the after command:

```text
--pltHm --hm-periods training --num-trainings 1
--hm-sync-bucket 1 --hm-sync-bucket-head-minutes 5
--hm-pair-export exports/film_after.json
```

Biological T3 is local training 1 in the second recording. Export mode runs the
analysis and writes eligibility instead of rendering positional heatmaps; other
normal analysis outputs, including `learning_stats.csv`, can still be written.
It does not generate new heatmap images or remove older ones.

## Render both sides from the shared reports

Rerun each original command, removing `--hm-pair-export` and appending:

```text
--hm-pair-manifest pairs.csv
--hm-pair-before-report exports/film_before.json
--hm-pair-after-report exports/film_after.json
--hm-pair-side before
--hm-pair-audit exports/film_before_pair_audit.csv
```

For the after command use `--hm-pair-side after` and
`--hm-pair-audit exports/film_after_pair_audit.csv`. Both runs must use the same
manifest and two reports. Copy/rename each run's usual heatmap output before
running the other side, as for ordinary heatmaps.

The helper's render stage uses logarithmic color scaling by default. Pass
`linear` after the render stage to additionally produce a linear-scale version:

```bash
bash scripts/run_film_slide_paired_heatmaps.sh film-slide render linear
```

Linear figures receive a `_linear` filename suffix, such as
`T2_SB5_last5min_paired_linear.pdf`, and therefore do not overwrite the default
log-scale figures. The pairing reports and audits are shared because scale does
not change cohort eligibility. The older `... render` command and the explicit
`... render log` form both retain the existing filenames.

The helper uses `0` to `4e-4` for linear renders so lower-density spatial
structure occupies more of the color scale. Logarithmic renders retain their
existing `1e-6` to `1e-3` bounds. Each mode uses the same bounds for both
timeframes and every cohort so the resulting heatmaps remain comparable.

Each render verifies its selected training, bucket, head/tail duration, bucket
length, recording/fly identities, eligibility, and heatmap count digests against
its saved report. It fails if they differ. Reports are snapshots: regenerate
**both** after changing source tracking, exclusions, or analysis settings. A run
cannot revalidate the opposite recording's source data; use a consistent set of
reports for both renders.

Reports now use version 2 to include early-skipped identities. Regenerate both
eligibility reports and the candidate manifest if you generated them with the
earlier implementation; rendering rejects version 1 reports.

## Inclusion policy

A pair is included only if both sides have:

- A trajectory not marked bad, synchronization to an actual reward, and a
  complete selected sync bucket, using the slide-circle calculation's bounds.
- At least one frame in that full bucket not marked missing by the trajectory's
  `nan` mask. This tests whether circle occupancy would be defined; it does not
  require `--pref-circle-slide` or a radius, since radius does not affect whether
  the occupancy denominator exists.
- A usable heatmap in the selected head/tail window: nonzero accumulated walking
  position counts under the existing heatmap calculation.

No reward PI minimum-entry threshold or experimental/yoked pairing is applied.
The experimental fly alone determines whether the recording pair is retained;
both the experimental and corresponding yoked-control heatmaps are then rendered
using those retained recordings. Yoked-control trajectory validity does not alter
the paired cohort, although unusable yoked maps continue to be omitted by the
ordinary heatmap aggregation. The full-bucket check precedes the five-minute
heatmap check. Thus a fly can pass occupancy eligibility but fail heatmap
eligibility. Heatmaps continue to use their existing walking-frame selection and
normalization; their values are not circle occupancy.

The audit contains every manifest row, `included`, `before_reason`, and
`after_reason`. Reasons include `missing_recording`, `rejected_trajectory`,
`missing_sync`, `incomplete_bucket`, `no_valid_bucket_frames`, and
`empty_heatmap_window`, as well as the early-skip reasons described above.
The JSON reports also expose `occupancy_ok` and
`heatmap_ok` separately. If no pairs survive, the audit is written and plotting
fails with a clear error. For comparison to an occupancy-delta plot, use the audit
to identify any additional heatmap exclusions.
