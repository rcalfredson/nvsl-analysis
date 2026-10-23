# Manual circle-entry review of tracking gaps

This workflow follows the endpoint visualization. It tests whether offline
calculated circle entries agree with visible behavior in selected intervals.
It does not assume that wall-associated gaps are harmless and does not change
the analyzer's tracking-loss cutoff, reward counting, or manuscript exports.

## 1. Build a review bundle

Run from the repository root with the project's analysis environment:

```bash
MPLCONFIGDIR=/tmp/frame_loss_spatial_audit_mpl \
  /home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/review_tracking_gap_entries.py build --pilot
```

Default output: `results/tracking_gap_entry_review/pilot/`. Open its `index.html`
for a case list and individual player commands. The pilot uses the three
previously reviewed trajectories, not every fly in the three cohorts. They are
one Ctrl>Kir yoked fly and two antennae-removed experimental flies. Each has
defined final SLI. Another recording can be selected using
`build --video '/full/path.avi' --pair 1 --role exp --output-dir 'new/directory'`.
Fly IDs are zero-based and `--pair` has the same experimental-fly numbering as
the existing analyzer's `-f`; `exp`/`yok` selects which member to inspect.

An initial pilot bundle has already been generated during implementation. Start
with step 2 to review it. To regenerate a sample, supply a new empty directory
with `--output-dir`; the build intentionally refuses to overwrite a bundle.

The pilot uses large2 production geometry with control radius 15 mm. Every
candidate is a gap/circle combination intersecting Training 2, or recovering
exactly at its onset. Both reward and control circles are considered. Recorded
circle state is classified as inside, border, outside, or unavailable. Candidate
strata are:

- stable-state change between observed endpoints;
- a border or unavailable endpoint;
- matching states with an endpoint near a circle boundary;
- matching states and a long gap;
- other matching-state gaps, including outside-to-outside gaps.

The default near-boundary distance is 1 mm and the default long gap is 2 s.
These are configurable triage thresholds, not validated exclusion criteria.
Candidates are randomly sampled within each fly/role/circle/stratum, up to two
per stratum, with a fixed seed. Missing strata naturally have no samples.
`sampling.csv` and `sampled_candidates.csv` preserve selection counts and IDs.
No gap category is declared safe by its endpoint states.

Selected review windows have one second of padding and are clipped to Training
2. Overlapping windows for the same fly/circle are merged to prevent duplicate
counting. A fully observed two-second comparison interval is also sampled per
fly/circle, when possible. It does not overlap other scored intervals for that
fly/circle. More comparison intervals can be requested with `--observed-controls`.
Sampling occurs at the candidate level before merging; this is a diagnostic
sample, not a representative estimate of frame-weighted or visit-weighted error.

`episodes.csv` is the immutable list of scored intervals. Its `sync_buckets`
column indicates the 10-minute actual-reward-synchronized buckets that overlap
each interval. This helps locate SB5 and SB2–5 cases; it does not imply complete
coverage of either endpoint window. Candidate windows can span bucket boundaries.
The synchronization ranges come from the production metadata routine.

## 2. Inspect actual video, frame by frame

```bash
/home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/review_tracking_gap_entries.py play \
  --bundle results/tracking_gap_entry_review/pilot
```

Add `--review-id ID` to start at one case, or `--context-seconds 5` for more
orientation outside the scored interval. This command opens a local OpenCV GUI;
it requires a graphical session. Build and summarize do not require a GUI.

Controls:

| Key | Action |
|---|---|
| Space | Play/pause |
| a / d, left / right | Previous/next frame |
| b / f | Back/forward ten frames |
| n / p | Next/previous review case |
| Digits, then g | Jump to an absolute frame within the displayed context |
| q | Quit |

The selected circle and its outer entry-border limit are yellow; the other circle
is gray. A pink cross is drawn only for a finite observed target position. During
loss, the raw video continues and a label explicitly says tracking is missing;
no interpolated target position is drawn. The header identifies fly, exp/yok,
absolute frame number, and whether the frame is inside the scored interval.
Calculated-entry cues are hidden by default so manual scoring need not follow
the automated result. Use `--show-calculated` to display those cues during a
subsequent comparison. The case index also lists calculated entries. Those are
offline reconstructions, not independently observed events or actual light commands.

The event reconstruction calls the production `Trajectory.calc_in_circle` and
`Trajectory._calcEnEx` methods and holds the pre-gap state over originally
missing frames. It also preserves the first-training pre/training split. Tests
compare it with the actual `_calcRewards` method, including gaps whose
interpolated coordinates would otherwise fabricate circle crossings.

## 3. Transcribe observations into annotations.csv

The bundle also has an editable **worksheet.html**, prepopulated in player case
order with video filename/full path, fly ID, role, circle, scored frames,
training/bucket labels, duration, and case type. It provides fields for review
status, observed entry frames, confidence, notes, and reviewer, plus player-command
copy buttons. Calculated events are not shown in this worksheet.

For a bundle created before the worksheet was added, generate it with:

```bash
/home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/review_tracking_gap_entries.py worksheet \
  --bundle results/tracking_gap_entry_review/pilot
```

It loads the current `annotations.csv` without modifying it. A pre-existing
worksheet requires a different `--output` or explicit `--overwrite`.

You can take notes directly in the browser and download an editable HTML draft
containing the table and current notes, including incomplete observations.
Browser autosave is best-effort and does not update the original disk files.
Use **Download annotations.csv** for a validated file with the pipeline's exact
schema. Keep the downloaded file separately and pass it to summarize using
`--annotations '/path/to/downloaded/annotations.csv'`, or deliberately replace
the bundle's CSV yourself. Export includes every case regardless of table filters.

Alternatively, edit the CSV directly as described below.

Edit **annotations.csv only**, keeping every review ID. Use the index or
`episodes.csv` to identify the target circle and half-open scored interval
`[review_start_frame, review_stop_frame_exclusive)`. Score all entries in the
whole interval, including its padding, not only entries during the selected gap.
If nearby gaps fall in that interval, they are part of the same scoring task.
Context outside the scored interval is for orientation and is not counted.

Count an entry using the same stable-state convention as the analyzer: an
outside state beyond the outer border followed by an inside state within the
nominal circle. Intermediate border observations do not reset the last stable
state. An animal already inside at the start is not automatically an entry.
Use preceding context to establish the transition. Record its first visibly
inside frame; use notes for ambiguous boundary placement. Assess the animal's
center consistently with the centroid used for tracking.

| Column | What to enter |
|---|---|
| review_status | `pending`, `reviewed`, or `unscorable` |
| observed_entry_frames | For reviewed cases, JSON list of absolute integer frames, e.g. `[41234, 41291]`; explicitly `[]` for zero entries |
| confidence | `high`, `medium`, or `low` for reviewed cases |
| reviewer | Your name/initials |
| notes | Visibility, suspected tracking identity errors, uncertain boundary placement, etc. |

Use **unscorable** if the actual behavior cannot be determined for the whole
interval, and leave its entry list blank. Do not turn uncertainty into a zero.
Partially visible behavior can be described in notes but is not a complete count.
The viewer never auto-scores or modifies the annotation file. A build refuses a
nonempty output directory so rerunning cannot erase manual work. For a different
sample, use a new output directory; review IDs remain tied to source identity,
circle, and scored frame interval.

## 4. Compare observations with calculated entries

```bash
/home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/review_tracking_gap_entries.py summarize \
  --bundle results/tracking_gap_entry_review/pilot
```

Outputs: `review_comparison.csv`, `review_summary.csv`, and `review_summary.html`.
Only `reviewed` intervals contribute observed counts. Pending and unscorable
coverage is reported explicitly, separately for gap and observed-control tasks.
Low-confidence reviewed cases are flagged in the summaries.

The comparison reports observed minus calculated counts and order-preserving
event matching within two frames (configurable with `--tolerance-frames`).
Unmatched events are not automatically called false/missed entries: timing
differences can create unmatched events even when total counts agree. Inspect
the offset lists and videos to distinguish timing shifts from count errors.

`manifest.json` records source-file size/mtime snapshots, script/production-code
hashes, geometry, and sampling options. Player startup checks that the source
files have not changed. The episode file's content hash is checked by the player
and summarizer so observations cannot silently be reassigned to edited windows.
Source size/mtime checks are not full recording content hashes. Existing bundles
freeze the calculated event lists produced when they were built.

## Limits and next decisions

### Continue across more flies

`build --cases-csv selections.csv` accepts `video_path,fly,role` columns,
plus optional `cohort` and selection metadata. Use `--case-offset 52` to continue
display numbering from case 53. Repeat `--exclude-bundle PATH` to exclude earlier
scored intervals for the same video/fly/role/circle. The original bundle stays
unchanged. New bundles include `review_table.tsv` for spreadsheets and
`wiki_table.html` for lightweight notes. Their codes distinguish G (missing frames
inside the scored interval), C (gap-related selection with missingness outside
the scored interval), and T (fully tracked comparison). Legacy pilot tables used
G for both G and C; preserve those original labels when importing that pilot.

The 2026-10-22 continuation is `results/tracking_gap_entry_review/batch02_20261022`.
It contains cases 53–151 from six additional final-SLI-eligible trajectories,
one exp and one yok per cohort, chosen by highest loss among previously
unreviewed trajectories. Selection and source hashes are preserved in
`results/tracking_gap_entry_review/selection_20261022`.

### Reconcile spreadsheet notes

```bash
/home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/import_tracking_gap_audit.py \
  --spreadsheet /path/to/tracking_gap_audit.ods \
  --bundle results/tracking_gap_entry_review/pilot \
  --output-dir results/tracking_gap_entry_review/new_reconciliation
```

The importer expects the nine-column notes table and validates case identity,
target circle, and scored bounds against the immutable bundle. It freezes a
copy of the ODS and exports imported annotations, all comparisons, flagged
recheck cases, and HTML/Markdown reports. Blank entry observations are unscorable;
explicit `none` means zero. It does not infer confidence or correct annotations
from prose. Legacy imported annotations can be summarized with the explicit
`summarize --allow-missing-confidence --annotations PATH` option. A count
discrepancy is a reconciliation task, not automatically a tracking-caused error.

For the completed cumulative spreadsheet, repeat `--bundle` to include both
sets of globally numbered cases:

```bash
/home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/import_tracking_gap_audit.py \
  --spreadsheet '/media/Synology4/Robert/nvsl-stash/tracking_gap_audit.ods' \
  --bundle results/tracking_gap_entry_review/pilot \
  --bundle results/tracking_gap_entry_review/batch02_20261022 \
  --output-dir results/tracking_gap_entry_review/new_complete_reconciliation
```

To check just one batch, select its bundle and add `--case-start 123 --case-end 132`
(both bounds are inclusive). Spreadsheet rows outside the selected bundle/range
are explicitly listed as excluded in `selection.json` and the reports. Every
selected bundle case must have a spreadsheet row. For a partially filled sheet,
use an explicit range or `--allow-partial`; the latter lists absent case numbers
and excludes them from every reviewed denominator. It does not invent zero
observations or mark omitted rows reviewed. Duplicate global case numbers across
bundles are rejected, even outside the selected range. A legacy bundle without
`case_number` uses its original one-based row order.

An empty observation in an included row is different from an absent row: blank,
`unknown`, `n/a`, and `unscorable` mean unscorable; `none` or `[]` explicitly means
zero entries. Notes and uncertainty are preserved, but prose is never used to
automatically change the entry column or confidence. Inspect uncertain notes
before calling a matching count validated.

Imported annotation files contain only selected cases. The standard `summarize`
command requires every case of one bundle, so a partial or combined import should
be assessed using its reconciliation report rather than passed directly to that
command. A complete single-bundle import can use `--allow-missing-confidence`.

Reconciliation separates count disagreements, matching counts with unmatched
events, matching counts with small offsets only, and exact agreement. Unmatched
events take precedence over small matched offsets. Recheck exports include count
disagreements, temporal mismatches, and heuristic note flags, including notes
claiming no interactions in the named circle with a nonempty entry list. These
flags prompt human review; they do not establish tracking-error causation.

### Code and record layout

- `src/utils/tracking_gap_review_io.py`: shared CSV and hash-checked bundle I/O.
- `src/analysis/tracking_gap_review.py`: candidate selection, production-entry
  reconstruction, and annotation/event comparison.
- `src/analysis/tracking_gap_review_import.py`: ODS reading, case selection,
  strict identity matching, and discrepancy reconciliation.
- `src/exporting/tracking_gap_review_report.py` and
  `src/exporting/tracking_gap_review_worksheet.py`: report and worksheet rendering.
- The scripts provide command-line orchestration; the player remains in
  `scripts/review_tracking_gap_entries.py`.

Episode records use `review_id` as stable identity, `case_number` as display
identity, and `review_start_frame`/`review_stop_frame_exclusive` as zero-based
half-open scored bounds. `source` distinguishes gap selections from tracked
comparisons. New bundles additionally store `scored_missing_frames` and
`interval_type`; legacy missingness is reconstructed from candidate gap ranges.
`calculated_entry_frames` contains frozen production-rule reconstructions, not
ground truth. Annotation records attach status, observed-entry JSON, confidence,
reviewer, and notes to an exact `review_id`.

Reconciliation outputs go to a new/empty directory. Source ODS bytes are read
once, parsed, frozen, and hashed consistently. `provenance.json` records source
and bundle hashes and selection options. Existing bundles and saved observations
are never rewritten by the importer.

This phase produces manually assessed examples and a reproducible discrepancy
ledger. It does not extrapolate a cohort error rate from the stratified sample,
fill in unreviewed intervals, or silently revise PI/SLI. Before estimating a
change to final or mean SLI, establish review coverage over the relevant complete
bucket windows for **both** members of each experimental/yoked pair, retain the
original count/eligibility rules, and track unresolved intervals explicitly.
Frame timing shifts near bucket boundaries require separate attention.

The fully observed comparison tasks help check the counting convention and
reviewer consistency, but a handful of short windows cannot establish an
absence of errors. This validation of circle entries also does not validate
interpolated path length, tortuosity, wall contacts, or occupancy metrics.
