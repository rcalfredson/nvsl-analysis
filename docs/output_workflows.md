# Output Generation Workflows

This guide highlights four important entry points for producing analysis
artifacts. Use the paper-panel notebook for the numbered figure panels and one of
the secondary workflows below for GraphPad tables, the broader exploratory
analysis matrix, or the configurable turn home-vector analysis.

Run all commands from the repository root with the project environment activated
and the Cython extensions built (see the top-level `README.md`). The shell scripts
are Bash scripts and are intended for Linux/macOS or a Bash environment on
Windows.

| Entry point | Purpose | Main outputs |
| --- | --- | --- |
| [`notebooks/paper_figure_panels.ipynb`](../notebooks/paper_figure_panels.ipynb) | Main, numbered paper-panel workflow | Panel `.npz` bundles and figures |
| [`notebooks/graphpad_exports.ipynb`](../notebooks/graphpad_exports.ipynb) | GraphPad Prism-friendly per-fly tables | CSV files under `exports/` |
| [`scripts/run_analysis_matrix.sh`](../scripts/run_analysis_matrix.sh) | Curated multi-analysis batch for turnback, home-vector, and tortuosity analyses | Dated bundles and plots under `exports/`; optional debug galleries under `imgs/` |
| [`scripts/run_turn_home_vector_alignment_analysis.sh`](../scripts/run_turn_home_vector_alignment_analysis.sh) | Focused, configurable turn home-vector alignment analysis | Bundles, plots, and statistics under `exports/turn_home_vector_alignment/` by default |

## GraphPad exports notebook

`notebooks/graphpad_exports.ipynb` is a small command orchestrator, not an
independent implementation of the metrics. It supports two workflows:

- Convert the per-fly scalar bundles produced by panels 21 and 26 of
  `paper_figure_panels.ipynb` into column-oriented GraphPad CSV files.
- Generate and convert percent-time-on-agarose results for the agarose HTL and
  flat HTL chambers. These analyses are not part of a numbered paper panel.

To use it:

1. Start Jupyter from the repository root and open
   `notebooks/graphpad_exports.ipynb`.
2. Run the setup cells, including `%cd ..`.
3. For the panel-derived exports, first run the Panel 21 or Panel 26 data-export
   stage in `paper_figure_panels.ipynb`. Check the dated `.npz` paths in the
   GraphPad notebook and update them if the panel exports use a different date.
4. Run the relevant GraphPad section once in its default preview mode. The
   notebook displays every command and whether its expected output exists.
5. Change only that section's `RUN_...` variable from `False` to `True`, then
   rerun the cell to execute it.

The percent-time-on-agarose sections contain repository-author paths under
`/media/Synology4`. Replace those video globs and intermediate CSV destinations
when using another data mount. Each upstream command runs `analyze.py --agarose`,
copies the root-level `learning_stats.csv` immediately to a group- and
chamber-specific path, and the final stage calculates the per-fly percentage-point
change `pre last 10m - T3 post last 10m`.

The final CSVs are written to `exports/`. All execution toggles default to
`False`, so opening and running the notebook does not start an analysis until a
toggle is explicitly enabled.

## Analysis matrix script

`scripts/run_analysis_matrix.sh` is a curated batch recipe rather than a generic
command-line interface. Its active blocks define the exact analysis set. At the
time of writing, the default run produces:

- dual-circle turnback ratios for 3/5, 8/10, and 13/15 mm radius pairs;
- turnback home-vector heading-alignment outputs for all flies, top-20% SLI
  flies, and bottom-50% SLI flies;
- return-leg tortuosity by maximum-distance bin; and
- post-wall-departure tortuosity.

Several older matrix variants remain commented out at the bottom of the script.
They are reference recipes and do not run unless the script itself is edited.

### Dataset inputs

The default matrix needs these shell variables:

```bash
export INTACT_CTRL='/path/to/intact-control/c*.avi'
export INTACT_PFND='/path/to/intact-pfnd/c*.avi'
export AR_CTRL='/path/to/ar-control/c*.avi'
export MBKC_KIR='/path/to/mbkc-kir/c*.avi'
```

Values use the same comma-separated paths and glob syntax accepted by
`analyze.py -v`. You can instead place the exports in a private, untracked shell
file and source it before the run:

```bash
source /path/to/my_video_lists.sh
scripts/run_analysis_matrix.sh
```

Unlike the focused script described below, the matrix does not automatically
load the three standard `export ...` lines from `video_lists.log`. With the
default `TURNBACK_COMPARISON_GROUP=mbkc_kir`, it can derive `MBKC_KIR` from the
matching header/subheader section of `video_lists.log`; setting `MBKC_KIR`
explicitly is less dependent on that file's layout. Set
`TURNBACK_COMPARISON_GROUP=ar_ctrl` to use the antennae-removed cohort instead.
This comparison setting affects the active turnback and home-vector blocks; the
active tortuosity blocks continue to use Ctrl, PFNd>Kir, and antennae-removed
cohorts.

### Recommended first run

Preview the generated commands before starting the long batch:

```bash
source /path/to/my_video_lists.sh
PRINT_ONLY=1 DATE_TAG=trial scripts/run_analysis_matrix.sh
```

Then run with a stable date tag. The tag is part of the filenames and defaults to
the current date:

```bash
DATE_TAG=2026-07-20 scripts/run_analysis_matrix.sh
```

The most useful restricted modes are:

```bash
# Only the default dual-circle turnback analysis
RUN_DUAL_CIRCLE_TURNBACK_ONLY=1 scripts/run_analysis_matrix.sh

# Only the home-vector analysis (all, top 20%, and bottom 50%)
RUN_TURNBACK_HOME_VECTOR_ALIGNMENT_ONLY=1 scripts/run_analysis_matrix.sh

# Replot matching dated bundles without re-exporting them
DATE_TAG=2026-07-20 RUN_DUAL_CIRCLE_TURNBACK_ONLY=1 \
  DUAL_CIRCLE_TURNBACK_REUSE_EXISTING_BUNDLES=1 \
  scripts/run_analysis_matrix.sh
```

`RUN_DUAL_CIRCLE_TURNBACK_ONLY` and
`RUN_TURNBACK_HOME_VECTOR_ALIGNMENT_ONLY` are mutually exclusive. Home-vector
bundles are reused by default when their exact dated filenames already exist;
set `TURNBACK_HOME_VECTOR_ALIGNMENT_REUSE_EXISTING_BUNDLES=0` to force their
regeneration. Use `DUAL_CIRCLE_TURNBACK_IMG_FORMAT=pdf` or
`TURNBACK_HOME_VECTOR_ALIGNMENT_IMG_FORMAT=pdf` for vector versions of those
plots. The configurable example-gallery variables are declared at the top of the
script.

The separate `RUN_FLAT_HTL_TURNBACK_PAIRS=1` branch uses
`FLAT_HTL_CTRL`, `FLAT_HTL_HIND_TARSI_GENITALIA_GLUED`, and
`FLAT_HTL_ANTENNAE_REMOVED`, then exits without running the default large-chamber
matrix. Its current executable analysis is the optional home-vector-threshold
variant, enabled with `RUN_FLAT_HTL_TURNBACK_HOME_VECTOR_VARIANT=1`.

## Focused turn home-vector alignment script

`scripts/run_turn_home_vector_alignment_analysis.sh` is the preferred entry point
when the desired output is specifically turn home-vector alignment and the
detection, grouping, or radial-bin settings need to be varied. It exports both
experimental and experimental-minus-yoked bundles, then creates overview plots,
changes from pre-training, and TSV statistics. By default it includes all flies,
top-20% SLI, and bottom-50% SLI subsets for Ctrl, PFNd>Kir, and antennae-removed
cohorts.

The script reads lines beginning exactly with `export INTACT_CTRL=`,
`export INTACT_PFND=`, and `export AR_CTRL=` from `video_lists.log`. Alternatively,
export those variables in the calling shell or point `VIDEO_LISTS_FILE` at a
different file:

```bash
PRINT_ONLY=1 scripts/run_turn_home_vector_alignment_analysis.sh
scripts/run_turn_home_vector_alignment_analysis.sh
```

The default output directory is `exports/turn_home_vector_alignment/`. Output
names include the detector configuration and `DATE_TAG`, so plotting or bundle
reuse must use the same settings and date tag as the export run.

Common configurations include:

```bash
# Compare against MBKC>Kir instead of antennae removed.
COMPARISON_GROUP=mbkc_kir MBKC_KIR='/path/to/mbkc-kir/c*.avi' \
  scripts/run_turn_home_vector_alignment_analysis.sh

# Split outputs into radial ranges.
TURN_RADIAL_BINS='3:5,8:10,13:15' \
  scripts/run_turn_home_vector_alignment_analysis.sh

# Replot an existing, exactly matching dated bundle set.
DATE_TAG=2026-07-20 PLOT_ONLY=1 \
  scripts/run_turn_home_vector_alignment_analysis.sh

# Keep analysis enabled but skip groups whose complete bundle sets exist.
DATE_TAG=2026-07-20 REUSE_EXISTING_BUNDLES=1 \
  scripts/run_turn_home_vector_alignment_analysis.sh
```

Important controls are:

| Variable | Default | Effect |
| --- | --- | --- |
| `COMPARISON_GROUP` | `ar_ctrl` | Third cohort: `ar_ctrl` or `mbkc_kir` |
| `PLOT_ONLY` | `0` | Skip analysis and require the matching bundles |
| `REUSE_EXISTING_BUNDLES` | `0` | Reuse a group's complete matching bundle set |
| `OUT_DIR` | `exports/turn_home_vector_alignment` | Bundle, plot, and TSV destination |
| `TURN_FILTER` | `all` | Use all turns or `exclude_wall_contact` |
| `TURN_HOME_TARGET` | `reward_center` | Home target; `opposite_reward_center` is the control |
| `TURN_RADIAL_BINS` | empty | Optional comma-separated `lo:hi` or `lo-hi` ranges in mm |
| `TURN_RADIAL_BIN_ASSIGNMENT` | `full_containment` | Or `max_distance_point` |
| `BEHAVIOR_STATE_DETECTOR` | `haberkern` | Turn detector; `schmitt_butterworth` enables the `SB_...` settings |
| `RUN_SLI_SUBSETS` | `1` | Also export the top/bottom SLI groups |
| `TOP_SLI_FRACTION` / `BOTTOM_SLI_FRACTION` | `0.2` / `0.5` | Learner subset sizes |
| `TRAININGS` | `1,2` | Training panels included alongside pre-training |
| `FLIES` | `0-1` | Fly selection passed to `analyze.py` |
| `RCC` | `15` | Reward-circle radius passed to `analyze.py` |

See the variable block at the top of the script for detector thresholds and sync
bucket controls. For the Schmitt-Butterworth detector's interpretation and
parameters, see
[`behavior_state_detector_schmitt_butterworth.md`](behavior_state_detector_schmitt_butterworth.md).

## Reproducibility notes

- Keep `DATE_TAG` fixed between export, reuse, and plot-only runs.
- Use `PRINT_ONLY=1` to inspect expanded commands. It suppresses analysis and
  plotting commands, though the focused script may still create its output and
  Matplotlib configuration directories.
- Do not run multiple `graphpad_exports.ipynb` upstream agarose stages
  concurrently: each `analyze.py` invocation writes the same root-level
  `learning_stats.csv` before it is copied.
- Treat the dataset path lists as local configuration. Avoid committing private
  mount paths or a `video_lists.log` containing environment-specific data.
