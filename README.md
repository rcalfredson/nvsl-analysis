# nvsl-analysis
Analysis code for "Spatial learning in feature-impoverished environments in Drosophila" (Yang Chen, Robert Alfredson, Dorsa Motevalli, Ulrich Stern, Chung-Hui Yang, bioRxiv 2024.09.28.615625; doi: https://doi.org/10.1101/2024.09.28.615625)

## Setup

This document provides instructions for setting up the data analysis environment for this project. The software has been tested on **Ubuntu 22.04** and **Windows 11**, but the instructions should be broadly applicable to other OS versions. No special hardware is required beyond a standard desktop or laptop computer.

### Prerequisites

- **Python 3.10** (either system-installed or provided via Conda)
- A C compiler (e.g., `gcc` on Linux, Xcode command line tools on macOS, or Visual Studio Build Tools on Windows)

---
### Installation Methods
You can set up this project using either of two recommended methods:
- **Method 1**: Virtualenv (requires system-wide Python installation)
- **Method 2**: Conda (provides isolated Python environments and package management)
Choose the method most suitable for your needs.
---

### Method 1: Virtualenv
Follow this method if you already have a system-wide Python installation and prefer using `virtualenv`.

#### 1. Install Python 3.10

Use your OS package manager (apt, brew, etc.) or download from the [official Python website](https://www.python.org/downloads/).

#### 2. Install `virtualenv` (if necessary)

```bash
pip install virtualenv
```

#### 3. Create a Virtual Environment

- Linux/macOS

  ```bash
  python3 -m venv myenv
  ```
- Windows

  ```cmd

  python -m venv myenv
  ```

#### 4. Activate the Virtual Environment

- Linux/macOS

  ```bash

  source myenv/bin/activate
  ```

- Windows

  ```cmd

  .\myenv\Scripts\activate
  ```

#### 5. Install Dependencies

```bash

pip install -r requirements.txt
```

#### 6. Build Cython Extensions

```bash

python scripts/setup.py build_ext --inplace
```
---
### Method 2: Conda

Follow this method for a more isolated environment, particularly if you do not have Python 3.10 installed system-wide or prefer easy package management.

#### 1. Install Miniconda
Download and install from the [official Miniconda website](https://docs.conda.io/en/latest/miniconda.html), selecting the appropriate installer for your OS.

#### 2. Create and Activate a Conda Environment
```bash
conda create -n nvsl-analysis python=3.10
conda activate nvsl-analysis
```

#### 3. Install Dependencies
With the environment activated, install packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### 4. Build Cython Extensions
```bash
python scripts/setup.py build_ext --inplace
```
---
### (Optional) Create Drive Mapping File

By default, the script will use OS-specific path conventions (e.g., `/home/you/your_video.avi` vs. `C:\Users\you\your_video.avi`). To universally use Unix-style paths on Windows, create a JSON file `drive_mapping.json`:

```json
{
    "/media/Synology4/": "Y:\\\\"
}
```

### (Optional) Create a Local Analyze Config File

If you want a machine-local override for selected `analyze.py` defaults without checking those preferences into Git, create a file named `.analyze.local.env` in the repository root. This file is ignored by Git.

At present, this local config supports toggling the default one-shot between-reward SLI-bracketed plots:

```env
# Local-only defaults for analyze.py
ENABLE_DEFAULT_BETWEEN_REWARD_SLI_PLOTS=false
```

Accepted boolean values are `1`, `true`, `yes`, `on`, `0`, `false`, `no`, and `off`.

Behavior:
- If `.analyze.local.env` is absent, the repository default behavior is used.
- If `ENABLE_DEFAULT_BETWEEN_REWARD_SLI_PLOTS=false`, the default one-shot between-reward SLI-bracketed plots are skipped for your local runs only.

---

### Typical Installation Time
If Python 3.10 is **already installed**, setting up the virtual environment and installing dependencies (including building Cython extensions) typically takes about **2 minutes** on a normal desktop computer. If you need to install Python or Miniconda from scratch, plan for an additional **3–5 minutes**, depending on your internet speed and operating system.

---

### Deactivation

Exit your environment:
- **Virtualenv:**
  ```bash
  deactivate
  ```
- **Conda:**
  ```bash
  conda deactivate
  ```

### Troubleshooting

- **Dependency errors**: Double-check you have Python 3.10 and your virtual environment is activated before installing.
- **Cython compilation issues**: Confirm that you have a working C compiler (e.g., gcc, Xcode CLT, or VS Build Tools).

---

## Basic Usage
Once your environment is ready, you can run a standard analysis with:
```bash
python analyze.py -v "/actual/path/to/your_video.avi" -f "0-9"
```
Where:
- `-v` is **required** and specifies the path to your video file.
- `-f` is **required** if analyzing a chamber that supports multiple flies; use a range like `0-9`.

### Additional Flags: Turn Detection

You can detect turn events in different areas with the --turn flag:
- `circle`
- `agarose`
- `boundary`

For example:
```bash
python analyze.py -v "/actual/path/to/your_video.avi" -f "0-9" --turn circle
```
For more details on optional flags, see the top of `analyze.py` or run `python analyze.py --help`.

---

## Analysis Workflow Catalog

Most analyses in this repository follow one of three patterns:

- **One-shot plots**: run `analyze.py` with the relevant flag and it writes figures, CSVs, logs, or debug files directly.
- **Bundle then plot**: run `analyze.py` once per cohort to export an `.npz` bundle, then call a `scripts/plot_*` script to overlay bundles across cohorts, SLI extremes, or conditions.
- **Cached result overlay**: run `analyze.py` with an export flag for a specialized plot result, then re-run `analyze.py` with matching import flags to overlay cached results without recomputing trajectories.

The common base command is:

```bash
python analyze.py -v "/actual/path/to/videos/c*.avi" -f "0-9" --gl "Group A"
```

For cross-group bundle workflows, repeat the `analyze.py` export command for each group, storing each bundle under a distinct filename. The plotting scripts usually accept either comma-separated bundles:

```bash
python scripts/plot_com_sli_bundles.py \
    --bundles exports/group_a_com_sli.npz,exports/group_b_com_sli.npz \
    --labels "Group A,Group B" \
    --metric commag \
    --out imgs/group_com_sli_overlay.png
```

or repeatable `Label=path` inputs:

```bash
python scripts/plot_overlay_training_metric_hist.py \
    --input "Group A=exports/group_a_reward_count_hist.npz" \
    --input "Group B=exports/group_b_reward_count_hist.npz" \
    --out imgs/reward_count_hist_overlay.png
```

### One-Shot Workflows

These metrics can be run directly from `analyze.py`; outputs are usually written under `imgs/` plus any CSV/NPZ paths named by the flag.

| Metric or view | Main `analyze.py` flags | Typical output | Notes |
| --- | --- | --- | --- |
| Standard learning/reward summaries | default command, often with `--rpd` | `learning_stats.csv`, reward and reward-per-distance plots | Good first pass for a new experiment. |
| SLI/reward PI by sync bucket | `--rpd`; optional `--best-worst-sli` | `imgs/rewards_per_dist...`, SLI-bracketed reward plots | `--best-worst-sli` also enables top/bottom SLI overlays for supported reward metrics. |
| Walking speed, stopped fraction, rewards/min | default post-analysis metrics | log summaries and standard plots | These are part of the core per-training summary path. |
| Circular motion, angular velocity, turn radius | `--circle` | angular velocity and turn-radius figures | Use with circular-motion analyses; `--jab` selects the JAABA classifier. |
| Turn events by area | `--turn circle`, `--turn agarose`, or `--turn boundary` | turn plots and turn statistics | Area-specific turn detection. |
| Turn probability by distance | `--turn-prob-by-dist 2,3,4,5,6` | distance-binned turn probability plots | Commonly used with `--turn circle`. |
| Outside-circle duration | `--outside-circle-radii ...` | outside-circle duration plots | Radius-thresholded residence outside the reward circle. |
| Reward raster | `--reward-raster` | `imgs/reward_raster.png` by default | Supports training selection, first-N rewards, SLI subset, and sorting flags. |
| First-N reward diagnostics | `--first-n-reward-diagnostics` | CSV plus `imgs/first_n_reward_diagnostics.png` | Anchors each fly to its nth reward; can also write an optional NPZ. |
| First-N reward SLI comparison | `--first-n-reward-sli-comparison` | CSV plus `imgs/first_n_reward_sli_comparison.png` | Compares a selected reward-timing/count metric against SLI. |
| Between-reward trajectory examples | `--btw-rwd-plots` | sampled trajectory panels under `imgs/` | Use `--btw-rwd-mode` to choose random, first-N, last-N, or first/last panels. |
| Between-reward distance histogram | `--btw-rwd-dist-hist` | `imgs/btw_rwd_dists.png` | Can export histogram data with `--btw-rwd-dist-export-hist`. |
| Normalized between-reward distance histogram | `--btw-rwd-norm-dist-hist` | `imgs/btw_rwd_norm_dists.png` | Supports normalization mode, transform, SLI subset, and wall/nonwalking filters. |
| Between-reward tortuosity histogram | `--btw-rwd-tortuosity-hist` | `imgs/btw_rwd_tortuosity_hist.png` | Can also export histogram data for overlay plotting. |
| Between-reward tortuosity by max radius | `--btw-rwd-tortuosity-box` | `imgs/btw_rwd_tortuosity_by_max_radius_box.png` | Add `--btw-rwd-tortuosity-box-export` for later group overlays. |
| Between-reward COM magnitude histogram | `--btw-rwd-com-mag-hist` | `imgs/btw_rwd_com_mag_hist.png` | Uses between-reward center-of-mass magnitude. |
| Distance-binned COM | `--btw-rwd-conditioned-com` | `imgs/btw_rwd_dist_binned_com.png` | Optional export/import NPZ flags make this reusable across groups. |
| Distance-binned distance traveled | `--btw-rwd-conditioned-disttrav` | `imgs/btw_rwd_dist_binned_disttrav.png` | Optional export/import NPZ flags support cached overlays. |
| Max distance vs. distance traveled | `--btw-rwd-conditioned-dmax-vs-disttrav` | `imgs/between_reward_dmax_vs_disttrav.png` | Optional export/import NPZ flags support cached overlays. |
| Between-reward polar occupancy | `--btw-rwd-polar` | `imgs/btw_rwd_polar__*.png` | Supports 1D angular, 2D theta-radius, pooled trainings, per-fly plots, and debug TSVs. |
| Reward count histogram | `--reward-count-hist` | `imgs/rwd_count_hist.png` | Add `--reward-count-export-hist` for script-based overlays. |
| Reward total bars | `--reward-count-total-bars` | `imgs/rwd_totals.png` | Add `--reward-count-total-export` for script-based overlays and stats. |
| Wall-contact PMF/totals | `--wall`, `--wall-contacts-pmf-*`, `--wall-contacts-per-reward-interval-total-bars` | wall-contact plots, optional NPZ exports | Some wall-contact outputs can be imported or plotted later. |

### Bundle-Then-Plot Workflows

These workflows are useful when you want cohort overlays, top/bottom SLI splits at plot time, stats annotations, or consistent axis handling across multiple runs.

| Metric family | Export from `analyze.py` | Follow-up script | Plot-time metric/options |
| --- | --- | --- | --- |
| COM magnitude + SLI | `--export-com-sli-bundle exports/group_a_com_sli.npz` | `scripts/plot_com_sli_bundles.py` | `--metric commag` or `--metric sli`; supports `--sli-extremes`. |
| Between-reward max distance + SLI | `--export-between-reward-maxdist-sli-bundle exports/group_a_maxdist_sli.npz` | `scripts/plot_com_sli_bundles.py` | `--metric between_reward_maxdist`. |
| Between-reward return-leg distance + SLI | `--export-btw-rwd-return-leg-dist-sli-bundle exports/group_a_return_leg_sli.npz` | `scripts/plot_com_sli_bundles.py` | `--metric between_reward_return_leg_dist`. |
| Cumulative-reward-aligned SLI | `--export-cum-reward-sli-bundle exports/group_a_cum_reward_sli.npz` | `scripts/plot_cum_reward_sli_bundles.py` | `--metric sli`, `reward_pi`, `reward_pi_exp`, or `reward_pi_yoked`. |
| Turnback dual-circle ratio + SLI | `--export-turnback-sli-bundle exports/group_a_turnback_sli.npz` | `scripts/plot_com_sli_bundles.py` | `--metric turnback`; choose `--turnback-mode exp`, `ctrl`, or `exp_minus_ctrl`. |
| Turnback by outer radius | `--export-turnback-outer-radius-sli-bundle exports/group_a_turnback_radius.npz` | `scripts/plot_turnback_outer_radius_sli_bundles.py` | Radius-response bars/curves with optional stats. |
| Turnback by excursion bin | `--export-turnback-excursion-bin-sli-bundle exports/group_a_turnback_excursion.npz` | `scripts/plot_turnback_excursion_bin_sli_bundles.py` | Realized excursion-bin ratio/success/failure/total plots. |
| Return probability by outer radius | `--export-return-prob-outer-radius-sli-bundle exports/group_a_return_prob_radius.npz` | `scripts/plot_return_prob_outer_radius_sli_bundles.py` | `--metric ratio`, `success`, `failure`, `total`, or `stacked`. |
| Return probability by excursion bin | `--export-return-prob-excursion-bin-sli-bundle exports/group_a_return_prob_excursion.npz` | `scripts/plot_return_prob_excursion_bin_sli_bundles.py` | Realized excursion-bin return probability. |
| Agarose dual-circle avoidance + SLI | `--export-agarose-sli-bundle exports/group_a_agarose_sli.npz` | `scripts/plot_com_sli_bundles.py`; stats via `scripts/stats_agarose_sli_bundles.py` or `scripts/stats_agarose_sli_anova.py` | `--metric agarose`; add `--agarose-sli-include-pre` when pre windows matter. |
| Wall-contact percent + SLI | `--export-wallpct-sli-bundle exports/group_a_wallpct_sli.npz` | `scripts/plot_com_sli_bundles.py` | `--metric wallpct`; useful with plot-time SLI filtering. |
| Weaving per exit + SLI | `--export-weaving-sli-bundle exports/group_a_weaving_sli.npz` | `scripts/plot_com_sli_bundles.py` | `--metric weaving`. |
| Large-turn start distance + SLI | `--export-lgturn-startdist-sli-bundle exports/group_a_lgturn_startdist.npz` | `scripts/plot_com_sli_bundles.py` | `--metric lgturn_startdist`. |
| Reward-anchored large-turn path length/prevalence + SLI | `--export-reward-lgturn-pathlen-sli-bundle exports/group_a_reward_lgturn.npz` | `scripts/plot_com_sli_bundles.py` | `--metric reward_lgturn_pathlen` or `reward_lgturn_prevalence`. |
| Reward local variation + SLI | `--reward-lv --export-reward-lv-sli-bundle exports/group_a_reward_lv.npz` | `scripts/plot_com_sli_bundles.py` | `--metric reward_lv`. |
| Between-reward shortest-tail efficiency | `--btw-rwd-shortest-tail --btw-rwd-shortest-tail-export-npz exports/group_a_shortest_tail.npz` | `scripts/plot_overlay_btw_rwd_shortest_tail.py` | Per-training shortest-tail overlays and stats. |
| Between-reward tortuosity histograms | `--btw-rwd-tortuosity-hist --btw-rwd-tortuosity-export-hist exports/group_a_tort_hist.npz` | `scripts/plot_overlay_training_metric_hist.py` or `scripts/plot_overlay_training_metric_filled_hist.py` | PDF/CDF or filled-distribution overlays. |
| Reward count histograms | `--reward-count-hist --reward-count-export-hist exports/group_a_reward_count_hist.npz` | `scripts/plot_overlay_training_metric_hist.py` or `scripts/plot_overlay_training_metric_filled_hist.py` | PDF/CDF or filled-distribution overlays. |
| Reward total bars | `--reward-count-total-bars --reward-count-total-export exports/group_a_reward_totals.npz` | `scripts/plot_overlay_training_metric_scalar_bars.py` | Scalar bar overlays with optional stats. |
| Wall contacts per reward interval totals | `--wall-contacts-per-reward-interval-total-bars --wall-contacts-per-reward-interval-total-export exports/group_a_wall_interval_totals.npz` | `scripts/plot_overlay_training_metric_scalar_bars.py` | Scalar bar overlays with optional stats. |
| Tortuosity mean swarm | `--btw-rwd-tortuosity-mean-swarm --btw-rwd-tortuosity-mean-swarm-export exports/group_a_tort_swarm.npz` | `scripts/plot_between_reward_tortuosity_mean_swarm_bundles.py` | Per-fly swarm overlays. |
| Tortuosity by radius box data | `--btw-rwd-tortuosity-box --btw-rwd-tortuosity-box-export exports/group_a_tort_box.npz` | `scripts/plot_between_reward_tortuosity_box_bundles.py` | Cross-group box plots by radius bin. |
| Tortuosity vs. wall-contact scatter data | `--btw-rwd-tortuosity-wall-scatter-export exports/group_a_tort_wall.npz` | `scripts/plot_between_reward_tortuosity_wall_scatter_bundles.py` or `scripts/plot_between_reward_tortuosity_wall_flycorr_bundles.py` | Segment-level scatter or per-fly correlation overlays. |

Some between-reward SLI plots are also emitted automatically by `analyze.py` for one or two groups: COM magnitude, between-reward max distance, and return-leg distance. This behavior can be locally disabled with `.analyze.local.env`:

```env
ENABLE_DEFAULT_BETWEEN_REWARD_SLI_PLOTS=false
```

Use explicit bundle exports when you need reproducible filenames, more than two groups, plot-time SLI extremes, or a figure that combines independently analyzed cohorts.

## Demo
Below is a short demo to verify the pipeline with *real* data.
1. **Download the sample data**
   Grab the files from [this Figshare dataset](https://figshare.com/articles/dataset/NVSL_Analysis_Demo_Sample_Experiments/28228976).
2. **Run an example command**
   ```bash
   python analyze.py \
       -v /path/to/your/demo/files/c*.avi \
       -f 0-9 \
       --turn circle \
       --turn-prob-by-dist 2,3,4,5,6
   ```
   - The analysis takes about **4 minutes** to complete on a typical desktop.
   - This produces:
       - `__analyze.log`: A log summarizing results per fly.
       - `learning_stats.csv`: Metrics such as walking speed, number of rewards, etc.
       - `imgs/`: A folder with around 48 plots covering various metrics.
3. **Compare outputs to expected**
    - Download the reference outputs from [this Figshare dataset](https://figshare.com/articles/dataset/NVSL_Analysis_Demo_Expected_Outputs/28229126).
    - Compare each file (log, CSV, images) to confirm the results match.

---

## Testing
A regression test checks whether the latest code matches prior reference outputs:
```bash
python scripts/regression_test.py
```
It:
  - Parses log/CSV commands from past runs
  - Executes them on your current code
  - Compares generated outputs (logs, CSVs, images) with the reference data
  - Prints any discrepancies

Logs like `__analyze_13F02.log` or `__analyze_blind.log` and CSVs like `learning_stats_alt.csv` or `learning_stats_blind.csv` are checked, and images in `test-imgs/key.json` are compared via binary checks.

*Thank you for using `nvsl-analysis`! If you have any feedback or questions, please open an issue on GitHub or email the authors.*
