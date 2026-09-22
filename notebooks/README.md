# Paper Figure Notebooks

This directory holds Jupyter notebooks used to assemble paper-ready figure panels.

The manuscript-ordered notebook is:

- `manuscript_figure_panels.ipynb`: uses manuscript panel labels and currently
  covers Figures 1e–1h, 1j, and 1m–1n from one shared flat HTL control analysis, and
  Figures 1i and 1k from a separate correlation analysis of the same flies.
  Figure 1l uses the first-ten-rewards diagnostic recipe from the older notebook.
  Figure 1q adapts its Panel 38 matched closed-loop/open-loop exports and
  correlation plot, with separate export and plotting toggles.
  Figures 2b–2c compare Ctrl and PFNd training SLI for Kir and GtACR1,
  respectively, with a fixed 0–1.2 y-axis range and the same two-training,
  27 pt time-plot style.
  Figure 2e adds the corresponding Ctrl>CsC versus PFNd>CsC comparison.
  Figure 2f compares Ctrl>CsC and PFNv>CsC with the same Figure 2 time-plot
  settings.
  Figure 2g compares the Figure 1 flat HTL controls with antennae-removed
  flies and reuses the control fly list defined for Figure 1.
  Figure 2o compares flat HTL controls with hind-tarsi-removed,
  genitalia-glued flies under the same Figure 2 time-plot settings.
  Figure 2q compares Ctrl>Kir and MBKC-1>Kir in the flat HTL chamber.
  Figure 2r compares Ctrl>GtACR1 and MBKC-1>GtACR1 in the same chamber.
  Figure 3c compares the Figure 1 flat HTL control cohort with its agarose
  counterpart with a fixed 0–2.0 SLI y-axis range.
  Figure 3d compares that agarose control cohort with antennae-removed flies
  using the same range.
  Figure 3g compares a handled agarose control cohort with flies whose hind
  tarsi were removed and genitalia glued, also using the 0–2.0 range.
  Figure 3j compares Ctrl>Kir and MBKC-1>Kir in the agarose HTL chamber with
  the same Figure 3 SLI range.
  Figure 3k compares Ctrl>Kir and PFNd>Kir in the agarose HTL chamber with
  the same range.
  Figure 4b contains the Ctrl>Kir Training 2 and Post 2 heatmaps copied from
  the recipes in `scripts/generate_heatmaps.sh`. Each run saves the shared
  `imgs/heatmaps2.pdf` output under a distinct manuscript artifact name.
  Figure 4c reproduces Panel 40 from the earlier notebook: the first and last
  ten Training 2 between-reward trajectories from one example fly. Its command
  is preserved verbatim, including PDF output and its original plot settings.
  Figure 4d compares training SLI for flat large-chamber controls and
  antennae-removed flies with a fixed −0.5–2.5 y-axis range.
  Figure 4i compares the same flat large-chamber Ctrl>Kir cohort with
  PFNd>Kir flies using the same Figure 4 SLI range.
  Figure 4s compares the Ctrl>Kir cohort across agarose and flat large
  chambers, with agarose first and the same Figure 4 SLI range.
  The notebook's plot-style cell sets the font family, time/correlation/heatmap
  font sizes, and image format. Figure 4b stays in PDF format to match its source recipe.
  Analysis commands run only when their panel toggle is enabled.
  Its final copy stage previews or copies named panels to
  `/media/Synology4/Robert/nvsl-analysis-plots/figN/`.

The earlier analysis notebook is:

- `paper_figure_panels.ipynb`: uses its own historical panel numbering, which
  does not correspond to manuscript panel labels.

Additional export notebooks:

- `graphpad_exports.ipynb`: converts selected per-fly scalar exports into
  GraphPad Prism-friendly CSV files, and contains the standalone agarose
  percent-time export workflow that is not part of the numbered panel notebook.

For prerequisites, execution toggles, input dependencies, and the related batch
scripts, see the repository-level
[`docs/output_workflows.md`](../docs/output_workflows.md) guide.

The intended pattern for each panel is:

1. A short markdown description of the panel and what biological comparison it shows.
2. One or more data export commands that write `.npz` bundles.
3. One or more plotting commands that consume those bundles and generate the final panel image.

The notebook keeps export and plotting commands separate so each panel can be rerun in stages.

Notes:

- Start Jupyter from the repository root, then run the notebook's setup cell (`%cd ..`) before executing the export or plotting stages.
- The notebook uses your existing CLI commands directly rather than reimplementing analysis logic in Python.
- This repository's `requirements.txt` does not currently install Jupyter itself, so you may want to install `jupyterlab` or `notebook` in your active environment if it is not already present.
