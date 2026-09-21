# Paper Figure Notebooks

This directory holds Jupyter notebooks used to assemble paper-ready figure panels.

The manuscript-ordered notebook is:

- `manuscript_figure_panels.ipynb`: uses manuscript panel labels and currently
  covers Figures 1e–1h, 1j, and 1m–1n from one shared flat HTL control analysis, and
  Figures 1i and 1k from a separate correlation analysis of the same flies.
  Figure 1l uses the first-ten-rewards diagnostic recipe from the older notebook.
  Its plot-style cell sets the font family, time/correlation font sizes, and
  image format. Analysis commands run only when their panel toggle is enabled.
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
