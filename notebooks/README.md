# Paper Figure Notebooks

This directory holds Jupyter notebooks used to assemble paper-ready figure panels.

The starter notebook is:

- `paper_figure_panels_template.ipynb`

The intended pattern for each panel is:

1. A short markdown description of the panel and what biological comparison it shows.
2. One or more data export commands that write `.npz` bundles.
3. One or more plotting commands that consume those bundles and generate the final panel image.

The prototype notebook keeps export and plotting commands separate so each panel can be rerun in stages.

Notes:

- Start Jupyter from the repository root, then run the notebook's setup cell (`%cd ..`) before executing the export or plotting stages.
- The notebook uses your existing CLI commands directly rather than reimplementing analysis logic in Python.
- This repository's `requirements.txt` does not currently install Jupyter itself, so you may want to install `jupyterlab` or `notebook` in your active environment if it is not already present.
