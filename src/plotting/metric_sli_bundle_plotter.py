"""Generic metric+SLI bundle plotting entry points.

This module is the preferred import location for plotting reusable NPZ bundles
that pair a plotted metric with SLI scalar/time-series data. The older
``com_sli_bundle_plotter`` module remains as a compatibility import path.
"""

from src.plotting.com_sli_bundle_plotter import (
    plot_com_sli_bundle_data as plot_metric_sli_bundle_data,
    plot_com_sli_bundles as plot_metric_sli_bundles,
)

# Backward-compatible function names for callers that have already migrated the
# module import but not the function names.
plot_com_sli_bundle_data = plot_metric_sli_bundle_data
plot_com_sli_bundles = plot_metric_sli_bundles

