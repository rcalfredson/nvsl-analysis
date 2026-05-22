#!/usr/bin/env python3
"""Compatibility wrapper for the renamed metric+SLI bundle plotter."""

import sys

from scripts.plot_metric_sli_bundles import main


if __name__ == "__main__":
    print(
        "[deprecated] scripts.plot_com_sli_bundles has been renamed to "
        "scripts.plot_metric_sli_bundles.",
        file=sys.stderr,
    )
    main()

