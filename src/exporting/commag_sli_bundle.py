"""COM-magnitude + SLI bundle export entry points.

The implementation still lives in ``com_sli_bundle`` for compatibility with
older imports. New code should import this module, whose name reflects that the
plotted metric is COM magnitude rather than a generic COM concept.
"""

from src.exporting.com_sli_bundle import (
    build_com_sli_bundle as build_commag_sli_bundle,
    export_com_sli_bundle as export_commag_sli_bundle,
    _compute_sli_scalar_and_timeseries_from_rpid,
    _safe_group_label,
)

# Compatibility aliases for callers that switch modules before function names.
build_com_sli_bundle = build_commag_sli_bundle
export_com_sli_bundle = export_commag_sli_bundle

