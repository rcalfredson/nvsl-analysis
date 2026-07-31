from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.agarose_time_summary import (
    DEFAULT_POST_COL,
    DEFAULT_PRE_COL,
    DEFAULT_SECTION,
    paired_test,
    parse_group,
)
from src.utils.parsers import parse_labeled_path
if TYPE_CHECKING:
    from src.plotting.overlay_training_metric_scalar_bars import (
        ExportedTrainingScalarBars,
    )


def _unique_headers(headers: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for header in headers:
        base = str(header)
        n = seen.get(base, 0) + 1
        seen[base] = n
        out.append(base if n == 1 else f"{base} {n}")
    return out


def _write_wide_numeric_csv(
    out_csv: str | Path,
    headers: Sequence[str],
    columns: Sequence[Sequence[float]],
) -> None:
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = _unique_headers(headers)
    col_arrays = [np.asarray(col, dtype=float).reshape(-1) for col in columns]
    n_rows = max((int(col.size) for col in col_arrays), default=0)

    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in range(n_rows):
            row = []
            for col in col_arrays:
                if r >= col.size or not np.isfinite(col[r]):
                    row.append("")
                else:
                    row.append(f"{float(col[r]):.12g}")
            writer.writerow(row)


def scalar_exports_to_graphpad_columns(
    exports: Sequence["ExportedTrainingScalarBars"],
    *,
    panel: int | str | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    if not exports:
        raise ValueError("at least one scalar export is required")

    panel_indices_by_export: list[list[int]] = []
    for export in exports:
        labels = list(export.panel_labels)
        if panel is None:
            panel_indices_by_export.append(list(range(len(labels))))
            continue

        if isinstance(panel, int):
            idx = int(panel) - 1
            if idx < 0 or idx >= len(labels):
                raise ValueError(
                    f"panel {panel} is out of range for group {export.group!r}; "
                    f"available panels: {labels}"
                )
            panel_indices_by_export.append([idx])
            continue

        wanted = str(panel)
        matches = [i for i, label in enumerate(labels) if str(label) == wanted]
        if not matches:
            raise ValueError(
                f"panel {wanted!r} not found for group {export.group!r}; "
                f"available panels: {labels}"
            )
        panel_indices_by_export.append([matches[0]])

    one_panel_each = all(len(indices) == 1 for indices in panel_indices_by_export)
    headers: list[str] = []
    columns: list[np.ndarray] = []
    for export, indices in zip(exports, panel_indices_by_export):
        for idx in indices:
            vals = np.asarray(export.per_unit_values_panel[idx], dtype=float).reshape(-1)
            vals = vals[np.isfinite(vals)]
            if one_panel_each:
                header = str(export.group)
            else:
                header = f"{export.group} | {export.panel_labels[idx]}"
            headers.append(header)
            columns.append(vals)
    return headers, columns


def write_scalar_exports_graphpad_csv(
    exports: Sequence["ExportedTrainingScalarBars"],
    out_csv: str | Path,
    *,
    panel: int | str | None = None,
) -> None:
    headers, columns = scalar_exports_to_graphpad_columns(exports, panel=panel)
    _write_wide_numeric_csv(out_csv, headers, columns)


def turnback_ratio_bundles_to_graphpad_columns(
    bundles: Sequence[tuple[str, Mapping[str, object]]],
    *,
    top_sli_fraction: float | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    """Extract per-fly experimental turnback ratios from SLI bundles.

    When ``top_sli_fraction`` is supplied, learner selection is performed
    independently within each cohort using the bundle's precomputed SLI.  This
    is the same selection used by the turnback bundle plotting pipeline.
    """
    if not bundles:
        raise ValueError("at least one turnback bundle is required")

    from src.analysis.sli_tools import select_fractional_groups

    selected_by_bundle: list[tuple[str, list[str], np.ndarray]] = []
    for label, bundle in bundles:
        n_flies = len(np.asarray(bundle["sli"]).reshape(-1))
        if top_sli_fraction is None:
            indices = np.arange(n_flies, dtype=int)
        else:
            _, top = select_fractional_groups(
                pd.Series(np.asarray(bundle["sli"], dtype=float).reshape(-1)),
                top_fraction=float(top_sli_fraction),
            )
            indices = np.asarray([] if top is None else top, dtype=int)

        values = np.asarray(
            bundle["turnback_excursion_bin_ratio_exp"], dtype=float
        )
        inner = np.asarray(
            bundle["turnback_excursion_bin_pair_inner_deltas_mm"], dtype=float
        ).reshape(-1)
        outer = np.asarray(
            bundle["turnback_excursion_bin_pair_outer_deltas_mm"], dtype=float
        ).reshape(-1)
        if values.shape != (n_flies, inner.size) or inner.size != outer.size:
            raise ValueError(
                f"turnback bundle {label!r} has inconsistent ratio/radius shapes"
            )
        panel_labels = [f"{i:g}/{o:g} mm" for i, o in zip(inner, outer)]
        selected_by_bundle.append((str(label), panel_labels, values[indices, :]))

    one_panel_each = all(len(labels) == 1 for _, labels, _ in selected_by_bundle)
    headers: list[str] = []
    columns: list[np.ndarray] = []
    for label, panel_labels, values in selected_by_bundle:
        for panel_idx, panel_label in enumerate(panel_labels):
            column = np.asarray(values[:, panel_idx], dtype=float).reshape(-1)
            columns.append(column[np.isfinite(column)])
            headers.append(label if one_panel_each else f"{label} | {panel_label}")
    return headers, columns


def write_turnback_ratio_bundles_graphpad_csv(
    bundles: Sequence[tuple[str, Mapping[str, object]]],
    out_csv: str | Path,
    *,
    top_sli_fraction: float | None = None,
) -> None:
    headers, columns = turnback_ratio_bundles_to_graphpad_columns(
        bundles,
        top_sli_fraction=top_sli_fraction,
    )
    _write_wide_numeric_csv(out_csv, headers, columns)


def agarose_time_to_graphpad_columns(
    groups: Sequence[tuple[str, str]],
    *,
    section: str = DEFAULT_SECTION,
    pre_col: str = DEFAULT_PRE_COL,
    post_col: str = DEFAULT_POST_COL,
) -> tuple[list[str], list[np.ndarray]]:
    if not groups:
        raise ValueError("at least one agarose group is required")
    headers: list[str] = []
    columns: list[np.ndarray] = []
    for label, path in groups:
        parsed = parse_group(
            label,
            path,
            section=section,
            numeric_cols=[pre_col, post_col],
        )
        test = paired_test(parsed, pre_col=pre_col, post_col=post_col)
        vals = np.asarray(test.reductions, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        headers.append(str(label))
        columns.append(vals)
    return headers, columns


def write_agarose_time_graphpad_csv(
    groups: Sequence[tuple[str, str]],
    out_csv: str | Path,
    *,
    section: str = DEFAULT_SECTION,
    pre_col: str = DEFAULT_PRE_COL,
    post_col: str = DEFAULT_POST_COL,
) -> None:
    headers, columns = agarose_time_to_graphpad_columns(
        groups,
        section=section,
        pre_col=pre_col,
        post_col=post_col,
    )
    _write_wide_numeric_csv(out_csv, headers, columns)
