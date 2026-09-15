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


def _write_grouped_repeated_measures_csv(
    out_csv: str | Path,
    groups: Sequence[tuple[str, Sequence[str], Sequence[object], np.ndarray]],
) -> None:
    """Write group-local subject IDs followed by aligned repeated measures."""
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    normalized = []
    for group, panel_labels, unit_ids, values in groups:
        panel_labels = [str(label) for label in panel_labels]
        if len(set(panel_labels)) != len(panel_labels):
            raise ValueError(
                f"repeated-measures group {group!r} contains duplicate panel labels"
            )
        ids = np.asarray(unit_ids, dtype=object).reshape(-1)
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape != (ids.size, len(panel_labels)):
            raise ValueError(
                f"repeated-measures group {group!r} has inconsistent ID/value shapes"
            )
        id_text = np.asarray([str(x) for x in ids], dtype=object)
        if len(set(id_text.tolist())) != id_text.size:
            raise ValueError(
                f"repeated-measures group {group!r} contains duplicate unit IDs"
            )
        headers.append(f"{group} | Subject ID")
        headers.extend(f"{group} | {label}" for label in panel_labels)
        normalized.append((id_text, matrix))

    n_rows = max((ids.size for ids, _ in normalized), default=0)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_unique_headers(headers))
        for row_idx in range(n_rows):
            row: list[str] = []
            for ids, matrix in normalized:
                if row_idx >= ids.size:
                    row.extend([""] * (matrix.shape[1] + 1))
                    continue
                row.append(str(ids[row_idx]))
                row.extend(
                    "" if not np.isfinite(value) else f"{float(value):.12g}"
                    for value in matrix[row_idx]
                )
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


def write_repeated_measures_scalar_exports_graphpad_csv(
    exports: Sequence["ExportedTrainingScalarBars"],
    out_csv: str | Path,
) -> None:
    """Align scalar exports by unit ID within groups encoded as GROUP|PANEL."""
    if not exports:
        raise ValueError("at least one scalar export is required")

    by_group: dict[str, list[tuple[str, object]]] = {}
    for export in exports:
        parts = [part.strip() for part in str(export.group).split("|", 1)]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "repeated-measures scalar labels must use GROUP|PANEL syntax; "
                f"got {export.group!r}"
            )
        group, panel = parts
        if len(export.panel_labels) != 1:
            raise ValueError(
                f"repeated-measures scalar input {export.group!r} must have one panel"
            )
        by_group.setdefault(group, []).append((panel, export))

    grouped = []
    for group, entries in by_group.items():
        union_ids: list[str] = []
        seen: set[str] = set()
        panel_maps = []
        panel_labels = []
        for panel, export in entries:
            ids = np.asarray(export.per_unit_ids_panel[0], dtype=object).reshape(-1)
            values = np.asarray(export.per_unit_values_panel[0], dtype=float).reshape(-1)
            if ids.size != values.size:
                raise ValueError(f"{export.group!r} has different ID and value counts")
            id_text = [str(x) for x in ids]
            if len(set(id_text)) != len(id_text):
                raise ValueError(f"{export.group!r} contains duplicate unit IDs")
            mapping = dict(zip(id_text, values))
            panel_maps.append(mapping)
            panel_labels.append(panel)
            for unit_id in id_text:
                if unit_id not in seen:
                    seen.add(unit_id)
                    union_ids.append(unit_id)
        matrix = np.full((len(union_ids), len(panel_maps)), np.nan, dtype=float)
        for panel_idx, mapping in enumerate(panel_maps):
            for row_idx, unit_id in enumerate(union_ids):
                if unit_id in mapping:
                    matrix[row_idx, panel_idx] = mapping[unit_id]
        grouped.append((group, panel_labels, union_ids, matrix))

    _write_grouped_repeated_measures_csv(out_csv, grouped)


def rpd_exp_minus_yok_exports_to_graphpad_columns(
    exports: Sequence["ExportedTrainingScalarBars"],
    *,
    panel: int | str | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    """Adapt validated pooled RPD exp-minus-yoked exports for GraphPad."""
    for export in exports:
        value_mode = str(export.meta.get("rpd_total_value_mode", ""))
        if value_mode != "exp_minus_yok":
            raise ValueError(
                f"RPD export {export.group!r} has value mode {value_mode!r}; "
                "expected 'exp_minus_yok'. Re-run analyze.py with "
                "--rpd-total-value-mode exp_minus_yok."
            )
    return scalar_exports_to_graphpad_columns(exports, panel=panel)


def write_rpd_exp_minus_yok_exports_graphpad_csv(
    exports: Sequence["ExportedTrainingScalarBars"],
    out_csv: str | Path,
    *,
    panel: int | str | None = None,
) -> None:
    headers, columns = rpd_exp_minus_yok_exports_to_graphpad_columns(
        exports,
        panel=panel,
    )
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
    if not bundles:
        raise ValueError("at least one turnback bundle is required")
    from src.analysis.sli_tools import select_fractional_groups

    groups = []
    for label, bundle in bundles:
        sli = np.asarray(bundle["sli"], dtype=float).reshape(-1)
        ids = np.asarray(bundle["video_ids"], dtype=object).reshape(-1)
        values = np.asarray(bundle["turnback_excursion_bin_ratio_exp"], dtype=float)
        inner = np.asarray(
            bundle["turnback_excursion_bin_pair_inner_deltas_mm"], dtype=float
        ).reshape(-1)
        outer = np.asarray(
            bundle["turnback_excursion_bin_pair_outer_deltas_mm"], dtype=float
        ).reshape(-1)
        if ids.size != sli.size or values.shape != (ids.size, inner.size):
            raise ValueError(f"turnback bundle {label!r} has inconsistent shapes")
        if inner.size != outer.size:
            raise ValueError(f"turnback bundle {label!r} has inconsistent radii")
        if top_sli_fraction is None:
            indices = np.arange(ids.size, dtype=int)
        else:
            _, top = select_fractional_groups(
                pd.Series(sli), top_fraction=float(top_sli_fraction)
            )
            indices = np.asarray([] if top is None else top, dtype=int)
        panel_labels = [f"{i:g}/{o:g} mm" for i, o in zip(inner, outer)]
        groups.append((str(label), panel_labels, ids[indices], values[indices, :]))
    _write_grouped_repeated_measures_csv(out_csv, groups)


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
