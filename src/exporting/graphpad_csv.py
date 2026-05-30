from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

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
    if any(
        dict(getattr(export, "meta", {}) or {}).get("metric")
        == "between_reward_tortuosity_mean"
        for export in exports
    ):
        from src.validation.between_reward_tortuosity import (
            validate_between_reward_tortuosity_graphpad_columns,
        )

        validate_between_reward_tortuosity_graphpad_columns(
            headers, columns, path=out_csv
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
