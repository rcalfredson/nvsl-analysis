#!/usr/bin/env python3
"""Stats report for turn home-vector alignment scalar-bar bundles."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'nvsl')}")

import numpy as np
from scipy.stats import ttest_ind, ttest_rel

from src.plotting.overlay_training_metric_scalar_bars import load_export_npz


@dataclass(frozen=True)
class GroupBundle:
    label: str
    path: str
    export: object


def _parse_input(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"--input must be LABEL=PATH, got: {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise SystemExit(f"--input must include both label and path, got: {spec!r}")
    return label, path


def _panel_index(export, panel_label: str) -> int:
    labels = [str(x) for x in export.panel_labels]
    if panel_label not in labels:
        raise SystemExit(
            f"Panel {panel_label!r} not found in {export.group!r}; available: {labels}"
        )
    return labels.index(panel_label)


def _panel_values_by_id(export, panel_label: str) -> dict[str, float]:
    p_idx = _panel_index(export, panel_label)
    ids = np.asarray(export.per_unit_ids_panel[p_idx], dtype=object).ravel()
    vals = np.asarray(export.per_unit_values_panel[p_idx], dtype=float).ravel()
    out: dict[str, float] = {}
    for unit_id, val in zip(ids, vals):
        if unit_id is None or not np.isfinite(val):
            continue
        out[str(unit_id)] = float(val)
    return out


def _mean(x: list[float]) -> float:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else math.nan


def _holm_adjust(pvals: list[float]) -> list[float]:
    finite = [(i, float(p)) for i, p in enumerate(pvals) if np.isfinite(p)]
    adjusted = [math.nan] * len(pvals)
    m = len(finite)
    if m == 0:
        return adjusted
    ordered = sorted(finite, key=lambda item: item[1])
    prev = 0.0
    for rank, (orig_i, p) in enumerate(ordered):
        adj = min((m - rank) * p, 1.0)
        adj = max(adj, prev)
        adjusted[orig_i] = adj
        prev = adj
    return adjusted


def _paired_panel_comparison(group: GroupBundle, baseline: str, target: str) -> dict:
    base = _panel_values_by_id(group.export, baseline)
    tgt = _panel_values_by_id(group.export, target)
    ids = sorted(set(base).intersection(tgt))
    before = np.asarray([base[k] for k in ids], dtype=float)
    after = np.asarray([tgt[k] for k in ids], dtype=float)
    diff = after - before
    stat = pval = math.nan
    if len(ids) >= 2 and np.any(np.isfinite(diff)):
        res = ttest_rel(after, before, nan_policy="omit")
        stat = float(res.statistic)
        pval = float(res.pvalue)
    return {
        "family": "within_group",
        "comparison": f"{target} - {baseline}",
        "group_a": group.label,
        "group_b": "",
        "panel": target,
        "n_a": len(ids),
        "n_b": len(ids),
        "mean_a": _mean(before.tolist()),
        "mean_b": _mean(after.tolist()),
        "effect": _mean(diff.tolist()),
        "test": "paired_t",
        "statistic": stat,
        "p": pval,
        "p_holm": math.nan,
    }


def _between_group_comparison(
    group_a: GroupBundle, group_b: GroupBundle, panel: str
) -> dict:
    vals_a = list(_panel_values_by_id(group_a.export, panel).values())
    vals_b = list(_panel_values_by_id(group_b.export, panel).values())
    stat = pval = math.nan
    if len(vals_a) >= 2 and len(vals_b) >= 2:
        res = ttest_ind(vals_a, vals_b, equal_var=False, nan_policy="omit")
        stat = float(res.statistic)
        pval = float(res.pvalue)
    return {
        "family": "between_group",
        "comparison": f"{group_a.label} vs {group_b.label}",
        "group_a": group_a.label,
        "group_b": group_b.label,
        "panel": panel,
        "n_a": len(vals_a),
        "n_b": len(vals_b),
        "mean_a": _mean(vals_a),
        "mean_b": _mean(vals_b),
        "effect": _mean(vals_a) - _mean(vals_b),
        "test": "welch_t",
        "statistic": stat,
        "p": pval,
        "p_holm": math.nan,
    }


def _format_float(x: object) -> str:
    try:
        val = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(val):
        return ""
    return f"{val:.6g}"


def _print_rows(rows: list[dict]) -> None:
    fieldnames = [
        "family",
        "comparison",
        "group_a",
        "group_b",
        "panel",
        "n_a",
        "n_b",
        "mean_a",
        "mean_b",
        "effect",
        "test",
        "statistic",
        "p",
        "p_holm",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _format_float(row.get(k, "")) for k in fieldnames})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Report paired within-group and unpaired between-group stats for "
            "turn home-vector alignment scalar exports."
        )
    )
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable LABEL=PATH scalar-bar NPZ input.",
    )
    p.add_argument("--baseline-panel", default="Pre-training")
    p.add_argument(
        "--within-target-panel",
        action="append",
        default=["Training 1", "Training 2"],
        help="Panel compared against --baseline-panel within each group.",
    )
    p.add_argument(
        "--between-panel",
        action="append",
        default=["Training 1", "Training 2"],
        help="Panel used for across-group comparisons.",
    )
    p.add_argument(
        "--out-tsv",
        default=None,
        help="Optional path to write the same TSV report.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    groups = [
        GroupBundle(label, path, load_export_npz(label, path))
        for label, path in (_parse_input(spec) for spec in args.input)
    ]
    if len(groups) < 2:
        raise SystemExit("At least two --input bundles are required.")

    rows: list[dict] = []
    for group in groups:
        for panel in args.within_target_panel:
            rows.append(_paired_panel_comparison(group, args.baseline_panel, panel))

    for panel in args.between_panel:
        for i, group_a in enumerate(groups):
            for group_b in groups[i + 1 :]:
                rows.append(_between_group_comparison(group_a, group_b, panel))

    for family in sorted({str(row["family"]) for row in rows}):
        idxs = [i for i, row in enumerate(rows) if row["family"] == family]
        adj = _holm_adjust([float(rows[i]["p"]) for i in idxs])
        for i, p_holm in zip(idxs, adj):
            rows[i]["p_holm"] = p_holm

    _print_rows(rows)

    if args.out_tsv:
        os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
        with open(args.out_tsv, "w", newline="") as fh:
            fieldnames = list(rows[0].keys()) if rows else []
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"[turn_home_vector_alignment_stats] wrote {args.out_tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
