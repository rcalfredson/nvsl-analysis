from __future__ import annotations

import csv
from pathlib import Path


def make_agarose_dual_circle_debug_rows(vas) -> list[dict]:
    rows: list[dict] = []
    for va in vas:
        payload = getattr(va, "agarose_dual_circle_debug_rows", None)
        if not payload:
            continue
        rows.extend(dict(row) for row in payload)
    return rows


def save_agarose_dual_circle_debug_table(vas, out_path: str | Path) -> None:
    rows = make_agarose_dual_circle_debug_rows(vas)
    if not rows:
        print("[agarose-debug] no dual-circle debug rows to write")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[agarose-debug] wrote dual-circle debug CSV: {out_path}")
