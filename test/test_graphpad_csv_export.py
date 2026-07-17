from pathlib import Path
import subprocess
import sys

import numpy as np

from src.exporting.graphpad_csv import (
    write_agarose_time_graphpad_csv,
    write_scalar_exports_graphpad_csv,
)
from src.plotting.overlay_training_metric_scalar_bars import load_export_npz
from src.analysis.agarose_time_summary import DEFAULT_SECTION


PRE = "% time spent on agarose- pre last 10m (exp)"
POST = "% time spent on agarose- T3 post last 10m (exp)"


def _write_scalar_npz(path: Path, values: list[float], label: str = "T2") -> None:
    arr = np.asarray(values, dtype=float)
    np.savez_compressed(
        path,
        panel_labels=np.asarray([label], dtype=object),
        per_unit_values_panel=np.asarray([arr], dtype=object),
        per_unit_ids_panel=np.asarray(
            [np.asarray([f"fly{i}" for i in range(arr.size)], dtype=object)],
            dtype=object,
        ),
        mean=np.asarray([float(np.nanmean(arr))], dtype=float),
        ci_lo=np.asarray([np.nan], dtype=float),
        ci_hi=np.asarray([np.nan], dtype=float),
        n_units_panel=np.asarray([arr.size], dtype=int),
        meta_json="{}",
    )


def _write_two_panel_scalar_npz(path: Path) -> None:
    np.savez_compressed(
        path,
        panel_labels=np.asarray(["training 1", "training 2"], dtype=object),
        per_unit_values_panel=np.asarray(
            [np.asarray([1.0, 2.0]), np.asarray([3.0])],
            dtype=object,
        ),
        per_unit_ids_panel=np.asarray(
            [
                np.asarray(["fly0", "fly1"], dtype=object),
                np.asarray(["fly0"], dtype=object),
            ],
            dtype=object,
        ),
        mean=np.asarray([1.5, 3.0], dtype=float),
        ci_lo=np.asarray([np.nan, np.nan], dtype=float),
        ci_hi=np.asarray([np.nan, np.nan], dtype=float),
        n_units_panel=np.asarray([2, 1], dtype=int),
        meta_json="{}",
    )


def _write_learning_stats(path: Path, rows: list[tuple[object, ...]]) -> None:
    path.write_text(
        "\n".join(
            [
                DEFAULT_SECTION,
                f"video,fly,{PRE},{POST}",
                *(",".join(map(str, row)) for row in rows),
                "",
            ]
        )
    )


def test_scalar_npz_graphpad_csv_is_wide_numeric_table(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    out = tmp_path / "graphpad.csv"
    _write_scalar_npz(a, [1.25, 2.5])
    _write_scalar_npz(b, [4.0])

    write_scalar_exports_graphpad_csv(
        [load_export_npz("Ctrl", a), load_export_npz("PFNd>Kir", b)],
        out,
    )

    assert out.read_text().splitlines() == [
        "Ctrl,PFNd>Kir",
        "1.25,4",
        "2.5,",
    ]


def test_agarose_graphpad_csv_exports_pre_minus_post_deltas(tmp_path):
    a = tmp_path / "control.csv"
    b = tmp_path / "mutant.csv"
    out = tmp_path / "agarose_graphpad.csv"
    _write_learning_stats(a, [("v1", 0, 10, 7), ("v2", 1, 14, 9)])
    _write_learning_stats(b, [("m1", 0, 8, 3)])

    write_agarose_time_graphpad_csv(
        [("Control", str(a)), ("upd3 KO", str(b))],
        out,
        section=DEFAULT_SECTION,
        pre_col=PRE,
        post_col=POST,
    )

    assert out.read_text().splitlines() == [
        "Control,upd3 KO",
        "3,5",
        "5,",
    ]


def test_scalar_panel_cli_aliases_select_internal_export_panel(tmp_path):
    npz_path = tmp_path / "multi.npz"
    out_new = tmp_path / "new.csv"
    out_old = tmp_path / "old.csv"
    _write_two_panel_scalar_npz(npz_path)

    base = [
        sys.executable,
        "scripts/export_graphpad_csv.py",
        "scalar-npz",
        "--input",
        f"Ctrl={npz_path}",
    ]
    subprocess.run(
        [*base, "--scalar-panel", "training 2", "--out", str(out_new)],
        check=True,
    )
    subprocess.run([*base, "--panel", "2", "--out", str(out_old)], check=True)

    assert out_new.read_text().splitlines() == ["Ctrl", "3"]
    assert out_old.read_text().splitlines() == ["Ctrl", "3"]
