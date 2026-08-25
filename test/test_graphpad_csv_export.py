import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.exporting.graphpad_csv import (
    rpd_exp_minus_yok_exports_to_graphpad_columns,
    write_agarose_time_graphpad_csv,
    write_rpd_exp_minus_yok_exports_graphpad_csv,
    write_scalar_exports_graphpad_csv,
    write_turnback_ratio_bundles_graphpad_csv,
)
from src.plotting.overlay_training_metric_scalar_bars import load_export_npz
from src.analysis.agarose_time_summary import DEFAULT_SECTION


PRE = "% time spent on agarose- pre last 10m (exp)"
POST = "% time spent on agarose- T3 post last 10m (exp)"

_RPD_RUNNER_ENV_VARS = {
    "LIST_DATASETS",
    "OUT_CSV",
    "OUT_DIR",
    "PRINT_ONLY",
    "PYTHON_BIN",
    "REFRESH_DATASETS",
    "REUSE_EXISTING_NPZ",
    "VIDEO_LISTS_FILE",
    "WRITE_CSV",
}


def _rpd_runner_env(**overrides: str) -> dict[str, str]:
    """Return a runner environment isolated from local RPD configuration."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("RPD_") and key not in _RPD_RUNNER_ENV_VARS
    }
    env.update(overrides)
    return env


def _write_scalar_npz(
    path: Path,
    values: list[float],
    label: str = "T2",
    meta_json: str = "{}",
) -> None:
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
        meta_json=meta_json,
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


def test_rpd_exp_minus_yok_graphpad_adapter_validates_and_writes(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    out = tmp_path / "rpd.csv"
    meta = '{"rpd_total_value_mode": "exp_minus_yok"}'
    _write_scalar_npz(a, [1.25, -2.5], meta_json=meta)
    _write_scalar_npz(b, [4.0], meta_json=meta)
    exports = [
        load_export_npz("Ctrl>Kir | flat large", a),
        load_export_npz("PFNd>Kir | flat large", b),
    ]

    write_rpd_exp_minus_yok_exports_graphpad_csv(exports, out, panel=1)

    assert out.read_text().splitlines() == [
        "Ctrl>Kir | flat large,PFNd>Kir | flat large",
        "1.25,4",
        "-2.5,",
    ]


def test_rpd_exp_minus_yok_graphpad_adapter_rejects_exp_only_export(tmp_path):
    export_path = tmp_path / "exp_only.npz"
    _write_scalar_npz(
        export_path,
        [1.0],
        meta_json='{"rpd_total_value_mode": "exp"}',
    )

    with np.testing.assert_raises_regex(ValueError, "expected 'exp_minus_yok'"):
        rpd_exp_minus_yok_exports_to_graphpad_columns(
            [load_export_npz("Ctrl", export_path)]
        )


def test_rpd_chamber_runner_resolves_twenty_matched_video_lists_in_preview():
    result = subprocess.run(
        ["bash", "scripts/run_rpd_exp_minus_yok_chamber_graphpad.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=_rpd_runner_env(
            PRINT_ONLY="1",
            PYTHON_BIN=sys.executable,
            REUSE_EXISTING_NPZ="0",
            REFRESH_DATASETS="all",
            WRITE_CSV="1",
            LIST_DATASETS="0",
        ),
    )
    commands = result.stdout.splitlines()
    analysis_commands = [
        line for line in commands if "--rpd-total-export" in line
    ]
    csv_commands = [
        line for line in commands if "rpd-exp-minus-yok-npz" in line
    ]

    assert len(analysis_commands) == 20
    assert len(csv_commands) == 1
    assert csv_commands[0].count("--input") == 20
    assert "2024-03-04/c3\\[12\\]_\\*" in analysis_commands[0]
    assert "2024-03-1\\[19\\]/c3\\[12\\]_\\*" in analysis_commands[2]
    # The four intact Ctrl>Kir/PFNd>Kir lists match new_experiment_lists.txt.
    assert "2023-02-19/c3\\[12\\]_\\*" in analysis_commands[4]
    assert "2023-02-2\\[02\\]/c6\\[12\\]_\\*" in analysis_commands[5]
    assert "2023-02-2\\[02\\]/c4\\[12\\]_\\*" in analysis_commands[6]
    assert "2023-02-19/c62_\\*" in analysis_commands[7]
    # The former glued slots now resolve the four 2024 antennae-removed lists.
    assert "2024-02-1\\[5678\\]/c3\\[12\\]_\\*" in analysis_commands[8]
    assert "2024-02-1\\[78\\]/c5\\[12\\]_\\*" in analysis_commands[9]
    assert "2024-02-1\\[56789\\]/c4\\[12\\]_\\*" in analysis_commands[10]
    assert "2024-02-17/c6\\[12\\]_\\*" in analysis_commands[11]
    assert "2025-07-1\\[24\\]/afternoon/c3\\[12\\]_\\*" in analysis_commands[18]
    assert "2025-07-1\\[25\\]/afternoon/c5\\[12\\]_\\*" in analysis_commands[19]
    assert "Antennae-removed\\ PFNd\\>Kir" in csv_commands[0]
    assert "Antennae-glued" not in csv_commands[0]
    assert "Antennae\\ removed\\ PFNd\\>Kir" in csv_commands[0]
    assert any(
        "[rpd_dataset] antenna_removed_pfnd_flat_htl | "
        "Antennae-removed PFNd>Kir | flat HTL | source:"
        in line
        for line in commands
    )


def test_rpd_chamber_runner_lists_names_sources_and_video_arguments():
    result = subprocess.run(
        ["bash", "scripts/run_rpd_exp_minus_yok_chamber_graphpad.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=_rpd_runner_env(
            REUSE_EXISTING_NPZ="0",
            REFRESH_DATASETS="all",
            WRITE_CSV="1",
            LIST_DATASETS="1",
        ),
    )

    lines = result.stdout.splitlines()
    assert len(lines) == 21
    assert lines[0].split("\t") == [
        "slug",
        "CSV column",
        "video_lists section",
        "video_lists cohort",
        "output NPZ",
        "-v video list",
    ]
    fields = next(
        line for line in lines if line.startswith("sensory_ctrl_flat_htl\t")
    ).split("\t")
    assert fields[1] == "Antennae-intact control (AR-matched) | flat HTL"
    assert fields[3] == "Flat"
    assert "/media/Synology4/Yang Chen/2024-03-04/c3[12]_*" in fields[5]


def test_rpd_chamber_runner_dataset_list_reflects_video_override():
    result = subprocess.run(
        ["bash", "scripts/run_rpd_exp_minus_yok_chamber_graphpad.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=_rpd_runner_env(
            REUSE_EXISTING_NPZ="0",
            REFRESH_DATASETS="all",
            LIST_DATASETS="1",
            RPD_FLAT_HTL_AR_KIR_CTRL="/tmp/custom_removed_ctrl_video_*",
        ),
    )

    fields = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("antenna_removed_ctrl_flat_htl\t")
    ).split("\t")
    assert fields[0] == "antenna_removed_ctrl_flat_htl"
    assert fields[5] == "/tmp/custom_removed_ctrl_video_*"


def test_rpd_chamber_runner_can_refresh_one_named_npz_without_csv():
    result = subprocess.run(
        ["bash", "scripts/run_rpd_exp_minus_yok_chamber_graphpad.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=_rpd_runner_env(
            PRINT_ONLY="1",
            PYTHON_BIN=sys.executable,
            REUSE_EXISTING_NPZ="0",
            REFRESH_DATASETS="antenna_removed_pfnd_flat_htl",
            WRITE_CSV="0",
            LIST_DATASETS="0",
        ),
    )

    commands = result.stdout.splitlines()
    analysis_commands = [
        line for line in commands if "--rpd-total-export" in line
    ]
    assert len(analysis_commands) == 1
    assert "antenna_removed_pfnd_flat_htl.npz" in analysis_commands[0]
    assert "2024-02-1\\[56789\\]/c4\\[12\\]_\\*" in analysis_commands[0]


def test_rpd_chamber_runner_never_reuses_old_glued_bundles_for_removed_cohorts(
    tmp_path,
):
    cohort_pairs = [
        ("antenna_glued_ctrl_flat_htl", "antenna_removed_ctrl_flat_htl"),
        ("antenna_glued_ctrl_agarose_htl", "antenna_removed_ctrl_agarose_htl"),
        ("antenna_glued_pfnd_flat_htl", "antenna_removed_pfnd_flat_htl"),
        ("antenna_glued_pfnd_agarose_htl", "antenna_removed_pfnd_agarose_htl"),
    ]
    for glued_slug, _ in cohort_pairs:
        (tmp_path / f"{glued_slug}.npz").write_bytes(b"wrong cohort")
    result = subprocess.run(
        ["bash", "scripts/run_rpd_exp_minus_yok_chamber_graphpad.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=_rpd_runner_env(
            PRINT_ONLY="1",
            PYTHON_BIN=sys.executable,
            REUSE_EXISTING_NPZ="0",
            REFRESH_DATASETS="none",
            WRITE_CSV="1",
            LIST_DATASETS="0",
            OUT_DIR=str(tmp_path),
            OUT_CSV=str(tmp_path / "out.csv"),
        ),
    )

    csv_command = next(
        line
        for line in result.stdout.splitlines()
        if "rpd-exp-minus-yok-npz" in line
    )
    for glued_slug, removed_slug in cohort_pairs:
        assert str(tmp_path / f"{glued_slug}.npz") not in csv_command
        assert str(tmp_path / f"{removed_slug}.npz") in csv_command


def test_rpd_chamber_runner_reuses_equivalent_legacy_bundle_name(tmp_path):
    legacy = tmp_path / "ar_ctrlKir_flat_htl.npz"
    legacy.write_bytes(b"legacy placeholder")
    result = subprocess.run(
        ["bash", "scripts/run_rpd_exp_minus_yok_chamber_graphpad.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=_rpd_runner_env(
            PRINT_ONLY="1",
            PYTHON_BIN=sys.executable,
            REUSE_EXISTING_NPZ="0",
            REFRESH_DATASETS="none",
            WRITE_CSV="1",
            LIST_DATASETS="0",
            OUT_DIR=str(tmp_path),
            OUT_CSV=str(tmp_path / "out.csv"),
        ),
    )

    assert (
        "[rpd_dataset] reusing legacy NPZ for ar_ctrl_flat_htl: "
        f"{legacy}"
    ) in result.stdout
    csv_command = next(
        line
        for line in result.stdout.splitlines()
        if "rpd-exp-minus-yok-npz" in line
    )
    assert str(legacy) in csv_command


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


def test_turnback_ratio_graphpad_csv_exports_all_fly_values_by_radius_pair(tmp_path):
    out = tmp_path / "turnback.csv"
    bundle = {
        "sli": np.asarray([0.1, 0.9, 0.5]),
        "video_ids": np.asarray(["a", "b", "c"]),
        "turnback_excursion_bin_ratio_exp": np.asarray(
            [[0.1, 0.2], [0.8, np.nan], [0.4, 0.6]]
        ),
        "turnback_excursion_bin_ratio_ctrl": np.zeros((3, 2)),
        "turnback_excursion_bin_pair_inner_deltas_mm": np.asarray([3, 8]),
        "turnback_excursion_bin_pair_outer_deltas_mm": np.asarray([5, 10]),
    }

    write_turnback_ratio_bundles_graphpad_csv([("Ctrl", bundle)], out)

    assert out.read_text().splitlines() == [
        "Ctrl | 3/5 mm,Ctrl | 8/10 mm",
        "0.1,0.2",
        "0.8,0.6",
        "0.4,",
    ]


def test_turnback_ratio_graphpad_csv_selects_top_sli_fraction_per_group(tmp_path):
    out = tmp_path / "turnback_top.csv"
    bundle = {
        "sli": np.asarray([0.1, np.nan, 0.9, 0.5]),
        "video_ids": np.asarray(["a", "b", "c", "d"]),
        "turnback_excursion_bin_ratio_exp": np.asarray([[0.1], [0.2], [0.9], [0.5]]),
        "turnback_excursion_bin_ratio_ctrl": np.zeros((4, 1)),
        "turnback_excursion_bin_pair_inner_deltas_mm": np.asarray([3]),
        "turnback_excursion_bin_pair_outer_deltas_mm": np.asarray([5]),
    }

    write_turnback_ratio_bundles_graphpad_csv(
        [("Ctrl", bundle)], out, top_sli_fraction=2 / 3
    )

    assert out.read_text().splitlines() == ["Ctrl", "0.5", "0.9"]
