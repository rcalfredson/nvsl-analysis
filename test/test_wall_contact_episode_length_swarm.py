import csv
from types import SimpleNamespace

import matplotlib.pyplot as plt

from src.plotting.wall_contact_episode_length_swarm import (
    load_episode_csvs,
    plot_wall_contact_episode_length_swarm,
)


def _write_csv(path, group, rows):
    fieldnames = [
        "group",
        "unit_id",
        "contains_wall_contact",
        "included_in_metric",
        "trajectory_length_mm",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"group": group, **row})


def test_load_and_plot_wall_contact_episode_length_swarm(tmp_path):
    ctrl = tmp_path / "ctrl.csv"
    ar = tmp_path / "ar.csv"
    _write_csv(
        ctrl,
        "Ctrl",
        [
            {
                "unit_id": "a",
                "contains_wall_contact": False,
                "included_in_metric": True,
                "trajectory_length_mm": 12.0,
            },
            {
                "unit_id": "a",
                "contains_wall_contact": True,
                "included_in_metric": True,
                "trajectory_length_mm": 45.0,
            },
            {
                "unit_id": "b",
                "contains_wall_contact": True,
                "included_in_metric": False,
                "trajectory_length_mm": 80.0,
            },
        ],
    )
    _write_csv(
        ar,
        "Antennae removed",
        [
            {
                "unit_id": "c",
                "contains_wall_contact": True,
                "included_in_metric": True,
                "trajectory_length_mm": 90.0,
            }
        ],
    )

    data, group_order = load_episode_csvs([str(ctrl), str(ar)])

    assert group_order == ["Ctrl", "Antennae removed"]
    assert len(data) == 3
    fig = plot_wall_contact_episode_length_swarm(
        data,
        group_order=group_order,
        opts=SimpleNamespace(fontSize=16, fontFamily=None),
    )
    ax = fig.axes[0]
    assert len(ax.collections) == 3
    assert "1 flies, 2 episodes" in ax.get_xticklabels()[0].get_text()
    assert ax.get_ylabel() == "Trajectory length (mm)"
    plt.close(fig)
