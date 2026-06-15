import csv
from types import SimpleNamespace

import matplotlib.pyplot as plt

from src.plotting.contactless_episode_maxdist_swarm import (
    load_episode_csvs,
    plot_contactless_episode_maxdist_swarm,
)


def _write_csv(path, group, rows):
    fieldnames = [
        "group",
        "unit_id",
        "contains_wall_contact",
        "included_in_metric",
        "max_distance_from_reward_center_mm",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"group": group, **row})


def test_load_and_plot_contactless_episode_swarm(tmp_path):
    ctrl = tmp_path / "ctrl.csv"
    pfn = tmp_path / "pfn.csv"
    _write_csv(
        ctrl,
        "Ctrl",
        [
            {
                "unit_id": "a",
                "contains_wall_contact": False,
                "included_in_metric": True,
                "max_distance_from_reward_center_mm": 4.0,
            },
            {
                "unit_id": "a",
                "contains_wall_contact": True,
                "included_in_metric": True,
                "max_distance_from_reward_center_mm": 20.0,
            },
            {
                "unit_id": "b",
                "contains_wall_contact": False,
                "included_in_metric": False,
                "max_distance_from_reward_center_mm": 7.0,
            },
        ],
    )
    _write_csv(
        pfn,
        "PFNd",
        [
            {
                "unit_id": "c",
                "contains_wall_contact": True,
                "included_in_metric": True,
                "max_distance_from_reward_center_mm": 18.0,
            }
        ],
    )

    data, group_order = load_episode_csvs([str(ctrl), str(pfn)])

    assert group_order == ["Ctrl", "PFNd"]
    assert len(data) == 3
    fig = plot_contactless_episode_maxdist_swarm(
        data,
        group_order=group_order,
        opts=SimpleNamespace(fontSize=16, fontFamily=None),
    )
    ax = fig.axes[0]
    assert len(ax.collections) == 3
    assert "1 flies, 2 episodes" in ax.get_xticklabels()[0].get_text()
    assert ax.get_ylabel() == "Maximum distance from\nreward center (mm)"
    plt.close(fig)
