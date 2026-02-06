# src/plotting/wall_contacts_per_sync_bkt_pmf.py
from __future__ import annotations

from dataclasses import dataclass

from src.plotting.wall_contacts_pmf import (
    WallContactsPMFResult,
    plot_wall_contacts_pmf_overlay as _plot_wall_contacts_pmf_overlay,
)


@dataclass
class WallContactsPerSyncBktResult(WallContactsPMFResult):
    """
    Backward-compatible wrapper for the legacy sync-bucket PMF plotter.
    """

    @classmethod
    def load_npz(cls, path: str) -> "WallContactsPerSyncBktResult":
        res = WallContactsPMFResult.load_npz(path)
        if res.episode_kind != "sync_bucket":
            raise ValueError(
                "Expected sync-bucket NPZ (counts_per_bucket), "
                f"got episode_kind={res.episode_kind!r}"
            )
        return cls(**res.__dict__)


def plot_wall_contacts_pmf_overlay(
    *, results, labels, out_file, opts, customizer, log_tag="wall_contacts_pmf"
):
    # For sync-bucket NPZs, the generic plotter will label correctly via episode_kind.
    return _plot_wall_contacts_pmf_overlay(
        results=results,
        labels=labels,
        out_file=out_file,
        opts=opts,
        customizer=customizer,
        log_tag=log_tag,
    )
