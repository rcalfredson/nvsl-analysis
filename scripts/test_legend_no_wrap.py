#!/usr/bin/env python3
"""Generate synthetic plots with deliberately long legend labels.

Run from the repository root after activating the project environment:

    conda activate analysis3.13
    python scripts/test_legend_no_wrap.py

The script writes visual smoke-test PNGs to /tmp/nvsl_legend_no_wrap_smoke by default.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = Path("/tmp/nvsl_matplotlib_config")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


DEFAULT_LABELS = [
    "Control condition with a deliberately long legend label that should stay on one line",
    "Experimental condition with embedded newline\nand enough words to reveal wrapping behavior",
]


def _parse_labels(raw: str | None) -> list[str]:
    if raw is None:
        return DEFAULT_LABELS
    labels = [part.strip() for part in raw.split("|")]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError("--labels must contain at least one non-empty label")
    return labels


def _import_analyze_without_consuming_script_args():
    """Import analyze.py without letting its global argparse parse this script's args."""
    import sys

    old_argv = sys.argv[:]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        sys.argv = [old_argv[0]]
        import analyze
    finally:
        sys.argv = old_argv
    return analyze


def _make_plot_rewards_data(rng, *, n_groups: int, n_trains: int, n_buckets: int):
    n_flies = 1
    videos_per_group = 24
    n_videos = n_groups * videos_per_group
    data = np.zeros((n_videos, n_trains, n_flies * (n_buckets + 1)), dtype=float)

    x = np.linspace(-0.15, 0.25, n_buckets)
    for video_idx in range(n_videos):
        group_idx = video_idx // videos_per_group
        for train_idx in range(n_trains):
            offset = 0.12 * group_idx + 0.05 * train_idx
            trend = x + offset
            noise = rng.normal(0.0, 0.08, size=n_buckets)
            values = np.concatenate([trend + noise, [np.nan]])
            data[video_idx, train_idx, :] = values

    group_indices = np.repeat(np.arange(n_groups), videos_per_group)
    return data, group_indices


def run_plot_rewards_smoke(out_dir: Path, labels: list[str], *, font_size: float) -> Path:
    analyze = _import_analyze_without_consuming_script_args()
    analyze.customizer.update_font_size(font_size)
    analyze.opts.imageFormat = "png"
    analyze.opts.plot_rewards_xlabel = None
    analyze.opts.plot_rewards_ylabel = None
    analyze.REWARD_PI_DIFF_IMG_FILE = str(
        out_dir / "plotRewards_long_legend__%s_min_buckets.png"
    )

    rng = np.random.default_rng(12345)
    data, group_indices = _make_plot_rewards_data(
        rng,
        n_groups=len(labels),
        n_trains=2,
        n_buckets=5,
    )

    class DummyTraining:
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name

        def hasSymCtrl(self):
            return False

    class DummyVA:
        def __init__(self):
            self.flies = [0]
            self.saved_auc = {}
            self.numPostBuckets = None
            self.numNonPostBuckets = 5
            self.rpiNumPostBuckets = None
            self.rpiNumNonPostBuckets = 0
            self.speed = [0]
            self.stopFrac = [0]
            self.ct = analyze.CT.htl

        def _bad(self, _fly):
            return False

    fake_va = DummyVA()
    trainings = [DummyTraining("Training 1"), DummyTraining("Training 2")]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        analyze.plotRewards(
            va=fake_va,
            tp="rpid",
            a=data,
            trns=trainings,
            gis=group_indices,
            gls=labels,
            vas=[fake_va] * len(group_indices),
        )

    matches = sorted(out_dir.glob("plotRewards_long_legend__*_min_buckets.png"))
    if not matches:
        raise FileNotFoundError("plotRewards did not produce the expected smoke-test PNG")
    return matches[-1]


def _make_metric_bundle(rng, *, label: str, bundle_idx: int):
    n_videos = 24
    n_trains = 2
    n_buckets = 5
    x = np.linspace(0.0, 1.0, n_buckets)
    commag = np.empty((n_videos, n_trains, n_buckets), dtype=float)

    for video_idx in range(n_videos):
        for train_idx in range(n_trains):
            offset = 0.08 * bundle_idx + 0.04 * train_idx
            commag[video_idx, train_idx, :] = (
                0.35 + offset + 0.2 * x + rng.normal(0.0, 0.05, size=n_buckets)
            )

    return {
        "sli": rng.normal(0.0, 0.35, size=n_videos),
        "group_label": label,
        "bucket_len_min": 5.0,
        "training_names": np.array(["Training 1", "Training 2"], dtype=object),
        "video_ids": np.array(
            [f"bundle{bundle_idx}_video{idx:02d}" for idx in range(n_videos)],
            dtype=object,
        ),
        "sli_training_idx": 1,
        "sli_use_training_mean": False,
        "commag_exp": commag,
    }


def run_metric_bundle_smoke(out_dir: Path, labels: list[str], *, font_size: float) -> Path:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src.plotting.metric_sli_bundle_plotter import plot_metric_sli_bundles

    rng = np.random.default_rng(67890)
    bundle_paths = []
    for idx, label in enumerate(labels):
        bundle = _make_metric_bundle(rng, label=label, bundle_idx=idx)
        path = out_dir / f"synthetic_metric_bundle_{idx + 1}.npz"
        np.savez(path, **bundle)
        bundle_paths.append(str(path))

    out_path = out_dir / "metric_sli_bundle_long_legend.png"
    opts = SimpleNamespace(
        wspace=0.35,
        imageFormat="png",
        fontSize=font_size,
        fontFamily=None,
    )

    plot_metric_sli_bundles(
        bundle_paths,
        str(out_path),
        labels=labels,
        opts=opts,
        metric="commag",
        show_legend=True,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--which",
        choices=("all", "plotRewards", "metric"),
        default="all",
        help="Which visual smoke test to run.",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/nvsl_legend_no_wrap_smoke",
        help="Directory for generated PNGs and synthetic bundles.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Pipe-separated legend labels. Example: 'Very long control label|Very long experimental label'",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=23.0,
        help="Large font size makes wrapping/clipping behavior easier to inspect.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = _parse_labels(args.labels)

    outputs = []
    if args.which in ("all", "plotRewards"):
        outputs.append(run_plot_rewards_smoke(out_dir, labels, font_size=args.font_size))
    if args.which in ("all", "metric"):
        outputs.append(run_metric_bundle_smoke(out_dir, labels, font_size=args.font_size))

    print("Generated legend no-wrap smoke plots:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
