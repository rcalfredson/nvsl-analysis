#!/usr/bin/env python3
"""Read-only sensitivity audit of current manuscript walking occupancy maps.

Run with the analysis environment:
  python scripts/audit_walking_scale.py --output exports/diagnostics/walking_scale

Only diagnostic files are written. Production loading, interpolation, speeds,
exclusions, windows and baseline histograms are reused, without invoking the
VideoAnalysis constructor's downstream analyses or any figure rendering.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import contextlib
import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECIPE_NAMES = [
    "figure_4b_training",
    "figure_4b_post",
    *[
        f"extended_data_figure_13a_{floor}_{period}"
        for floor in ("flat", "agarose")
        for period in ("pre", "training")
    ],
    *[
        f"extended_data_figure_15{letter}_{period}"
        for letter in "dgjm"
        for period in ("training", "post")
    ],
]


def literal_recipe_value(node, values):
    """Evaluate data assignments only; never execute notebook cells."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal_recipe_value(node.left, values) + literal_recipe_value(node.right, values)
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [literal_recipe_value(x, values) for x in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.Dict):
        return {
            literal_recipe_value(key, values): literal_recipe_value(value, values)
            for key, value in zip(node.keys, node.values)
        }
    if (
        isinstance(node, ast.Call)
        and not node.keywords
        and len(node.args) == 1
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "quote"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "shlex"
    ):
        return shlex.quote(literal_recipe_value(node.args[0], values))
    raise ValueError("Not a supported recipe-data expression")


def read_recipes(notebook):
    """Read the explicitly supported manuscript recipes without executing cells."""
    values = {}
    for cell in json.loads(Path(notebook).read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    value = literal_recipe_value(node.value, values)
                except (ValueError, KeyError, TypeError):
                    continue
                values[node.targets[0].id] = value
    missing = set(RECIPE_NAMES) - values.keys()
    if missing:
        raise ValueError(f"Missing supported notebook recipes: {sorted(missing)}")
    return {name: values[name] for name in RECIPE_NAMES}


def walking_masks(sp, ppmm, factor):
    """Compare strict thresholds on the same camera-pixel/second speed array."""
    if not np.isfinite(ppmm) or ppmm <= 0 or not np.isfinite(factor) or factor <= 0:
        raise ValueError("Positive finite physical scale required")
    return np.asarray(sp) > 2 * ppmm, np.asarray(sp) > 2 * ppmm * factor


def histogram(xy, mask, bins, rng, post_offset=None):
    """Match production NaN policy and post slicing, including edge inclusivity."""
    xy = [np.array(a, copy=True) for a in xy]
    if post_offset is not None:
        for a in xy:
            a[:post_offset] = np.nan
    selected = [a[mask] for a in xy]
    np.testing.assert_array_equal(np.isnan(selected[0]), np.isnan(selected[1]))
    paired = ~np.isnan(selected[0])
    points = [a[paired] for a in selected]
    mp = np.histogram2d(*points, bins=bins, range=rng)[0].T
    return mp, int(paired.sum())


def map_metrics(current, scaled, ppmm, bin_sizes=(2, 2)):
    """Measure occupancy mass and unit-mass shape separately.

    Inputs are per-fly or group maps already divided by window length. Relative
    changes and shape_tv are fractions, not percentages. Centroid shifts use
    template bin widths and reference template pixels/mm, without camera scaling.
    """
    current = np.asarray(current)
    scaled = np.asarray(scaled)
    mass_current = float(current.sum())
    mass_scaled = float(scaled.sum())
    l1 = float(np.abs(scaled - current).sum())
    result = dict(
        mass_current=mass_current,
        mass_scaled=mass_scaled,
        mass_delta=mass_scaled - mass_current,
        relative_mass_delta=(mass_scaled / mass_current - 1) if mass_current else None,
        l1=l1,
        relative_l1=l1 / mass_current if mass_current else None,
        max_abs_bin_delta=float(np.max(np.abs(scaled - current))),
    )
    if mass_current and mass_scaled:
        shape_current = current / mass_current
        shape_scaled = scaled / mass_scaled
        yy, xx = np.indices(current.shape)
        shape_difference = shape_scaled - shape_current
        shift_x = (shape_difference * xx).sum() * bin_sizes[0]
        shift_y = (shape_difference * yy).sum() * bin_sizes[1]
        result.update(
            shape_tv=float(np.abs(shape_current - shape_scaled).sum() / 2),
            centroid_shift_mm=float(np.hypot(shift_x, shift_y) / ppmm),
        )
    else:
        result.update(shape_tv=None, centroid_shift_mm=None)
    return result


def distribution(values):
    """Summarize finite individual-fly values, retaining their sample count."""
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(a):
        return {"n": 0}
    quantiles = dict(zip(
        ("min", "p05", "p25", "median", "p75", "p95", "max"),
        map(float, np.quantile(a, [0, .05, .25, .5, .75, .95, 1])),
    ))
    return dict(n=len(a), mean=float(a.mean()), **quantiles)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def heatmap_window(va, opts, t, period):
    """Return half-open bounds and the production post-start masking index."""
    if period == "pre":
        bounds = va._heatmapPreTrainingFrameRange()
        return (*bounds, None) if bounds is not None else None
    if period == "training":
        bounds = va._heatmapTrainingFrameRange(t)
        return (*bounds, None) if bounds is not None else None
    from src.utils import util
    fi = util.none2val(va._postSyncBucket(t, skip=0))
    la = fi + va._min2f(opts.rpiPostBucketLenMin)
    if not np.isfinite(fi) or la > t.postStop:
        return None
    fi_ri = util.none2val(va._idxSync(va._heatmapPostStartType(), t, fi, la, skip=0), la)
    return int(fi), int(la), int(fi_ri)


def region_masks(va, trx, t, role, xym, xyM, shape):
    """Bin-center regions; physical labels use the hypothesized template calibration."""
    ny, nx = shape
    bin_width = (xyM[0] - xym[0]) / nx
    bin_height = (xyM[1] - xym[1]) / ny
    xx, yy = np.meshgrid(
        np.linspace(xym[0], xyM[0], nx, endpoint=False) + bin_width / 2,
        np.linspace(xym[1], xyM[1], ny, endpoint=False) + bin_height / 2,
    )
    from src.utils.common import CT
    if va.ct is CT.htl:
        lo, hi = np.asarray(va.ct.value["floor"])
    else:
        corners = [
            va.mirror(va.xf._shift(corner, va.trxf[role], toTL=False))
            for corner in va.ct.value["floor"]
        ]
        lo, hi = np.min(corners, axis=0), np.max(corners, axis=0)
    edge_distance = np.minimum.reduce([xx - lo[0], hi[0] - xx, yy - lo[1], hi[1] - yy])
    inside = edge_distance >= 0
    edge_band_width = 2 * trx.pxPerMmFloor
    regions = {
        "floor_interior": edge_distance >= edge_band_width,
        "floor_edge_2mm": inside & (edge_distance < edge_band_width),
        "outside_floor": ~inside,
    }
    if t.circles(role):
        cx, cy, radius = t.circles(role)[0]
        if va.ct is CT.htl:
            cx, cy = va.xf.f2t(cx, cy, f=va.trxf[role])
        else:
            cx, cy = va.mirror(va.xf.f2t(cx, cy))
        radius /= va.xf.fctr
        d = np.hypot(xx-cx, yy-cy)
        regions.update(
            reward_disk=d <= radius,
            reward_outer_annulus_2mm=(d > radius) & (d <= radius + edge_band_width),
        )
    return regions


@dataclass
class AuditResults:
    """Accumulate diagnostic records without retaining video-analysis objects."""

    rows: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    scales: list = field(default_factory=list)
    maps: dict = field(default_factory=lambda: defaultdict(list))
    mask_checks: int = 0
    histogram_checks: int = 0


@contextlib.contextmanager
def diagnostic_environment(out):
    """Keep incidental analysis files in the audit directory; restore process state."""
    config = ROOT / ".analyze.local.env"
    if config.exists():
        shutil.copy2(config, out / "work/.analyze.local.env")

    old_cwd = Path.cwd()
    old_argv = sys.argv
    old_path = sys.path.copy()
    old_bytecode = sys.dont_write_bytecode
    old_environment = {key: os.environ.get(key) for key in ("MPLBACKEND", "MPLCONFIGDIR")}
    import __main__
    old_detector = getattr(__main__, "FlyDetector", None)
    try:
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = str(out / "work/matplotlib")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(ROOT))
        os.chdir(out / "work")
        # analyze defines the authoritative parser and also parses argv on import.
        sys.argv = ["analyze.py"]
        import analyze
        sys.argv = old_argv
        # Legacy .data pickles reference this exact production compatibility stub.
        __main__.FlyDetector = analyze.FlyDetector
        yield analyze
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if old_detector is None:
            if hasattr(__main__, "FlyDetector"):
                delattr(__main__, "FlyDetector")
        else:
            __main__.FlyDetector = old_detector


def analysis_options(analyze, command):
    """Use production defaults; reject options outside the read-only audit scope."""
    options = analyze.p.parse_args(shlex.split(command)[2:])
    options.cTurnAnlyz = False
    options.rTurnAnlyz = False
    analyze._normalize_com_options(options)
    unsafe = (
        options.showTrackIssues,
        options.showRewardMismatch,
        options.chooseOrientations,
        options.hm_pair_export,
        options.hm_pair_manifest,
    )
    if any(unsafe):
        raise ValueError("Recipe enables unsupported plotting, orientation or pairing options")
    if len(options.hm_periods) != 1:
        raise ValueError("Each audit recipe must select exactly one heatmap period")
    return options


def cohort_name(recipe_name):
    """Pair pre/training/post recipes without conflating experimental cohorts."""
    for suffix in ("_training", "_post", "_pre"):
        recipe_name = recipe_name.removesuffix(suffix)
    return recipe_name


def prepare_recipes(recipes, analyze, max_videos, provenance):
    """Resolve notebook recipes, video lists and historical cache provenance."""
    from src.plotting.heatmap_workflow import heatmap_cache_mismatch, heatmap_cache_provenance
    from src.utils import util

    cohorts = defaultdict(list)
    for name, recipe in recipes.items():
        options = analysis_options(analyze, recipe["command"])
        if options.fly not in ("0-9", "0-1"):
            raise ValueError(f"Unsupported manuscript fly selector: {options.fly}")
        if len(recipe["panels"]) != 1:
            raise ValueError("The audit currently supports one panel per recipe")
        panel = next(iter(recipe["panels"]))
        paths = util.fileList(options.video, pattern=util.AVI_X)
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate recording within recipe {name}")
        paths = sorted(paths)
        if max_videos is not None:
            paths = paths[:max_videos]
        if not paths:
            raise ValueError(f"No matching recordings for {name}")
        cohort = cohort_name(name)
        entry = dict(
            name=name,
            panel=panel,
            recipe=recipe,
            opts=options,
            paths=paths,
            flies=list(range(10 if options.fly == "0-9" else 2)),
            cohort=cohort,
        )
        cohorts[cohort].append(entry)
        cache = ROOT / "exports/manuscript/heatmaps" / f"{panel}.npz"
        expected = heatmap_cache_provenance(ROOT, recipe["command"])
        provenance["recipes"][name] = dict(
            command=recipe["command"],
            cohort=cohort,
            panel=panel,
            videos=paths,
            cache=str(cache),
            cache_mismatch=heatmap_cache_mismatch(cache, expected),
            cache_sha256=sha256(cache) if cache.exists() else None,
            analysis_provenance=expected,
        )
    return cohorts


def record_input_files(path, cohort, provenance):
    """Hash tracking/protocol/notes; record video metadata without hashing full videos."""
    record = provenance["recordings"].setdefault(path, dict(cohorts=[], files={}))
    record["cohorts"].append(cohort)
    if record["files"]:
        return
    video = Path(path)
    inputs = [
        video,
        video.with_suffix(".data"),
        video.with_suffix(".trx"),
        video.with_name(video.stem + "__note.txt"),
    ]
    for input_path in inputs:
        if input_path.exists():
            stat = input_path.stat()
            record["files"][str(input_path)] = dict(
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=sha256(input_path) if input_path.suffix != ".avi" else None,
            )


def load_analysis(path, experimental_fly, entry, analyze, log):
    """Run only production loading, trajectory initialization and note exclusions.

    VideoAnalysis.__init__ also performs unrelated analyses and writes outputs.
    This adapter deliberately calls the ordered initialization subset it needs.
    The returned object's capture is owned by the caller and must be released.
    """
    from src.analysis.video_analysis import VideoAnalysis

    va = VideoAnalysis.__new__(VideoAnalysis)
    va.opts = analysis_options(analyze, entry["recipe"]["command"])
    va.gidx = 0
    va.f = experimental_fly
    va._skipped = True
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            va._loadData(path)
            if not va.trns or getattr(va, "isyc", False):
                return va
            va.flies = (0,) if va.noyc else (0, 1)
            va.setOptionDefaults()
            va._initTrx()
            va._readNoteFile(path)
        return va
    except BaseException:
        if hasattr(va, "cap"):
            va.cap.release()
        raise


def validate_walking_masks(va, results):
    """Require exact full-trajectory baseline agreement before comparing windows."""
    masks = {}
    for role, trajectory in enumerate(va.trx):
        if hasattr(trajectory, "walking"):
            current, scaled = walking_masks(
                trajectory.sp, trajectory.pxPerMmFloor, va.xf.fctr
            )
            np.testing.assert_array_equal(current, trajectory.walking)
            results.mask_checks += 1
            masks[role] = (current, scaled, trajectory.walking.copy())
    return masks


def compare_fly_window(va, trajectory, role, training, training_index, period, bounds, masks, base):
    """Compare one window and validate its baseline against production calcHm()."""
    from src.utils.constants import HEATMAP_DIV

    first, last, post_start = bounds
    if not 0 <= first < last <= len(trajectory.x):
        raise ValueError(f"Invalid frame range {first}:{last} for {base}")
    current_full, scaled_full, saved = masks
    current = current_full[first:last]
    scaled = scaled_full[first:last]
    xy = va._heatmapCoords(trajectory, role, first, last)
    lower, upper = va._heatmapBounds(role)
    bins = [int(value) for value in (upper - lower) / HEATMAP_DIV]
    histogram_range = np.vstack((lower, upper)).T
    post_offset = None if post_start is None else post_start - first
    current_counts, selected_current = histogram(xy, current, bins, histogram_range, post_offset)
    scaled_counts, selected_scaled = histogram(xy, scaled, bins, histogram_range, post_offset)

    if period == "pre":
        production = va.heatmapPre[role]
    else:
        production_maps = va.heatmapPost if period == "post" else va.heatmap
        production = production_maps[role][training_index]
    np.testing.assert_array_equal(current_counts, production[0])
    n_frames = last - first
    if n_frames != production[1]:
        raise AssertionError("Diagnostic and production histogram denominators differ")

    # Normalize by all window frames, never by the selected walking positions.
    current_map = current_counts / n_frames
    scaled_map = scaled_counts / n_frames
    bin_sizes = (upper - lower) / np.asarray(bins)
    metrics = map_metrics(current_map, scaled_map, trajectory.pxPerMmFloor, bin_sizes)
    finite_xy = np.isfinite(trajectory.x[first:last]) & np.isfinite(trajectory.y[first:last])
    finite_speed = np.isfinite(trajectory.sp[first:last])
    changed = current != scaled
    missing_step = trajectory.nan.copy()
    missing_step[1:] |= trajectory.nan[:-1]
    changed_speeds = trajectory.sp[first:last][changed]
    speed_quantiles = np.quantile(changed_speeds, [0, .5, 1]) if changed.any() else [None] * 3
    count_current = int(round(current_map.sum() * n_frames))
    count_scaled = int(round(scaled_map.sum() * n_frames))

    row = dict(
        **base,
        status="ok",
        fi=first,
        la=last,
        post_start=post_start,
        n_frames=n_frames,
        bin_width=float(bin_sizes[0]),
        bin_height=float(bin_sizes[1]),
        original_missing=int(trajectory.nan[first:last].sum()),
        finite_positions=int(finite_xy.sum()),
        finite_speeds=int(finite_speed.sum()),
        finite_joint=int((finite_xy & finite_speed).sum()),
        walking_current=int(current.sum()),
        walking_scaled=int(scaled.sum()),
        walking_fraction_current=float(current.mean()),
        walking_fraction_scaled=float(scaled.mean()),
        walking_delta_pp=float((scaled.mean() - current.mean()) * 100),
        changed_frames=int(changed.sum()),
        changed_fraction=float(changed.mean()),
        changed_originally_missing=int((changed & trajectory.nan[first:last]).sum()),
        changed_step_touches_original_missing=int((changed & missing_step[first:last]).sum()),
        changed_speed_px_s_min=speed_quantiles[0],
        changed_speed_px_s_median=speed_quantiles[1],
        changed_speed_px_s_max=speed_quantiles[2],
        added=int((scaled & ~current).sum()),
        removed=int((current & ~scaled).sum()),
        effective_threshold_current_mm_s=2 / va.xf.fctr,
        selected_positions_current=selected_current,
        selected_positions_scaled=selected_scaled,
        histogram_counts_current=count_current,
        histogram_counts_scaled=count_scaled,
        oob_or_nonfinite_current=selected_current - count_current,
        oob_or_nonfinite_scaled=selected_scaled - count_scaled,
        eligible_current=bool(current_map.sum() > 0),
        eligible_scaled=bool(scaled_map.sum() > 0),
        **metrics,
    )
    regions = []
    for region, mask in region_masks(va, trajectory, training, role, lower, upper, current_map.shape).items():
        regions.append(dict(
            **base,
            region=region,
            mass_current=float(current_map[mask].sum()),
            mass_scaled=float(scaled_map[mask].sum()),
            absolute_change=float(np.abs(scaled_map - current_map)[mask].sum()),
        ))
    np.testing.assert_array_equal(trajectory.walking, saved)
    return row, regions, current_map, scaled_map


def compare_recipe(va, entry, masks, analyze, log, results):
    """Apply a recipe's production windows and eligibility to initialized trajectories."""
    va.opts = entry["opts"]
    va.setOptionDefaults()
    with contextlib.redirect_stdout(log):
        va.calcHm()  # Numeric baseline only; no renderer is invoked.
    period, = va.opts.hm_periods
    indices = [0] if period == "pre" else analyze._resolve_heatmap_training_indices(va.trns, va.opts.num_trainings)
    if len(indices) != 1:
        raise ValueError("Each audit recipe must select exactly one training")
    for training_index in indices:
        training = va.trns[training_index]
        bounds = heatmap_window(va, va.opts, training, period)
        for role, trajectory in enumerate(va.trx):
            base = dict(
                panel=entry["panel"], recipe=entry["name"], cohort=entry["cohort"],
                chamber=va.ct.name, video=va.fn, experimental_fly=va.f,
                absolute_fly=va.trxf[role], role=role, period=period,
                training=training.n, factor=va.xf.fctr,
            )
            if trajectory.bad() or bounds is None or role not in masks:
                results.rows.append(dict(
                    **base,
                    status="bad_trajectory" if trajectory.bad() else "unavailable_window",
                    eligible_current=False, eligible_scaled=False,
                ))
                continue
            row, regions, current, scaled = compare_fly_window(
                va, trajectory, role, training, training_index, period, bounds, masks[role], base
            )
            results.rows.append(row)
            results.regions.extend(regions)
            results.maps[(entry["panel"], role)].append((len(results.rows) - 1, current, scaled))
            results.histogram_checks += 1


def collect_cohort(cohort, entries, analyze, log, provenance, results):
    """Load each experimental/yoked pair once, then compare its related recipes."""
    from src.analysis.training import Training

    first = entries[0]
    if not all(entry["paths"] == first["paths"] and entry["flies"] == first["flies"] for entry in entries):
        raise ValueError(f"Inconsistent video/fly selections within {cohort}")
    Training._exp = Training._expVals = None
    for video_index, path in enumerate(first["paths"]):
        print(f"{cohort}: recording {video_index + 1}/{len(first['paths'])}", flush=True)
        record_input_files(path, cohort, provenance)
        for experimental_fly in first["flies"]:
            va = load_analysis(path, experimental_fly, first, analyze, log)
            try:
                skip = dict(video=path, experimental_fly=experimental_fly, cohort=cohort)
                if not va.trns or getattr(va, "isyc", False):
                    results.skipped.append(dict(**skip, reason="no_training_or_yoked_selector"))
                    continue
                if experimental_fly == first["flies"][0]:
                    tm = va.dt["protocol"]["tm"]
                    results.scales.append(dict(
                        cohort=cohort, video=path, chamber=va.ct.name, tm_fctr=tm["fctr"],
                        rszFctr=tm.get("rszFctr"), resize_default="rszFctr" not in tm,
                        effective_factor=va.xf.fctr, fps=va.fps, frame_size=va.xf.frameSize,
                    ))
                masks = validate_walking_masks(va, results)
                if va.trx[0].bad():
                    results.skipped.append(dict(**skip, reason="experimental_trajectory_bad"))
                    continue
                va._skipped = False
                for entry in entries:
                    compare_recipe(va, entry, masks, analyze, log, results)
                for role, (_, _, saved) in masks.items():
                    np.testing.assert_array_equal(va.trx[role].walking, saved)
            finally:
                va.cap.release()
        log.flush()


def aggregate_maps(current, scaled):
    """Match production's separate nonempty-fly eligibility and equal-fly means."""
    keep_current = current.sum(axis=(1, 2)) > 0
    keep_scaled = scaled.sum(axis=(1, 2)) > 0
    if not keep_current.any() or not keep_scaled.any():
        raise ValueError("No eligible group heatmaps under one or both walking rules")
    group_current = current[keep_current].mean(axis=0)
    group_scaled = scaled[keep_scaled].mean(axis=0)
    return group_current, group_scaled, keep_current, keep_scaled


def compare_cached_group(panel, role, group_current, recipe_record, validation):
    """Record historical cache agreement separately from exact per-fly checks."""
    cache = Path(recipe_record["cache"])
    if not cache.exists():
        return None
    with np.load(cache, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        panels = [panel for row in metadata["rows"] for panel in row if panel["fly"] == role]
        if len(panels) != 1:
            raise ValueError(f"Expected one cached map for {panel}, role {role}")
        cached_map = archive[panels[0]["array"]]
        max_error = float(np.max(np.abs(group_current - cached_map)))
        validation.append(dict(
            panel=panel, role=role,
            allclose=bool(np.allclose(group_current, cached_map, rtol=1e-12, atol=1e-15)),
            max_abs_error=max_error, cache_mismatch=recipe_record["cache_mismatch"],
        ))
    return max_error


def summarize_groups(results, provenance, out, partial_run):
    """Export per-fly arrays and derive group metrics without plotting."""
    from src.utils.common import CT

    fields = (
        "walking_fraction_current", "walking_fraction_scaled", "walking_delta_pp",
        "changed_fraction", "relative_mass_delta", "relative_l1", "shape_tv", "centroid_shift_mm",
    )
    groups = []
    for (panel, role), items in results.maps.items():
        indices = np.array([index for index, _, _ in items])
        current = np.stack([current for _, current, _ in items])
        scaled = np.stack([scaled for _, _, scaled in items])
        group_current, group_scaled, keep_current, keep_scaled = aggregate_maps(current, scaled)
        common = keep_current & keep_scaled
        first = results.rows[indices[0]]
        ppmm = CT[first["chamber"]].pxPerMmFloor()
        bin_sizes = (first["bin_width"], first["bin_height"])
        common_metrics = None
        if common.any():
            common_metrics = map_metrics(current[common].mean(axis=0), scaled[common].mean(axis=0), ppmm, bin_sizes)
        summary = dict(
            panel=panel, role=role, cohort=first["cohort"], chamber=first["chamber"], period=first["period"],
            n_current=int(keep_current.sum()), n_scaled=int(keep_scaled.sum()),
            eligibility_added=int((keep_scaled & ~keep_current).sum()),
            eligibility_removed=int((keep_current & ~keep_scaled).sum()),
            **map_metrics(group_current, group_scaled, ppmm, bin_sizes),
            common_fly_metrics=common_metrics,
            fly_distributions={name: distribution(results.rows[index].get(name) for index in indices) for name in fields},
        )
        recipe_record = next(record for record in provenance["recipes"].values() if record["panel"] == panel)
        if not partial_run:
            error = compare_cached_group(panel, role, group_current, recipe_record, provenance["validation"]["cached_groups"])
            if error is not None:
                summary["cached_baseline_max_abs_error"] = error
        groups.append(summary)
        np.savez_compressed(
            out / "maps" / f"{panel}_role{role}.npz",
            row_indices=indices, current=current, scaled=scaled,
            group_current=group_current, group_scaled=group_scaled,
            difference=group_scaled - group_current,
            eligible_current=keep_current, eligible_scaled=keep_scaled,
        )
    return groups


def new_provenance():
    config = ROOT / ".analyze.local.env"
    return dict(
        script_sha256=sha256(__file__),
        notebook_sha256=sha256(ROOT / "notebooks/manuscript_figure_panels.ipynb"),
        python=sys.executable, numpy=np.__version__,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        local_config_sha256=sha256(config) if config.exists() else None,
        recipes={}, recordings={}, overlaps={}, validation={},
    )


def write_outputs(out, results, groups, provenance):
    csv_write(out / "flies.csv", results.rows)
    csv_write(out / "regions.csv", results.regions)
    csv_write(out / "skipped.csv", results.skipped)
    for name, content in (("scales", results.scales), ("groups", groups), ("provenance", provenance)):
        (out / f"{name}.json").write_text(json.dumps(content, indent=2) + "\n")


def run(args):
    out = Path(args.output).resolve()
    allowed = ROOT / "exports/diagnostics"
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError("Output must be a dedicated subdirectory of exports/diagnostics")
    if args.max_videos is not None and args.max_videos < 1:
        raise ValueError("--max-videos must be positive")
    for directory in (out, out / "work", out / "maps"):
        directory.mkdir(parents=True, exist_ok=True)
    recipes = read_recipes(ROOT / "notebooks/manuscript_figure_panels.ipynb")
    if args.recipes:
        if not set(args.recipes).issubset(recipes):
            raise ValueError("Unknown recipe")
        recipes = {name: recipe for name, recipe in recipes.items() if name in args.recipes}
    results = AuditResults()
    with diagnostic_environment(out) as analyze:
        provenance = new_provenance()
        cohorts = prepare_recipes(recipes, analyze, args.max_videos, provenance)
        with (out / "initialization.log").open("w") as log:
            for cohort, entries in cohorts.items():
                collect_cohort(cohort, entries, analyze, log, provenance, results)
        provenance["overlaps"] = {
            path: record["cohorts"] for path, record in provenance["recordings"].items()
            if len(record["cohorts"]) > 1
        }
        provenance["validation"].update(
            exact_full_trajectory_mask_checks=results.mask_checks,
            exact_histogram_checks=results.histogram_checks,
            walking_arrays_unchanged=True, partial_run=args.max_videos is not None, cached_groups=[],
        )
        groups = summarize_groups(results, provenance, out, args.max_videos is not None)
        provenance["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_outputs(out, results, groups, provenance)
    print(
        f"Completed: {len(results.rows)} fly/window records, {results.mask_checks} mask checks, "
        f"{results.histogram_checks} histogram checks; {out}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="exports/diagnostics/walking_scale")
    parser.add_argument("--recipes", nargs="+", choices=RECIPE_NAMES)
    parser.add_argument("--max-videos", type=int, help="Explicit partial smoke test per cohort; disables cache comparison")
    run(parser.parse_args())
