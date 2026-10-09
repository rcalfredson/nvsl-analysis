# Annotation layout performance

`resolve_annotation_text_overlaps` previously repainted the entire figure when
moving each count or significance label, and again during each collision pass.
Every repaint also drew the other subplots, curves, legends, and tick labels.
The work therefore grew with both annotation count and figure complexity.

The resolver now draws once to settle the figure and obtain a renderer, then
uses `Text.get_window_extent(renderer)` to measure labels after moving them.
Matplotlib calculates these extents from the current text properties and
transform; it does not require repainting the figure. Placement order, collision
padding, iteration limits, linked-stack gaps, and headroom calculations are
unchanged. Callers still draw or save the figure to render the final positions.

Automatic layout engines, draw callbacks, and text objects with custom draw
methods retain the original full-redraw path because their drawing may itself
change positions. Renderer reuse is local to each resolver call, so later
changes to limits, fonts, DPI, or subplot geometry receive a fresh initial draw.

## Reproduce the focused comparison

Before committing the optimization, save the original implementation and run:

```bash
git show HEAD:src/plotting/annotation_layout.py > /tmp/annotation_layout_before.py
python -m scripts.benchmark_annotation_layout --reference /tmp/annotation_layout_before.py
```

After committing, use the revision preceding the optimization instead of `HEAD`.
The script asserts exact equality of annotation coordinates, count placement
sides, returned axis limits, and PNG/PDF/SVG bytes. Export timestamps and SVG
identifier salts are fixed for reproducibility. It exits with an error if any
comparison differs. Font size, DPI, subplot count, and bucket count are adjustable.
Run without `--reference` and with `--profile` to inspect the current workload.

A three-panel, eight-bucket comparison using Python 3.13.7 and Matplotlib 3.10.6
took 55.57 seconds and 855 figure draws before the change, versus 0.67 seconds
and 3 draws after it (about 83 times faster). All comparisons were identical.
These are resolver timings on a synthetic workload, not whole-analysis timings;
speedup depends on figure complexity, annotation collisions, and machine load.

## Real-data comparison

The following command was run with the original and optimized resolver:

```bash
python analyze.py -v '/media/Synology4/April/2026-09-28/morning/c4[12]_*' -f 0-9 --skpFT --rmCC 5
```

Both runs used the same Python/Matplotlib environment described above. A temporary
harness profiled only `postAnalyze`, counted Agg figure draws, and redirected
outputs to separate directories under `/tmp/nvsl-layout-real`. This avoids
overwriting working plots. Each run included 19 accepted video/fly analyses.

| Measurement | Original | Optimized |
| --- | ---: | ---: |
| `postAnalyze` wall time | 95.55 s | 28.03 s |
| Figure draws during `postAnalyze` | 1,224 | 360 |
| Time in 34 resolver calls directly from `analyze.py` | 60.53 s | 3.30 s |

All 31 summary PNGs produced by `postAnalyze` were byte-identical. The separate
`analysis.png` and four randomly selected reward example images produced after
`postAnalyze` were excluded from this comparison. The resolver subtotal excludes
calls made internally by bundle plotters; the total stage time includes them.
Profiling and draw counters add overhead, so these measurements are comparative
single-run results, not predictions for every analysis.

The remaining work includes other full-figure layout draws, text/font rendering,
and between-reward bundle calculations. The latter took about 4.9 seconds in the
optimized profile. Those paths have not been changed by this optimization.

## Regression checks

```bash
python -m pytest -q \
  test/test_annotation_layout.py \
  test/test_annotation_layout_performance.py \
  test/test_flexible_overlay_layout.py \
  test/test_com_sli_bundle_plotter.py \
  test/test_time_series_auc.py \
  test/test_plot_typography.py
```

The performance regression tests compare the optimized path with forced full
redraws at several font sizes and DPIs. They assert identical exports and a
bounded draw count, avoiding machine-dependent timing thresholds. Separate
cases verify that automatic layout, callbacks, and custom text drawing still
use full redraws.
