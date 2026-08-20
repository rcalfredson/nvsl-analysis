# Agarose dual-circle spatial control

The `--agarose-virtual-control` option adds a spatial negative control to the
existing `--agarose-dual-circle` analysis. It does not replace the physical-well
measurement.

For each fly, the analysis rigidly rotates the four agarose-well centers around
the transformed arena center. The default rotation is 45 degrees, which places
each virtual site midway between adjacent physical wells. Inner radius, outer
padding, angular symmetry, and distance from the arena center are unchanged. In
these experiments the reward circle is offset from that center, so a rotation
does not preserve each site's distance from the reward circle. Episodes and
avoidance outcomes are then calculated with exactly the same code used for the
physical sites.

Example:

```bash
python analyze.py \
  -v "/path/to/video.avi" -f "0-9" \
  --agarose-dual-circle \
  --agarose-virtual-control \
  --export-agarose-sli-bundle exports/group_agarose_control.npz
```

The bundle retains the existing `agarose_*` arrays and adds paired arrays named
`agarose_virtual_ratio_*`, `agarose_virtual_total_*`, and
`agarose_virtual_avoid_*`. It also records `agarose_virtual_rotation_deg`. When
`--agarose-sli-include-pre` is used, corresponding virtual pre-training arrays
are included.

The primary validity contrast is physical minus virtual avoidance ratio within
the same video, fly role, training, and sync bucket. A positive contrast supports
agarose-specific avoidance. Similar or negative values indicate that generic
path structure or reward-directed navigation can explain as much or more of the
measured behavior. Counts should be inspected alongside ratios because the
rotated sites need not receive the same number of approaches.

Run the paired comparison for (for example) the last bucket of training 2 with:

```bash
python scripts/stats_agarose_virtual_control.py \
  --bundle exports/group_agarose_control.npz \
  --mode exp --training-index 2 --sync-bucket-index -1 \
  --csv-out exports/group_agarose_control_paired.csv
```

The script reports the mean paired difference, its 95% confidence interval, and
a paired t-test. A video enters that test only when both its physical and virtual
ratios pass the configured minimum-episode filter in the selected bucket.

For a contiguous timeframe, replace `--sync-bucket-index` with an inclusive
window. For example, this pools buckets 2 through 5:

```bash
python scripts/stats_agarose_virtual_control.py \
  --bundle exports/group_agarose_control.npz \
  --mode exp --training-index 2 \
  --sync-bucket-start-index 2 --sync-bucket-end-index 5
```

Window ratios are recomputed as the sum of avoidance episodes divided by the
sum of all episodes across the selected buckets. They are not unweighted means
of the bucket-level ratios. The minimum-episode threshold is applied to that
pooled denominator.

Use `--agarose-virtual-rotation-deg` to run a rotation sensitivity analysis.
Angles other than 45 degrees preserve radial distance but generally provide less
separation from the physical wells.

### Farthest-from-reward site subset

Add `--agarose-farthest-from-reward-only` to restrict both measurements to the
site or sites at the greatest center-to-center distance from the applicable
reward circle. Selection is recalculated for every training and fly, so mirrored
chambers are handled automatically. Sites within 0.25 mm of the maximum are
retained as symmetry ties. In the representative upper-right reward geometry,
this retains the left and bottom physical wells and the southwest virtual site.

The avoidance numerator and denominator are pooled over episodes at the retained
sites. Because this subset contains fewer sites, fewer flies may pass the usual
minimum-episode filter. Use a different output filename so the all-site bundle is
not overwritten:

```bash
python analyze.py \
  -v "/path/to/videos/*.avi" -f "0-1" --rCC 15 \
  --agarose-dual-circle \
  --agarose-virtual-control \
  --agarose-farthest-from-reward-only \
  --export-agarose-sli-bundle exports/group_agarose_control_far_reward.npz
```

The existing statistics command reads the subset flag from the bundle and
reports `sites=farthest-from-reward` in its selection line.
The plotting command also reads this metadata automatically: with two subset
bundles, it labels the summary as a farthest-from-reward comparison and limits
the chamber schematic to the retained physical and virtual sites.

## Slide-ready visualizations

The two completed chamber analyses can be plotted together as paired bar/swarm
ratios plus a physical-minus-virtual interaction panel. Supplying a representative
video also creates an annotated chamber image using the same reward and
dual-circle geometry as the analysis:

```bash
python scripts/plot_agarose_virtual_control.py \
  --agarose-bundle exports/agarose_control.npz \
  --flat-bundle exports/agarose_control_flatLgc.npz \
  --mode exp --training-index 2 \
  --sync-bucket-start-index 2 --sync-bucket-end-index 5 \
  --out imgs/agarose_virtual_control_summary.png \
  --background-video "/path/to/a/representative/agarose-large.avi" \
  --geometry-out imgs/agarose_virtual_control_geometry.png
```

The bar heights are video-level means and whiskers are 95% confidence intervals.
Open circles show individual videos. Lines in the placement panel connect the
physical and virtual ratios from the same video. The second panel displays those
paired differences and labels the between-chamber Welch interaction contrast.
