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

### Wall-facing entry subset

Add `--agarose-wall-facing-entry-only` to retain episodes whose first point
inside the outer circle lies on its outward-facing semicircle. For entry point
`p`, site center `c`, and chamber-floor center `a`, the criterion is
`dot(p - c, c - a) > 0`. It is evaluated identically for physical and rotated
virtual sites and therefore handles mirrored chambers without hard-coded
directions. The normalized dot product is retained in episode debug output as
`entry_wall_alignment` for later angular-margin sensitivity checks.

For an initial all-site analysis, use the wall-facing option without
`--agarose-farthest-from-reward-only`:

```bash
python analyze.py \
  -v "/path/to/videos/*.avi" -f "0-1" --rCC 15 \
  --agarose-dual-circle \
  --agarose-virtual-control \
  --agarose-wall-facing-entry-only \
  --export-agarose-sli-bundle exports/group_agarose_control_wall_entry.npz
```

To inspect the actual episodes accepted by this filter, request an annotated
image gallery from the same analysis run:

```bash
python analyze.py \
  -v "/path/to/videos/*.avi" -f "0-1" --rCC 15 \
  --agarose-dual-circle \
  --agarose-virtual-control \
  --agarose-wall-facing-entry-only \
  --agarose-dual-circle-debug-images-dir imgs/agarose_wall_entry_debug \
  --agarose-dual-circle-debug-max-images 12 \
  --agarose-dual-circle-debug-training-index 2 \
  --agarose-dual-circle-debug-sync-bucket-start-index 2 \
  --agarose-dual-circle-debug-sync-bucket-end-index 5
```

Each image shows both circle sets with light outlines, the local trajectory,
the selected episode segment, the entry vector `p - c`, the wall direction
`c - a`, and the normalized alignment score. The translucent semicircle is the
wall-facing half accepted by the filter. The gallery samples the experimental
fly and balances physical/virtual geometry and avoidance/contact outcome where
examples are available. Add `--agarose-dual-circle-debug-csv <path>` to save
the complete episode table as well.

### Concentric radius offsets and reward reference

The analysis circles remain concentric with each nominal agarose area. Their
sizes can be controlled independently:

- `--agarose-inner-radius-offset-mm` sets the inner-circle radius relative to
  the nominal agarose radius.
- `--agarose-outer-delta-mm` sets the outer-circle radius relative to that same
  nominal radius, rather than relative to the analysis inner circle.

The outer offset must be greater than the inner offset. For example, an inner
circle 1 mm beyond the agarose boundary and an outer circle 2 mm beyond it use
`--agarose-inner-radius-offset-mm 1 --agarose-outer-delta-mm 2`.

Separately:

- `--agarose-wall-facing-reference reward` defines the retained half using the
  applicable training's reward-circle center rather than the chamber-floor
  center.

With entry point `p`, site center `c`, and reward center `r`, the latter
criterion is `dot(p - c, c - r) > 0`. It therefore selects the reward-away half
of each outer circle. The reward center is resolved separately for each fly and
training; the experiment-wide pre period uses training 1's reward center, while
each training-specific pre period uses that training's center.

Run the reward-referenced semicircle variant with 1 mm and 2 mm concentric
offsets using:

```bash
python analyze.py \
  -v "/path/to/videos/*.avi" -f "0-1" --rCC 15 \
  --agarose-dual-circle \
  --agarose-inner-radius-offset-mm 1 \
  --agarose-outer-delta-mm 2 \
  --agarose-wall-facing-entry-only \
  --agarose-wall-facing-reference reward \
  --export-agarose-sli-bundle exports/group_agarose_radius1-2_rewardref.npz
```

The bundle records both offsets and the reference point. The debug gallery and
spatial-control schematic read the same settings, so their radii and entry
divider match the analyzed geometry.

### Reward-facing intersection-arc exclusion

`--agarose-exclude-reward-facing-arc-entries` replaces the semicircle filter
with a narrower, geometry-defined exclusion. For each site, the analysis grows
a circle around the applicable reward center until it reaches the nearest edge
of the nominal agarose area. The portion of the analysis outer circle lying
inside that reward-centered circle is the rejected entry arc; entries elsewhere
on the outer circle are retained.

The nominal agarose center and radius determine the reward-centered circle
regardless of the analysis inner-circle offset. Flat chambers use the same
nominal agarose geometry, and virtual sites use the corresponding rotated
nominal geometry.

The boundary crossing is normally calculated from the intersection of the
segment joining the last pre-entry position to the first inside position with
the outer circle. A radial projection of the first inside position is used only
when that segment intersection is unavailable. This makes the filter depend on
where the trajectory crossed the circumference rather than tracking-step depth
inside the circle.

The semicircle and intersection-arc filters are mutually exclusive. To test the
new arc filter with inner and outer offsets of 1 mm and 2 mm:

```bash
python analyze.py \
  -v "/path/to/videos/*.avi" -f "0-1" --rCC 15 \
  --agarose-dual-circle \
  --agarose-inner-radius-offset-mm 1 \
  --agarose-outer-delta-mm 2 \
  --agarose-exclude-reward-facing-arc-entries \
  --export-agarose-sli-bundle exports/group_agarose_reward_arc.npz
```

The debug CSV records the arc width in degrees, the expanded reward-circle
radius, the boundary-crossing coordinates and method, and whether the entry was
rejected. Debug galleries balance retained and rejected entries where both are
available. They shade the rejected arc, draw the expanded reward-centered
circle, outline the nominal agarose boundary, and mark the inferred boundary
crossing. This makes narrow or empty exclusion arcs directly visible.

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
