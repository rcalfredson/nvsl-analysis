# Schmitt–Butterworth Behavior-State Detector

## Status and purpose

This document specifies the proposed alternative detector for assigning every
trajectory frame one of three states:

- `REST`: part of a stopped bout;
- `RUN`: walking and not part of an accepted turn;
- `TURN`: walking and part of an accepted turn.

The detector is intended to coexist with the current Haberkern-derived detector
in `src/analysis/behavior_states.py`. The public state values remain unchanged.
The proposed command-line selector is:

```text
--behavior-state-detector haberkern
--behavior-state-detector schmitt_butterworth
```

`haberkern` remains the default so that existing analyses are reproducible. The
new detector is a 7.5-FPS adaptation of a method originally applied at 60 FPS;
frame-count parameters that cannot be transferred literally are identified
below.
The name `schmitt_butterworth` is provisional and can be replaced by the source
paper's short name once its citation is attached to the implementation.

## Coordinate, time, and interval conventions

- Trajectory positions are the interpolated centroid coordinates `trj.x` and
  `trj.y`.
- Pixel distances are converted with the same effective pixels/mm scale used by
  the existing behavior-state detector.
- Timestamps are used when valid. Otherwise, the sample interval is `1 / fps`.
- Frame intervals in this document are inclusive unless explicitly described as
  half-open Python slices.
- Signed physical angles use Cartesian coordinates `(x, -y)`, compensating for
  image y increasing downward. Positive angles therefore mean counterclockwise
  (CCW), and negative angles mean clockwise (CW).
- All stated comparisons are inclusive unless the rule explicitly says
  "less than." Walking begins at speed `>= 2 mm/s`, stopping begins at speed
  `<= 1 mm/s`, and a peak qualifies at `>= 120 deg/s`.

## Configuration

The initial implementation should expose a detector-specific configuration
object rather than reusing fields whose meanings belong to the Haberkern
detector.

| Parameter | Initial value | Notes |
| --- | ---: | --- |
| Position smoother | Savitzky–Golay | Applied separately to x and y |
| Position smoothing window | 3 frames | Provisional; configurable |
| Position smoothing polynomial order | 1 | Produces actual smoothing with a 3-frame window |
| Walking-start threshold | 2 mm/s | Rising Schmitt threshold |
| Stopping-start threshold | 1 mm/s | Falling Schmitt threshold |
| Minimum stopped bout | 0.2 s | Converted per trajectory to a minimum frame count |
| Minimum walking bout | 0.05 s | Converted per trajectory to a minimum frame count |
| Stop-extension radius | 0.5 mm | Measured from the relevant original bout endpoint |
| Stop-extension window | 1.0 s per side | One non-recursive pass |
| Angular Butterworth order | 4 | Low-pass |
| Angular Butterworth cutoff | 2.0 Hz | Provisional; specified in physical Hz |
| Angular filtering phase | Zero-phase | SOS `filtfilt` implementation |
| Angular moving average | 1 frame (disabled) | Configurable as 2 or 3 frames for tuning |
| Turn peak threshold | 120 deg/s | Applied to absolute filtered angular velocity |
| Turn flank size | 2 frames | Peak frame excluded from both fits |
| Turn merge radius | 0.5 mm | Distance between smoothed peak positions |
| Minimum overall turn angle | 20 degrees | Discard only when absolute angle is `< 20` |

The source detector's ten-frame moving average represented about 0.167 seconds
at 60 FPS, equivalent to only 1.25 frames at 7.5 FPS. It is therefore disabled
for the first pass rather than applied literally over 1.33 seconds. Similarly,
four source frames per turn flank would scale to half of one current frame; two
frames is the smallest possible line fit. These parameters and the provisional
position smoother must be recorded in exported metadata.

## Processing pipeline

### 1. Validate and prepare the trajectory

Reject trajectories under the same basic conditions as the current detector
(bad trajectory, missing coordinates, unavailable video calibration, or invalid
FPS). Use the already interpolated coordinate arrays. Filtering must preserve
the original array length.

Short arrays that cannot support a requested filter or a complete turn flank
are handled conservatively:

- reduce or bypass positional smoothing while retaining finite coordinates;
- do not emit peaks without the configured number of complete flank frames;
- do not allow filter padding artifacts to create turn peaks at the ends.

### 2. Smooth position and compute centroid speed

Apply the detector-specific positional smoother independently to x and y,
producing `x_smooth` and `y_smooth`.

Compute backward-looking frame-to-frame centroid speed:

```text
distance[i] = hypot(x_smooth[i] - x_smooth[i-1],
                    y_smooth[i] - y_smooth[i-1]) / pixels_per_mm
speed[i] = distance[i] / dt[i]
```

The first frame uses the same fallback convention as the existing detector so
the output length equals the trajectory length.

Do not use `Trajectory.sp` as detector input. `sp` is derived from unsmoothed
positions, is stored in pixels/second, and normally assumes a fixed FPS. It may
be exported alongside the new signal for validation.

### 3. Initial walking/stopped classification with a Schmitt trigger

Traverse speed chronologically:

1. Speed `>= 2 mm/s` sets the state to walking.
2. Speed `<= 1 mm/s` sets the state to stopped.
3. Speed strictly between the thresholds retains the previous state.
4. Before the first threshold crossing, initialize conservatively as stopped.

This produces the preliminary binary walking mask.

### 4. Extend preliminary stopped bouts

Find maximal stopped bouts in the preliminary mask. Build every extension from
this frozen set of preliminary bouts so that one extension does not recursively
seed another.

For each bout `[start, stop]`:

- Backward extension uses the smoothed coordinate at the original `start` as a
  fixed anchor. Examine at most one second of immediately preceding frames,
  moving outward from `start - 1`. Extend through the contiguous frames whose
  smoothed centroid remains within `0.5 mm` of that anchor; stop at the first
  frame outside the radius.
- Forward extension analogously uses the coordinate at the original `stop` as a
  fixed anchor and examines at most one second after `stop`.
- The two extensions are a single, non-recursive pass. Overlapping extended stop
  masks are combined by union.

The fixed anchor is not updated as frames are added. Stop extension occurs before
minimum-duration cleanup.

### 5. Enforce minimum bout durations

An undersized bout must be relabeled into the opposite state. Cleanup is
deterministic and terminating:

1. Recompute maximal binary bouts.
2. In chronological order, first select the earliest stopped bout shorter than
   `0.2 s`. If none exists, select the earliest walking bout shorter than
   `0.05 s`.
3. Relabel the entire selected bout to the opposite state.
4. Repeat until no eligible undersized bout remains.

At an interior bout, both neighboring bouts necessarily have the opposite label,
so relabeling merges all three. At an edge it merges with the sole neighbor.
Each operation strictly reduces the number of state boundaries; therefore the
procedure cannot oscillate and needs no arbitrary iteration cap. A trajectory
containing only one bout has no neighboring state into which it can merge and is
left unchanged, with a diagnostic if it is shorter than its nominal minimum.

For fixed-rate data, a minimum duration `d` corresponds to
`ceil(d * fps)` frames. When valid timestamps are available, the implementation
should use elapsed sample duration and produce the same result as the frame-count
rule on fixed-rate data.

The cleaned mask is the authoritative stopped/walking classification used when
turn candidates are filtered.

### 6. Derive path angular velocity

Use the same smoothed positions from step 2. Compute the backward-looking path
bearing from `(dx, -dy)`, unwrap it through time, and differentiate it by the
per-frame time interval. Convert to degrees/second.

This is path-derived angular velocity, not body-orientation velocity from
`trj.theta`. Frames with effectively zero displacement have undefined path
bearing and must not independently create angular-velocity peaks. The precise
finite-data handling should be covered by tests and included in diagnostics.

### 7. Filter path angular velocity

1. Apply a fourth-order low-pass Butterworth filter at the calibrated cutoff in
   Hz. Use second-order sections and forward/backward filtering to avoid phase
   shift.
2. Apply the configurable centered moving average to the Butterworth output. A
   width of one returns the Butterworth output unchanged; widths of two and three
   are supported for the planned sweep.

For reproducibility with an even-sized window, place the extra sample on the
future side (a two-frame output at `i` averages `i` and `i+1`), using reflected
edge padding. Suppress peak detection in edge regions influenced by incomplete
Butterworth padding or lacking complete turn flanks.

Preserve both the signed filtered signal and its absolute value. The absolute
value is used for peak selection; the sign at the representative peak records
CW/CCW direction.

### 8. Detect elementary turn candidates

A frame `p` is an elementary peak when:

```text
abs_angular_velocity[p] >= 120 deg/s
abs_angular_velocity[p] > abs_angular_velocity[p - 1]
abs_angular_velocity[p] > abs_angular_velocity[p + 1]
```

The strict neighbor comparisons intentionally give flat plateaus no peak. Do not
apply prominence, width, or minimum-peak-separation constraints initially.

With the initial two-frame flank, the elementary turn interval is
`[p - 2, p + 2]`. More generally it is `[p - flank, p + flank]`. Fit
ordinary least-squares
lines to the smoothed positions at:

- incoming frames `[p - flank, p - 1]`;
- outgoing frames `[p + 1, p + flank]`.

Fit each line parametrically against frame/time rather than fitting `y` as a
function of `x`, so vertical paths are valid. Orient both line vectors forward
in time. The signed candidate angle is the wrapped outgoing-minus-incoming angle
in `[-180, 180]` degrees.

### 9. Consolidate spatially adjacent peaks

Sort elementary candidates chronologically. Consecutive candidates belong to
the same merged turn when their smoothed peak coordinates are at most `0.5 mm`
apart. Apply this relation transitively: if A merges with B and B merges with C,
all three form one turn even if A and C are farther than `0.5 mm` apart.

The merged turn:

- spans from one flank before its first peak through one flank after its last
  peak;
- uses the first peak's incoming fit and the last peak's outgoing fit to
  calculate its overall signed heading change;
- uses the constituent peak with the largest absolute filtered angular velocity
  as its representative peak and direction sign; ties choose the earliest peak.

The largest-amplitude representative is a deterministic bookkeeping choice. The
outer flank fits, rather than that representative peak, determine the merged
turn's interval and overall angle.

The supplied rule has no temporal merge limit. Initially, only chronologically
adjacent candidates are compared, but a diagnostic should record peak-to-peak
time gaps. Validation should check for biologically separate turns being merged
after a fly revisits nearly the same coordinate; if this occurs, a temporal
criterion will need to be added and documented.

### 10. Filter merged turns

Process filters after consolidation so the overall maneuver, rather than each
noise-split peak, is evaluated. Discard a merged turn if any of these is true:

- `abs(overall signed angle) < 20 degrees`;
- any frame in its merged inclusive interval is stopped according to the cleaned
  mask from step 5;
- any frame in its merged inclusive interval is in wall contact.

Exactly `20 degrees` is retained. Stop overlap is evaluated before any accepted
turn overwrites walking labels.

Wall contact should use the repository's standard ellipse-edge wall-contact
regions. Selecting this detector must arrange for wall contact to be computed
before behavior-state classification. Missing wall-contact data should be an
explicit error for this detector rather than silently skipping a required
filter. This implies revising the current analysis order, which presently runs
behavior-state detection before boundary-contact analysis.

### 11. Emit frame-wise states

Initialize every cleaned stopped frame as `REST` and every cleaned walking frame
as `RUN`. For every accepted merged turn, overwrite its entire inclusive interval
with `TURN`. Turn intervals cannot overwrite stopped frames because such turns
were rejected in step 10.

Attach detector-specific intermediate arrays and event diagnostics to the
trajectory without changing the meanings of the existing Haberkern diagnostic
attributes. At minimum retain:

- smoothed x/y;
- smoothed centroid speed in mm/s;
- preliminary, extended, and cleaned walking masks;
- raw, Butterworth-filtered, and moving-average path angular velocity;
- elementary peaks;
- merged candidate intervals, representative peaks, signed angles, and rejection
  reasons;
- final turn mask and behavior-state array.

## Selecting the Butterworth cutoff

The cutoff should be chosen empirically in physical frequency units because the
same normalized cutoff means different things at different video frame rates.
The optional moving average also has an FPS-dependent response, so widths one,
two, and three must be included in calibration rather than considered
separately.

### Calibration sample

Choose representative trajectory windows across:

- videos and frame rates;
- genotypes/conditions;
- straight runs, gradual curves, sharp turns, stops, tracking jitter, and wall
  contact;
- both visually clear turns and plausible non-turn controls.

Where possible, create a small blinded set of manually marked turn centers and
directions. It need not be large; it is primarily an anchor against choosing a
cutoff solely by visual smoothness.

### Diagnostics

For each sample, export aligned plots of:

- smoothed x/y trajectory;
- raw path angular velocity;
- Butterworth output;
- final optional moving-average output;
- the `120 deg/s` threshold, detected peaks, consolidated intervals, turn angles,
  and rejection reasons.

Also examine Welch power spectra of straight-run controls and clear turns. The
goal is to identify frequencies dominated by frame-to-frame tracking noise while
retaining the bandwidth of genuine turns.

### Cutoff sweep and selection

At 7.5 FPS, sweep a practical grid below the 3.75-Hz Nyquist frequency, for
example `0.75, 1, 1.25, 1.5, 2, 2.5, 3 Hz`. Measure:

- agreement with manual turn centers and CW/CCW direction;
- false peaks during straight-run controls;
- accepted-turn count and duration;
- peak timing shift;
- signed turn-angle stability;
- sensitivity of results to the neighboring cutoff values.

Prefer the broad stable region in which clear turns remain above threshold and
straight-run peaks are suppressed. Select one physical cutoff shared across
datasets if possible. If frame rates differ enough that this is untenable, any
FPS-dependent rule must be explicit and stored in metadata.

The positional smoothing parameters should be swept at the same time on a much
smaller grid (including no smoothing and the proposed three-frame Savitzky–Golay
filter), because positional smoothing changes both speed classification and the
angular signal feeding the Butterworth filter.

## Tests required before integration

Unit tests should cover:

- inclusive Schmitt thresholds and initial state;
- one-pass, fixed-anchor stop extension in both directions;
- terminating minimum-bout cleanup, including alternating short bouts and edge
  bouts;
- FPS and timestamp duration conversion;
- signed CW/CCW path angular velocity in image coordinates;
- strict local maxima and plateau rejection;
- vertical-path line fits and wrapped signed angles;
- correctly sized elementary intervals for the configured flanks;
- transitive consolidation and deterministic representative-peak selection;
- recomputation of merged angles from outer flanks;
- exact `20 degree` retention;
- stopped-frame and wall-contact rejection;
- final state precedence (`TURN` over `RUN`, never over `REST`);
- short trajectories and filter edges;
- backward compatibility when `haberkern` is selected or the selector is omitted.

An integration test should run both detectors on the same synthetic trajectory
and verify that selecting one does not alter the other's configuration,
diagnostics, or output.

## Remaining empirical decisions

The algorithm is sufficiently specified for scaffolding and tests, with these
values intentionally left for calibration before scientific use:

1. Whether the provisional 2-Hz Butterworth cutoff is retained.
2. Whether the provisional three-frame positional Savitzky–Golay parameters are
   retained.
3. Whether a one-, two-, or three-frame post-filter average performs best.
4. Whether two-, three-, or four-frame turn flanks perform best.
5. Whether spatial consolidation also needs a maximum time gap after real-data
   diagnostics are inspected.
