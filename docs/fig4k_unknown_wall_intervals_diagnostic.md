# Figure 4k unknown wall-state diagnostic

This standalone audit measures the effect of rejecting recorded-reward intervals
containing any unknown wall-contact frame. It does not change production analysis,
contact detection, episode eligibility, figures, archives, or GraphPad inputs.

Run from the analysis environment, using a **new** output directory for each run:

```bash
/home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/diagnose_fig4k_unknown_wall_intervals.py \
  --cohort ctrl --out exports/diagnostics/wall_unknown_new_run/ctrl
/home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/diagnose_fig4k_unknown_wall_intervals.py \
  --cohort pfnd_kir --out exports/diagnostics/wall_unknown_new_run/pfnd_kir
/home/tracking/miniconda3/envs/analysis3.13/bin/python \
  scripts/diagnose_fig4k_unknown_wall_intervals.py \
  --cohort antennae_removed --out exports/diagnostics/wall_unknown_new_run/antennae_removed
```

The driver reads each archived Figure 4k provenance command, parses its options
with the existing analysis parser, and constructs the same video/fly inputs in
the production ordering. It applies the relevant entry-point normalizations for
these commands, enables standard wall detection, and disables wall caching.
Acquisition pickles use the exact production `FlyDetector` placeholder class.
The driver does not execute `analyze.py`'s main, postprocessing, or export stages.

A temporary wrapper measures states immediately before the existing cleanup
method, then calls that method unchanged. Each video analysis is discarded after
its measurements are captured. All incidental relative paths are isolated in
the diagnostic output directory. Input files and wall caches are not written.

## Fixed analysis definitions

- Recorded reward-on events (`calc=False`), not calculated circle entries.
- Training 2, skip one sync bucket and retain four (SB2–5).
- Existing reward sanitization and half-open intervals `[reward_i,reward_{i+1})`;
  the closing reward frame is excluded from missingness evaluation.
- Existing `wall/all/edge` episode regions, including initial-region exclusion
  and complete-episode eligibility. Episode assignment uses its start frame.
- Existing bad-fly, target-sync-bucket, five-interval minimum, and final T2 SB5
  SLI eligibility policies for the manuscript population.

Classifiable states must be finite and binary. An interval is eligible only if
every frame in its existing slice is classifiable. Empty/reversed/out-of-range
intervals, nonfloating fresh traces, and unexpected finite states are errors;
they are not repaired or treated as observed noncontact.

## Baseline gate and outputs

`summary.json` sets `baseline_verified` only for a complete run with no detected
discrepancies. Checks compare all archived/recomputed interval identities, window
bounds, contact counts, target eligibility, scalar inclusion/means, and final
manuscript inclusion/means. The archived scalar/eligibility combination must also
reproduce the actual GraphPad CSV values in archive source order. Float means
use relative tolerance `1e-10` and absolute tolerance `1e-12`.

Any differences go to `baseline_discrepancies.csv`. A `--limit` pilot always fails
the full-cohort verification gate. Strict results are still written for inspection
when the gate fails, but must not be interpreted as an isolated policy effect.

- `intervals.csv`: existing boundaries/counts, classifiable frame counts and
  fractions, strict eligibility, and original target eligibility.
- `flies.csv`: total/eligible/unknown/entirely unknown interval counts; total and
  discarded episodes; both means and their difference; observed-fraction
  quantiles; baseline and strict manuscript eligibility.
- `attempts.csv`: every attempted video/fly and whether analysis was skipped.
- `summary.json`: group summaries, baseline comparison, input hashes, original
  command, effective arguments, interpreter, and compiled wall-extension identity.
- `analysis.log`: complete recomputation output.

Group summaries distinguish all flies with intervals, the fixed baseline
manuscript population, and the population remaining after strict exclusion plus
the existing five-interval minimum. A fly with zero eligible intervals has an
undefined strict mean; one with one to four has a mathematical mean but would
fail the manuscript minimum. Paired changes use flies with defined strict means.

The recorded/calculated event distinction, legacy initial-region omission, and
cache conversion of NaN states to integers remain separate findings. No fixes
to these behaviors are included in this diagnostic.

## Validation

```bash
/home/tracking/miniconda3/envs/analysis3.13/bin/python -m pytest -q \
  test/test_fig4k_unknown_wall_intervals_diagnostic.py \
  test/paper_metrics/test_wall_contacts_per_reward_interval_truth.py \
  test/test_boundary_contact_frame_policy.py
```

Focused tests cover half-open missingness boundaries, all-unknown and empty
interval sets, loss of an observed complete episode, unchanged mean despite
exclusions, invalid traces/bounds, and the distinction between undefined means
and the five-interval threshold. Each real fly's baseline reduction is also
checked against the unchanged production collector.
