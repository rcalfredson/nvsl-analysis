# Paper Metric Bundle Manifests

These manifests record canonical digests for local paper export artifacts. They
are regression guards: if a metric implementation changes intentionally,
regenerate the affected export artifacts first, then refresh only the matching
manifest.

## Check Manifests

Run the full paper-metric suite:

```bash
pytest -q test/paper_metrics
```

Or check one manifest directly:

```bash
python scripts/check_bundle_digest.py check-manifest test/reference/bundles/<manifest>.json
```

Manifest tests skip when the referenced local exports are absent.

## Refresh Workflow

### 1. Regenerate Artifacts

Regenerate the affected `exports/...` artifacts using the corresponding
`notebooks/paper_figure_panels.ipynb` command or checked-in export script.

### 2. Run Focused Tests

Run the focused truth and bundle-invariant tests for the metric before updating
the manifest. This keeps semantic failures separate from expected digest drift.

### 3. Rewrite The Manifest

For an existing manifest, use `refresh-manifest`. It reuses the checked-in
bundle names, paths, artifact types, and regression keys, then recomputes the
digests from the regenerated local artifacts:

```bash
python scripts/check_bundle_digest.py refresh-manifest \
  test/reference/bundles/commag_paper_manifest.json
```

When the intended regression key set changes, override the stored keys with a
named preset:

```bash
python scripts/check_bundle_digest.py refresh-manifest \
  test/reference/bundles/commag_paper_manifest.json \
  --key-preset commag_sli
```

Use `write-manifest` when creating a new manifest, adding or removing artifacts,
or intentionally changing artifact order. Pass every artifact with a stable
`name=path` pair, and use `--key-preset` instead of hand-typing long key lists:

```bash
python scripts/check_bundle_digest.py write-manifest \
  --out test/reference/bundles/example_manifest.json \
  --bundle panel_name_a=exports/example_a.npz \
  --bundle panel_name_b=exports/example_b.npz \
  --key-preset sli
```

A minimal one-artifact example:

```bash
python scripts/check_bundle_digest.py write-manifest \
  --out test/reference/bundles/example_manifest.json \
  --bundle example=exports/example.npz \
  --key-preset sli
```

For CSV artifacts, the same command works when the artifact path ends in
`.csv`.

### 4. Verify And Review

Re-run the manifest test and the paper-metric suite:

```bash
pytest -q test/paper_metrics
```

Review the manifest diff. Expect digest changes only for artifacts whose metric
values, shapes, row counts, or selected regression keys changed.

## Key Presets

| Manifest | Key preset |
| --- | --- |
| `sli_paper_manifest.json` | `sli` |
| `return_prob_excursion_bin_paper_manifest.json` | `return_prob_excursion_bin` |
| `turnback_excursion_bin_paper_manifest.json` | `turnback_excursion_bin` |
| `between_reward_distance_hist_paper_manifest.json` | `between_reward_distance_hist` |
| `between_reward_conditioned_disttrav_paper_manifest.json` | `between_reward_conditioned_disttrav` |
| `between_reward_maxdist_paper_manifest.json` | `between_reward_maxdist_sli` |
| `between_reward_return_leg_dist_paper_manifest.json` | `between_reward_return_leg_dist_sli` |
| `commag_paper_manifest.json` | `commag_sli` |
| `turnback_ratio_paper_manifest.json` | `turnback_ratio` |
| `agarose_avoidance_paper_manifest.json` | `agarose_sli` |
| `first_n_reward_diagnostics_paper_manifest.json` | `first_n_reward_diagnostics` |
