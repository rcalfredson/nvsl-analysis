# Agarose-Time P-Value Trace

This workflow separates two causes of changes in the JAK–STAT agarose-time
comparison:

1. whether lost/interpolated frames are included in the contact numerator; and
2. whether the group comparison uses control-family Welch tests with
   Holm–Bonferroni adjustment or Welch's one-way ANOVA followed by Games–Howell.

The production default includes interpolated frames consistently in both the
numerator and denominator. To reproduce the historical numerator behavior,
rerun each upstream `analyze.py --agarose` command with:

```text
--agarose-time-lost-frame-policy legacy
```

To exclude interpolated frames from both the numerator and denominator, use:

```text
--agarose-time-lost-frame-policy corrected
```

Write each policy to a distinct path such as `learning_stats.*.legacy.csv`,
`learning_stats.*.corrected.csv`, or
`learning_stats.*.interpolated_inclusive.csv`; do not overwrite another
policy's files. The command recorded at the top of each output CSV then serves
as provenance for the frame policy.

After all three groups have been generated under the policies being compared,
run the tracer. The legacy and corrected policies are required; the
interpolated-inclusive policy can be included when its dataset is available.
For the agarose HTL chamber, the command has this form:

```bash
python scripts/trace_agarose_time_pvalues.py \
  --legacy-group Control=PATH_TO_LEGACY_CONTROL.csv \
  --legacy-group 'upd3 KO=PATH_TO_LEGACY_UPD3.csv' \
  --legacy-group 'upd2+3 KO=PATH_TO_LEGACY_UPD2_PLUS_3.csv' \
  --corrected-group Control=PATH_TO_CORRECTED_CONTROL.csv \
  --corrected-group 'upd3 KO=PATH_TO_CORRECTED_UPD3.csv' \
  --corrected-group 'upd2+3 KO=PATH_TO_CORRECTED_UPD2_PLUS_3.csv' \
  --interpolated-inclusive-group Control=PATH_TO_INCLUSIVE_CONTROL.csv \
  --interpolated-inclusive-group 'upd3 KO=PATH_TO_INCLUSIVE_UPD3.csv' \
  --interpolated-inclusive-group 'upd2+3 KO=PATH_TO_INCLUSIVE_UPD2_PLUS_3.csv' \
  --control-group Control \
  --comparison-group 'upd3 KO' \
  --posthoc-scope control \
  --section '% time over agarose (contact events begin when body center crosses agarose border)' \
  --out exports/agarose_time_ctrl_vs_upd3_pvalue_trace.csv
```

When interpolated-inclusive inputs are supplied, the summary output contains
six rows: the three frame policies crossed with Holm-Welch and
Welch-ANOVA/Games–Howell. Without those optional inputs, the original four-row
legacy/corrected summary is retained. `p_value_reported` is the value to compare
across cells. For Holm-Welch it is the Holm-adjusted p-value; for Games–Howell
it is the studentized-range p-value. In the Holm-Welch rows, `p_value_raw`
retains the unadjusted Welch p-value. In the Games–Howell rows it is the same
studentized-range p-value as `p_value_reported`; the Welch-ANOVA omnibus
statistics are retained in their own columns.

The command also writes a second CSV whose name ends in
`_multiple_comparisons.csv`. For each numerator policy, that audit reports:

- the unadjusted Welch t-test;
- Bonferroni with the two planned control contrasts;
- Bonferroni with all three possible pairs;
- Holm–Bonferroni with the two planned control contrasts;
- Holm–Bonferroni with all three possible pairs;
- Games–Howell fit to only the two target groups; and
- Games–Howell fit to all three groups.

For Bonferroni, `bonferroni_multiplier` is exactly the number of comparisons
in the chosen family. For Holm, `target_p_rank` and
`effective_p_multiplier` expose why its result need not equal raw p times the
family size. `raw_welch_p_value` is retained on every row as a common reference.
It is not the literal pre-adjustment input to Games–Howell; Games–Howell obtains
its p-value from the studentized range distribution.

`--posthoc-scope control` has two related but different meanings in the summary:
it defines the two-test family for Holm, but only filters which Games–Howell
pairs are reported. Games–Howell itself is still computed from every supplied
group. The detailed audit makes this distinction explicit using
`groups_in_family` and `comparisons_in_family`.

Keep the third group in the inputs even when the target pair is Control versus
upd3 KO. It defines the same two-comparison control family used for the Holm
adjustment and the same three-group family used by Games–Howell.
