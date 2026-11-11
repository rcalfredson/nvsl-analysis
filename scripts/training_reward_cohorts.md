# Reward-table cohort alignment

The active manifest follows all 79 data rows of `tmp/rewards_per_minute_Yang.csv`
in order, with its four metadata columns copied verbatim. The generated CSVs add
`n`, `training 1`, `training 2`, and `training 3`, yielding the same eight columns.
HTML uses the same layout. Values retain `mean ±CI` and `(n)` when a training's
sample size differs from the cohort default. RPD is experimental minus yoked.

47 existing cohorts retain their IDs and video patterns; 32 new cohorts use IDs
`cohort_54` through `cohort_85`. IDs are stable identifiers, not row numbers.
Each active entry records its one-based template data-row number and the command
line and preceding label in `tmp/full_video_list.txt` used to identify its videos.
The input files in `tmp` are local working inputs, not runtime dependencies.

Six original cohorts do not occur in the new table. Their original definitions
remain under `excluded_cohorts` in the manifest and are not analyzed or rendered:

- `cohort_19`: HTL, flat, aristae removed alone. This is distinct from the new
  combined aristae + front tarsi removal cohorts.
- `cohort_20`, `cohort_36`, `cohort_37`: flat-floor upd control, upd3 KO, upd2+3 KO.
- `cohort_24`, `cohort_53`: HTL, flat, control/PFNd>Kir with antennae glued.
  These are distinct from the new antennae-removed Kir cohorts.

The fixed/random assignment rows (16–17) retain `-f 0-19 --ayc` from the
source commands. `--ayc` allows the analyzer to skip yoked fly indices and use
the recorded experimental/yoked pairing, including randomized assignments.
The section heading mentions `--yc R`, but this is not a current `analyze.py`
option. Other HTL rows use `-f 0-9 --rmCC 5`; Large rows use `-f 0-1 --rCC 15`.
Source plotting flags and `--num-trainings 2` are not copied: this adapter requests
all three whole-training summaries, as before.

Template spelling and genotype text are preserved, including `agaroe` in row 12,
`0273-GAL4//LexAop` in MBKC-1 Kir genotypes, and spaces in some labels/genotypes.
These labels do not control chamber selection or video matching. In particular,
rows 13–15 select the no-agarose, center-only, and edges-only commands, respectively;
the source's center-plus-edges command is not included.

## Row mapping

Data row excludes the CSV header. Source line is the command line in the supplied
video list; retained patterns were also checked against that command.

| Data row | Cohort ID | Group | Chamber | Floor/substrate | Video-list line | Origin |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | cohort_15 | Ctrl for antennae manipulations | HTL | flat | 19 | retained |
| 2 | cohort_17 | Antennae removed | HTL | flat | 23 | retained |
| 3 | cohort_18 | Antennae glued | HTL | flat | 25 | retained |
| 4 | cohort_54 | Aristae + front tarsi removed | HTL | flat | 29 | added |
| 5 | cohort_16 | Ctrl for antennae manipulations | HTL | agarose | 31 | retained |
| 6 | cohort_55 | Antennae removed | HTL | agarose | 33 | added |
| 7 | cohort_56 | Antennae glued | HTL | agarose | 35 | added |
| 8 | cohort_57 | Aristae + front tarsi removed | HTL | agarose | 39 | added |
| 9 | cohort_14 | Ctrl for hind tarsi removed + genitalia glued | HTL | flat | 249 | retained |
| 10 | cohort_48 | Hind tarsi removed + genitalia glued | HTL | flat | 270 | retained |
| 11 | cohort_58 | Ctrl for hind tarsi removed + genitalia glued | HTL | agarose | 252 | added |
| 12 | cohort_59 | Hind tarsi removed + genitalia glued | HTL | agaroe | 272 | added |
| 13 | cohort_60 | No agarose (Fig. 3h) | HTL | No agarose (Fig. 3h) | 278 | added |
| 14 | cohort_61 | Agarose at center (Fig. 3h) | HTL | Agarose at center (Fig. 3h) | 282 | added |
| 15 | cohort_62 | Agarose at edges (Fig. 3h) | HTL | Agarose at edges (Fig. 3h) | 290 | added |
| 16 | cohort_63 | Fixed assignment (Extended Data Fig. 2c) | HTL | flat | 302 | added |
| 17 | cohort_64 | Random assignment (Extended Data Fig. 2c) | HTL | flat | 304 | added |
| 18 | cohort_25 | Blind flies | HTL | flat | 14 | retained |
| 19 | cohort_65 | Blind flies | HTL | agarose | 16 | added |
| 20 | cohort_11 | Ctrl>Kir (for PFNd>Kir and MBKC-2>Kir) | HTL | flat | 62 | retained |
| 21 | cohort_51 | PFNd>Kir | HTL | flat | 73 | retained |
| 22 | cohort_40 | MBKC-2>Kir | HTL | flat | 195 | retained |
| 23 | cohort_66 | Ctrl>Kir antennae removed | HTL | flat | 67 | added |
| 24 | cohort_67 | PFNd>Kir antennae removed | HTL | flat | 77 | added |
| 25 | cohort_68 | Ctrl>Kir (for PFNd>Kir) | HTL | agarose | 65 | added |
| 26 | cohort_69 | PFNd>Kir | HTL | agarose | 75 | added |
| 27 | cohort_70 | Ctrl>Kir antennae removed | HTL | agarose | 69 | added |
| 28 | cohort_71 | PFNd>Kir antennae removed | HTL | agarose | 79 | added |
| 29 | cohort_72 | Ctrl>Kir (for MBKC-1>Kir) | HTL | flat | 182 | added |
| 30 | cohort_73 | MBKC-1>Kir | HTL | flat | 187 | added |
| 31 | cohort_74 | Ctrl>Kir (for MBKC-1>Kir) | HTL | agarose | 199 | added |
| 32 | cohort_75 | MBKC-1>Kir | HTL | agarose | 201 | added |
| 33 | cohort_76 | Ctrl>Kir (for MBKC-2>Kir) | HTL | agarose | 185 | added |
| 34 | cohort_77 | MBKC-2>Kir | HTL | agarose | 189 | added |
| 35 | cohort_03 | Ctrl>Kir (for EPG>Kir and PFR>Kir) | HTL | flat | 145 | retained |
| 36 | cohort_26 | EPG>Kir | HTL | flat | 150 | retained |
| 37 | cohort_31 | PFR>Kir | HTL | flat | 154 | retained |
| 38 | cohort_07 | Ctrl>GtACR1 (for PFNd>GtACR1) | HTL | flat | 85 | retained |
| 39 | cohort_45 | PFNd>GtACR1 | HTL | flat | 92 | retained |
| 40 | cohort_10 | Ctrl>GtACR1 (for MBKC-1>GtACR1 and MBKC-2GtACR1) | HTL | flat | 206 | retained |
| 41 | cohort_41 | MBKC-1>GtACR1 | HTL | flat | 210 | retained |
| 42 | cohort_42 | MBKC-2>GtACR1 | HTL | flat | 214 | retained |
| 43 | cohort_78 | Ctrl>GtACR1 (for MBKC-1>GtACR1 and MBKC-2GtACR1) | HTL | agarose | 208 | added |
| 44 | cohort_79 | MBKC-1>GtACR1 | HTL | agarose | 212 | added |
| 45 | cohort_80 | MBKC-2>GtACR1 | HTL | agarose | 216 | added |
| 46 | cohort_04 | Ctrl>GtACR1 (for EPG>GtACR1) | HTL | flat | 134 | retained |
| 47 | cohort_29 | EPG>GtACR1 | HTL | flat | 139 | retained |
| 48 | cohort_05 | Ctrl>GtACR1 (for PFR>GtACR1) | HTL | flat | 160 | retained |
| 49 | cohort_32 | PFR>GtACR1 | HTL | flat | 165 | retained |
| 50 | cohort_09 | Ctrl>CsC(for PFNd>CsC and EPG>CsC) | HTL | flat | 100 | retained |
| 51 | cohort_46 | PFNd>CsC | HTL | flat | 107 | retained |
| 52 | cohort_30 | EPG>CsC | HTL | flat | 113 | retained |
| 53 | cohort_08 | Ctrl>CsC(for PFNv>CsC) | HTL | flat | 119 | retained |
| 54 | cohort_47 | PFNv>CsC | HTL | flat | 125 | retained |
| 55 | cohort_06 | Ctrl>CsC(for PFR>CsC) | HTL | flat | 171 | retained |
| 56 | cohort_33 | PFR>CsC | HTL | flat | 176 | retained |
| 57 | cohort_12 | Orco2/+ | HTL | flat | 221 | retained |
| 58 | cohort_34 | Orco1/Orco2 | HTL | flat | 226 | retained |
| 59 | cohort_13 | Orco2/+ | HTL | agarose | 223 | retained |
| 60 | cohort_35 | Orco1/Orco2 | HTL | agarose | 228 | retained |
| 61 | cohort_21 | Ctrl for upd2 + upd3 KO | HTL | agarose | 348 | retained |
| 62 | cohort_38 | upd3 KO | HTL | agarose | 353 | retained |
| 63 | cohort_39 | upd2 + upd3 KO | HTL | agarose | 358 | retained |
| 64 | cohort_81 | Ctrl for Or7a mutant | HTL | flat | 319 | added |
| 65 | cohort_82 | Or7a mutant | HTL | flat | 324 | added |
| 66 | cohort_83 | 0273-GAL4 & P1 overlapping neurons(Extended Data Fig. 1c) | HTL | flat | 428 | added |
| 67 | cohort_84 | 0273-GAL4 & P1 overlapping neurons (Extended Data Fig. 1c) | HTL | agarose | 425 | added |
| 68 | cohort_01 | Ctrl for large chamber | Large | flat | 368 | retained |
| 69 | cohort_49 | PFNd>Kir | Large | flat | 384 | retained |
| 70 | cohort_22 | Antennae removed | Large | flat | 370 | retained |
| 71 | cohort_52 | PFNd>Kir, antennae removed | Large | flat | 386 | retained |
| 72 | cohort_02 | Ctrl for large chamber | Large | agarose | 372 | retained |
| 73 | cohort_50 | PFNd>Kir | Large | agarose | 388 | retained |
| 74 | cohort_23 | Antennae removed | Large | agarose | 375 | retained |
| 75 | cohort_85 | PFNd>Kir, antennae removed | Large | agarose | 390 | added |
| 76 | cohort_43 | MBKC-1>Kir | Large | flat | 379 | retained |
| 77 | cohort_27 | EPG>Kir | Large | flat | 393 | retained |
| 78 | cohort_44 | MBKC-1>Kir | Large | agarose | 381 | retained |
| 79 | cohort_28 | EPG>Kir | Large | agarose | 395 | retained |
