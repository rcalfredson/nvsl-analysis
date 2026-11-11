"""Full-window aggregation for the conditional additional-loss experiment."""
from __future__ import annotations

import hashlib
import numpy as np

METRICS = ('tortuosity', 'wall_contacts', 'wall_contact_length')
CONTRASTS = (('pfnd_kir', 'ctrl'), ('antennae_removed', 'ctrl'),
             ('antennae_removed', 'pfnd_kir'))


def stable_rng(seed, *identity):
    """Independent reproducible streams, unaffected by subject processing order."""
    import json
    digest = hashlib.sha256(json.dumps([seed, *identity]).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], 'little'))


def aggregate(episodes, minimum=5):
    values = np.asarray([e['value'] for e in episodes], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Native episode values must be finite')
    return dict(n_episodes=len(values), episode_value_sum=float(values.sum()),
                eligible=bool(len(values) >= minimum),
                value=float(values.mean()) if len(values) >= minimum else np.nan)


def episode_changes(before, after):
    """Exact-boundary membership diagnostics; changed boundaries are not misses."""
    def index(episodes):
        result = {(e['start'], e['stop']): e['value'] for e in episodes}
        if len(result) != len(episodes):
            raise ValueError('Duplicate episode boundaries')
        return result
    a, b = index(before), index(after)
    common = a.keys() & b.keys()
    return dict(episodes_no_longer_present=len(a.keys()-b.keys()),
                episodes_newly_present=len(b.keys()-a.keys()),
                same_boundary_episodes=len(common),
                same_boundary_values_changed=sum(not np.isclose(a[k], b[k], rtol=1e-10, atol=1e-10)
                                                  for k in common))


def video_bootstrap_weights(rows, rng):
    """Resample recordings within cohort; carry all their flies/repeats together."""
    by_cohort, membership = {}, {}
    for row in rows:
        c, v = row['cohort'], row['video_path']
        if v in membership and membership[v] != c:
            raise ValueError('A recording belongs to multiple cohorts; specify joint cluster strata')
        membership[v] = c
        by_cohort.setdefault(c, set()).add(v)
    weights = {}
    for cohort in sorted(by_cohort):
        videos = sorted(by_cohort[cohort])
        names, counts = np.unique(rng.choice(videos, len(videos), replace=True), return_counts=True)
        weights.update(zip(names, counts))
    return weights


def contrast_components(rows, positive, negative, weights=None):
    """Equal-fly contrasts, averaged over repeats; mask repeats are not flies.

    Rows are supported baseline-eligible fly/repeat comparisons. Optional video
    multiplicities apply the same bootstrap draw to every repeat and both flies
    in a video. Baseline/common and after/common use identical individuals.
    """
    def mean(group, field):
        vals = np.asarray([r[field] for r in group], dtype=float)
        ww = np.asarray([1 if weights is None else weights.get(r['video_path'], 0)
                         for r in group], dtype=float)
        return float(np.average(vals, weights=ww)) if ww.sum() else np.nan
    summaries = []
    for rep in sorted({r['repeat'] for r in rows}):
        groups = {c: [r for r in rows if r['repeat'] == rep and r['cohort'] == c]
                  for c in (positive, negative)}
        baseline = {c: [r for r in g if np.isfinite(r['before_value'])] for c, g in groups.items()}
        survivors = {c: [r for r in g if r['after_eligible']] for c, g in groups.items()}
        common = {c: [r for r in g if r['after_eligible']] for c, g in baseline.items()}
        before = mean(baseline[positive], 'before_value')-mean(baseline[negative], 'before_value')
        before_common = mean(common[positive], 'before_value')-mean(common[negative], 'before_value')
        after_common = mean(common[positive], 'after_value')-mean(common[negative], 'after_value')
        after = mean(survivors[positive], 'after_value')-mean(survivors[negative], 'after_value')
        summaries.append((before, before_common, after_common, after))
    values = np.asarray(summaries, dtype=float).reshape(-1, 4)
    # Do not hide an undefined repeat by averaging only its defined counterparts.
    before, before_common, after_common, after = values.mean(axis=0) if len(values) else [np.nan]*4
    return dict(before_contrast=float(before), before_common_contrast=float(before_common),
                after_common_contrast=float(after_common),
                after_contrast=float(after), total_change=float(after-before),
                common_fly_change=float(after_common-before_common),
                eligibility_loss_shift=float(before_common-before),
                eligibility_gain_shift=float(after-after_common),
                composition_change=float(before_common-before+after-after_common))
