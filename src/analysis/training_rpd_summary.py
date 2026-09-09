"""Training-wide RPD readouts using the same interval as rewards per minute."""

import numpy as np

from src.utils import util


def training_rewards_per_distance(va, trn):
    """Return experimental RPD and paired exp-minus-yoked RPD, in m⁻¹.

    Match rewardsPerMinute: start at _syncBucket(skip=0), count calculated
    target entries through training stop, and include the final partial bucket.
    Neither PI masks nor sync-bucket pooled-plot count/window options apply.
    Each fly contributes total rewards / total distance, including zero rewards.
    """
    if getattr(va, "_skipped", False):
        return np.nan, np.nan
    start = va._syncBucket(trn, skip=0)[0]
    stop = trn.stop
    if start is None or not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        return np.nan, np.nan
    start, stop = int(start), int(stop)
    px_per_mm = float(va.xf.fctr) * float(va.ct.pxPerMmFloor())
    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
        return np.nan, np.nan

    values = [np.nan, np.nan]
    for f, traj in enumerate(va.trx[:2]):
        if traj.bad():
            continue
        distance_m = traj.distTrav(start, stop) / px_per_mm / 1000.0
        rewards = va._countOn(start, stop, calc=True, ctrl=False, f=f)
        if np.isfinite(distance_m) and distance_m > 0 and np.isfinite(rewards):
            values[f] = rewards / distance_m
    exp, yoked = values
    return exp, exp - yoked


def report_training_rewards_per_distance(vas, trns, gis, gls=None):
    """Print fly-level means, Student-t 95% CI half-widths, and sample sizes.

    Differences are formed within valid pairs before calculating uncertainty.
    Group membership and training selection are supplied by postAnalyze.
    """
    gis = np.asarray(gis)
    groups = sorted(set(gis.tolist()))
    values = np.full((len(vas), len(trns), 2), np.nan)
    for v_idx, va in enumerate(vas):
        for t_idx, trn in enumerate(trns):
            if trn.n <= len(va.trns):
                values[v_idx, t_idx] = training_rewards_per_distance(
                    va, va.trns[trn.n - 1]
                )

    for mode, label in enumerate(("exp fly", "exp-minus-yoked")):
        print(f"\nrewards per distance traveled [m⁻¹], {label}:")
        print("interval: RPM window (first synchronized reward through training end)")
        print("means with 95% confidence intervals:")
        ns = [np.count_nonzero(gis == g) for g in groups]
        print('  n = %s  (in "()" below if different)' % util.join(", ", ns))
        for t_idx, trn in enumerate(trns):
            for g, nominal_n in zip(groups, ns):
                mean, delta, n = util.meanConfInt(
                    values[gis == g, t_idx, mode], asDelta=True
                )
                group_label = (
                    label if len(groups) == 1 else (gls[g] if gls else f"group {g + 1}")
                )
                suffix = f" ({n})" if n != nominal_n else ""
                print(f"  t{trn.n}, {group_label}: {mean:.2f} ±{delta:.2f}{suffix}")
    return values
