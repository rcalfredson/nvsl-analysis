import numpy as np


def turnback_counts_exp_by_bucket(
    va,
    *,
    inner_delta_mm: float,
    outer_delta_mm: float,
    border_width_mm: float,
    radius_offset_px: float = 0.0,
):
    """
    Return (turn_counts, total_counts) for exp fly only.
    Each is a list over trainings; each element is a 1D np.ndarray of length nbuckets for that training.
    """
    sync_ranges = getattr(va, "sync_bucket_ranges", None)
    if not sync_ranges:
        return [], []

    turn_by_t = []
    total_by_t = []

    # exp is index 0 by your convention
    if not getattr(va, "trx", None) or len(va.trx) < 1:
        return [], []
    trj = va.trx[0]
    if getattr(trj, "_bad", False):
        return [], []

    for t_idx, bucket_ranges in enumerate(sync_ranges):
        nb = len(bucket_ranges) if bucket_ranges else 0
        turn = np.zeros(nb, dtype=int)
        total = np.zeros(nb, dtype=int)

        if nb == 0 or t_idx >= len(getattr(va, "trns", [])):
            turn_by_t.append(turn)
            total_by_t.append(total)
            continue

        trn = va.trns[t_idx]
        if trn is None or not trn.isCircle():
            turn_by_t.append(turn)
            total_by_t.append(total)
            continue

        episodes = trj.reward_turnback_dual_circle_episodes_for_training(
            trn=trn,
            inner_delta_mm=inner_delta_mm,
            outer_delta_mm=outer_delta_mm,
            border_width_mm=border_width_mm,
            debug=False,
            radius_offset_px=radius_offset_px,
        )
        for ep in episodes:
            event_t = int(ep["stop"]) - 1
            turns_back = bool(ep.get("turns_back", False))
            for b_idx, (sb_start, sb_stop) in enumerate(bucket_ranges):
                if sb_start <= event_t < sb_stop:
                    total[b_idx] += 1
                    if turns_back:
                        turn[b_idx] += 1
                    break

        turn_by_t.append(turn)
        total_by_t.append(total)

    return turn_by_t, total_by_t


def report_turnback_sensitivity(vas, opts):
    if not getattr(opts, "turnback_dual_circle_sensitivity", False):
        return

    top_k = int(getattr(opts, "turnback_dual_circle_sensitivity_top_k", 15) or 15)
    # Recommended: require BOTH baseline and jittered totals >= min_total for "stable"
    stable_require_both = bool(
        getattr(opts, "turnback_dual_circle_sensitivity_stable_require_both", False)
    )

    # parse jitters
    jitter_str = getattr(opts, "turnback_dual_circle_sensitivity_jitter_px", "0.5,1.0")
    jitters = []
    for tok in jitter_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            jitters.append(float(tok))
        except ValueError:
            pass
    jitters = tuple(sorted({abs(j) for j in jitters if abs(j) > 1e-12}))
    if not jitters:
        print("[turnback sensitivity] No valid jitters provided.")
        return

    min_total = int(
        getattr(opts, "turnback_dual_circle_sensitivity_min_total", 10) or 0
    )

    # Pull the same params used in the metric
    inner_delta_mm = float(getattr(opts, "turnback_inner_delta_mm", 0.0) or 0.0)
    outer_delta_mm = float(getattr(opts, "turnback_outer_delta_mm", 2.0) or 2.0)
    border_width_mm = float(getattr(opts, "turnback_border_width_mm", 0.1) or 0.1)

    # Accumulate per-video maxima per training
    per_trn_max = {
        d: [] for d in jitters
    }  # dict: jitter -> list of arrays per video, each array length n_trn (ragged tolerated)
    per_trn_max_stable = {d: [] for d in jitters}

    # Capture top-K outlier cases: dict[jitter][t_idx] -> list[case-dict]
    outliers_all = {d: {} for d in jitters}
    outliers_stable = {d: {} for d in jitters}

    # determine a reference n_trn for reporting (max seen)
    n_trn_ref = 0
    for va in vas:
        n_trn_ref = max(n_trn_ref, len(getattr(va, "sync_bucket_ranges", []) or []))

    for va in vas:
        va_name = repr(va)
        # baseline
        turn0, total0 = turnback_counts_exp_by_bucket(
            va,
            inner_delta_mm=inner_delta_mm,
            outer_delta_mm=outer_delta_mm,
            border_width_mm=border_width_mm,
            radius_offset_px=0.0,
        )
        if not turn0 and not total0:
            continue

        # baseline ratio per training/bucket
        ratio0 = []
        for t, (tu, to) in enumerate(zip(turn0, total0)):
            r = np.full_like(tu, np.nan, dtype=float)
            np.divide(tu, to, out=r, where=(to > 0))
            ratio0.append(r)

        sync_ranges = getattr(va, "sync_bucket_ranges", []) or []

        for d in jitters:
            # +d and -d
            max_abs = np.zeros(len(ratio0), dtype=float)
            max_abs_stable = np.zeros(len(ratio0), dtype=float)

            for sign in (+1.0, -1.0):
                turnX, totalX = turnback_counts_exp_by_bucket(
                    va,
                    inner_delta_mm=inner_delta_mm,
                    outer_delta_mm=outer_delta_mm,
                    border_width_mm=border_width_mm,
                    radius_offset_px=sign * d,
                )

                for t_idx in range(min(len(ratio0), len(turnX))):
                    tu = turnX[t_idx]
                    to = totalX[t_idx]
                    rX = np.full_like(tu, np.nan, dtype=float)
                    np.divide(tu, to, out=rX, where=(to > 0))

                    dr = rX - ratio0[t_idx]
                    if dr.size == 0:
                        continue

                    # ignore buckets with tiny denominators if requested
                    if min_total > 0 and t_idx < len(total0):
                        mask = total0[t_idx] >= min_total
                        if (
                            stable_require_both
                            and t_idx < len(totalX)
                            and totalX[t_idx].shape == mask.shape
                        ):
                            mask &= (totalX[t_idx] >= min_total)
                        if np.any(mask):
                            absdr = np.abs(dr[mask])
                            if np.any(np.isfinite(absdr)):
                                m = float(np.nanmax(absdr))
                                max_abs_stable[t_idx] = max(max_abs_stable[t_idx], m)
                                # record argmax bucket for stable
                                # map masked index back to original bucket index
                                idxs = np.nonzero(mask)[0]
                                j_local = int(np.nanargmax(absdr))
                                b_idx = int(idxs[j_local])
                                # store case
                                if t_idx < len(sync_ranges) and b_idx < len(
                                    sync_ranges[t_idx]
                                ):
                                    sb_start, sb_stop = sync_ranges[t_idx][b_idx]
                                else:
                                    sb_start = sb_stop = None
                                case = dict(
                                    va=va_name,
                                    training=int(t_idx + 1),
                                    bucket=int(b_idx + 1),
                                    sign=("+%g" % d) if sign > 0 else ("-%g" % d),
                                    abs_delta=float(abs(dr[b_idx])),
                                    delta=float(dr[b_idx]),
                                    base_turn=int(turn0[t_idx][b_idx]),
                                    base_total=int(total0[t_idx][b_idx]),
                                    base_ratio=(
                                        float(ratio0[t_idx][b_idx])
                                        if np.isfinite(ratio0[t_idx][b_idx])
                                        else np.nan
                                    ),
                                    jit_turn=int(turnX[t_idx][b_idx]),
                                    jit_total=int(totalX[t_idx][b_idx]),
                                    jit_ratio=(
                                        float(rX[b_idx])
                                        if np.isfinite(rX[b_idx])
                                        else np.nan
                                    ),
                                    sb_start=(
                                        int(sb_start) if sb_start is not None else None
                                    ),
                                    sb_stop=(
                                        int(sb_stop) if sb_stop is not None else None
                                    ),
                                )
                                outliers_stable[d].setdefault(t_idx, []).append(case)
                    m_all = np.nanmax(np.abs(dr)) if dr.size else np.nan
                    if np.isfinite(m_all):
                        max_abs[t_idx] = max(max_abs[t_idx], float(m_all))
                        # record argmax bucket for “all”
                        absdr_all = np.abs(dr)
                        if np.any(np.isfinite(absdr_all)):
                            b_idx = int(np.nanargmax(absdr_all))
                            if t_idx < len(sync_ranges) and b_idx < len(
                                sync_ranges[t_idx]
                            ):
                                sb_start, sb_stop = sync_ranges[t_idx][b_idx]
                            else:
                                sb_start = sb_stop = None
                            case = dict(
                                va=va_name,
                                training=int(t_idx + 1),
                                bucket=int(b_idx + 1),
                                sign=("+%g" % d) if sign > 0 else ("-%g" % d),
                                abs_delta=float(absdr_all[b_idx]),
                                delta=float(dr[b_idx]),
                                base_turn=int(turn0[t_idx][b_idx]),
                                base_total=int(total0[t_idx][b_idx]),
                                base_ratio=(
                                    float(ratio0[t_idx][b_idx])
                                    if np.isfinite(ratio0[t_idx][b_idx])
                                    else np.nan
                                ),
                                jit_turn=int(turnX[t_idx][b_idx]),
                                jit_total=int(totalX[t_idx][b_idx]),
                                jit_ratio=(
                                    float(rX[b_idx])
                                    if np.isfinite(rX[b_idx])
                                    else np.nan
                                ),
                                sb_start=(
                                    int(sb_start) if sb_start is not None else None
                                ),
                                sb_stop=int(sb_stop) if sb_stop is not None else None,
                            )
                            outliers_all[d].setdefault(t_idx, []).append(case)

            per_trn_max[d].append(max_abs)
            per_trn_max_stable[d].append(max_abs_stable)

    # Print summary
    print("\n[turnback dual-circle] sensitivity to inner-radius offset (exp only)")
    print(f"  jitters_px={jitters}  min_total_per_bucket={min_total}")
    print(f"  stable_require_both_totals={stable_require_both}  top_k={top_k}")

    def _summ(arrs, t_idx):
        vals = [a[t_idx] for a in arrs if t_idx < len(a) and np.isfinite(a[t_idx])]
        if not vals:
            return None
        v = np.asarray(vals, float)
        return dict(
            n=int(v.size),
            med=float(np.median(v)),
            p90=float(np.quantile(v, 0.90)),
            mx=float(np.max(v)),
        )

    for t_idx in range(n_trn_ref):
        print(f"\n  Training {t_idx+1}:")
        for d in jitters:
            s_all = _summ(per_trn_max[d], t_idx)
            s_stb = _summ(per_trn_max_stable[d], t_idx) if min_total > 0 else None
            if not s_all:
                print(f"    ±{d:g}px: (no data)")
                continue
            line = f"    ±{d:g}px: n={s_all['n']} med={s_all['med']:.4f} p90={s_all['p90']:.4f} max={s_all['mx']:.4f}"
            if s_stb:
                line += f" | stable(min_total): med={s_stb['med']:.4f} p90={s_stb['p90']:.4f} max={s_stb['mx']:.4f}"
            print(line)

    # Print top-K offenders
    def _print_top(label, cases_by_t):
        print(f"\n  Top outliers ({label}):")
        for t_idx in range(n_trn_ref):
            print(f"    Training {t_idx+1}:")
            for d in jitters:
                cases = cases_by_t[d].get(t_idx, [])
                if not cases:
                    print(f"      ±{d:g}px: (none)")
                    continue
                cases_sorted = sorted(
                    cases, key=lambda c: c["abs_delta"], reverse=True
                )[:top_k]
                print(f"      ±{d:g}px:")
                for i, c in enumerate(cases_sorted, 1):
                    # concise single-line record
                    print(
                        f"        #{i:02d} |Δ|={c['abs_delta']:.4f} (Δ={c['delta']:+.4f}) "
                        f"{c['va']} b{c['bucket']} {c['sign']}px "
                        f"base={c['base_turn']}/{c['base_total']}({c['base_ratio']:.3f}) "
                        f"jit={c['jit_turn']}/{c['jit_total']}({c['jit_ratio']:.3f}) "
                        f"frames=[{c['sb_start']},{c['sb_stop']})"
                    )

    _print_top("all buckets", outliers_all)
    if min_total > 0:
        _print_top("stable buckets", outliers_stable)
