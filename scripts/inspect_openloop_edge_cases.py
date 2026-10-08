#!/usr/bin/env python3
"""Match LED-on positional PI to walking and inspect low-activity edge cases."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_openloop_walking import period_masks, walking_trajectory
from src.analysis.agarose_preinvestigation import load_pickle
from src.utils.common import CT, Xformer
from src.utils.util import prefIdx


def led_side_preference(trj, masks, midline):
    """Match posPrefOL for on/off protocols (top side is the LED side).

    Like posPrefOL, unresolved NaNs anywhere in the session invalidate both
    session preferences. Interpolated gaps remain in the denominator.
    """
    if trj is None or np.any(np.isnan(trj.y[masks['session']])):
        return {'led_on': float('nan'), 'led_off': float('nan')}
    result = {}
    for period in ('led_on', 'led_off'):
        mask = masks[period]
        top = int(np.count_nonzero((trj.y < midline) & mask))
        result[period] = float(prefIdx(top, int(mask.sum()) - top))
    return result


def classify(pi, walking_pct, threshold):
    if not np.isfinite(pi) or not np.isfinite(walking_pct):
        return 'missing_measurement'
    if pi < 0:
        return 'negative_PI_low_walking' if walking_pct < threshold else 'negative_PI_other_walking'
    return 'nonnegative_PI_low_walking' if walking_pct < threshold else 'nonnegative_PI_other_walking'


def apply_reference_exports(table, *, open_loop_export=None, closed_loop_export=None):
    """Validate 1q PI and label preceding roles using optional supplied CSVs.

    None skips that operation; a supplied path must exist. Returns a copy so
    callers can reuse their reconstructed measurements without role changes.
    """
    table = table.copy()
    if open_loop_export is not None:
        existing = pd.read_csv(open_loop_export)
        comparison = table.loc[table.panel.eq('1q')].merge(existing, on='subject_key',
                         validate='one_to_one', suffixes=('', '_existing'))
        if len(comparison) != len(existing):
            raise ValueError('Existing 1q export contains unmatched flies')
        for p in ('led_on', 'led_off'):
            if not np.allclose(comparison[f'preference_pi_{p}'], comparison[f'preference_pi_{p}_existing'],
                               equal_nan=True, atol=1e-12, rtol=0):
                raise ValueError(f'Reconstructed 1q {p} PI differs from manuscript export')
        print(f'Validated LED-on/off PI against all {len(existing)} existing 1q export rows: {open_loop_export}')
    else:
        print('1q reference PI validation skipped: no --open-loop-export supplied.')
    table['closed_loop_role'] = 'standalone'
    table.loc[table.panel.eq('1q'), 'closed_loop_role'] = 'not_checked'
    if closed_loop_export is not None:
        roles = pd.read_csv(closed_loop_export)
        table.loc[table.panel.eq('1q'), 'closed_loop_role'] = 'not_in_closed_export'
        for field, label in [('exp_subject_key', 'experimental'), ('yoked_subject_key', 'yoked')]:
            table.loc[table.panel.eq('1q') & table.subject_key.isin(roles[field]), 'closed_loop_role'] = label
        print(f'1q closed-loop roles read from: {closed_loop_export}')
    else:
        print('1q closed-loop role lookup skipped: no --closed-loop-export supplied.')
    return table


def run(walking_csv, out_dir, threshold=10, period='session', *,
        open_loop_export=None, closed_loop_export=None):
    for reference in (open_loop_export, closed_loop_export):
        if reference is not None and not Path(reference).is_file():
            raise FileNotFoundError(f'Reference export does not exist: {reference}')
    table = pd.read_csv(walking_csv)
    if table.subject_key.duplicated().any():
        raise ValueError('Duplicate walking subject keys')
    preferences = []
    for video, selected in table.groupby('open_loop_video', sort=False):
        path = Path(video)
        proto = load_pickle(path.with_suffix('.data'))['protocol']
        if proto.get('pt') != 'openLoop' or proto.get('alt', True):
            raise ValueError(f'Expected on/off open-loop protocol: {video}')
        raw = load_pickle(path.with_suffix('.trx'))
        cap = cv2.VideoCapture(video)
        try:
            ok, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        if not ok or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Cannot read video metadata: {video}')
        xf = Xformer(proto.get('tm'), CT.htl, frame, proto.get('fy', False))
        for row in selected.itertuples():
            fly = int(row.fly_id)
            if not np.isclose(fps, row.fps):
                raise ValueError(f'Walking CSV frame rate differs for {video}')
            masks = period_masks(proto['frameNums'][fly], len(raw['x'][fly]))
            trj = walking_trajectory(raw['x'][fly], raw['y'][fly], fps, CT.htl)
            ytop, ybottom = (proto['lines'][k][fly] for k in ('yTop', 'yBottom'))
            midline = xf.t2fY(CT.htl.center()[1], f=fly) if ytop is None else ytop
            if ytop != ybottom:
                raise ValueError('Expected single positional preference midline')
            pi = led_side_preference(trj, masks, midline)
            preference_status = ('no_trajectory' if trj is None else
                                 'unresolved_session_tracking' if np.any(np.isnan(trj.y[masks['session']])) else 'ok')
            for p in masks:
                value = 100 * np.count_nonzero(trj.walking & masks[p]) / masks[p].sum() if trj is not None else np.nan
                if not np.isclose(value, getattr(row, f'{p}_walking_pct'), equal_nan=True):
                    raise ValueError(f'Walking CSV differs from current tracking: {row.subject_key}')
            preferences.append(dict(subject_key=row.subject_key, preference_pi_led_on=pi['led_on'],
                                    preference_pi_led_off=pi['led_off'], led_side_midline_y=midline,
                                    preference_status=preference_status))
    table = table.merge(pd.DataFrame(preferences), on='subject_key', validate='one_to_one')
    table = apply_reference_exports(table, open_loop_export=open_loop_export,
                                    closed_loop_export=closed_loop_export)
    walking_field = f'{period}_walking_pct'
    table['walking_period_for_classification'] = period
    table['low_walking_threshold_pct'] = threshold
    table['edge_case_category'] = [classify(pi, w, threshold) for pi, w in
                                 zip(table.preference_pi_led_on, table[walking_field])]
    table['negative_led_on_pi'] = table.preference_pi_led_on.lt(0)
    table['low_walking'] = table[walking_field].lt(threshold)
    table = table.sort_values(['panel', walking_field], na_position='last')
    edges = table.loc[table.negative_led_on_pi | table.low_walking]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / 'walking_preference_per_fly.csv', index=False)
    edges.to_csv(out_dir / 'walking_preference_edge_cases.csv', index=False)
    table.groupby(['panel', 'edge_case_category']).size().rename('n_flies').reset_index().to_csv(
        out_dir / 'walking_preference_counts.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, panel in zip(axes, ('1p', '1q')):
        selected = table.loc[table.panel.eq(panel)]
        ax.scatter(selected[walking_field], selected.preference_pi_led_on, s=20, color='0.65')
        highlighted = edges.loc[edges.panel.eq(panel)]
        for row in highlighted.itertuples():
            x, y = getattr(row, walking_field), row.preference_pi_led_on
            if np.isfinite(x) and np.isfinite(y):
                ax.scatter(x, y, color='#d55e00', s=30)
                ax.annotate(f'{row.camera_id}/f{row.fly_id}', (x, y), xytext=(5, 5),
                            textcoords='offset points', fontsize=8)
        ax.axhline(0, color='0.4', lw=1)
        ax.axvline(threshold, color='0.4', lw=1, ls='--')
        ax.set(title=f'Figure {panel}', xlabel=f'{period.replace("_", " ").capitalize()} walking (%)',
               xlim=(-2, 100), ylim=(-1.05, 1.05))
        ax.grid(alpha=.15)
    axes[0].set_ylabel('LED-on positional light PI')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out_dir / f'walking_vs_light_pi.{suffix}', dpi=180)
    plt.close(fig)
    columns = ['panel', 'camera_id', 'fly_id', 'closed_loop_role', 'session_walking_pct',
               'led_on_walking_pct', 'preference_pi_led_on', 'session_raw_missing_pct', 'edge_case_category']
    print(edges[columns].to_string(index=False))
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--walking-csv', type=Path, default=ROOT / 'exports/openloop_walking/walking_per_fly.csv')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'exports/openloop_walking/edge_cases')
    parser.add_argument('--walking-threshold-pct', type=float, default=10)
    parser.add_argument('--walking-period', choices=['session', 'led_on', 'led_off'], default='session')
    parser.add_argument('--open-loop-export', type=Path,
                        help='Optional 1q reference PI CSV for validation; omit to skip. Supplied paths must exist.')
    parser.add_argument('--closed-loop-export', type=Path,
                        help='Optional preceding closed-loop CSV for 1q role lookup; omit to skip. Supplied paths must exist.')
    args = parser.parse_args()
    if not np.isfinite(args.walking_threshold_pct) or not 0 <= args.walking_threshold_pct <= 100:
        parser.error('--walking-threshold-pct must be between 0 and 100')
    run(args.walking_csv, args.out_dir, args.walking_threshold_pct, args.walking_period,
        open_loop_export=args.open_loop_export, closed_loop_export=args.closed_loop_export)
