"""Whole-episode wall exclusion and tracking-loss exposure, with explicit bounds."""
from __future__ import annotations
import numpy as np

METRICS = ('tortuosity', 'turnback_ratio', 'home_vector_alignment', 'agarose_avoidance')
POLICIES = ('wall_allowed', 'exclude_episode_contact', 'exclude_support_contact')


def frame_counts(mask, starts, stops):
    mask = np.asarray(mask, dtype=bool)
    starts, stops = np.asarray(starts, dtype=int), np.asarray(stops, dtype=int)
    if np.any(starts < 0) or np.any(stops > len(mask)) or np.any(stops <= starts):
        raise ValueError('Invalid half-open intervals')
    prefix = np.r_[0, np.cumsum(mask, dtype=np.int64)]
    return prefix[stops]-prefix[starts]


def classify(episodes, contact, observed, near_wall, near_wall_gap, window):
    """Return frame counts; tortuosity's primary wall rule clips to its window.

    Primary bounds match production tortuosity/turnback/alignment exclusion.
    Agarose uses the shared half-open episode rule as an explicit new audit
    policy. The stricter support rule includes boundary and sampled headings.
    Proximity uses observed original positions; it never certifies gap interiors.
    """
    a = episodes.start_frame.to_numpy(dtype=int)
    b = episodes.stop_frame_exclusive.to_numpy(dtype=int)
    sa = episodes.support_start_frame.to_numpy(dtype=int)
    sb = episodes.support_stop_frame_exclusive.to_numpy(dtype=int)
    wa, wb = a.copy(), b.copy()
    tort = episodes.metric.eq('tortuosity').to_numpy()
    if tort.any():
        if window is None:
            raise ValueError('Tortuosity needs its production window')
        wa[tort] = np.maximum(wa[tort], window[0])
        wb[tort] = np.minimum(wb[tort], window[1])
    return dict(wall_filter_start=wa, wall_filter_stop_exclusive=wb,
        wall_contact_frames=frame_counts(contact, wa, wb),
        support_wall_contact_frames=frame_counts(contact, sa, sb),
        support_observed_frames=frame_counts(observed, sa, sb),
        support_observed_near_wall_frames=frame_counts(near_wall, sa, sb),
        support_lost_frames_with_near_wall_gap_endpoint=frame_counts(near_wall_gap, sa, sb))


def reduce_cell(episodes, policy, minimum=5):
    if policy not in POLICIES:
        raise ValueError(policy)
    selected = episodes
    if policy != 'wall_allowed':
        field = 'wall_contact_frames' if policy == 'exclude_episode_contact' else 'support_wall_contact_frames'
        selected = episodes[episodes[field].eq(0)]
    n = len(selected)
    total = float(selected.value.sum())
    return dict(n_episodes=n, episode_value_sum=total,
        measured_value=total/n if n >= minimum else np.nan,
        exposed_episodes=int(selected.support_lost_frames.gt(0).sum()),
        episode_exposed_episodes=int(selected.episode_lost_frames.gt(0).sum()),
        exposed_episode_fraction=float(selected.support_lost_frames.gt(0).mean()) if n else np.nan,
        episodes_with_observed_wall_proximity=int(selected.support_observed_near_wall_frames.gt(0).sum()),
        exposed_episodes_with_near_wall_gap_endpoint=int(selected.support_lost_frames_with_near_wall_gap_endpoint.gt(0).sum()))
