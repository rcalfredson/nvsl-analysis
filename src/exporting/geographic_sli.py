"""Manifest validation and cohort selection for geographic final-SLI exports."""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path


def load_geographic_sli_cohorts(path):
    """Read ordered line/chamber cohorts without accessing source recordings."""
    cohorts = json.loads(Path(path).read_text())
    if not isinstance(cohorts, list) or not cohorts:
        raise ValueError('Provide a nonempty list of geographic SLI cohorts')
    ids, labels, pairs = set(), set(), set()
    for cohort in cohorts:
        if not isinstance(cohort, dict):
            raise ValueError('Each cohort must be an object')
        cohort_id = cohort.get('id', '')
        label = cohort.get('label', '')
        line, chamber = cohort.get('line'), cohort.get('chamber')
        patterns = cohort.get('video_globs')
        if not isinstance(cohort_id, str) or not re.fullmatch(r'[a-z0-9_]+', cohort_id):
            raise ValueError(f'Invalid cohort id: {cohort_id!r}')
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f'Missing label for {cohort_id}')
        valid_line = (
            (type(line) is int and line >= 0)
            or (isinstance(line, str) and re.fullmatch(r'[a-z][a-z0-9_]*', line))
        )
        if not valid_line or chamber not in ('flat', 'agarose'):
            raise ValueError(f'Invalid line/chamber for {cohort_id}')
        if (not isinstance(patterns, list) or not patterns
                or any(not isinstance(p, str) or not p.strip() or ',' in p or '|' in p
                       for p in patterns)):
            raise ValueError(f'Invalid video_globs for {cohort_id}')
        if len(set(patterns)) != len(patterns):
            raise ValueError(f'Duplicate video patterns for {cohort_id}')
        if cohort_id in ids or label in labels or (line, chamber) in pairs:
            raise ValueError(f'Duplicate cohort id, label, or line/chamber: {cohort_id}')
        ids.add(cohort_id)
        labels.add(label)
        pairs.add((line, chamber))
    return cohorts


def geographic_sli_video_inventory(cohorts):
    """Resolve AVI inputs; reject unmatched patterns and overlapping cohorts."""
    inventory, seen = {}, {}
    for cohort in cohorts:
        videos = []
        for pattern in cohort['video_globs']:
            matches = sorted(Path(p).resolve() for p in glob.glob(pattern)
                             if Path(p).suffix.lower() == '.avi' and Path(p).is_file())
            if not matches:
                raise ValueError(f"No AVI videos for {cohort['id']}: {pattern}")
            for video in matches:
                if video in seen:
                    raise ValueError(f"Overlapping video in {seen[video]} and {cohort['id']}: {video}")
                seen[video] = cohort['id']
                videos.append(str(video))
        inventory[cohort['id']] = videos
    return inventory


def select_geographic_sli_stages(stages, cohort_ids=None):
    """Select analysis cohorts while preserving the full export's column order."""
    if cohort_ids is None:
        return list(stages)
    if not isinstance(cohort_ids, (list, tuple, set)):
        raise ValueError('Analysis cohorts must be None or a collection of cohort ids')
    known = {stage['cohort_id'] for stage in stages}
    unknown = set(cohort_ids) - known
    if unknown:
        raise ValueError(f'Unknown analysis cohort ids: {sorted(unknown)}')
    return [stage for stage in stages if stage['cohort_id'] in cohort_ids]
