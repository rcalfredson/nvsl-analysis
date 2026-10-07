"""Inventory manuscript SLI data exports without running notebook analyses."""
from __future__ import annotations

import argparse
import ast
import builtins
import csv
import hashlib
import json
import re
import shlex
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

POLICY_COMMIT = 'c8ca3a8aaec0e723ddb4966c6e466e9fa9bc10f5'
EASTERN = ZoneInfo('America/New_York')


def load_sli_recipes(notebook):
    """Evaluate only recipe assignments and dependencies in this trusted notebook."""
    assignments, names = {}, []
    for cell in json.loads(Path(notebook).read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                assignments[name] = node
                if isinstance(node.value, ast.Dict) and any(
                    isinstance(k, ast.Constant) and k.value == 'output_type'
                    and isinstance(v, ast.Constant) and v.value == 'data'
                    for k, v in zip(node.value.keys, node.value.values)
                ):
                    names.append(name)
    scope = {'shlex': shlex, 'Path': Path}

    def resolve(name):
        if name in scope or hasattr(builtins, name):
            return
        node = assignments[name]
        for child in ast.walk(node.value):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in assignments:
                resolve(child.id)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(notebook), 'exec'), scope)

    recipes = []
    for name in names:
        resolve(name)
        recipe = scope[name]
        if any(tool in recipe['export_command'] for tool in
               ('export_reward_pi_difference_graphpad.py', 'mean-sli-csv')):
            recipes.append((name, recipe))
    return recipes


def assess_source(path, cutoff, revision_state):
    """Use embedded code revision and policy override before falling back to mtime."""
    path = Path(path)
    info = {'mtime': '', 'revision': '', 'status': 'missing', 'reason': 'File does not exist'}
    if not path.is_file():
        return info
    stamp = datetime.fromtimestamp(path.stat().st_mtime, EASTERN)
    info['mtime'] = stamp.isoformat(timespec='seconds')
    with path.open() as f:
        header = f.readline()
    match = re.search(r'\[r([0-9a-f]{7,40})(-M)?\]', header)
    policy = re.search(r'--reward-pi-sync(?:\s+|=)(\w+)', header)
    if policy and policy.group(1) != 'reward':
        info.update(status='stale', reason=f'Explicit legacy/different reward PI policy: {policy.group(1)}')
    elif match:
        revision = match.group(1)
        info['revision'] = revision + ('-M' if match.group(2) else '')
        state = revision_state(revision)
        if state == 'before':
            info.update(status='stale', reason='Recorded analysis revision predates the policy commit')
        elif state == 'after' and not match.group(2):
            info.update(status='current', reason='Recorded analysis revision contains the policy commit')
        else:
            info.update(status='recent_unverified', reason='Recorded revision is unknown or the analysis tree was modified')
    elif stamp < cutoff:
        info.update(status='stale', reason='Source modification time predates the policy boundary; no recorded revision')
    else:
        info.update(status='recent_unverified', reason='Source dated after boundary, but has no recorded analysis revision')
    return info


def assess_keyed_source(path, root, cutoff, revision_state):
    info = assess_source(path, cutoff, revision_state)
    sidecar = Path(path).with_suffix('.provenance.json')
    if not sidecar.is_file() or not Path(path).is_file():
        return info
    try:
        meta = json.loads(sidecar.read_text())
        stats = root / meta['learning_stats']
        valid = (
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == meta['source_sha256']
            and stats.is_file()
            and hashlib.sha256(stats.read_bytes()).hexdigest() == meta['learning_stats_sha256']
        )
        if valid:
            source = assess_source(stats, cutoff, revision_state)
            info.update(status=source['status'], revision=source['revision'],
                        reason='Verified source/statistics hashes; ' + source['reason'])
        else:
            info.update(status='recent_unverified', reason='Provenance hashes no longer match source/statistics')
    except (ValueError, KeyError, OSError):
        info.update(status='recent_unverified', reason='Unreadable or incomplete provenance sidecar')
    return info


def inventory(recipes, root, cutoff, revision_state):
    rows, panels = [], []
    for name, recipe in recipes:
        panel = next(iter(recipe['panels']))
        stages = recipe['analysis_stages']
        args = shlex.split(recipe['export_command'])
        groups = dict(args[i + 1].split('=', 1)[::-1] for i, v in enumerate(args) if v == '--group')
        sources, source_paths = [], set()
        for stage in stages:
            rel = stage['archive_output']
            source_paths.add(rel)
            info = assess_keyed_source(root / rel, root, cutoff, revision_state)
            row = dict(panel=panel, recipe=name, kind='source_archive', group=groups.get(rel, ''),
                       path=rel, **info)
            rows.append(row)
            sources.append(row)
            stats = stage.get('learning_stats_output')
            if stats:
                source_paths.add(stats)
                info = assess_source(root / stats, cutoff, revision_state)
                if info['status'] == 'missing':
                    info['status'] = 'missing_share_snapshot'
                    info['reason'] = 'Run this cohort analysis to archive its full learning_stats.csv'
                rows.append(dict(panel=panel, recipe=name, kind='learning_stats', group=groups.get(rel, ''), path=stats, **info))
        stale = [s['group'] for s in sources if s['status'] == 'stale']
        missing = [s['group'] for s in sources if s['status'] == 'missing']
        unverified = [s['group'] for s in sources if s['status'] == 'recent_unverified']
        for rel in recipe['artifacts']:
            if rel in source_paths:
                continue
            path = root / rel
            info = assess_source(path, cutoff, revision_state)
            if path.is_file():
                if stale or missing:
                    info.update(status='stale_dependency', reason='Source archives are stale/missing: ' + ', '.join(stale + missing))
                elif any(path.stat().st_mtime < (root / s['path']).stat().st_mtime for s in sources):
                    info.update(status='stale_conversion', reason='Output is older than an input archive')
                elif datetime.fromtimestamp(path.stat().st_mtime, EASTERN) < cutoff:
                    info.update(status='stale', reason='Output modification time predates the policy boundary')
                elif unverified:
                    info.update(status='recent_unverified', reason='Depends on sources without verified post-policy provenance')
                else:
                    info.update(status='current', reason='Sources contain policy commit; output is at least as recent as inputs')
            rows.append(dict(panel=panel, recipe=name, kind='graphpad' if rel == recipe['panels'][panel]['output'] else 'audit', group='', path=rel, **info))
        panels.append(dict(panel=panel, recipe=name, title=recipe['title'], stale_groups=stale,
                           missing_groups=missing, unverified_groups=unverified,
                           missing_outputs=[r['path'] for r in rows if r['panel'] == panel and r['kind'] != 'source_archive' and r['status'].startswith('missing')],
                           refresh_commands=[s['command'] for s in stages if any(row['path'] == s['archive_output'] and row['status'] in {'stale', 'missing'} for row in sources)],
                           conversion_command=recipe['export_command']))
    return rows, panels


def write_report(rows, panels, root, boundary, cutoff, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix('.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['panel', 'recipe', 'kind', 'group', 'path', 'mtime', 'revision', 'status', 'reason'])
        w.writeheader(); w.writerows(rows)
    refresh_path = out.with_name(out.stem + '_refresh.csv')
    with refresh_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['panel', 'recipe', 'kind', 'group', 'path', 'mtime', 'revision', 'status', 'reason'])
        w.writeheader()
        w.writerows(row for row in rows if row['status'].startswith(('stale', 'missing')))
    payload = {'policy_commit': boundary, 'boundary_eastern': cutoff.isoformat(), 'root': str(root), 'panels': panels, 'files': rows}
    out.with_suffix('.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Manuscript SLI export freshness', '',
             f'Policy boundary: `{boundary}` — **{cutoff.strftime("%B %d, %Y, %I:%M:%S %p %Z")}**.', '',
             'The new default counts training rewards after the first actual reward; the former default waited for a midline/control crossing. '
             'The audit uses recorded analysis revisions, explicit policy overrides, file times, and input dependencies. '
             'Newly converted GraphPad files still need refreshing when their input archives predate the policy.', '',
             '`recent_unverified` means the file is dated after the boundary but its analysis revision cannot be confirmed. '
             'A modified (`-M`) tree does not prove which policy was executed. '
             '`missing_share_snapshot` means the numerical source exists but its full learning_stats snapshot has not been archived. '
             'The audit does not rerun analyses or change exports.', '',
             '| Panel | Stale cohort archives | Missing cohort archives | Unverified cohorts | Missing outputs/snapshots |',
             '|---|---|---|---|---|']
    for p in panels:
        lines.append(f"| {p['panel']} | {len(p['stale_groups'])} | {len(p['missing_groups'])} | {len(p['unverified_groups'])} | {len(p['missing_outputs'])} |")
    lines += ['', '## Files requiring analysis or conversion refresh', '']
    for p in panels:
        todo = [r for r in rows if r['panel'] == p['panel'] and (r['status'].startswith('stale') or r['status'] == 'missing')]
        if not todo:
            continue
        lines += [f"### {p['panel']}", '']
        for row in todo:
            lines.append(f"- `{row['path']}` — {row['status']}: {row['reason']}; modified {row['mtime'] or '—'}; revision `{row['revision'] or 'not recorded'}`.")
        lines += ['', f"Regenerate sources with the `{p['recipe']}` analysis toggle, then convert. "
                  'If refreshing shared cohorts through another recipe, rerun the dependent conversion afterward.', '']
    lines += ['## Full file inventory', '', 'See the sibling CSV and JSON for every source, GraphPad output, audit, and missing full-statistics snapshot. '
              'The `_refresh.csv` file lists only stale or missing artifacts. '
              'The JSON includes refresh and conversion commands. Figures 3f and ED12g reuse source cohorts, '
              'so refreshing the originals does not automatically update their separately archived copies or conversions.', '']
    out.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', default='notebooks/manuscript_figure_panels.ipynb')
    parser.add_argument('--policy-commit', default=POLICY_COMMIT)
    parser.add_argument('--out', default='exports/manuscript/final_sli_freshness_report.md')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stamp = subprocess.check_output(['git', 'show', '-s', '--format=%cI', args.policy_commit], cwd=root, text=True).strip()
    cutoff = datetime.fromisoformat(stamp).astimezone(EASTERN)

    @lru_cache(None)
    def revision_state(revision):
        if subprocess.run(['git', 'cat-file', '-e', revision + '^{commit}'], cwd=root, capture_output=True).returncode:
            return 'unknown'
        result = subprocess.run(['git', 'merge-base', '--is-ancestor', args.policy_commit, revision], cwd=root, capture_output=True)
        if result.returncode == 0:
            return 'after'
        result = subprocess.run(['git', 'merge-base', '--is-ancestor', revision, args.policy_commit], cwd=root, capture_output=True)
        return 'before' if result.returncode == 0 else 'unknown'

    rows, panels = inventory(load_sli_recipes(root / args.notebook), root, cutoff, revision_state)
    out = root / args.out
    write_report(rows, panels, root, args.policy_commit, cutoff, out)
    print('Panel: stale / missing / unverified source archives')
    for p in panels:
        print(f"{p['panel']}: {len(p['stale_groups'])} / {len(p['missing_groups'])} / {len(p['unverified_groups'])}")
    print(f'Wrote {out} and sibling CSV/JSON ({len(rows)} file records).')


if __name__ == '__main__':
    main()
