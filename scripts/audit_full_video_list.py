#!/usr/bin/env python3
"""Read-only full-paper video audit. Stdlib only; never executes notebook code.

Run: python3 scripts/audit_full_video_list.py
Outputs: tmp/full_video_audit/{report.html,*.csv,manifest.json,README.md}
The full-paper source is authoritative. Heading inference is deliberately shallow:
all heading text and provenance survive; uncertain biological meaning stays unknown.
"""
import argparse
import ast
import csv
import fnmatch
import hashlib
import html
import itertools
from functools import lru_cache
import json
import os
from pathlib import Path
import posixpath
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

VIDEO = re.compile(r'\.avi$', re.I)
START = re.compile(r'/media/')
REVIEWED_SOURCE_SHA256 = '18926db48933cbc59fac88be7793c352aacc241a7fb59985bbdae6122040e4d3'
# Source-specific structural anchors, recognized by heading text.
ANCHORS = {
    'High-throughput task:': 'task', 'Large chamber tasks': 'task',
    'Large chamber Kir exps (used in manuscript)': 'experiment',
    'Blind flies: NorpA; +; 0273Gal4/UAS-CSCV': 'cohort',
    'Sight flies UAS-CSCV; +; 0273Gal4': 'cohort',
    'both hind tarsi removed and end of belly glued': 'cohort',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids(text):
    if not text:
        return None
    out = set()
    try:
        for part in text.split(','):
            a = part.split('-')
            if len(a) == 1:
                out.add(int(a[0]))
            elif len(a) == 2:
                lo, hi = map(int, a)
                if hi < lo or hi - lo > 10000:
                    return None
                out.update(range(lo, hi + 1))
            else:
                return None
        return sorted(out)
    except ValueError:
        return None


@lru_cache(maxsize=None)
def expand_classes(pattern):
    """Finite expansion of positive character classes; leave stars and ? intact."""
    m = re.search(r'\[([^\]]+)\]', pattern)
    if not m:
        return [pattern]
    chars = m[1]
    if chars.startswith(('!', '^')):
        raise ValueError('negated class requires general glob intersection')
    chars = re.sub(r'(.)-(.)', lambda x: ''.join(chr(i) for i in range(ord(x[1]), ord(x[2]) + 1)), chars)
    result = []
    for c in dict.fromkeys(chars):
        result.extend(expand_classes(pattern[:m.start()] + c + pattern[m.end():]))
        if len(result) > 10000:
            raise ValueError('class expansion exceeds 10000')
    return result


def pattern_witness(a, b):
    """Complete intersection for literals, positive [] classes and terminal *.
    Returns a symbolic intersection, NOT evidence that a recording exists.
    Other glob constructs return unsupported rather than a false disjoint result.
    """
    aa, bb = expand_classes(a), expand_classes(b)
    if any(any(c in posixpath.dirname(s) for c in '*?') for s in aa + bb):
        raise ValueError('non-finite directory wildcard intersection unsupported')
    for x, y in itertools.product(aa, bb):
        for s in (x, y):
            if '?' in s or '*' in s.rstrip('*') or s.endswith('**'):
                raise ValueError('unsupported wildcard intersection')
        if posixpath.dirname(x) != posixpath.dirname(y):
            continue
        if x == y:
            return x
        if x.endswith('*') and y.startswith(x[:-1]):
            return y
        if y.endswith('*') and x.startswith(y[:-1]):
            return x
    return ''


def extract_payload(line):
    """Return exact -v contents or raw inline path list, and its starting column."""
    m = re.search(r'(?:^|\s)-v\s+(["\'])', line)
    if m:
        start = m.end()
        end = line.find(m[1], start)
        if end < 0:
            raise ValueError('unterminated -v quote')
        return line[start:end], start
    m = START.search(line)
    if m:
        return line[m.start():], m.start()
    return '', 0


def parse_source(path):
    lines = Path(path).read_text().splitlines()
    refs, lists, coverage, headings, issues = [], [], [], [], []
    state = dict(task='', experiment='High-throughput task:', cohort='', condition='', round='', panel='')
    provenance = {}
    list_by_key = {}
    for n, raw in enumerate(lines, 1):
        s = raw.strip()
        row = dict(line=n, category='', reference_count=0, original=raw)
        if not s:
            row['category'] = 'blank'
        elif '/media/' in s:
            row['category'] = 'references'
            try:
                payload, col = extract_payload(raw)
            except ValueError as e:
                issues.append(dict(line=n, kind='extraction', detail=str(e), original=raw))
                row['category'] = 'unparsed_reference_line'
                coverage.append(row)
                continue
            prefix = raw[:raw.find('python analyze.py')].strip(' :') if 'python analyze.py' in raw else raw[:col].strip(' :')
            # Inline named openloop lists are siblings; unlabelled pooled command is its own list.
            label = prefix
            list_kind = ('named inline list' if prefix else
                         'unlabelled open-loop command' if re.search(r'open[ -]*loop', state['condition'], re.I) and '-v ' in raw else
                         'source list')
            context = ' > '.join(f'{k}: {v} [L{provenance.get(k, "?")}]' for k, v in state.items() if v)
            key = (tuple(sorted(provenance.items())), tuple(state.items()), label)
            if key not in list_by_key:
                lid = f'L{n:04}'
                li = dict(list_id=lid, first_line=n, last_line=n, **state, inline_label=label,
                          heading_lines=json.dumps(provenance), context=context, list_kind=list_kind,
                          chamber_type=('agarose' if 'agarose' in state['condition'].lower() else 'flat' if 'flat' in state['condition'].lower() else 'unspecified'),
                          apparatus_hint=('high-throughput' if state['task'] == 'High-throughput task:' else 'large chamber' if 'Large chamber Kir' in state['experiment'] or 'LC slide' in state['experiment'] else 'uncertain; enclosing Large chamber tasks heading'),
                          structure_status='inferred; original headings retained')
                lists.append(li)
                list_by_key[key] = li
            li = list_by_key[key]
            li['last_line'] = n
            fly = re.search(r'(?:^|\s)-f\s+([0-9,|\-]+)', raw)
            flygroups = fly[1].split('|') if fly else []
            for group, grouptext in enumerate(payload.split('|')):
                for token in grouptext.split(','):
                    if not token.strip():
                        issues.append(dict(line=n, kind='empty token', detail='Empty video-list entry', original=token))
                        continue
                    norm = token.strip()
                    selection = flygroups[min(group, len(flygroups)-1)] if flygroups else ''
                    notes = []
                    m = re.search(r':([0-9,\-]+)$', norm)
                    if m:
                        selection = m[1]
                        norm = norm[:m.start()]
                    if '(' in norm:
                        notes.append('inline prose inside path; not silently repaired')
                        m = re.search(r'only\s+f([0-9,\-]+)', norm)
                        if m:
                            selection = m[1]
                    if not norm.startswith('/'):
                        notes.append('relative path/shorthand unresolved')
                    if norm.startswith('/'):
                        norm = posixpath.normpath(norm)
                    r = dict(reference_id=f'R{len(refs)+1:04}', list_id=li['list_id'], line=n,
                             group=group, original=token, normalized=norm, selection_raw=selection,
                             selection_ids=json.dumps(ids(selection)), flags=raw[payload.__len__()+col:].strip(),
                             annotation='; '.join(notes), context=context, source_line=raw)
                    refs.append(r)
                    row['reference_count'] += 1
                    if notes:
                        issues.append(dict(line=n, kind='path annotation', detail='; '.join(notes), original=token))
                    if not selection:
                        issues.append(dict(line=n, kind='unspecified selection', detail='No -f or path-local fly/pair selector; no inheritance from adjacent commands', original=token))
            if row['reference_count'] != raw.count('/media/'):
                issues.append(dict(line=n, kind='coverage mismatch', detail='Extracted count differs from /media/ starts', original=raw))
        elif s.startswith(('-', '–')):
            row['category'] = 'option_continuation'
            row['preceding_command_line'] = refs[-1]['line'] if refs else ''
            row['list_id'] = refs[-1]['list_id'] if refs else ''
        elif s == 'Video list for the paper only':
            row['category'] = 'title'
        else:
            kind = ANCHORS.get(s)
            if re.match(r'summary for\b', s, re.I):
                kind = 'experiment'
            elif s.startswith(('Ctrl group:', 'exp group:', 'Exp group:', 'Ctrl-1 ', 'ctrl-2 ', 'Ctrl (#', 'UAS>>', 'upd3-mutant;', 'upd2 and upd3-mutant;', '71G01-')):
                kind = 'cohort'
            elif s in ('1st round', '2nd round'):
                kind = 'round'
            elif s.startswith('(For Fig.'):
                kind = 'panel'
            elif not kind:
                kind = 'condition'
            if kind == 'task':
                state = dict(task=s, experiment='', cohort='', condition='', round='', panel='')
                provenance = {'task': n}
            else:
                resets = {'experiment': ['cohort','condition','round','panel'], 'cohort': ['condition','round'], 'condition': ['round']}
                for k in resets.get(kind, []):
                    state[k] = ''
                    provenance.pop(k, None)
                state[kind], provenance[kind] = s, n
            row['category'] = 'heading_' + kind
            headings.append(dict(line=n, role=kind, original=raw, inference='source-specific anchor' if s in ANCHORS else 'lexical rule; biological meaning not inferred'))
        coverage.append(row)
    for li in lists:
        entries = [r for r in refs if r['list_id'] == li['list_id']]
        # JSON strings keep the evidence available in lists.csv as well as classification.
        li['expression_inventory'] = json.dumps(sorted({r['normalized'] for r in entries}))
        li['selection_inventory'] = json.dumps(sorted(
            json.dumps([r['normalized'], r['group'], ids(r['selection_raw']), r['selection_raw'] if ids(r['selection_raw']) is None else ''])
            for r in entries))
    return lines, refs, lists, coverage, headings, issues


class Resolver:
    def __init__(self):
        self.cache = {}
        self.directories = {}
        self.branches = []

    def directory(self, path):
        if path not in self.directories:
            try:
                with os.scandir(path) as it:
                    names = [e.name for e in it if e.is_file() and VIDEO.search(e.name)]
                self.directories[path] = (sorted(names), 'available')
            except FileNotFoundError:
                self.directories[path] = ([], 'missing/unavailable directory')
            except OSError as e:
                self.directories[path] = ([], f'{type(e).__name__}: {e}')
        return self.directories[path]

    def resolve(self, pattern, annotation=''):
        if (pattern, annotation) in self.cache:
            return self.cache[pattern, annotation]
        paths, unavailable, available, empty = set(), [], [], []
        status = 'unresolved'
        try:
            if annotation:
                raise ValueError(annotation)
            for p in expand_classes(pattern):
                parent, base = posixpath.split(p)
                if any(c in parent for c in '*?[') or not parent.startswith('/'):
                    raise ValueError('directory shorthand/glob not supported by resolver')
                names, ds = self.directory(parent)
                if ds != 'available':
                    unavailable.append(parent)
                    self.branches.append(dict(expression=pattern, branch=p, directory=parent, status=ds, matched_recordings=0))
                else:
                    available.append(parent)
                    matched = [posixpath.join(parent, name) for name in names if fnmatch.fnmatchcase(name, base)]
                    paths.update(matched)
                    if not matched:
                        empty.append(p)
                    self.branches.append(dict(expression=pattern, branch=p, directory=parent, status='matched' if matched else 'unmatched in available directory', matched_recordings=len(matched)))
            status = ('partial' if unavailable or empty else 'expanded') if paths else ('unavailable' if unavailable else 'unmatched in available directories')
        except ValueError as e:
            status = f'unresolved: {e}'
        result = dict(paths=sorted(paths), expansion_status=status, available_directories=sorted(set(available)), unavailable_directories=sorted(set(unavailable)), unmatched_branches=sorted(set(empty)))
        self.cache[pattern, annotation] = result
        return result


def relation(a, b):
    x, y = ids(a['selection_raw']), ids(b['selection_raw'])
    if x is None or y is None:
        return 'unknown', ''
    shared = sorted(set(x) & set(y))
    return ('intersecting' if shared else 'disjoint'), json.dumps(shared)


def heading_key(li):
    """Compare explicit heading meaning, independent of source line positions."""
    return tuple(' '.join(li.get(k, '').casefold().split())
                 for k in ('task', 'experiment', 'cohort', 'condition', 'round', 'panel'))


def inventory(li, field):
    return json.loads(li.get(field, '[]'))


def classify(a, b, lists):
    """Triage backed by heading context AND reference inventories, never line numbers.

    Labels suggest reasons for reuse; they do not validate biological membership.
    Unrecognized relationships remain for review even when labels look similar.
    """
    if a['list_id'] == b['list_id']:
        return 'within-list repetition'
    x, y = lists[a['list_id']], lists[b['list_id']]
    same_headings = heading_key(x) == heading_key(y)
    xi, yi = inventory(x, 'selection_inventory'), inventory(y, 'selection_inventory')
    xe, ye = set(inventory(x, 'expression_inventory')), set(inventory(y, 'expression_inventory'))
    labels = {x.get('inline_label', '').casefold(), y.get('inline_label', '').casefold()}
    if same_headings and labels == {'samples for seq', 'all data'}:
        sample, all_data = (x, y) if x.get('inline_label', '').casefold() == 'samples for seq' else (y, x)
        sample_counts = Counter(inventory(sample, 'selection_inventory'))
        all_counts = Counter(inventory(all_data, 'selection_inventory'))
        if sample_counts and all(all_counts[k] >= v for k, v in sample_counts.items()):
            return 'legitimate-looking reuse: samples versus all data; matching selections'
    if same_headings and {x.get('list_kind'), y.get('list_kind')} == {'named inline list', 'unlabelled open-loop command'}:
        named, pooled = (xe, ye) if x.get('list_kind') == 'named inline list' else (ye, xe)
        if named and named < pooled:
            return 'legitimate-looking reuse: named open-loop subset and pooled command'
    # A repeated experiment block can appear under a different enclosing task
    # container. Require all experiment-level headings and the full inventory to
    # agree, and expose the task discrepancy instead of inferring apparatus type.
    same_experiment_context = (bool(x.get('experiment', '').strip())
                               and heading_key(x)[1:] == heading_key(y)[1:])
    if (same_experiment_context and xi and xi == yi
            and x.get('inline_label', '') == y.get('inline_label', '')):
        suffix = '' if same_headings else '; enclosing task headings differ'
        return 'legitimate-looking reuse: repeated experiment context and identical source inventory' + suffix

    def control_label(li):
        label = ' '.join(li.get('condition', '').casefold().split())
        # An ISO date suffix is scheduling context; preserve treatment/genotype words.
        label = re.sub(r'\s+\d{4}-\d{2}(?:-\d{2})?$', '', label)
        return label if re.search(r'\b(?:ctrl|control)\b', label) else ''

    control = control_label(x)
    if (control and control == control_label(y) and xi and xi == yi
            and all(x.get(k, '') == y.get(k, '') for k in ('task', 'cohort', 'round'))):
        return 'legitimate-looking reuse: identical control inventory; verify shared-control intent'
    return 'potentially conflicting assignments; requires review'


def compare(refs, listmap, resolver):
    findings = []
    expansions, members = [], defaultdict(list)
    for r in refs:
        result = resolver.resolve(r['normalized'], r['annotation'])
        r.update({k: json.dumps(v) if isinstance(v,list) else v for k,v in result.items() if k != 'paths'})
        r['matched_recordings'] = len(result['paths'])
        for path in result['paths']:
            expansions.append(dict(reference_id=r['reference_id'], list_id=r['list_id'], line=r['line'], path=path, selection_raw=r['selection_raw']))
            members[path].append(r)
    actual_pairs = defaultdict(list)
    for path, rr in members.items():
        for a,b in itertools.combinations(rr,2):
            actual_pairs[a['reference_id'],b['reference_id']].append(path)
    def add(a,b,evidence,identity,recording='',witness=''):
        rel, shared = relation(a,b)
        findings.append(dict(finding_id=f'F{len(findings)+1:05}', category=classify(a,b,listmap),
                             evidence=evidence, identity_level=identity, recording=recording, symbolic_intersection=witness,
                             reference_a=a['reference_id'],reference_b=b['reference_id'],list_a=a['list_id'],list_b=b['list_id'],
                             line_a=a['line'],line_b=b['line'],original_a=a['original'],original_b=b['original'],
                             normalized_a=a['normalized'],normalized_b=b['normalized'],selection_a=a['selection_raw'],selection_b=b['selection_raw'],
                             selection_relation=rel,shared_selection_ids=shared,context_a=a['context'],context_b=b['context']))
    for a,b in itertools.combinations(refs,2):
        found = actual_pairs.get((a['reference_id'],b['reference_id']),[])
        for path in found:
            add(a,b,'confirmed existing recording','exact full path',recording=path)
        if a['normalized'] == b['normalized']:
            add(a,b,'confirmed repeated reference expression','exact normalized expression',witness=a['normalized'])
        elif not a['annotation'] and not b['annotation']:
            try:
                w = pattern_witness(a['normalized'],b['normalized'])
                if w:
                    add(a,b,'overlapping patterns; existence separate','symbolic full-path intersection',witness=w)
            except ValueError:
                # Recorded separately in coverage; do not call unsupported comparisons disjoint.
                pass
    # Identical AVI basename in different directories is only a possible identity.
    bybase = defaultdict(dict)
    for path, rr in members.items():
        bybase[posixpath.basename(path)][path] = rr
    for base, paths in bybase.items():
        for (pa, aa),(pb,bb) in itertools.combinations(paths.items(),2):
            for a,b in itertools.product(aa,bb):
                add(a,b,'possible match only; different full paths','basename only',recording=f'{pa} | {pb}')
                findings[-1]['category'] = 'possible identity; basename alone cannot establish recording reuse'
    return findings, expansions, members


def notebook_scan(paths, refs, resolver):
    """Extract Python literal strings with AST, including implicitly joined strings.
    Never execute/eval cells. Dynamic recipes are not reconstructed; audit coverage says so.
    """
    rows, cov, unparsed = [], [], []
    for path in paths:
        nb = json.loads(path.read_text())
        for ci, cell in enumerate(nb.get('cells',[])):
            source = ''.join(cell.get('source',[]))
            if cell.get('cell_type') != 'code' or '/media/' not in source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                cov.append(dict(notebook=str(path),cell_index=ci,status='syntax not parsed',detail=str(e),media_starts=source.count('/media/')))
                continue
            parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            constants = [n for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and '/media/' in n.value]
            count = 0
            for node in constants:
                value = node.value
                ancestor = node
                symbol = ''
                complete_literal = False
                while ancestor in parents:
                    ancestor = parents[ancestor]
                    if isinstance(ancestor, (ast.Assign, ast.AnnAssign)):
                        targets = ancestor.targets if isinstance(ancestor, ast.Assign) else [ancestor.target]
                        symbol = ','.join(ast.unparse(t) for t in targets)
                        complete_literal = isinstance(ancestor.value, ast.Constant) and isinstance(ancestor.value.value, str)
                        break
                local_fly = re.search(r'(?:^|\s)-f\s+([0-9,|\-]+)', value)
                # Stop at delimiters; paths can include spaces. One reference ends at *, .avi or :selector.
                node_matches = list(re.finditer(r'/media/[^\n\r,|\'"*]*?(?:\*(?::[0-9,-]+)?|\.avi(?::[0-9,-]+)?)',value))
                if len(node_matches) != value.count('/media/'):
                    unparsed.append(dict(notebook=str(path),cell_index=ci,code_line=node.lineno,symbol=symbol,media_starts=value.count('/media/'),extracted=len(node_matches),original=value,reason='Some /media/ occurrences are directory literals, shorthand, or unsupported reference syntax; no reconstruction attempted'))
                for m in node_matches:
                    token = m[0].strip()
                    matched = [r['reference_id'] for r in refs if r['normalized'] == token]
                    rows.append(dict(notebook=str(path),cell_index=ci,cell_id=cell.get('id',''),code_line=node.lineno, symbol=symbol, complete_assignment_literal=complete_literal, literal_selection_raw=local_fly[1] if local_fly else '',cell_selector_candidates=json.dumps(re.findall(r'(?:^|\s)-f\s+([0-9,|\-]+)', source)),
                                     original=token,normalized=token,fullpaper_exact_expression_refs=json.dumps(matched),
                                     literal_context=value,selection_status='literal only; enclosing dynamic recipe not executed'))
                    count += 1
            cov.append(dict(notebook=str(path),cell_index=ci,status='literal AST scan; dynamic commands not reconstructed',detail='',media_starts=source.count('/media/'),literal_media_starts=sum(n.value.count('/media/') for n in constants),extracted=count))
    return rows,cov,unparsed


def reconcile_epg(lists, notebook_rows, notebook_unparsed):
    """Compare the targeted EPG literal list without relying on document positions.

    A unique, complete source on each side is required. This deliberately does not
    infer runtime selector bindings or choose between multiple candidate cohorts.
    """
    symbol = 'EXTENDED_DATA_FIGURE_14I_EPG_VIDEO_GLOBS'
    paper_candidates = [li for li in lists
                        if 'large chamber kir' in li.get('experiment', '').casefold()
                        and re.search(r'\bEPG\b', li.get('cohort', ''), re.I)
                        and re.search(r'\bKir\b', li.get('cohort', ''), re.I)
                        and re.search(r'\bflat\b', li.get('condition', ''), re.I)
                        and not re.search(r'\bAR\b|antenna', li.get('condition', ''), re.I)]
    nb = [r for r in notebook_rows if r['symbol'] == symbol]
    groups = {(r['notebook'], r['cell_index'], r['code_line']) for r in nb}
    result = dict(status='', summary='', fullpaper_candidates=len(paper_candidates),
                  notebook_candidates=len(groups), fullpaper_line='', list_id='',
                  notebook='', cell_index='', code_line='', symbol=symbol,
                  comparison_scope='literal video expressions only; runtime recipes and selector bindings not reconstructed')
    differences = []
    if len(paper_candidates) == 1:
        paper = paper_candidates[0]
        result.update(fullpaper_line=paper['first_line'], list_id=paper['list_id'])
    if len(groups) == 1:
        result.update(zip(('notebook', 'cell_index', 'code_line'), next(iter(groups))))
    if len(paper_candidates) > 1 or len(groups) > 1:
        result.update(status='ambiguous', summary='Multiple EPG source candidates found; no automatic comparison or assignment was made.')
    elif not paper_candidates or not groups:
        result.update(status='unavailable', summary='A unique full-paper flat EPG/Kir list and notebook literal assignment could not both be found. Agreement is unverified.')
    elif (not all(r.get('complete_assignment_literal') is True for r in nb)
          or any(r.get('symbol') == symbol for r in notebook_unparsed)
          or not inventory(paper, 'expression_inventory')):
        result.update(status='incomplete', summary='The EPG source could not be extracted as a complete literal list. Agreement is unverified.')
    else:
        paper_set = set(inventory(paper, 'expression_inventory'))
        notebook_set = {r['normalized'] for r in nb}
        paper_only, notebook_only = paper_set - notebook_set, notebook_set - paper_set
        location = f"full-paper L{paper['first_line']} and {result['notebook']} cell {result['cell_index']} (zero-based)"
        if not paper_only and not notebook_only:
            result.update(status='matching', summary=f'The EPG literal video expressions agree between {location}. Runtime recipes and fly selections were not compared.')
        else:
            result.update(status='different', summary=f'The EPG literal lists differ between {location}: {len(paper_only)} full-paper-only and {len(notebook_only)} notebook-only expressions. The full-paper document remains the primary source; review the notebook differences.')
            # Supplemental notebook expansion must not contaminate source coverage counts.
            resolver = Resolver()
            for origin, expressions in [('full-paper only', paper_only), ('notebook only', notebook_only)]:
                for expression in sorted(expressions):
                    matched = resolver.resolve(expression)
                    differences.append(dict(status=origin, expression=expression,
                                            recordings=json.dumps(matched['paths']),
                                            expansion_status=matched['expansion_status'],
                                            fullpaper_line=paper['first_line'], notebook=result['notebook'],
                                            cell_index=result['cell_index'], code_line=result['code_line'],
                                            symbol=symbol, selection_candidates=nb[0]['cell_selector_candidates'],
                                            interpretation='Current literal expression difference; primary source is the full-paper document. Runtime recipe not executed.'))
    return result, differences


def write_csv(path, rows, fields=None):
    keys = fields or list(dict.fromkeys(k for r in rows for k in r)) or ['no_rows']
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def table(rows, fields=None):
    if not rows:
        return '<p>None found in the covered inputs.</p>'
    fields = fields or list(rows[0])
    def val(k,v):
        s = html.escape(str(v))
        if k in ('line','line_a','line_b') and str(v).isdigit():
            return f'<a href="#source-{v}">{s}</a>'
        return s
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+val(k,r.get(k,''))+'</td>' for k in fields)+'</tr>' for r in rows)+'</tbody></table></div>'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('tmp/full_video_list.txt'))
    ap.add_argument('--out',type=Path,default=Path('tmp/full_video_audit'))
    args=ap.parse_args()
    protected=[args.source,*sorted(Path('notebooks').glob('*.ipynb'))]
    before={str(p):sha(p) for p in protected}
    lines,refs,lists,coverage,headings,issues=parse_source(args.source)
    lm={r['list_id']:r for r in lists}
    resolver=Resolver()
    findings,expansions,members=compare(refs,lm,resolver)
    nbrows,nbcoverage,nbunparsed=notebook_scan(protected[1:],refs,resolver)
    for li in lists:
        rr = [r for r in refs if r['list_id'] == li['list_id']]
        li['reference_count'] = len(rr)
        li['unique_expression_count'] = len({r['normalized'] for r in rr})
        li['selection_values'] = json.dumps(sorted({r['selection_raw'] for r in rr}))
        li['resolved_recording_count'] = len({e['path'] for e in expansions if e['list_id'] == li['list_id']})
        li['unresolved_reference_count'] = sum(r['expansion_status'] != 'expanded' for r in rr)
    # Group findings by list pair for investigation, separately from expression evidence.
    grouped=defaultdict(list)
    for f in findings:
        grouped[f['list_a'],f['list_b'],f['category']].append(f)
    pairs=[]
    for (a,b,c),ff in grouped.items():
        existing={f['recording'] for f in ff if f['evidence']=='confirmed existing recording'}
        pairs.append(dict(list_a=a,list_b=b,category=c,existing_recording_count=len(existing),
                          reference_pair_count=len({(f['reference_a'],f['reference_b']) for f in ff}),
                          selection_relations='; '.join(sorted({f['selection_relation'] for f in ff})),
                          recordings=json.dumps(sorted(existing)),source_a=lm[a]['context'],source_b=lm[b]['context'],
                          line_a=lm[a]['first_line'],line_b=lm[b]['first_line']))
    dirs=[dict(directory=p,status=v[1],avi_count=len(v[0])) for p,v in sorted(resolver.directories.items())]
    unresolved=[r for r in refs if r['expansion_status']!='expanded']
    for r in refs:
        try:
            pattern_witness(r['normalized'],r['normalized'])
        except ValueError as e:
            issues.append(dict(line=r['line'],kind='symbolic comparison unsupported',detail=str(e),original=r['original']))
    # Structural ambiguities and source notes are explicit; do not silently assign meanings.
    structural=[
        (2,'Heading hierarchy is inferred from lexical rules and reviewed anchors. Per-list heading paths remain authoritative context, not validated biological labels.'),
        (12,'Blind/sighted genotype headings are treated as cohorts under high-throughput tasks.'),
        (40,'AR, AG and LA shorthand retained verbatim; AR is contextually antenna removed, but AG/LA are not expanded. Named open-loop entries and pooled command are separate lists.'),
        (51,'Both arista and front tarsi inline command is a separate sibling list; it is not the label for the unlabelled pooled command on L52.'),
        (60,'Summary mentions antenna glued while subordinate AR headings may mean antenna removed; preserve both and verify terminology.'),
        (96,'"c31 (only f0-9)_*" is malformed as a literal path. Selector 0-9 recorded, path not repaired or expanded.'),
        (247,'Bare agarose/openloop headings lack repeated treatment labels. Do not propagate treatment from a previous condition; enclosing experiment retained.'),
        (279,'Trailing numbers 88/102/82/106/83 retained as headings, not parsed as fly counts or IDs.'),
        (300,'--ayc permits yoked IDs; --yc R appears in the heading. Preserve flags and numeric selections; do not infer actual accepted flies or exclusions without protocol/analysis execution.'),
        (343,'The upd2/upd3 experiment block is repeated at L660-675. Reuse is likely document duplication, not independent cohorts.'),
        (364,'Large chamber tasks remains the enclosing task for later summaries. Some later commands select 0-9/0-19: actual chamber type of those experiments is uncertain from this layout; see chamber inference note.'),
        (366,'Ctrl-lexA cohort plus AR/agarose/flat conditions inferred from adjacent headings. AR does not establish genotype membership by itself.'),
        (396,'PFR lexA driver is missing before -lexA; do not fill it from other experiments.'),
        (400,'Slide conditions and first/second rounds are nested; round paths remain distinct recordings. Shared cohort over rounds is not exact video reuse.'),
        (430,'Washed/unwashed and learned/naive mixed conditions may split roles within recordings. Numeric selectors and original descriptions preserved; fly-role mapping is unresolved.'),
        (439,'Wild-caught numbered strains are encoded in condition headings; strain number is not a fly ID. RNAseq headings preserve sampling distinctions.'),
        (623,'L624 samples for seq versus L625 all data is a nested subset of the same #174 cohort; likely intentional reuse.'),
    ]
    source_reviewed=sha(args.source)==REVIEWED_SOURCE_SHA256
    if not source_reviewed:
        issues.append(dict(line=1,kind='source revision',detail='Source differs from reviewed snapshot. Snapshot-specific structural notes are omitted; review the inferred hierarchy for the revised document.',original=''))
    for n,d in structural if source_reviewed else []:
        issues.append(dict(line=n,kind='structure/interpretation',detail=d,original=lines[n-1] if n<=len(lines) else ''))
    epg, differences = reconcile_epg(lists, nbrows, nbunparsed)
    # Verify the two user-provided cross-checks by literal pattern membership in each source.
    checks=[]
    for base in ('c61__2025-07-21__20-41-33.avi','c62__2025-07-21__20-41-36.avi'):
        path='/media/Synology4/Yang Chen/2025-07-21/night/'+base
        checks.append(dict(recording=path,exists=Path(path).is_file(),
                           fullpaper_memberships=json.dumps([dict(reference_id=r['reference_id'],line=r['line'],context=r['context'],selection=r['selection_raw']) for r in refs if fnmatch.fnmatchcase(path,r['normalized'])]),
                           notebook_literal_memberships=json.dumps([dict(notebook=r['notebook'],cell_index=r['cell_index'],code_line=r['code_line'],symbol=r['symbol'],literal_selection=r['literal_selection_raw'],cell_selector_candidates=r['cell_selector_candidates'],expression=r['normalized']) for r in nbrows if fnmatch.fnmatchcase(path,r['normalized'])]),
                           interpretation='Memberships are computed from the current literal expressions. EPG comparison: ' + epg['status'] + '. No assignment edit performed.'))
    after={str(p):sha(p) for p in protected}
    if before!=after:
        raise RuntimeError('Protected input hash changed during audit')
    args.out.mkdir(parents=True,exist_ok=True)
    data={'references':refs,'lists':lists,'headings':headings,'line_coverage':coverage,'findings':findings,'list_pairs':pairs,
          'expanded_recordings':expansions,'directories':dirs,'pattern_branches':resolver.branches,'command_continuations':[r for r in coverage if r['category']=='option_continuation'],'unresolved':unresolved,'issues':issues,
          'notebook_literals':nbrows,'notebook_coverage':nbcoverage,'notebook_unparsed_literals':nbunparsed,'cross_checks':checks,'epg_comparison':[epg],'source_discrepancies':differences}
    for name,rows in data.items():
        write_csv(args.out/(name+'.csv'),rows)
    confirmed=[f for f in findings if f['evidence']=='confirmed existing recording']
    stats=dict(source_lines=len(lines),line_categories=dict(Counter(r['category'] for r in coverage)),
               media_path_starts=sum(s.count('/media/') for s in lines),extracted_references=len(refs),lists=len(lists),headings=len(headings),
               distinct_expressions=len({r['normalized'] for r in refs}),unique_existing_recordings=len(members),
               recording_memberships=len(expansions),expansion_statuses=dict(Counter(r['expansion_status'] for r in refs)),
               finding_evidence_counts=dict(Counter(f['evidence'] for f in findings)),
               confirmed_recording_categories=dict(Counter(f['category'] for f in confirmed)),
               confirmed_unique_shared_recordings=len({f['recording'] for f in confirmed}),
               within_list_duplicate_expression_pairs=sum(f['category']=='within-list repetition' and f['evidence']=='confirmed repeated reference expression' for f in findings),
               potentially_conflicting_existing_recording_rows=sum(f['category'].startswith('potentially conflicting') for f in confirmed),
               unavailable_directories=sum(d['status']!='available' for d in dirs),
               unmatched_finite_pattern_branches=sum(b['status']=='unmatched in available directory' for b in resolver.branches),
               notebook_literal_occurrences=len(nbrows),notebook_cells_scanned=len(nbcoverage),
               unsupported_symbolic_references=sum(i['kind']=='symbolic comparison unsupported' for i in issues),
               protected_inputs_unchanged=before==after)
    manifest=dict(generated_utc=datetime.now(timezone.utc).isoformat(),source=str(args.source),input_sha256=before,
                  reviewed_source_snapshot=source_reviewed,checker_sha256=sha(__file__),command='python3 scripts/audit_full_video_list.py --source '+str(args.source)+' --out '+str(args.out),stats=stats,
                  epg_comparison=epg,
                  user_context='Historical context: user identified the full-paper document as corrected. Current notebook agreement is computed each run.',
                  limitations=['File identity means same lexical full AVI path, not content hash/inode; different full paths never merged.',
                               'No AVI contents or protocol pickles loaded; selectors describe admitted IDs, not post-exclusion sample sizes.',
                               'Unmatched/unavailable patterns remain unresolved. Available glob expansion is a filesystem snapshot.',
                               'Symbolic comparisons support literal paths, positive character classes and terminal stars. Unsupported syntax is reported.',
                               'Basename-only comparison covers resolved AVIs. Similar camera/date names alone never establish identity.',
                               'Notebook AST scan extracts literal paths only; dynamic recipe execution and comprehensive notebook reconciliation are out of scope.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    readme='''# Full-paper video source audit

Run from repository root: `python3 scripts/audit_full_video_list.py` (Python 3, standard library only).
Open `report.html`. All tables are CSV; `manifest.json` records hashes, counts and filesystem snapshot time.
Source inputs and notebooks are read-only and checked by SHA-256 before/after.

`findings.csv` has separate rows for expression repetition, symbolic overlap, existing full-path recording matches,
and basename-only possibilities. These evidence rows must not be summed as unique recordings.
`list_pairs.csv` deduplicates recordings per list pair; a recording can still occur in multiple pairs.
`expanded_recordings.csv` is the long reference-to-AVI membership table, preserving selectors.
`references.csv` retains original entries and command lines, normalization, selectors, flags and heading context.
`line_coverage.csv` accounts for every line; `headings.csv`, `lists.csv`, `issues.csv`, `unresolved.csv`
and `directories.csv` expose interpretation and expansion gaps. `pattern_branches.csv` records every finite character-class branch, including unmatched branches inside otherwise matching patterns.
`notebook_literals.csv` is a nonexecuting AST literal scan with zero-based cell index and one-based cell code line.
It is supplemental, incomplete for dynamic recipes, and never overrides the full-paper text.

Normalization trims token-edge whitespace and applies lexical POSIX normalization only; case, dates, camera names,
round subdirectories and spellings are preserved. Fly IDs from -f or : overrides are numeric admitted selections.
No selection is inherited from adjacent commands. Inline prose is not silently stripped from paths.
Actual protocol roles, exclusions and final sample membership are not asserted.

Hierarchy is inferred using lexical rules plus explicit anchors in the checker.
Curated structural notes use line anchors from the reviewed source snapshot and are omitted when its hash
changes. Classification uses heading context and source inventories, independent of line positions; unknown
relationships require review. Shared controls require identical source/selection inventories and matching
control labels (ignoring only a trailing ISO date). The targeted EPG comparison requires one explicit flat
EPG/Kir cohort under Large chamber Kir experiments and one complete literal assignment to
EXTENDED_DATA_FIGURE_14I_EPG_VIDEO_GLOBS. Missing, ambiguous or incomplete comparisons are reported without
claiming agreement. Agreement covers literal expressions, not reconstructed runtime commands.
Automatic extraction and recording matching always use the supplied source contents. Source line IDs anchor each list.
Every heading and original line is retained. In particular the Large chamber task heading remains an enclosing
layout container for later summaries; it does not prove their physical chamber types. Reuse classifications are
triage suggestions that require intent verification; no assignments are changed.
'''
    (args.out/'README.md').write_text(readme)
    reuse_counts=Counter(f['category'] for f in confirmed)
    summary_text=(f"The full-paper lists contain {stats['confirmed_unique_shared_recordings']} shared existing recordings. "
                  f"Within-list duplicate expression pairs: {stats['within_list_duplicate_expression_pairs']}; potentially conflicting existing-recording rows: {stats['potentially_conflicting_existing_recording_rows']}. "
                  'Reuse categories and source contexts are shown below. No absence-of-overlap claim extends to unresolved references.')
    parts=['''<!doctype html><html><head><meta charset="utf-8"><title>Full-paper video audit</title>
<style>body{font:15px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#17222b}h1,h2{color:#153c55}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #ccd6dd;padding:6px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#edf3f6;position:sticky;top:0}.scroll{max-height:650px;overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:15px}details{margin:15px 0}.note{background:#fff1cd;padding:14px}a{color:#15638c}.line{display:block;border-bottom:1px solid #eee;font:12px monospace;white-space:pre-wrap;overflow-wrap:anywhere}.line:target{background:#ffe2a5}</style></head><body>
<h1>Full-paper video source audit</h1><p>Inputs preserved. Existing recordings, repeated expressions and hypothetical pattern intersections are counted separately.</p>''']
    parts.append('<p>Primary source: '+html.escape(str(args.source))+'</p>')
    parts.append('<p class="note"><b>EPG comparison:</b> '+html.escape(epg['summary'])+'</p>')
    if not source_reviewed:
        parts.append('<p class="note">Source differs from reviewed snapshot. Snapshot-specific structural notes have been omitted. Review the inferred hierarchy for the revised document.</p>')
    parts += ['<h2>Findings at a glance</h2><p>'+html.escape(summary_text)+'</p>'+table([dict(category=k,existing_recording_rows=v) for k,v in reuse_counts.items()])+'<p>Basename-only candidates remain unconfirmed recording identities. Similar camera names alone do not establish identity.</p><h2>Coverage and counts</h2><pre>'+html.escape(json.dumps(stats,indent=2))+'</pre>',
              '<p>Confirmed counts are reference-pair/recording rows, not unique animals. Physical chamber type and fly-role mappings are not validated. Chamber surface is inferred from condition headings; apparatus hints remain uncertain where the layout mixes task types.</p>',
              '<h2>Confirmed recording overlaps by list</h2>'+table([p for p in pairs if p['existing_recording_count']],['list_a','list_b','category','existing_recording_count','selection_relations','line_a','line_b','source_a','source_b']),
              '<h2>Possible overlaps and expression-only findings</h2>'+table([p for p in pairs if not p['existing_recording_count']]),
              '<h2>User-provided recording cross-checks</h2>'+table(checks)+'<h2>EPG literal source comparison</h2>'+table([epg])+'<h3>Expression differences</h3>'+table(differences),
              '<h2>Unresolved references</h2><p>An unmatched or unavailable pattern does not establish absence of overlap. Each missing bracket-expanded directory is included in directories.csv.</p>'+table(unresolved,['reference_id','line','original','selection_raw','expansion_status','available_directories','unavailable_directories','unmatched_branches','context']),
              '<h2>Structure, shorthand and selection gaps</h2>'+table(issues),
              '<h2>All findings (filter with browser search; full CSV linked below)</h2>'+table(findings),
              '<details><summary>Inferred list hierarchy</summary>'+table(lists)+'</details>',
              '<details><summary>Directory availability</summary>'+table(dirs)+'</details>',
              '<details><summary>Notebook scan coverage</summary>'+table(nbcoverage)+'</details>'+'<details><summary>Notebook literals not fully extracted</summary>'+table(nbunparsed)+'</details>',
              '<h2>Method and reproducibility</h2><pre>'+html.escape(readme)+'</pre><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in manifest['limitations'])+'</ul>',
              '<h2>Download tables</h2><ul>'+''.join(f'<li><a href="{name}.csv">{name}.csv</a> ({len(rows)} rows)</li>' for name,rows in data.items())+'<li><a href="manifest.json">manifest.json</a></li><li><a href="README.md">README.md</a></li></ul>',
              '<h2>Original source with line references</h2>'+''.join(f'<span class="line" id="source-{n}"><a href="#source-{n}">{n:4}</a>  {html.escape(s)}</span>' for n,s in enumerate(lines,1)), '</body></html>']
    (args.out/'report.html').write_text('\n'.join(parts))
    print(json.dumps(stats,indent=2))
    print('Report:',args.out/'report.html')


if __name__=='__main__':
    main()
