"""Audit invariants: identity, uncertain patterns, selection and source preservation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('video_audit', Path(__file__).parents[1] / 'scripts/audit_full_video_list.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class FullVideoAuditTests(unittest.TestCase):
    def test_symbolic_overlap_does_not_require_available_recording(self):
        self.assertEqual(audit.pattern_witness('/missing/day[12]/c[12]_*', '/missing/day2/c1_*'), '/missing/day2/c1_*')
        self.assertFalse(audit.pattern_witness('/missing/day1/c1_*', '/other/day1/c1_*'))
        self.assertFalse(audit.pattern_witness('/missing/day1/c1_*', '/missing/day1/c11_*'))
        with self.assertRaises(ValueError):
            audit.pattern_witness('/missing/c?_*', '/missing/c1_*')

    def test_same_recording_disjoint_ids_are_not_same_flies(self):
        a = {'selection_raw': '0-1'}
        b = {'selection_raw': '2-3'}
        self.assertEqual(audit.relation(a,b), ('disjoint','[]'))
        self.assertEqual(audit.relation(a,{'selection_raw':'1-2'}), ('intersecting','[1]'))
        self.assertEqual(audit.relation(a,{'selection_raw':''}), ('unknown',''))

    def test_partial_globs_expose_missing_and_empty_branches(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'day1').mkdir()
            (root/'day1'/'c1__recording.avi').touch()
            (root/'day1'/'c1__recording.data').touch()
            r=audit.Resolver()
            result=r.resolve(str(root/'day[12]'/'c[12]_*'))
            self.assertEqual(len(result['paths']),1)
            self.assertEqual(result['expansion_status'],'partial')
            self.assertEqual(result['unavailable_directories'],[str(root/'day2')])
            self.assertEqual(result['unmatched_branches'],[str(root/'day1'/'c2_*')])
            self.assertEqual(len(r.branches),4)
            self.assertTrue(r.resolve(str(root/'day1'/'c9_*'))['expansion_status'].startswith('unmatched'))

    def test_repeated_blocks_remain_distinct_and_prose_is_not_repaired(self):
        source='''Video list for the paper only
High-throughput task:
Summary for group
Flat
python analyze.py -v " /media/absent/c1_* ,/media/absent/c1_*:2-3" -f 0-1
Summary for group
Flat
python analyze.py -v "/media/absent/c1_*" -f 0-1
openloop
/media/absent/c31 (only f0-9)_*
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'source.txt'
            p.write_text(source)
            _,refs,lists,cov,_,issues=audit.parse_source(p)
            self.assertEqual(len(lists),3)
            self.assertEqual(refs[0]['normalized'],'/media/absent/c1_*')
            self.assertEqual(refs[0]['original'],' /media/absent/c1_* ')
            self.assertEqual(refs[1]['selection_raw'],'2-3')
            self.assertNotEqual(refs[0]['list_id'],refs[2]['list_id'])
            self.assertIn('(only f0-9)',refs[3]['normalized'])
            self.assertEqual(refs[3]['selection_raw'],'0-9')
            self.assertEqual(len(cov),len(source.splitlines()))
            self.assertEqual(p.read_text(),source)
            resolver=audit.Resolver()
            findings,expanded,_=audit.compare(refs,{x['list_id']:x for x in lists},resolver)
            self.assertFalse(expanded)
            self.assertTrue(any(f['evidence']=='confirmed repeated reference expression' and f['selection_relation']=='disjoint' for f in findings))
            self.assertFalse(any(f['evidence']=='confirmed existing recording' for f in findings))

    def test_similar_names_and_basename_match_keep_different_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            refs=[]
            lists={}
            for i,folder in enumerate(('one','two')):
                d=root/'media'/folder
                d.mkdir(parents=True)
                (d/'c1__recording.avi').touch()
                lid=str(i)
                lists[lid]=dict(experiment=folder,cohort='',condition='Flat',first_line=i+1)
                refs.append(dict(reference_id=lid,list_id=lid,line=i+1,normalized=str(d/'c1_*'),original=str(d/'c1_*'),annotation='',selection_raw='0-1',context=folder))
            findings,_,members=audit.compare(refs,lists,audit.Resolver())
            self.assertEqual(len(members),2)
            self.assertEqual([f['identity_level'] for f in findings],['basename only'])

    def parse_fixture(self, source):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'source.txt'
            p.write_text(source)
            return audit.parse_source(p)

    def test_reuse_classification_survives_shifted_lines_and_task_containers(self):
        source = '''Video list for the paper only
High-throughput task:
Summary for openloop experiment
open-loop data
Flat 7p: /media/absent/a_*
Flat 25p: /media/absent/b_*
python analyze.py -v "/media/absent/a_*,/media/absent/b_*" -f 0-19
Summary for sampling experiment
#12 RNAseq Flat
samples for seq: python analyze.py -v "/media/absent/sample_*" -f 0-1
all data: python analyze.py -v "/media/absent/sample_*,/media/absent/extra_*" -f 0-1
Summary for repeated experiment
Flat
python analyze.py -v "/media/absent/repeat_*" -f 0-1
Large chamber tasks
Summary for repeated experiment
Flat
python analyze.py -v "/media/absent/repeat_*" -f 0-1
'''
        def classifications(text):
            _, refs, lists, *_ = self.parse_fixture(text)
            lm = {li['list_id']: li for li in lists}
            # Include only pairs with equal expressions; physical expansion is irrelevant.
            return [audit.classify(a, b, lm)
                    for i, a in enumerate(refs) for b in refs[i + 1:]
                    if a['normalized'] == b['normalized']]
        expected = classifications(source)
        self.assertEqual(expected, classifications('\n' * 100 + 'Inserted heading\n' + source))
        self.assertEqual(sum('named open-loop subset' in c for c in expected), 2)
        self.assertTrue(any('samples versus all data' in c for c in expected))
        self.assertTrue(any('identical source inventory; enclosing task headings differ' in c for c in expected))

    def test_similar_cohort_labels_do_not_establish_legitimate_reuse(self):
        cases = [
            ('#174 for RNAseq Flat', '', '#174 for RNAseq Agarose', '', '0-1', '0-1'),
            ('ctrl>Kir-flat', '', 'ctrl>Kir-flat', '', '0-1', '2-3'),
            ('#12 RNAseq Flat', 'samples for seq: ', '#12 RNAseq Flat', 'all data: ', '0-1', '2-3'),
        ]
        for condition_a, prefix_a, condition_b, prefix_b, selection_a, selection_b in cases:
            with self.subTest(conditions=(condition_a, condition_b), selections=(selection_a, selection_b)):
                source = (f'Summary for experiment\n{condition_a}\n'
                          f'{prefix_a}python analyze.py -v "/media/absent/shared_*" -f {selection_a}\n'
                          f'{condition_b}\n'
                          f'{prefix_b}python analyze.py -v "/media/absent/shared_*" -f {selection_b}\n')
                _, refs, lists, *_ = self.parse_fixture(source)
                result = audit.classify(refs[0], refs[1], {li['list_id']: li for li in lists})
                self.assertIn('requires review', result)
        # A repeated control name with a different full source inventory also needs review.
        source = '''Summary for first experiment
ctrl>Kir-flat
python analyze.py -v "/media/absent/shared_*,/media/absent/one_*" -f 0-1
Summary for second experiment
ctrl>Kir-flat
python analyze.py -v "/media/absent/shared_*,/media/absent/two_*" -f 0-1
'''
        _, refs, lists, *_ = self.parse_fixture(source)
        self.assertIn('requires review', audit.classify(refs[0], refs[2], {li['list_id']: li for li in lists}))

    def test_identical_controls_require_matching_treatment_and_selection(self):
        source = '''Summary for first experiment
ctrl>Kir-flat 2023-02
python analyze.py -v "/media/absent/shared_*" -f 0-1
Summary for second experiment
ctrl>Kir-flat
python analyze.py -v "/media/absent/shared_*" -f 0-1
'''
        _, refs, lists, *_ = self.parse_fixture(source)
        lm = {li['list_id']: li for li in lists}
        self.assertIn('identical control inventory', audit.classify(*refs, lm))
        lists[1]['condition'] = 'ctrl>Kir-flat AR'
        self.assertIn('requires review', audit.classify(*refs, lm))

    def epg_fixture(self, notebook_expression, prefix='', notebook_value=None):
        """Use a real JSON notebook/AST scan, without relying on the repository input."""
        source = prefix + '''Large chamber tasks
Large chamber Kir exps (used in manuscript)
UAS>>CsC; 60D05-lexA (EPG); lexAop>>Kir
Flat-lower chamber
python analyze.py -v "/media/absent/c3[12]_*" -f 0-1
'''
        _, refs, lists, *_ = self.parse_fixture(source)
        symbol = 'EXTENDED_DATA_FIGURE_14I_EPG_VIDEO_GLOBS'
        code = f'{symbol} = {notebook_value or repr(notebook_expression)}\ncommand_flags = " -f 0-1"\n'
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'fixture.ipynb'
            path.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': [code]}]}))
            rows, _, unparsed = audit.notebook_scan([path], refs, audit.Resolver())
        return lists, rows, unparsed

    def test_epg_narrative_tracks_correction_and_shifted_source(self):
        lists, rows, unparsed = self.epg_fixture('/media/absent/c6[12]_*', prefix='\n' * 200)
        result, differences = audit.reconcile_epg(lists, rows, unparsed)
        self.assertEqual(result['status'], 'different')
        self.assertEqual(len(differences), 2)
        self.assertIn(f"L{lists[0]['first_line']}", result['summary'])
        self.assertEqual({d['fullpaper_line'] for d in differences}, {lists[0]['first_line']})
        # Correct a copy of the notebook literal; no production notebook is edited.
        corrected = [dict(r, normalized='/media/absent/c3[12]_*') for r in rows]
        result, differences = audit.reconcile_epg(lists, corrected, [])
        self.assertEqual(result['status'], 'matching')
        self.assertFalse(differences)
        self.assertIn('agree', result['summary'])
        self.assertNotIn('stale', result['summary'])
        self.assertNotIn('pending', result['summary'])

    def test_epg_missing_ambiguous_and_dynamic_sources_remain_unverified(self):
        lists, rows, unparsed = self.epg_fixture('/media/absent/c3[12]_*')
        self.assertEqual(audit.reconcile_epg([], rows, [])[0]['status'], 'unavailable')
        self.assertEqual(audit.reconcile_epg(lists, [], [])[0]['status'], 'unavailable')
        self.assertEqual(audit.reconcile_epg(lists * 2, rows, [])[0]['status'], 'ambiguous')
        more_rows = rows + [dict(r, cell_index=1) for r in rows]
        self.assertEqual(audit.reconcile_epg(lists, more_rows, [])[0]['status'], 'ambiguous')
        dynamic = self.epg_fixture('', notebook_value="'/media/absent/c3[12]_*' if flag else '/media/absent/other_*'")
        result, differences = audit.reconcile_epg(*dynamic)
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(differences)
        partial = self.epg_fixture('/media/absent/c3[12]_*,/media/absent/directory')
        result, differences = audit.reconcile_epg(*partial)
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(differences)

    def test_cli_report_updates_after_notebook_correction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'notebooks').mkdir()
            source = root / 'source.txt'
            source.write_text('\n' * 40 + '''Large chamber tasks
Large chamber Kir exps (used in manuscript)
UAS>>CsC; 60D05-lexA (EPG); lexAop>>Kir
Flat-lower chamber
python analyze.py -v "/media/absent/c3[12]_*" -f 0-1
''')
            nb_path = root / 'notebooks' / 'fixture.ipynb'
            checker = Path(audit.__file__).resolve()
            for camera, expected in [('c6', 'different'), ('c3', 'matching')]:
                code = f'EXTENDED_DATA_FIGURE_14I_EPG_VIDEO_GLOBS = "/media/absent/{camera}[12]_*"\n'
                nb_path.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': [code]}]}))
                source_before, notebook_before = source.read_bytes(), nb_path.read_bytes()
                subprocess.run([sys.executable, str(checker), '--source', str(source), '--out', 'out'],
                               cwd=root, check=True, capture_output=True, text=True)
                manifest = json.loads((root / 'out' / 'manifest.json').read_text())
                self.assertEqual(manifest['epg_comparison']['status'], expected)
                self.assertTrue(manifest['stats']['protected_inputs_unchanged'])
                report = (root / 'out' / 'report.html').read_text()
                self.assertIn(manifest['epg_comparison']['summary'], report)
                self.assertNotIn('stale notebook', report)
                self.assertNotIn('updates are pending', report)
                self.assertEqual(source.read_bytes(), source_before)
                self.assertEqual(nb_path.read_bytes(), notebook_before)


if __name__=='__main__':
    unittest.main()
