import csv
import tempfile
import unittest
from pathlib import Path

from scripts.export_reward_pi_difference_graphpad import export_differences


class RewardPiDifferenceExportTest(unittest.TestCase):
    def test_paired_subtraction_exclusions_and_unequal_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = ('__calculated__ reward PI by bucket:\n'
                      'video,fly,training 2 last SB (exp),training 2 last SB (yok)\n')
            first, second = root / 'first.csv', root / 'second.csv'
            first.write_text(header + 'a,0,0.535,0.172\na,1,nan,0.2\na,2,0.1,nan\na,3,-0.2,0.3\n\n')
            second.write_text(header + 'b,0,0.7,-0.1\n\n')
            out, audit = root / 'out.csv', root / 'audit.csv'
            self.assertEqual(export_differences([('Ctrl', first), ('AR', second)], out, audit),
                             [('Ctrl', 2, 2), ('AR', 1, 0)])
            with out.open() as f:
                self.assertEqual(list(csv.reader(f)), [['Ctrl', 'AR'], ['0.363', '0.8'], ['-0.5', '']])
            with audit.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)
            self.assertEqual([r['included'] for r in rows], ['True', 'False', 'False', 'True', 'True'])
            self.assertEqual(rows[0]['fly'], '0')
            self.assertEqual(rows[0]['training 2 last SB (yok)'], '0.172')

    def test_rejects_duplicate_and_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.csv'
            for body in [
                'video,fly,training 2 last SB (exp)\na,0,0.2\n',
                'video,fly,training 2 last SB (exp),training 2 last SB (yok)\na,0,0.2,0.1\na,0,0.4,0.1\n',
            ]:
                source.write_text('__calculated__ reward PI by bucket:\n' + body)
                with self.assertRaises(ValueError):
                    export_differences([('Ctrl', source)], root / 'out.csv', root / 'audit.csv')


if __name__ == '__main__':
    unittest.main()
