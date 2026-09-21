import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'psrc'))
from main_utils.export_negation_conll import export, token_indices
from main_utils.negation_agreement import agreement, read_conll, nominal_alpha, pair_metrics, fleiss_kappa


class AgreementTests(unittest.TestCase):
    def test_known_coefficients(self):
        a, b = list('XXOO'), list('XOOO')
        self.assertAlmostEqual(pair_metrics(a, b)['cohen_kappa'], .5)
        self.assertAlmostEqual(nominal_alpha(list(map(list, zip(a, b)))), 8/15)
        self.assertAlmostEqual(fleiss_kappa(list(map(list, zip(a, b))), 2), 7/15)
        # Singleton ratings must not affect alpha's marginals.
        self.assertAlmostEqual(nominal_alpha([['X','O'], ['O','O','X'], ['X']]), -1/3)
        self.assertIsNone(nominal_alpha([['O', 'O']]))
        self.assertIsNone(pair_metrics([], [])['cohen_kappa'])

    def test_scope_alignment_uses_cues_not_group_order(self):
        tokens = ('not', 'never')
        first = (tokens, ('O', 'X'), ('X', 'O'))
        second = (tokens, ('X', 'O'), ('O', 'X'))
        a = {'name': 'a', 'blocks': {('s', (0,)): first, ('s', (1,)): second}}
        b = {'name': 'b', 'blocks': {('s', (1,)): second, ('s', (0,)): first}}
        report = agreement([a, b], 'scope')
        self.assertEqual(report['pairwise'][0]['cohen_kappa'], 1)
        del b['blocks'][('s', (1,))]
        report = agreement([a, b], 'scope')
        self.assertEqual(report['pairwise'][0]['unmatched_blocks']['a'], 1)
        self.assertEqual(report['tokens_with_at_least_two_ratings'], 2)

    def test_bad_spans(self):
        for entry in ({'i': -1}, {'i': 1}, {'i': 0, 'start': 2, 'end': 1}):
            with self.assertRaises(ValueError):
                token_indices([entry], ['word'])

    def test_export_and_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db, sample, out = root/'annotations.sqlite', root/'sample.csv', root/'out'
            with sample.open('w') as f:
                writer = csv.DictWriter(f, fieldnames=['item_uuid', 'text'])
                writer.writeheader()
                writer.writerows([{'item_uuid': str(i), 'text': text} for i, text in enumerate(
                    ['unhappy never fine.', 'Fine.', 'garbled'])])
            groups = [{'cue': [{'i': 0, 'start': 0, 'end': 2}], 'scope': [{'i': 0}]},
                      {'cue': [{'i': 1}], 'scope': [{'i': 2}]}]
            with sqlite3.connect(db) as conn:
                conn.execute('CREATE TABLE annotations (sentence_id TEXT, annotator TEXT, groups_json TEXT, no_negation INT, garbled INT)')
                conn.executemany('INSERT INTO annotations VALUES (?,?,?,?,?)', [
                    ('0', 'A', json.dumps(groups), 0, 0), ('1', 'A', '[]', 0, 0),
                    ('2', 'A', '[]', 0, 1)])
            report = export(db, sample, out)
            self.assertEqual(report['annotators']['A']['cue_sentences'], 2)
            self.assertEqual((out/'A.cue.conll').read_text(),
                             'unhappy\tX\nnever\tX\nfine.\tO\n\nFine.\tO\n\n')
            scope = read_conll(out/'A.scope.conll', 'scope')
            self.assertEqual(len(scope['blocks']), 2)
            self.assertEqual(scope['blocks'][('0', (0,))][1], ('X', 'O', 'O'))
            with (out/'A.scope.conll').open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(ValueError, 'stale'):
                read_conll(out/'A.scope.conll', 'scope')
            export(db, sample, out, include_garbled=True)
            self.assertEqual(len(read_conll(out/'A.cue.conll', 'cue')['blocks']), 3)


if __name__ == '__main__':
    unittest.main()
