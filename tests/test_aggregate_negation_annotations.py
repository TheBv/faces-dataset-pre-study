import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'psrc'))
from main_utils.aggregate_negation_annotations import (
    aggregate, align_groups, decide, fit_model, posterior,
)
from main_utils.negation_agreement import read_conll


def proposal(name, cue, scope=()):
    return {'annotator': name, 'cue': list(cue), 'scope': list(scope), 'source_group_index': 0}


def group(cue, scope):
    return {'cue': [{'i': i} for i in cue], 'scope': [{'i': i} for i in scope]}


class AggregationTests(unittest.TestCase):
    def test_partial_multicue_and_separate_negations(self):
        proposals = [proposal('A', [1, 2]), proposal('B', [2]),
                     proposal('A', [5]), proposal('B', [5])]
        aligned = align_groups(proposals)
        self.assertEqual([g['cue_indices'] for g in aligned], [[1, 2], [5]])
        self.assertEqual(aligned[0]['alignment'], 'overlap_union')
        self.assertEqual(align_groups(list(reversed(proposals)))[0]['cue_indices'], [1, 2])

    def test_ambiguous_many_to_one_does_not_collapse_negations(self):
        aligned = align_groups([proposal('A', [1]), proposal('A', [2]), proposal('B', [1, 2])])
        self.assertEqual([g['cue_indices'] for g in aligned], [[1], [1, 2], [2]])
        self.assertTrue(all(g['ambiguous_alignment'] for g in aligned))

    def test_posterior_hand_calculation_and_missingness(self):
        model = {'prevalence': .2, 'annotators': {
            'A': {'sensitivity': .8, 'specificity': .9},
            'B': {'sensitivity': .7, 'specificity': .95}}}
        expected = (.2 * .8 * .3) / (.2 * .8 * .3 + .8 * .1 * .95)
        self.assertAlmostEqual(posterior({'A': 1, 'B': 0}, model), expected)
        self.assertAlmostEqual(posterior({'A': 1}, model), 2/3)
        self.assertEqual(decide({'A': 1}, model), 1)
        self.assertEqual(decide({'A': 0, 'B': 0}, model), 0)

    def test_model_convergence_and_no_single_rating_fit(self):
        units = [{'A': 0, 'B': 0}]*50 + [{'A': 1, 'B': 1}]*20 + [{'A': 1, 'B': 0}]*5
        model = fit_model(units, ['A', 'B', 'C'])
        self.assertTrue(model['converged'])
        self.assertEqual(model, fit_model(units + [{'A': 1}]*100, ['A', 'B', 'C']))
        self.assertEqual(model['annotators']['C']['sensitivity'], .9)
        self.assertEqual(model['training_ratings_per_annotator']['C'], 0)
        with self.assertRaises(ValueError):
            fit_model([{'A': 1}], ['A'])
        with self.assertRaises(ValueError):
            fit_model(units, ['A', 'B'], strength=float('nan'))

    def test_integration_split_isolation_provenance_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db, sample = root/'annotations.sqlite', root/'sample.csv'
            with sample.open('w') as f:
                writer = csv.DictWriter(f, fieldnames=['item_uuid', 'text', 'split'])
                writer.writeheader()
                writer.writerows({'item_uuid': sid, 'text': 'not at all fine', 'split': split}
                                 for sid, split in [('1', 'train'), ('2', 'train'), ('3', 'test'), ('4', 'val')])
            with sqlite3.connect(db) as conn:
                conn.execute('CREATE TABLE annotations (sentence_id TEXT, annotator TEXT, groups_json TEXT, no_negation INT, garbled INT)')
                conn.executemany('INSERT INTO annotations VALUES (?,?,?,?,?)', [
                    ('1', 'A', json.dumps([group([0], [1, 2])]), 0, 0),
                    ('1', 'B', json.dumps([group([0], [2, 3])]), 0, 0),
                    ('2', 'A', '[]', 0, 0), ('2', 'B', '[]', 0, 0),
                    ('3', 'A', json.dumps([group([0], [3])]), 0, 0),
                    ('3', 'B', '[]', 0, 0),
                    ('4', 'A', json.dumps([group([0], [2])]), 0, 1),
                    ('4', 'B', '[]', 0, 1)])
            before = hashlib.sha256(db.read_bytes()).hexdigest()
            report = aggregate(db, sample, root/'out')
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), before)
            self.assertEqual(report['counts']['sentences'], 3)
            self.assertEqual(report['sentences_with_no_usable_annotation'], ['4'])
            self.assertEqual(report['counts']['single_source_scope_groups'], 1)
            self.assertEqual(report['counts']['probabilistic_scope_groups'], 1)
            records = [json.loads(line) for line in (root/'out/reference.jsonl').read_text().splitlines()]
            test = next(r for r in records if r['split'] == 'test')
            self.assertEqual(test['groups'][0]['scope_indices'], [3])
            self.assertEqual(test['groups'][0]['scope_posterior_x'], [None]*4)
            self.assertEqual(test['cue_unanimous_indices'], [])
            for split in ('train', 'val', 'test', ''):
                for path in (root/'out'/split).glob('*.conll'):
                    task = path.name.split('.')[-2]
                    read_conll(path, task)
            aggregate(db, sample, root/'again')
            for path in (root/'out').rglob('*'):
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (root/'again'/path.relative_to(root/'out')).read_bytes())
            with sqlite3.connect(db) as conn:
                conn.execute('UPDATE annotations SET groups_json=? WHERE sentence_id="3" AND annotator="B"',
                             (json.dumps([group([0], [1, 2, 3])]),))
            changed = aggregate(db, sample, root/'changed')
            self.assertEqual(report['model'], changed['model'])


if __name__ == '__main__':
    unittest.main()
