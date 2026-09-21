"""Recall-oriented cue union and regularized Dawid-Skene scope aggregation.

Standard-library implementation. Parameters are MAP point estimates, not a
fully Bayesian posterior. Defaults fit only shared-cue training examples.
See NEGATION_CONLL.md for equations, matching policy, and limitations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3

try:
    from .export_negation_conll import ROOT, token_indices, write_dataset
except ImportError:
    from export_negation_conll import ROOT, token_indices, write_dataset


def align_groups(proposals):
    """Merge overlap components only if each annotator contributes <=1 group.

    Ambiguous many-to-one components fall back to exact cue-set matching.
    Thus separate negations from the same annotator are never collapsed.
    """
    pending = set(range(len(proposals)))
    result = []
    while pending:
        component = {min(pending)}
        pending -= component
        frontier = list(component)
        while frontier:
            i = frontier.pop()
            neighbors = {j for j in pending if proposals[i]['annotator'] != proposals[j]['annotator']
                         and set(proposals[i]['cue']) & set(proposals[j]['cue'])}
            pending -= neighbors
            component |= neighbors
            frontier.extend(sorted(neighbors))
        members = [proposals[i] for i in sorted(component)]
        ambiguous = len({p['annotator'] for p in members}) < len(members)
        partitions = defaultdict(list)
        for p in members:
            partitions[tuple(p['cue']) if ambiguous else ()].append(p)
        for group in partitions.values():
            result.append({'cue_indices': sorted({i for p in group for i in p['cue']}),
                           'sources': group, 'ambiguous_alignment': ambiguous,
                           'alignment': 'exact_only_ambiguous' if ambiguous else (
                               'exact' if len({tuple(p['cue']) for p in group}) == 1 else 'overlap_union')})
    return sorted(result, key=lambda g: tuple(g['cue_indices']))


def posterior(ratings, model):
    """P(scope=X | observed ratings, fitted point parameters)."""
    log_odds = math.log(model['prevalence'] / (1 - model['prevalence']))
    for name, value in ratings.items():
        rates = model['annotators'][name]
        sensitivity, specificity = rates['sensitivity'], rates['specificity']
        positive = sensitivity if value else 1 - sensitivity
        negative = 1 - specificity if value else specificity
        log_odds += math.log(positive / negative)
    return 1 / (1 + math.exp(-log_odds)) if log_odds >= 0 else (
        math.exp(log_odds) / (1 + math.exp(log_odds)))


def fit_model(units, names, strength=10.0, initial_accuracy=.9, max_iter=1000, tolerance=1e-8):
    """MAP-EM with Beta(1+.9*s, 1+.1*s) priors on sensitivity/specificity.

    Prevalence has Beta(2,2). Only multiply rated units enter fitting. No hard
    consensus labels are assumed in fitting; consensus is preserved at export.
    Compress equal response patterns to avoid repeated token calculations.
    """
    if not math.isfinite(strength) or strength <= 0 or not 0 < initial_accuracy < 1:
        raise ValueError('Prior strength must be positive; initialization must be inside (0,1)')
    patterns = Counter(tuple(sorted(u.items())) for u in units if len(u) >= 2)
    if not patterns:
        raise ValueError('No multiply annotated scope tokens in fitting split')
    patterns = [(dict(p), count) for p, count in sorted(patterns.items())]
    total = sum(count for _, count in patterns)
    model = {'prevalence': .5, 'annotators': {
        name: {'sensitivity': initial_accuracy, 'specificity': initial_accuracy} for name in sorted(names)}}
    convergence = False
    for iteration in range(1, max_iter + 1):
        sufficient = {name: [0., 0., 0., 0.] for name in names}
        positive_total = 0.
        for ratings, count in patterns:
            q = posterior(ratings, model)
            positive_total += count * q
            for name, value in ratings.items():
                stats = sufficient[name]
                stats[0] += count * q * value
                stats[1] += count * q
                stats[2] += count * (1-q) * (1-value)
                stats[3] += count * (1-q)
        updated = {'prevalence': (positive_total + 1) / (total + 2), 'annotators': {}}
        delta = abs(updated['prevalence'] - model['prevalence'])
        for name, (tp, pos, tn, neg) in sufficient.items():
            rates = {'sensitivity': (tp + .9*strength)/(pos + strength),
                     'specificity': (tn + .9*strength)/(neg + strength)}
            delta = max(delta, *(abs(rates[k] - model['annotators'][name][k]) for k in rates))
            updated['annotators'][name] = rates
        model = updated
        if delta < tolerance:
            convergence = True
            break
    model.update({'converged': convergence, 'iterations': iteration, 'max_parameter_change': delta,
                  'training_tokens': total, 'response_patterns': len(patterns),
                  'prior_strength': strength, 'prior_accuracy_mode': .9,
                  'initial_accuracy': initial_accuracy, 'tolerance': tolerance,
                  'max_iterations': max_iter,
                  'training_ratings_per_annotator': {
                      name: sum(count for ratings, count in patterns if name in ratings) for name in sorted(names)}})
    return model


def load_sentences(database, sample, include_garbled=False):
    samples = {}
    with sample.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['item_uuid'] in samples:
                raise ValueError(f"Duplicate CSV ID: {row['item_uuid']}")
            samples[row['item_uuid']] = row
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute('SELECT * FROM annotations ORDER BY sentence_id, annotator')]
    sentences = {}
    excluded = []
    for row in rows:
        sid, name = row['sentence_id'], row['annotator']
        if sid not in samples:
            raise ValueError(f'Sentence missing from CSV: {sid}')
        if row['garbled'] and not include_garbled:
            excluded.append({'sentence_id': sid, 'annotator': name})
            continue
        source = samples[sid]
        tokens = source['text'].split()
        if not tokens:
            raise ValueError(f'Empty sentence: {sid}')
        if source['split'] not in {'train', 'val', 'test'}:
            raise ValueError(f"Unknown split: {source['split']}")
        sentence = sentences.setdefault(sid, {
            'sentence_id': sid, 'split': source['split'], 'tokens': tokens,
            'annotations': [], 'proposals': []})
        if any(a['annotator'] == name for a in sentence['annotations']):
            raise ValueError(f'Duplicate annotation: {sid}/{name}')
        groups = json.loads(row['groups_json'])
        if not isinstance(groups, list) or (row['no_negation'] and groups):
            raise ValueError(f'Invalid groups/no_negation: {sid}/{name}')
        sentence['annotations'].append({'annotator': name, 'groups': groups,
                                         'garbled': bool(row['garbled'])})
        seen = set()
        for index, group in enumerate(groups):
            cue = sorted(token_indices(group['cue'], tokens))
            scope = sorted(token_indices(group['scope'], tokens))
            if not cue or tuple(cue) in seen:
                raise ValueError(f'Empty/duplicate cue: {sid}/{name}')
            seen.add(tuple(cue))
            sentence['proposals'].append({'annotator': name, 'source_group_index': index,
                                           'cue': cue, 'scope': scope})
    for sentence in sentences.values():
        sentence['groups'] = align_groups(sentence.pop('proposals'))
    return list(sentences.values()), excluded, hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ratings_for(group, length):
    sources = [(p['annotator'], set(p['scope'])) for p in group['sources']]
    return [{name: int(i in scope) for name, scope in sources} for i in range(length)]


def decide(ratings, model):
    values = set(ratings.values())
    if len(values) == 1:
        return next(iter(values))
    return int(posterior(ratings, model) >= .5)


def aggregate(database, sample, output, include_garbled=False, fit_split='train', strength=10.):
    sentences, excluded, source_hash = load_sentences(database, sample, include_garbled)
    names = sorted({a['annotator'] for s in sentences for a in s['annotations']})
    training = [r for s in sentences if fit_split == 'all' or s['split'] == fit_split
                for g in s['groups'] if len(g['sources']) >= 2 and not g['ambiguous_alignment']
                for r in ratings_for(g, len(s['tokens']))]
    model = fit_model(training, names, strength)
    variants = {'weaker_prior': fit_model(training, names, strength/2),
                'stronger_prior': fit_model(training, names, strength*2),
                'initial_accuracy_0.7': fit_model(training, names, strength, .7),
                'initial_accuracy_0.97': fit_model(training, names, strength, .97)}
    counts = Counter()
    changes = Counter()
    baseline_changes = Counter()
    split_counts = defaultdict(Counter)
    records = []
    for s in sentences:
        tokens, groups = s['tokens'], s['groups']
        union_cues = sorted({i for g in groups for i in g['cue_indices']})
        cue_votes = {a['annotator']: set() for a in s['annotations']}
        for g in groups:
            for source in g['sources']:
                cue_votes[source['annotator']].update(source['cue'])
        intersection = set.intersection(*cue_votes.values())
        counts['sentences'] += 1
        counts['positive_sentences'] += bool(union_cues)
        counts['single_annotator_sentences'] += len(cue_votes) == 1
        counts['cue_union_tokens'] += len(union_cues)
        if len(cue_votes) >= 2:
            counts['cue_tokens_added_beyond_unanimity'] += len(set(union_cues) - intersection)
        split_counts[s['split']]['sentences'] += 1
        for gi, group in enumerate(groups):
            ratings = ratings_for(group, len(tokens))
            scopes = {tuple(p['scope']) for p in group['sources']}
            decision = ('single_source' if len(group['sources']) == 1 else
                        'agreed' if len(scopes) == 1 else 'probabilistic')
            counts[decision + '_scope_groups'] += 1
            counts['scope_groups'] += 1
            counts['ambiguous_alignment_groups'] += group['ambiguous_alignment']
            counts['overlap_union_groups'] += group['alignment'] == 'overlap_union'
            split_counts[s['split']]['scope_groups'] += 1
            labels = [decide(r, model) for r in ratings]
            disputed = [i for i, r in enumerate(ratings) if len(set(r.values())) > 1]
            counts['disputed_scope_tokens'] += len(disputed)
            counts['single_source_scope_tokens'] += len(tokens) if decision == 'single_source' else 0
            probabilities = [posterior(r, model) if len(r) >= 2 else None for r in ratings]
            counts['low_confidence_disputed_scope_tokens'] += sum(
                .4 <= probabilities[i] <= .6 for i in disputed)
            sensitivity = {}
            for name, alternate in variants.items():
                changed = [i for i in disputed if decide(ratings[i], alternate) != labels[i]]
                sensitivity[name] = changed
                changes[name] += len(changed)
            for method, operation in [('scope_union', max), ('scope_intersection', min)]:
                baseline_changes[method] += sum(labels[i] != operation(r.values()) for i, r in enumerate(ratings))
            group.update({'group_index': gi, 'scope_indices': [i for i, value in enumerate(labels) if value],
                          'scope_decision': decision, 'disputed_scope_indices': disputed,
                          'scope_posterior_x': probabilities,
                          'low_confidence_disputed_indices': [i for i in disputed if .4 <= probabilities[i] <= .6],
                          'sensitivity_changed_indices': sensitivity})
        records.append({**s, 'cue_union_indices': union_cues,
                        'cue_unanimous_indices': sorted(intersection),
                        'cue_support': {str(i): sorted(name for name, cues in cue_votes.items() if i in cues)
                                        for i in union_cues}})
    output.mkdir(parents=True, exist_ok=True)
    for split in ('all', 'train', 'val', 'test'):
        selected = [s for s in records if split == 'all' or s['split'] == split]
        destination = output if split == 'all' else output / split
        destination.mkdir(parents=True, exist_ok=True)
        cue_blocks, scope_blocks = [], []
        union_blocks, intersection_blocks, unanimous_cue_blocks = [], [], []
        for s in selected:
            sid, tokens = s['sentence_id'], s['tokens']
            for blocks, indices in [(cue_blocks, s['cue_union_indices']),
                                    (unanimous_cue_blocks, s['cue_unanimous_indices'])]:
                blocks.append({'sentence_id': sid,
                               'rows': [(t, 'X' if i in indices else 'O') for i, t in enumerate(tokens)]})
            for g in s['groups']:
                scopes = [set(p['scope']) for p in g['sources']]
                for blocks, indices in [(scope_blocks, g['scope_indices']),
                                        (union_blocks, set.union(*scopes)),
                                        (intersection_blocks, set.intersection(*scopes))]:
                    blocks.append({'sentence_id': sid, 'cue_indices': g['cue_indices'],
                                   'rows': [(t, 'X' if i in g['cue_indices'] else 'O',
                                             'X' if i in indices else 'O') for i, t in enumerate(tokens)]})
        for filename, task, blocks in [('reference.cue', 'cue', cue_blocks),
                                       ('reference.scope', 'scope', scope_blocks),
                                       ('baseline_scope_union.scope', 'scope', union_blocks),
                                       ('baseline_scope_intersection.scope', 'scope', intersection_blocks),
                                       ('baseline_cue_unanimous.cue', 'cue', unanimous_cue_blocks)]:
            write_dataset(destination / (filename + '.conll'), filename, task, blocks)
    report = {'method': 'cue union + regularized binary Dawid-Skene MAP-EM scope aggregation',
              'version': 1, 'fit_split': fit_split, 'include_garbled': include_garbled,
              'ambiguous_alignment_sentence_ids': [s['sentence_id'] for s in records
                  if any(g['ambiguous_alignment'] for g in s['groups'])],
              'counts': dict(counts), 'splits': {k: dict(v) for k, v in split_counts.items()},
              'excluded_garbled_annotations': excluded,
              'sentences_with_no_usable_annotation': sorted({r['sentence_id'] for r in excluded} -
                                                          {s['sentence_id'] for s in records}),
              'source_annotation_rows_sha256': source_hash,
              'sample_sha256': hashlib.sha256(sample.read_bytes()).hexdigest(),
              'model': model, 'sensitivity_models': variants,
              'sensitivity_changed_scope_tokens': dict(changes),
              'baseline_changed_scope_tokens': dict(baseline_changes),
              'warnings': ([f'Model did not converge: {name}' for name, m in {'primary': model, **variants}.items()
                            if not m['converged']] +
                           [f'No fitting ratings for annotator: {name}' for name, n in
                            model['training_ratings_per_annotator'].items() if n == 0]),
              'limitations': ['Model probabilities are conditional on MAP parameters, not calibrated confidence.',
                              'Single-source scopes are copied; their posterior is null.',
                              'Ambiguous cue components retain exact-set proposals and can have overlapping cues.',
                              'Token decisions preserve unanimous votes; only disagreements use the model.',
                              'Cue/scope subword boundaries are preserved in sources, not inferred in the reference.']}
    (output / 'aggregation_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'reference.jsonl').write_text(''.join(json.dumps(s, ensure_ascii=False, allow_nan=False) + '\n'
                                                  for s in records), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / 'data/neg_annotations/annotations.sqlite')
    parser.add_argument('--sample', type=Path, default=ROOT / 'data/neg_samples/neg_anno_sample.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/neg_annotations/aggregated')
    parser.add_argument('--include-garbled', action='store_true')
    parser.add_argument('--fit-split', choices=('train', 'all'), default='train')
    parser.add_argument('--prior-strength', type=float, default=10.)
    args = parser.parse_args()
    report = aggregate(args.database, args.sample, args.output_dir, args.include_garbled,
                       args.fit_split, args.prior_strength)
    print(json.dumps({k: report[k] for k in ('counts', 'sensitivity_changed_scope_tokens', 'warnings')}, indent=2))
    print(f'Output: {args.output_dir}')


if __name__ == '__main__':
    main()
