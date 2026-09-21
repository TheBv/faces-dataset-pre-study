"""Token-level nominal agreement for two or more annotator CoNLL files.

Default alignment uses export sidecars: sentence ID for cue; sentence ID plus
exact cue-token set for scope. Unmatched scope groups are reported, not padded
with O. Scope agreement is therefore conditional on an exactly shared cue.
Use --alignment position ONLY for already aligned external CoNLL files.

Alpha uses all units with >=2 ratings, with coincidence weighting for missing
ratings. Fleiss' kappa uses complete units; Cohen's kappa uses each pair's
intersection. Undefined coefficients (empty data or zero expected disagreement)
are JSON null. Formulas: https://www.nltk.org/_modules/nltk/metrics/agreement.html
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path


def read_conll(path: Path, task: str, alignment='metadata') -> dict:
    raw = path.read_bytes()
    blocks, current = [], []
    for line in raw.decode('utf-8').splitlines() + ['']:
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        if line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != (2 if task == 'cue' else 3) or any(
                value not in {'X', 'O'} for value in fields[1:]):
            raise ValueError(f'{path}: invalid CoNLL row: {line!r}')
        current.append(fields)
    name = path.stem
    if alignment == 'metadata':
        sidecar = path.with_suffix(path.suffix + '.json')
        if not sidecar.exists():
            raise ValueError(f'Missing {sidecar}; use --alignment position only if blocks already align')
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
        if meta['task'] != task or meta['sha256'] != hashlib.sha256(raw).hexdigest():
            raise ValueError(f'{path}: wrong task or stale alignment sidecar')
        if len(meta['blocks']) != len(blocks):
            raise ValueError(f'{path}: sidecar block count differs')
        name = meta['annotator']
        identities = [(b['sentence_id'], tuple(b['cue_indices']) if task == 'scope' else ())
                      for b in meta['blocks']]
    else:
        identities = [(str(i), ()) for i in range(len(blocks))]
    result = {}
    for key, rows in zip(identities, blocks):
        if key in result:
            raise ValueError(f'{path}: duplicate alignment key {key}')
        if task == 'scope' and alignment == 'metadata':
            cue = tuple(i for i, row in enumerate(rows) if row[1] == 'X')
            if key[1] != cue:
                raise ValueError(f'{path}: cue labels disagree with sidecar')
        result[key] = (tuple(row[0] for row in rows), tuple(row[-1] for row in rows),
                       tuple(row[1] for row in rows))
    return {'name': name, 'blocks': result}


def corrected(observed: float, expected: float):
    return (observed - expected) / (1 - expected) if expected < 1 else None


def pair_metrics(a: list[str], b: list[str]) -> dict:
    n = len(a)
    if len(b) != n:
        raise ValueError('Unequal rating lengths')
    counts = Counter(zip(a, b))
    observed = sum(x == y for x, y in zip(a, b)) / n if n else None
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2 if n else None
    denominator = ca['X'] + cb['X']
    return {'tokens': n, 'observed_agreement': observed,
            'cohen_kappa': corrected(observed, expected) if n else None,
            'krippendorff_alpha_nominal': nominal_alpha([list(u) for u in zip(a, b)]),
            'positive_dice_f1': 2 * counts['X', 'X'] / denominator if denominator else None,
            'confusion': {x + '/' + y: counts[x, y] for x in ('O', 'X') for y in ('O', 'X')}}


def nominal_alpha(units: list[list[str]]):
    units = [u for u in units if len(u) >= 2]
    total = sum(map(len, units))
    if not total:
        return None
    counts = Counter(v for u in units for v in u)
    observed = sum((len(u)**2 - sum(n*n for n in Counter(u).values())) / (len(u)-1)
                   for u in units) / total
    expected = (total**2 - sum(n*n for n in counts.values())) / (total * (total-1))
    return 1 - observed / expected if expected else None


def fleiss_kappa(units: list[list[str]], raters: int):
    complete = [u for u in units if len(u) == raters]
    if not complete:
        return None
    observed = sum(sum(n*(n-1) for n in Counter(u).values()) / (raters*(raters-1))
                   for u in complete) / len(complete)
    counts = Counter(v for u in complete for v in u)
    expected = sum((n / (len(complete)*raters))**2 for n in counts.values())
    return corrected(observed, expected)


def agreement(datasets: list[dict], task: str, alignment='metadata') -> dict:
    if len(datasets) < 2 or len({d['name'] for d in datasets}) != len(datasets):
        raise ValueError('Provide at least two distinct annotators')
    keys = set().union(*(d['blocks'] for d in datasets))
    if alignment == 'position' and any(set(d['blocks']) != keys for d in datasets):
        raise ValueError('Positional alignment requires equal block counts')
    units = []
    for key in sorted(keys):
        blocks = [d['blocks'][key] for d in datasets if key in d['blocks']]
        if any(b[0] != blocks[0][0] for b in blocks):
            raise ValueError(f'Token mismatch at {key}')
        if task == 'scope' and any(b[2] != blocks[0][2] for b in blocks):
            raise ValueError(f'Scope blocks have different cues at {key}')
        units.extend([list(values) for values in zip(*(b[1] for b in blocks))])
    pairs = []
    for a, b in itertools.combinations(datasets, 2):
        shared = sorted(a['blocks'].keys() & b['blocks'].keys())
        aa = [v for k in shared for v in a['blocks'][k][1]]
        bb = [v for k in shared for v in b['blocks'][k][1]]
        pairs.append({'annotators': [a['name'], b['name']], 'shared_blocks': len(shared),
                      'unmatched_blocks': {d['name']: len(d['blocks']) - len(shared) for d in (a, b)},
                      'exact_block_agreement': sum(a['blocks'][k][1] == b['blocks'][k][1]
                                                   for k in shared) / len(shared) if shared else None,
                      **pair_metrics(aa, bb)})
    return {'task': task, 'alignment': alignment,
            'scope_policy': 'conditional on exact cue-token agreement' if task == 'scope' else None,
            'blocks_per_annotator': {d['name']: len(d['blocks']) for d in datasets},
            'tokens_with_at_least_two_ratings': sum(len(u) >= 2 for u in units),
            'tokens_with_all_ratings': sum(len(u) == len(datasets) for u in units),
            'krippendorff_alpha_nominal': nominal_alpha(units),
            'fleiss_kappa_complete_units': fleiss_kappa(units, len(datasets)),
            'pairwise': pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', type=Path, nargs='+')
    parser.add_argument('--task', choices=('cue', 'scope'), required=True)
    parser.add_argument('--alignment', choices=('metadata', 'position'), default='metadata')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = agreement([read_conll(p, args.task, args.alignment) for p in args.files],
                       args.task, args.alignment)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
