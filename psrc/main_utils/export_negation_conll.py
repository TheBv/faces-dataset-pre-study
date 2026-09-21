"""Export saved negation annotations to plain CoNLL plus alignment sidecars.

Tokens retain the annotation UI's whitespace segmentation. Partial-word spans
become whole-token X labels. Scope blocks are emitted once per negation group;
negative sentences occur only in the cue dataset. No database writes are made.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def token_indices(entries: list, tokens: list[str]) -> set[int]:
    indices = set()
    for entry in entries:
        i = entry['i']
        if type(i) is not int or not 0 <= i < len(tokens):
            raise ValueError(f'Invalid token index: {entry!r}')
        start, end = entry.get('start', 0), entry.get('end', len(tokens[i]))
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(tokens[i])):
            raise ValueError(f'Invalid span for token {tokens[i]!r}: {entry!r}')
        indices.add(i)
    return indices


def write_dataset(path: Path, annotator: str, task: str, blocks: list) -> None:
    content = ''.join(''.join('\t'.join(row) + '\n' for row in b['rows']) + '\n'
                      for b in blocks)
    path.write_text(content, encoding='utf-8')
    metadata = {'version': 1, 'annotator': annotator, 'task': task,
                'sha256': hashlib.sha256(content.encode('utf-8')).hexdigest(),
                'blocks': [{k: v for k, v in b.items() if k != 'rows'} for b in blocks]}
    path.with_suffix(path.suffix + '.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def export(database: Path, sample: Path, output: Path, include_garbled=False) -> dict:
    sentences = {}
    with sample.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            sid = row['item_uuid']
            if sid in sentences:
                raise ValueError(f'Duplicate sentence ID in CSV: {sid}')
            sentences[sid] = row
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        annotations = conn.execute('SELECT * FROM annotations ORDER BY annotator, sentence_id').fetchall()
    datasets = defaultdict(lambda: {'cue': [], 'scope': [], 'excluded_garbled': []})
    for annotation in annotations:
        sid, name = annotation['sentence_id'], annotation['annotator']
        data = datasets[name]
        if sid not in sentences:
            raise ValueError(f'{name}: sentence {sid} is absent from CSV')
        if annotation['garbled'] and not include_garbled:
            data['excluded_garbled'].append(sid)
            continue
        tokens = sentences[sid]['text'].split()
        if not tokens:
            raise ValueError(f'Empty sentence: {sid}')
        groups = json.loads(annotation['groups_json'])
        if not isinstance(groups, list):
            raise ValueError(f'{name}/{sid}: groups must be a list')
        if annotation['no_negation'] and groups:
            raise ValueError(f'{name}/{sid}: no_negation conflicts with groups')
        all_cues, seen_cues = set(), set()
        for group in groups:
            cue = token_indices(group['cue'], tokens)
            scope = token_indices(group['scope'], tokens)
            if not cue or tuple(sorted(cue)) in seen_cues:
                raise ValueError(f'{name}/{sid}: empty or duplicate cue group')
            seen_cues.add(tuple(sorted(cue)))
            all_cues.update(cue)
            data['scope'].append({'sentence_id': sid, 'cue_indices': sorted(cue),
                                  'rows': [(t, 'X' if i in cue else 'O',
                                            'X' if i in scope else 'O')
                                           for i, t in enumerate(tokens)]})
        data['cue'].append({'sentence_id': sid,
                            'rows': [(t, 'X' if i in all_cues else 'O')
                                     for i, t in enumerate(tokens)]})
    # Validate all annotations before creating output files.
    output.mkdir(parents=True, exist_ok=True)
    report = {'database': str(database), 'sample': str(sample),
              'include_garbled': include_garbled, 'annotators': {}}
    used = set()
    for name, data in datasets.items():
        slug = re.sub(r'[^\w.-]+', '_', name).strip('.') or 'annotator'
        if slug.casefold() in used:
            raise ValueError(f'Annotator filename collision: {name!r}')
        used.add(slug.casefold())
        for task in ('cue', 'scope'):
            write_dataset(output / f'{slug}.{task}.conll', name, task, data[task])
        report['annotators'][name] = {
            'cue_sentences': len(data['cue']), 'scope_groups': len(data['scope']),
            'excluded_garbled': data['excluded_garbled']}
    (output / 'export_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / 'data/neg_annotations/annotations.sqlite')
    parser.add_argument('--sample', type=Path, default=ROOT / 'data/neg_samples/neg_anno_sample.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/neg_annotations/conll')
    parser.add_argument('--include-garbled', action='store_true')
    args = parser.parse_args()
    report = export(args.database, args.sample, args.output_dir, args.include_garbled)
    for name, counts in report['annotators'].items():
        print(f"{name}: {counts['cue_sentences']} cue sentences, "
              f"{counts['scope_groups']} scope groups, "
              f"{len(counts['excluded_garbled'])} garbled excluded")
    print(f'Output: {args.output_dir}')


if __name__ == '__main__':
    main()
