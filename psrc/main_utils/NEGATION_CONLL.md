# Negation datasets and agreement

Run from the repository root (Python standard library only):

```bash
python psrc/main_utils/export_negation_conll.py
python psrc/main_utils/negation_agreement.py --task cue data/neg_annotations/conll/*.cue.conll --output data/neg_annotations/conll/cue_agreement.json
python psrc/main_utils/negation_agreement.py --task scope data/neg_annotations/conll/*.scope.conll --output data/neg_annotations/conll/scope_agreement.json
```

The exporter reads `data/neg_annotations/annotations.sqlite` read-only and joins
`annotations.sentence_id` to `data/neg_samples/neg_anno_sample.csv:item_uuid`.
Override paths with `--database`, `--sample`, and `--output-dir`.

Each annotator gets two UTF-8, tab-separated files, with blank lines between
blocks and no headers/comments:

- `NAME.cue.conll`: `token cue_label`, one block per saved sentence, marking the
  union of all cues. Saved empty groups are negative examples, even when the
  legacy `no_negation` flag is zero. Unannotated sentences are never negatives.
- `NAME.scope.conll`: `token cue_label scope_label`, one block per negation
  group, repeating the sentence for multiple negations. Only that group's cue
  and scope are marked. Negative sentences have no scope block.

`X` means annotated; `O` means outside. Tokenization is **exactly `text.split()`**,
matching the annotation UI; punctuation remains attached where it was attached
in the original text. Any nonempty subword annotation marks the entire token;
cue and scope may both be X on the same token. No linguistic inference is added.
Garbled rows are excluded unless `--include-garbled` is given. The summary lists
excluded IDs. Invalid IDs, indices, spans, or contradictory flags cause errors.
Re-running replaces the corresponding generated files.

Keep the `.conll.json` sidecars with the datasets. They contain annotator names,
sentence IDs, cue indices, and a checksum to reject stale metadata. The agreement
script uses sentence IDs rather than block positions or sentence text (which
can repeat). For independently supplied files without sidecars, use
`--alignment position` only when all blocks are already aligned; block counts,
tokens, and scope cues must agree. The positional mode cannot detect swapped
identical sentences.

Cue agreement compares token labels on shared annotated sentences. Scope
agreement compares scope labels only for **exactly matching cue-token sets in
the same sentence**. Group order does not matter. This is conditional scope
agreement, not end-to-end negation agreement. Unmatched blocks are reported;
they can reflect non-overlapping annotation assignments as well as cue
mismatches, and are not automatically counted as disagreements or filled with O.
Subword boundary agreement is not measured by this whole-token representation.

Reports include nominal Krippendorff's alpha (overall and per pair), pairwise
Cohen's kappa, observed token agreement, positive Dice/F1, O/X confusion counts,
and exact block agreement. Alpha uses units with at least two ratings and
coincidence weighting for missing ratings. Cohen uses each pair's intersection.
Fleiss' kappa is also reported on units rated by every supplied annotator.
Undefined statistics are `null`, including Fleiss' kappa when no units have all
raters. Scores are token-weighted, so long sentences contribute more.
Formula reference: [NLTK agreement implementation](https://www.nltk.org/_modules/nltk/metrics/agreement.html).

The current database has pairwise overlapping assignments, with no sentence
rated by all three annotators. Consequently the three-file reports have a
valid alpha and pairwise kappas, but no complete-unit Fleiss' kappa.

Run focused checks:

```bash
python -m unittest discover -s tests -p test_negation_conll.py
```

# Automatic consolidated reference

```bash
python psrc/main_utils/aggregate_negation_annotations.py
```

The output is `data/neg_annotations/aggregated/`. This is an automatically
consolidated reference, not manually adjudicated gold. The implementation uses
only the Python standard library. It reads the original database without
modifying it and retains original subword annotations in the JSONL provenance.

Outputs:

- `reference.cue.conll` and `reference.scope.conll`, with alignment sidecars.
- `train/`, `val/`, and `test/` contain the same exports restricted to existing
  CSV splits. No sentence is reassigned to another split.
- `reference.jsonl` records sentence IDs, splits, tokens, original annotations,
  cue support, group matching, final scope labels, decision source, model
  probabilities, and prior/initialization sensitivity per group.
- `aggregation_report.json` records input hashes, exclusion details, counts,
  fitted parameters, convergence, and sensitivity statistics.
- `baseline_cue_unanimous.cue.conll` provides the intersection of cue-token
  votes from usable annotators. With only one usable annotator, it retains that
  vote; it must not be interpreted as two-person agreement.
- `baseline_scope_union.scope.conll` and
  `baseline_scope_intersection.scope.conll` provide alternative scope decisions
  on the **same union cues and aligned groups**. Missing scopes do not count as O.
  These files also appear in each split directory. Use them to assess sensitivity
  of downstream model results; this script does not run downstream models.

## Cue fusion and group matching

The cue dataset marks the union of all cue tokens from usable human annotations.
A group retains its cue-to-scope link. Within each sentence, construct an overlap
graph of group proposals: two proposals from different annotators connect when
at least one cue-token index overlaps. An overlap component is fused only if it
contains at most one group per annotator. Its cue is the union of their indices.
This allows a partially missed multiword cue to be recovered.

If a component contains multiple groups from the same annotator, its matching is
ambiguous. Fall back to exact cue-set matching inside that component and retain
the distinct proposals. Flag all resulting groups and list the sentence ID in
the report. They may contain overlapping cues; they are alternative unresolved
human groupings, not a claim that every proposal is a distinct linguistic
negation. They are excluded from model fitting. Audit or exclude these flagged
sentences in an additional downstream sensitivity evaluation if relevant.
Disjoint cue fragments are not guessed to be one multiword cue. Subword span
boundaries are not inferred: aggregation is at whole-token resolution.

## Scope model and decision rule

Use a binary Dawid–Skene latent-label model with MAP-EM fitting. For annotator
`a`, sensitivity is `P(y_a=X | z=X)` and specificity is `P(y_a=O | z=O)`.
For a token's available ratings, posterior odds are:

```text
P(z=X | ratings) / P(z=O | ratings)
  = prevalence / (1-prevalence)
    * product_a [ P(y_a | z=X) / P(y_a | z=O) ]
```

Ratings are conditionally independent given the latent token label. The model
has no sequence transitions, lexical features, or contiguity constraint, so
scopes can remain discontinuous. This is a regularized token model, not Bayesian
sequence combination. Background on the base model:
[Dawid–Skene, 1979](https://doi.org/10.2307/2346806) and
[Crowd-Kit documentation](https://crowd-kit.readthedocs.io/en/latest/classification/).

Sensitivity and specificity each receive `Beta(1 + 0.9*s, 1 + 0.1*s)` priors,
where `s=10` by default. Their prior mode is 0.9; prevalence has `Beta(2,2)`.
The EM M-step uses expected sufficient counts plus the prior's `alpha-1` and
`beta-1` terms. Parameters start at accuracy 0.9 and prevalence 0.5. Iteration
stops when maximum parameter change is below `1e-8`, up to 1,000 iterations.
The report exposes convergence, parameters, and training coverage per annotator.
These are MAP point estimates; probabilities are conditional on those fitted
parameters, not fully Bayesian credible intervals or calibrated confidence.

By default parameters are fit **only on training-split groups with at least two
scope annotations and unambiguous cue alignment**, using all their token labels,
including agreements. Validation and test annotations never update parameters.
The same fixed parameters infer labels in all splits. `--fit-split all` is an
explicit transductive alternative and is recorded in the report. Do not describe
that alternative as train-only fitting. Fitting fails if no eligible units exist.
An annotator absent from eligible fitting data retains the prior rates and is
reported in warnings.

Final scope decisions:

1. If only one annotator provides the aligned cue, copy their entire scope.
   Record `single_source` and null model probabilities.
2. If scopes agree, preserve that scope (`agreed`).
3. Otherwise record `probabilistic`: preserve individual token agreements and
   decide disputed tokens using posterior `P(X) >= 0.5` (ties select X).

A missing cue produces **no scope rating**, never an all-O scope. Model
probabilities on shared groups are recorded even where consensus is preserved;
they need not equal 0 or 1 and do not override those preserved labels.

Sensitivity runs refit with half/double prior strength and initial accuracies
0.7/0.97. Reports count changed disputed-token decisions; per-group metadata
records their positions. All variants obey the same fitting-split restriction.
Zero changed labels is evidence of stability only for these particular settings,
not evidence that the labels are correct. Scope-union/intersection baseline
changes are also counted. No settings are selected using downstream test scores.

## Usability and reproducibility

Garbled annotation rows are excluded by default. If another usable annotation
exists, the sentence remains with that single source. If all annotations are
excluded, the sentence is omitted and listed. `--include-garbled` overrides this.
Saved empty groups remain negative cue examples; missing annotations never do.

Optional flags: `--database`, `--sample`, `--output-dir`, `--prior-strength`,
`--fit-split {train,all}`, `--include-garbled`. Runs are deterministic for identical
inputs/settings. Output files with matching names are overwritten. The JSONL
retains source groups, while the report hashes the annotation-table snapshot
and the exact CSV bytes. Keep these with the code version for reproducibility.

Use the counts and policy descriptions above in the methods section. Preserve
pre-aggregation agreement results separately. Union intentionally favors cue
recall and can admit false positives; single-source scopes and ambiguous group
matching remain limitations. Avoid presenting estimated annotator parameters as
independently measured human accuracy.

```bash
python -m unittest discover -s tests -p '*negation*.py'
```
