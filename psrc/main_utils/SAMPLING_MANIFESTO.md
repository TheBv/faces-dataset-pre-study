# Sampling Manifesto

## Purpose

This procedure creates a reproducible sample of 1,601 sentences from 27
interviews for subsequent manual negation annotation. Sampling is independent
of whether a sentence contains negation. Both `Interviewer` and `Interviewee`
speech are eligible. The sampled data are partitioned by complete interview so
that no interview—and therefore no sentence from an interview—can occur in
more than one dataset split.

## Source and sentence construction

The database query returns chronologically ordered dialogue turns formed from
consecutive audio chunks attributed to the same speaker role. These turns are
not treated as sentences.

Before sentence segmentation, whitespace-only turns are removed. Turns of the
same role that consequently become adjacent are joined in their original
order. A conservative deterministic rule may also reconnect an unfinished
speaker-A fragment across one intervening speaker-B turn when:

1. A's first fragment has no sentence-final `.`, `!`, `?`, or `…`;
2. B's interruption contains at most eight non-punctuation tokens; and
3. A's return provides evidence of continuation: it starts with a lowercase
   letter, A ends in a comma or dash, or B is a recognized backchannel and A
   ends in a predefined linking/function word. A return after an unambiguous
   unfinished function word can also be reconnected within the same eight-token
   interruption limit.

The joined text is parsed with the German spaCy model `de_core_news_sm`, but a
parser boundary is not accepted blindly. Explicit transcript punctuation is
preserved, abbreviations are protected, and unsupported parser-only boundaries
are merged when their left side ends in incomplete syntax. Conversely, a
conservative set of question, subject, and discourse restarts can recover a
missing boundary in an ASR run-on when the preceding clause is syntactically
complete. Boundaries inside unbalanced brackets are not accepted.

Candidates then pass a deterministic, negation-independent quality gate. Its
first stage is a mandatory sentence-form requirement that every eligible
candidate must satisfy:

1. at least four punctuation-excluding words;
2. a sentence-initial capital letter, ignoring a leading quotation mark,
   bracket, or dash; a candidate that begins with a lowercase letter or with a
   digit is rejected; and
3. a terminal `.`, `!`, or `?`, optionally followed by a closing quotation mark
   or bracket, and not a truncating `...` or `…`.

The second stage removes unbalanced bracket spans, subtitle-credit artifacts
(including an artifact prefix attached to spoken content), implausibly long
spans, and remaining incomplete syntactic tails such as a clause ending in a
preposition, conjunction, or determiner. The hard length limit is 120 words.
These checks depend only on sentence form, not on negation cues or labels. A
candidate is reported under the first rule it violates, so the recorded
exclusion counts are disjoint. Conversational ellipses are no longer exempt:
short responses, bare numeric or nominal answers, greetings, and Likert-scale
responses are eligible only when they also meet the mandatory form
requirement — `Trifft eher nicht zu.` remains eligible, while `Ja.` and
`Sehr schön.` do not.

Thus, one database turn may produce several sentences, and an interrupted
sentence may draw on more than one turn. Every retained sentence keeps its
speaker role, role-specific sentence index, source-turn provenance,
role-specific source-turn indices, and, where applicable, the IDs of
intervening turns. Automatically reconstructed cross-turn sentences and every
candidate without a detected finite verb are marked `review_required=True`;
because the mandatory form requirement already guarantees terminal punctuation
and four or more words, unpunctuated retained sentences no longer exist.
Turn IDs are taken from the query result when present; otherwise, stable
one-based positions in the interview's returned turn array are used. Sentence
segmentation and cross-turn reconstruction remain automatic rather than
gold-standard manual annotation.

## Stratified sampling

The final sampled/output unit is one automatically segmented sentence. Audio
chunks are preprocessing units. Dialogue turns are reconstruction, provenance,
position, and quota-allocation units, but are not emitted as sentence records.
The effective hierarchy is:

| Level | Grouping variable | Function in sampling |
| --- | --- | --- |
| 1 | Interview | Prevents long interviews from dominating the corpus. |
| 2 | Speaker identity | Targets equal recurring-interviewer totals and guarantees every distinct speaker at least 10 rows, or its complete eligible pool when that pool holds fewer than 10 sentences. |
| 3 | Speaker role within interview | Preserves natural role proportions and role coverage. |
| 4 | Position in the role-specific turn sequence | Covers early-to-late dialogue regions. |
| 5 | Individual turn | Spreads each positional quota across turns before selecting sentences. |

The chronological turn sequence is therefore used directly in sampling. For
each interview-role participant stream, cleaned non-empty turns receive
zero-based role-specific indices. Those turn indices define the five position
strata. Sentences inherit the stratum of their primary source turn. For a
sentence reconstructed across two turns, the primary turn is the earlier turn
where the sentence begins; all contributing turns remain in its provenance.
Within an interview, the interview-role combination represents the
conversational participant available from the query.

One NumPy random generator is initialized once for the complete operation:

```python
rng = np.random.default_rng(42)
```

Sampling is performed without replacement at five levels:

1. **Interview.** The target is divided approximately equally across all 27
   interviews. With sufficient capacity, 19 interviews receive 59 sentences
   and eight receive 60; the seeded generator selects the larger quotas. If
   an interview lacks enough sentences, all available sentences are used and
   the deficit is redistributed approximately equally among interviews with
   remaining capacity. Consequently, long interviews cannot dominate merely
   because they contain more sentences.
2. **Speaker identity.** Each interviewee is unique to one interview, so the
   interview ID is also used as that interviewee's speaker ID. Interviewer IDs
   are obtained by applying `get_mapping()` to the experiment/interview ID;
   the mapping contains three recurring interviewers. After preliminary role
   allocation, interviewer quotas initially target the same corpus-level count
   for all three recurring interviewer IDs. Adjustment remains within fixed
   interview quotas and available role capacities and minimizes deviation from
   preliminary proportional quotas. If the three feasible count intervals do
   not overlap, each interviewer instead receives the closest feasible total
   around the midpoint of the narrowest interval gap. Exact equality is a
   preferred constraint, not stronger than the requirements of 1,601
   unique-enough rows, equal interview representation, and minimum speaker
   coverage. Every distinct speaker identity must receive at least 10 sampled
   sentences when at least 10 eligible sentences are available.
3. **Speaker role.** Within each interview, its quota is allocated between
   `Interviewer` and `Interviewee` approximately in proportion to their
   available sentence counts. For interview quota `q`, interviewer count `n_I`,
   and interviewee count `n_E`, the preliminary interviewer quota is
   `round(q × n_I / (n_I + n_E))`; the interviewee receives the remainder.
   For example, 120 interviewer and 180 interviewee sentences with `q=59`
   yield approximately 24 and 35 sampled sentences. When feasible, each
   available role receives at least 10 sentences. A role with fewer than 10
   available sentences is fully sampled when the interview quota permits, with
   the remainder assigned to the other role. No artificial 50:50 role balance
   is imposed. The 10-sentence role minimum is retained while balancing the
   recurring interviewers whenever the constraints are jointly feasible. The
   minimum is never relaxed for validation or test interviews. Preliminary
   recurring-interviewer balancing may relax training-interview role floors,
   but final validation enforces at least 10 rows per distinct speaker. If
   similarity deduplication requires role redistribution, the final fallback
   initially reserves 10 rows for both roles before filling the flexible
   remainder. The subsequent recurring-interviewer rebalance may lower only a
   training interview's interviewer-role count below 10. It never lowers its
   unique interviewee below 10, never lowers either held-out role below 10, and
   retains interviewer-role presence and all five non-empty position strata.
4. **Turn position within role.** The chronological cleaned-turn indices for
   each interview-role combination are divided into five sequential strata
   with `np.array_split`: first 20%, 20–40%, 40–60%, 60–80%, and final 20%.
   Eligible sentences are assigned according to the turn where they begin.
   The role quota is divided approximately equally across these strata.
   Remainder strata are selected by the same seeded generator. Insufficient
   stratum capacity is redistributed only within that speaker role.
5. **Individual turn, then sentence.** Within each position stratum, its quota
   is divided approximately equally across turns that contain eligible
   sentences, subject to each turn's sentence capacity. Candidate sentences
   from all turn cells are then selected jointly under the corpus-wide
   similarity constraint described below.

This fifth level prevents a long, multi-sentence turn from dominating a
position stratum. Selection never uses negation cues, annotation labels, a
negation lexicon, or the number of negations. Sentence surface form is used
only for the explicit duplicate-control constraint.

## Corpus-wide duplicate control

Before the final draw, every eligible sentence from every interview and both
speaker roles is assigned to a surface-similarity cluster. Normalisation uses
Unicode NFKC, case-folding, punctuation removal, and whitespace collapse. Two
sentences with identical normalised forms belong to the same cluster,
irrespective of interview, speaker ID, or role. This includes recurring
scripted wording such as a repeated interview question.

Near-duplicate clustering is available but disabled in the recorded run:
`near_duplicate_similarity_threshold` is `1.0`, so only identical normalised
forms are merged. Below `1.0`, two sentences are additionally merged when both
contain at least six tokens, share a consecutive three-token sequence, and
their normalised character sequences obtain a `difflib.SequenceMatcher` ratio
of at least the threshold; connected matches are merged transitively. The
threshold and minimum length are saved in the sampling summary and can be
changed through command-line options.

The threshold is a research parameter with a measured cost. Under the
mandatory sentence-form requirement, the eligible pool forms 2,152 clusters at
`1.0` but only 1,685 at `0.92`, because the interviewer stream consists largely
of the same scripted questions in slightly varying transcription. The
achievable sample size follows: about 1,100 rows at `0.92` against 1,601 at
`1.0`. The recorded run therefore keeps the strict sentence-form gate and
accepts near-identical—but not identical—wording rather than shrinking the
dataset by a third.

The 1,601 sentences are selected jointly through a capacity-constrained
matching procedure that permits at most one sentence from each similarity
cluster. Thus, a question repeated verbatim across interviews—or by different
speakers—can occur at most once in the final dataset; at a threshold below
`1.0` this also covers near-identical wording. Other sentences are selected to
fill the vacated quotas; rows are not simply deleted after sampling.

The matcher first attempts to preserve every allocated turn and position
quota. If global duplicate clusters make those exact turn quotas infeasible,
it successively relaxes constraints in this order:

1. redistribute among turns within the same position stratum;
2. redistribute position and turn quotas within the same interview-role;
3. preserve each 59/60-sentence interview quota and a 10-sentence floor for
   both roles during matching, but allow the remaining role allocation to adapt
   to unique-cluster availability.

Dataset splits and interview quotas are never relaxed. Exact recurring-
interviewer equality and proportional role quotas may be relaxed only at the
last stage. A duplicate-safe within-interview swap pass then minimizes squared
imbalance among recurring interviewer totals without changing interview
quotas, unique-interviewee or held-out role floors, positional coverage, split
membership, or cluster uniqueness. The interviewer count in a training
interview may fall as low as five—one sentence in each position stratum—when
needed to improve corpus-level speaker balance. If 1,601
unique-enough sentences still cannot be selected, the procedure stops with an
error instead of emitting a smaller or duplicate-containing sample. Feasibility
is not monotone in the requested size, because the size determines the
per-interview quota: the recorded 1,601 is feasible while 1,600 is not. The
summary records the activated relaxation level, matching-flow capacities, and
interviewer counts before and after rebalancing. Each selected row stores its
stable
`similarity_cluster_id` and the number of source sentences represented by that
cluster, so the loss of utterance multiplicity remains visible for analysis.
The output also gives the number of source interviews represented by the
cluster. An exact normalised form occurring in at least three interviews is
marked `scripted_recurrence=True`. This is an audit flag, not an exclusion
criterion: scripted questions remain eligible, while cluster uniqueness still
prevents the same or near-identical wording from appearing more than once.

The resulting method is identified as
`interview_speaker_role_turn_position_similarity_deduplicated`. Sentence and
role-specific turn indices are zero-based; position strata are numbered 1–5.

## Interview-level train/validation/test split

Using the same seeded generator, one complete interview from each of the three
interviewer IDs is assigned to `test`, and one different interview from each is
assigned to `val`. The remaining 21 interviews form `train`. Partitioning
occurs at the interview level, not by assigning individual sentences: every
sentence sampled from a given interview inherits that interview's single split
label. This guarantees three test and three validation interviews, represents
all interviewer IDs in both held-out splits, and prevents dialogue content or
repeated material from one interview leaking between training and evaluation
sets.

The split is deliberately not interviewer-disjoint: the same recurring
interviewer may occur in train, validation, and test through different
interviews. The evaluation therefore measures generalization to unseen
interviews, not to unseen interviewers.

Each validation and test interview preserves the configured role minimum—10
sentences per role when both roles have at least 10 eligible sentences and the
interview quota permits it. This prevents a held-out interviewer from being
nominally represented by only one sentence. Recurring-interviewer equality is
retained when compatible with the global duplicate constraint; otherwise the
summary reports the achieved totals and the reason for relaxation.

The split assignment is reproducible for the same set of interview IDs and
seed. The same generator is subsequently used for quota allocation and random
sampling, without being reset.

## Validation and records

When at least 1,601 eligible sentences exist, the procedure requires exactly
1,601 sampled rows. It also verifies that every interview is represented, both
roles are represented whenever both are available, and no source sentence is
sampled twice. It requires every sampled `similarity_cluster_id` to be unique,
which rejects exact and configured near-duplicate text even when the source
sentences have different interview, speaker, role, or sentence IDs. It
additionally requires exactly three test interviews, three validation
interviews, all three interviewer IDs in both held-out splits, at least 10 rows
for every sufficiently represented speaker identity, and mutually exclusive
interview membership across all splits. It also asserts the feasible
10-sentence role floor independently for every validation and test interview.
Independently of the preprocessing gate, the final validation re-checks the
mandatory sentence form on every sampled row: at least four words, a
sentence-initial capital letter, and terminal `.`, `!`, or `?` without a
truncating ellipsis. Re-running the procedure on the same ordered query result
with seed 42 and the same similarity settings produces the same split
assignment and sample.

Each sampled row includes a stable item UUID, interview ID, speaker role,
speaker ID, role-specific sentence index, primary role-specific turn index,
global chronological source-turn indices, positional stratum, text,
reconstruction provenance, quality/review flags, similarity-cluster metadata,
sampling method, and random seed. Context fields contain the actual preceding,
source, intervening, and following cleaned chronological turns rather than
aligning unrelated role-specific turn indices.

A seeded randomized `presentation_order` prevents annotation drift from being
confounded with interview, role, or split order. Ten percent of items are
systematically spread across that timeline and marked `double_annotate=True`.
The separate annotation-facing CSV is sorted by presentation order and omits
interview IDs, speaker IDs, and dataset split; the master and train/validation/
test CSVs retain them. Short-responsive, token-count, finite-verb, and fragment
flags support quality control. A candidate can be excluded by the separately
reported form-based rules above, while absence of a finite verb by itself is
not an exclusion: that would remove valid dialogue answers. Because the
mandatory form requirement admits only sentences of at least four words,
`short_responsive` is retained as a schema-stable diagnostic column and is
`False` for every sampled row.

The interview report records interviewer and interviewee IDs. Corpus-level
output reports speaker counts, role proportions, eligible and sampled turns,
split counts, interview quota range, reconstruction counts, duplicate clusters,
constraint relaxation, annotation overlap, and reproducibility settings.
SHA-256 provenance covers the `query2` result, interviewer mapping result, both
query texts, sampling source code, configuration, Python/NumPy versions, and
the spaCy model/version. Only `query2` and `query_mapping` are executed against
the database. Sentence cleanup does not add or modify any database query.

## CSV files and column dictionary

All CSV files are UTF-8 encoded with a header row. Boolean values are written
as `True` or `False`. Columns containing lists or objects are serialized as
JSON inside the CSV field and should be decoded with a JSON parser after the
CSV has been read. Sentence and turn indices are zero-based; position strata
are one-based (`1`–`5`).

### Master sample and split files

`neg_anno_sample.csv` is the complete research sample. The files
`neg_anno_sample_train.csv`, `neg_anno_sample_val.csv`, and
`neg_anno_sample_test.csv` have exactly the same columns and contain the
corresponding interview-level subsets. Their union equals the master sample,
and an interview occurs in exactly one of them.

| Column | Meaning |
| --- | --- |
| `interview_id` | Experiment/interview identifier returned by `query2`. |
| `split` | Interview-level dataset assignment: `train`, `val`, or `test`. |
| `item_uuid` | Stable UUID derived from interview, role, role-specific sentence index, and normalized sentence text. It identifies the annotation item across files and reruns with unchanged source data. |
| `presentation_order` | Seeded random order over all 1,601 items, numbered `0`–`1600`; this is the intended annotation order. It is independent of the physical row order in the master file. |
| `double_annotate` | Whether the item belongs to the systematic 10% double-annotation subset. |
| `speaker_role` | `Interviewer` or `Interviewee`. |
| `speaker_id` | Recurring interviewer ID from `get_mapping()` for interviewer rows; the interview ID for the unique interviewee in that interview. Role and ID should be used together as the speaker key. |
| `sentence_index_within_role` | Position of the sentence in that interview's complete segmented stream for the given role, starting at `0`; it is a source index, not its order in the sample. |
| `position_stratum` | Sequential fifth of the role-specific cleaned-turn sequence containing the sentence's primary turn: `1` is earliest and `5` latest. |
| `text` | Automatically segmented sentence selected for annotation. It always has at least four words, a capital-letter start, and terminal `.`, `!`, or `?`. For a conservatively reconstructed interruption, this contains the joined sentence. |
| `similarity_cluster_id` | Stable identifier for the corpus-wide exact/near-duplicate connected component. Every sampled row has a different cluster ID. |
| `similarity_cluster_size` | Number of eligible source sentences represented by that similarity cluster, including the selected sentence. |
| `similarity_cluster_interview_count` | Number of distinct source interviews represented in the similarity cluster. |
| `scripted_recurrence` | Audit flag indicating that the cluster contains an exact normalized form observed in at least three interviews. It is not an exclusion rule. |
| `n_tokens` | Punctuation-excluding word count used by the preprocessing quality heuristics. It is at least 4 for every eligible sentence. |
| `short_responsive` | Heuristic flag for a one- or two-word response beginning with a recognized responsive/backchannel word. Such candidates are excluded by the four-word minimum, so the column is `False` throughout and is kept only for schema stability. |
| `finite_verb` | Whether the German spaCy analysis found a finite `VERB` or `AUX` in the sentence. |
| `fragment` | Quality-control heuristic equal to the absence of a detected finite verb. It is not a definitive linguistic fragment annotation. |
| `primary_turn_index_within_role` | Zero-based index of the first contributing cleaned turn in this role's turn stream. This turn determines positional and individual-turn stratification. |
| `source_turn_indices_within_role` | JSON array of zero-based role-specific cleaned-turn indices that contributed text to the sentence. |
| `source_turn_indices_within_interview` | JSON array of zero-based indices in the complete cleaned chronological dialogue that contributed text. |
| `source_turn_ids` | JSON array of query-provided source-turn IDs. When IDs are absent, stable one-based positions in the original `query2` turns array are used. |
| `source_chunk_ids` | JSON array of underlying chunk IDs when supplied by the query result. It is empty for the current `query2` result shape. |
| `interrupted_by_turn_ids` | JSON array of turn IDs belonging to the short intervening other-speaker turn(s) bridged by reconstruction; empty for ordinary sentences. |
| `cross_turn_sentence` | Whether the sampled sentence was reconstructed across a detected other-speaker interruption. |
| `reconstruction` | `automatic` for a conservatively bridged cross-turn sentence and `none` otherwise. |
| `review_required` | Whether the sentence requires manual review; true for automatically reconstructed cross-turn sentences and for candidates without a detected finite verb. Terminal punctuation is no longer a review reason because it is mandatory for eligibility. |
| `previous_turn_role` | Role of the cleaned chronological turn immediately before the first contributing turn; blank at the beginning of an interview. |
| `previous_turn_text` | Full text of that preceding cleaned turn. |
| `source_turn_text` | Full text of all contributing cleaned turns joined in chronological order. It supplies context and can be broader than the segmented `text`. |
| `intervening_turns` | JSON array describing non-source turns between contributing turns. Each object contains chronological turn index, role, text, and source-turn IDs. |
| `following_turn_role` | Role of the cleaned chronological turn immediately after the last contributing turn; blank at the end of an interview. |
| `following_turn_text` | Full text of that following cleaned turn. |
| `sampling_method` | Fixed method label: `interview_speaker_role_turn_position_similarity_deduplicated`. |
| `random_seed` | Seed used for splitting, quota allocation, selection, presentation order, and double-annotation assignment; `42` in the recorded run. |

The physical order of the master and split files is deterministic—split,
interview, role, and source sentence index—but analyses should use the named
columns rather than infer meaning from row order.

### Blinded annotation file

`neg_anno_annotation_items.csv` contains the annotation-facing view of the
same 1,601 rows, sorted by `presentation_order`. The following columns retain
the meanings defined above:

`item_uuid`, `presentation_order`, `double_annotate`, `speaker_role`, `text`,
`previous_turn_role`, `previous_turn_text`, `source_turn_text`,
`intervening_turns`, `following_turn_role`, `following_turn_text`,
`cross_turn_sentence`, `review_required`, `n_tokens`, `short_responsive`,
`finite_verb`, and `fragment`.

It deliberately omits `interview_id`, `speaker_id`, `split`, source sentence
indices, similarity information, and sampling strata. This reduces annotator
exposure to grouping and sampling information. `item_uuid` is the lossless key
for joining completed annotations back to the master sample.

### Per-interview sampling report

`neg_anno_sampling_report.csv` contains one row per interview. Its columns
describe the eligible source corpus and the achieved sample, not individual
sentences.

| Column or pattern | Meaning |
| --- | --- |
| `interview_id` | Interview represented by the row. |
| `interviewer_id` | Recurring interviewer identity obtained from `get_mapping()`. |
| `interviewee_id` | Unique interviewee identity, inferred from `interview_id`. |
| `split` | The single split assigned to the complete interview. |
| `interviewer_available`, `interviewee_available` | Eligible segmented source sentences for each role after preprocessing. |
| `total_available` | Sum of the two role-specific available counts. |
| `interview_quota` | Number of rows the interview must contribute to the final sample, 59 or 60 in the recorded run. |
| `interviewer_sampled`, `interviewee_sampled` | Achieved sampled sentences for each role. Their sum equals `interview_quota`. |
| `interviewer_turns_total`, `interviewee_turns_total` | Cleaned non-empty chronological turns belonging to each role, including turns that yield no eligible sentence. |
| `interviewer_turns_eligible`, `interviewee_turns_eligible` | Distinct primary turns for which at least one eligible sentence exists. |
| `interviewer_turns_sampled`, `interviewee_turns_sampled` | Distinct primary turns represented by at least one sampled sentence. |
| `interviewer_share_available`, `interviewee_share_available` | Role-specific proportion of all eligible sentences in the interview, expressed from `0` to `1`. |
| `interviewer_share_sampled`, `interviewee_share_sampled` | Role-specific proportion of the achieved interview sample, expressed from `0` to `1`. |
| `{role}_s{k}_available` | Eligible sentences for `role` (`interviewer` or `interviewee`) in position stratum `k` (`1`–`5`). |
| `{role}_s{k}_sampled` | Sampled sentences from that role and position stratum. |
| `{role}_s{k}_turns_eligible` | Distinct eligible primary turns in that role and stratum. |
| `{role}_s{k}_turns_sampled` | Distinct primary turns represented by the sample in that role and stratum. |

For example, `interviewer_s3_turns_sampled` is the number of distinct
interviewer turns represented in the middle positional fifth of that
interview. The repeated four-column stratum pattern is present for both roles
and all five strata.

`neg_anno_sampling_summary.json` is not a CSV and therefore has no fixed
column schema. It records corpus-level totals, balance and relaxation results,
duplicate-control diagnostics, configuration, authorized query names, and
SHA-256 provenance for reproducing the recorded run.

## Achieved sample for the recorded seed-42 run

For the query results identified by the SHA-256 hashes in
`neg_anno_sampling_summary.json`, preprocessing produced 3,638 eligible
sentences after excluding 3,018 structural-quality failures and 21 technical
artifacts. The mandatory sentence-form requirement accounts for most of the
exclusions: 2,050 candidates under four words, 360 without a capital-letter
start, 342 without terminal punctuation, and 31 truncating ellipses; the
remaining 235 are incomplete final tails and leading-question fragments. The
eligible pool contains 2,730 interviewer and 908 interviewee sentences.

The final sample contains exactly 1,601 rows from all 27 interviews, with 19
interviews contributing 59 rows and eight contributing 60. Its
train/validation/test counts are 1,246 / 178 / 177, corresponding to 21 / 3 / 3
complete interviews.

The sample contains 977 interviewer rows (61.02%) and 624 interviewee rows
(38.98%). This reverses the role proportions of the earlier, more permissive
run, and it is a direct consequence of the sentence-form requirement rather
than of the sampling design: scripted interviewer questions are transcribed as
complete, punctuated sentences far more often than spontaneous interviewee
speech, so the interviewee share of the eligible pool falls from 39% of 6,005
candidates to 25% of 3,638. Role quotas remain proportional to availability, as
specified above.

The sample represents 1,195 distinct role-specific turns: 702 interviewer and
493 interviewee turns. Every position stratum that has eligible capacity is
represented in the sample. Because the run activates the
`interview_fixed_role_minima_only` relaxation, positional quotas adapt to
unique-cluster availability instead of being equal by construction, and the
achieved stratum totals are 485 / 202 / 271 / 299 / 344 for strata 1 to 5.

Of the 30 distinct speaker identities, 28 reach at least 10 rows. The two
exceptions are the interviewees of interviews 7 and 18, whose complete eligible
pools contain only 7 and 6 sentences; both are sampled exhaustively, which is
what the minimum-coverage rule requires when fewer than 10 sentences exist. All
validation and test roles remain at or above 10.

The recurring-interviewer totals are 102 for interviewer 3, 296 for interviewer
4, and 579 for interviewer 5. Exact equality is infeasible under the
simultaneous candidate-capacity, per-interview, role-floor, positional, and
duplicate constraints. The allocation and duplicate-safe rebalancing
diagnostics are recorded explicitly in the JSON summary rather than silently
claiming equality.

The 3,638 eligible sentences form 2,152 similarity clusters under
exact-duplicate control. Of these, 221 clusters contain duplicates, covering
1,707 source sentences; the largest cluster contains 46 source sentences. The
final sample contains 1,601 unique cluster IDs and zero repeated normalised
forms. There are 143 source clusters marked as scripted recurrence, of which
131 are represented in the sample; selected instances carry that audit flag
rather than being categorically excluded. The sample contains 277 reconstructed
cross-turn sentences, 405 rows marked for review under the combined review
policy, and 160 items assigned to systematic double annotation.

The sample size is itself a finding. Under the mandatory sentence-form
requirement and cluster uniqueness, the corpus cannot supply 2,000 rows by any
configuration: the measured ceilings are 1,102 rows with near-duplicate
clustering at `0.92`, 1,213 rows if the minimum is lowered to three words and
digit-initial sentences are admitted, 1,601 rows under the recorded
configuration, and 1,820 rows with every form rule and the near-duplicate rule
loosened together. The recorded configuration is the largest sample that
preserves the sentence-form requirement in full.

## Methodological limitation

Sentence boundaries and interruption bridges are algorithmic preprocessing
decisions. They should not be described as gold-standard manual sentence
annotations. The saved provenance and review flags make reconstructed cases
auditable before negation annotation or statistical analysis. Interviewee
identity is inferred from the one-interview-per-interviewee design rather than
from an independent participant identifier. Any violation of that design
assumption would make the inferred interviewee speaker IDs invalid.

Removing surface-form duplicates changes the statistical target: a frequent
utterance such as `Ja.` can appear only once. The resulting corpus is optimized
for diverse annotation or model evaluation, not an unweighted estimate of raw
utterance prevalence. `similarity_cluster_size` preserves source multiplicity;
analyses of prevalence must account for it. Exact analytical inclusion
probabilities are not reported because global similarity matching and
constraint relaxation make simple per-stratum probabilities invalid.
