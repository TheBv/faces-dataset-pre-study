"""Create a reproducible interview/speaker/role/turn-stratified sample.

The database query returns chronological speaker turns.  Python removes empty
turns, conservatively reconnects interrupted sentences, sentence-segments each
speaker stream, and finally samples without using negation-related information.
Sentence and turn indices written by this module are zero- and one-based,
respectively.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
import uuid
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .surreal_client import SurrealClientConfig, surreal_client
except ImportError:  # Support ``PYTHONPATH=psrc python .../create_neg_anno_sample.py``.
    from main_utils.surreal_client import SurrealClientConfig, surreal_client


TOTAL_SAMPLE_SIZE = 1_422
RANDOM_SEED = 42
MIN_ROLE_SAMPLE = 10
N_POSITION_STRATA = 5
MAX_INTERRUPTION_TOKENS = 8
MAX_SENTENCE_TOKENS = 120
MIN_SENTENCE_TOKENS = 2
MIN_FRAGMENT_REVIEW_TOKENS = 4
# Conservative near-duplicate matching is active: no two sampled sentences may
# be near-identical anywhere in the corpus. 1.0 would restrict clustering to
# identical normalised forms.
NEAR_DUPLICATE_SIMILARITY_THRESHOLD = 0.92
NEAR_DUPLICATE_MIN_TOKENS = 6
# Near-duplicate wording may not span dataset splits: a cluster formed at this
# threshold is given to one split, and the other splits' copies leave the
# candidate pool. Corpus-wide cluster uniqueness already implies split
# disjointness, so this rule is only needed when
# NEAR_DUPLICATE_SIMILARITY_THRESHOLD is loosened above it; at 1.0 it is off.
SPLIT_DISJOINT_SIMILARITY_THRESHOLD = 1.0
SPLIT_DISJOINT_MIN_TOKENS = 6
DOUBLE_ANNOTATION_FRACTION = 0.10
N_TEST_INTERVIEWS = 3
N_VAL_INTERVIEWS = 3
N_INTERVIEWERS = 3
SPEAKER_ROLES = ("Interviewer", "Interviewee")
DATASET_SPLITS = ("train", "val", "test")
SAMPLING_METHOD = (
    "interview_speaker_role_turn_position_similarity_deduplicated"
)
SPLIT_METHOD = "seeded_interview_group_split_stratified_by_interviewer"

DEFAULT_SAMPLE_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "data/neg_samples/neg_anno_sample.csv"
    )
DEFAULT_REPORT_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "data/neg_samples/neg_anno_sampling_report.csv"
)
DEFAULT_SUMMARY_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "data/neg_samples/neg_anno_sampling_summary.json"
)
DEFAULT_ANNOTATION_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "data/neg_samples/neg_anno_annotation_items.csv"
)

SAMPLE_FIELDNAMES = (
    "interview_id",
    "split",
    "item_uuid",
    "presentation_order",
    "double_annotate",
    "speaker_role",
    "speaker_id",
    "sentence_index_within_role",
    "position_stratum",
    "text",
    "similarity_cluster_id",
    "similarity_cluster_size",
    "similarity_cluster_interview_count",
    "scripted_recurrence",
    "n_tokens",
    "short_responsive",
    "finite_verb",
    "fragment",
    "primary_turn_index_within_role",
    "source_turn_indices_within_role",
    "source_turn_indices_within_interview",
    "source_turn_ids",
    "source_chunk_ids",
    "interrupted_by_turn_ids",
    "cross_turn_sentence",
    "reconstruction",
    "review_required",
    "previous_turn_role",
    "previous_turn_text",
    "source_turn_text",
    "intervening_turns",
    "following_turn_role",
    "following_turn_text",
    "sampling_method",
    "random_seed",
)

ANNOTATION_FIELDNAMES = (
    "item_uuid",
    "presentation_order",
    "double_annotate",
    "speaker_role",
    "text",
    "previous_turn_role",
    "previous_turn_text",
    "source_turn_text",
    "intervening_turns",
    "following_turn_role",
    "following_turn_text",
    "cross_turn_sentence",
    "review_required",
    "n_tokens",
    "short_responsive",
    "finite_verb",
    "fragment",
)

_SENTENCE_FINAL_RE = re.compile(r"[.!?…][\"'»”’)]*$")
_SENTENCE_OPENING_CHARACTERS = "\"'«»„“‚‘([{-–— \t"
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_TECHNICAL_ARTIFACT_RES = (
    re.compile(
        r"^untertitel(?:ung)?(?:\s+des\s+zdf)?(?:\s*,?\s*(?:19|20)\d{2})?"
        r"[,.!]?$",
        re.IGNORECASE,
    ),
    re.compile(r"^©\s*.+$", re.IGNORECASE),
    re.compile(r"^(?:19|20)\d{2}[.!]?$"),
    re.compile(
        r"^\[?(?:musik|pause|stille|unverständlich|unverständlich\s+\d+)\]?[.!]?$",
        re.IGNORECASE,
    ),
)
_TECHNICAL_ARTIFACT_PREFIX_RE = re.compile(
    r"^(?:(?:untertitel(?:ung)?(?:\s+des\s+zdf)?"
    r"(?:\s*,?\s*(?:19|20)\d{2})?)[,.!]?\s*)+",
    re.IGNORECASE,
)
_BACKCHANNEL_PHRASES = {
    "ach so",
    "ah",
    "alles klar",
    "äh",
    "entschuldigung",
    "genau",
    "hm",
    "ja",
    "ja genau",
    "ja okay",
    "mhm",
    "ne",
    "okay",
    "ok",
    "richtig",
}
_SHORT_RESPONSIVE_WORDS = {
    "achso",
    "aha",
    "ah",
    "danke",
    "doch",
    "genau",
    "gut",
    "hm",
    "hmm",
    "ja",
    "jaja",
    "mhm",
    "ne",
    "nee",
    "nein",
    "ok",
    "okay",
    "richtig",
    "stimmt",
}
_INCOMPLETE_FINAL_WORDS = {
    "als",
    "am",
    "an",
    "auf",
    "aber",
    "dann",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "die",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "für",
    "im",
    "in",
    "ist",
    "kann",
    "mit",
    "oder",
    "und",
    "von",
    "wenn",
    "wie",
    "wird",
    "zu",
}
_DEPENDENT_INITIAL_WORDS = {
    "als",
    "dass",
    "falls",
    "indem",
    "nachdem",
    "ob",
    "obgleich",
    "obwohl",
    "seitdem",
    "sobald",
    "sofern",
    "solange",
    "während",
    "weil",
    "wenn",
    "wobei",
}
_UNAMBIGUOUS_INCOMPLETE_FINAL_WORDS = {
    "als",
    "aber",
    "also",
    "am",
    "an",
    "auf",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "die",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "für",
    "im",
    "mit",
    "ob",
    "obwohl",
    "oder",
    "und",
    "von",
    "weil",
    "wenn",
    "wie",
    "wo",
    "zu",
}
_HIGH_CONFIDENCE_UNPUNCTUATED_STARTS = {
    "Aber",
    "Also",
    "Danke",
    "Danach",
    "Dann",
    "Das",
    "Dazu",
    "Du",
    "Er",
    "Es",
    "Genau",
    "Gut",
    "Hallo",
    "Haben",
    "Heute",
    "Ich",
    "Inwieweit",
    "Ja",
    "Jetzt",
    "Kann",
    "Können",
    "Man",
    "Möchten",
    "Nein",
    "Okay",
    "Sie",
    "Trifft",
    "Und",
    "Wann",
    "Warum",
    "Was",
    "Welche",
    "Welcher",
    "Welches",
    "Wem",
    "Wen",
    "Wer",
    "Wie",
    "Wieso",
    "Wir",
    "Wo",
    "Wodurch",
    "Womit",
    "Würden",
}
_PARSER_INCOMPLETE_FINAL_WORDS = {
    "darf",
    "denke",
    "denken",
    "dürfte",
    "glaube",
    "glauben",
    "kann",
    "könnte",
    "mag",
    "meine",
    "meinen",
    "möchte",
    "muss",
    "müsste",
    "sage",
    "sagen",
    "soll",
    "sollte",
    "will",
    "würde",
}

query = """(SELECT experiment
 FROM Audio
 WHERE experiment.validInterview
 GROUP BY experiment
).map(|$group| {

    RETURN {
        experiment: $group.experiment,

        speakers: ["Interviewer", "Interviewee"].map(|$role| {

            LET $audios = (
                SELECT
                    <~AudioChunk[WHERE player.role = $role]<~Word.text AS words,
                    endTime
                FROM Audio
                WHERE experiment = $group.experiment
                ORDER BY endTime ASC
            );

            RETURN {
                role: $role,
                text: array::join(
                    array::flatten(
                        array::flatten($audios.words)
                    ),
                    " "
                )
            };
        })
    };
});"""


query2 = """(SELECT experiment
 FROM Audio
 WHERE experiment.validInterview
 GROUP BY experiment
).map(|$group| {

    LET $chunks = (
        SELECT
            player.role AS role,

            array::join(
                array::flatten(
                    array::flatten(<~Word.text)
                ),
                " "
            ) AS text,

            startTime,
            endTime

        FROM AudioChunk
        WHERE experiment = $group.experiment
        ORDER BY startTime ASC, endTime ASC
    );

    LET $state = $chunks.fold(
        {
            turns: [],
            current: NONE
        },

        |$acc, $chunk| {

            -- First chunk starts the first turn
            RETURN IF $acc.current = NONE {

                {
                    turns: $acc.turns,

                    current: {
                        role: $chunk.role,
                        text: $chunk.text,
                        startTime: $chunk.startTime,
                        endTime: $chunk.endTime,
                        chunk_count: 1
                    }
                }

            -- Same speaker -> extend current turn
            } ELSE IF $acc.current.role = $chunk.role {

                {
                    turns: $acc.turns,

                    current: {
                        role: $acc.current.role,

                        text: array::join(
                            [
                                $acc.current.text,
                                $chunk.text
                            ],
                            " "
                        ),

                        startTime: $acc.current.startTime,
                        endTime: $chunk.endTime,
                        chunk_count: $acc.current.chunk_count + 1
                    }
                }

            -- Speaker changed -> finish old turn and start new one
            } ELSE {

                {
                    turns: array::append(
                        $acc.turns,
                        $acc.current
                    ),

                    current: {
                        role: $chunk.role,
                        text: $chunk.text,
                        startTime: $chunk.startTime,
                        endTime: $chunk.endTime,
                        chunk_count: 1
                    }
                }
            };
        }
    );

    -- fold() leaves the final turn in `current`,
    -- so append it once at the end
    LET $turns = IF $state.current != NONE {
        array::append(
            $state.turns,
            $state.current
        )
    } ELSE {
        []
    };

    RETURN {
        experiment: $group.experiment,
        turns: $turns
    };
});"""


query_mapping = """SELECT * FROM InterviewerAssignment"""

def get_mapping():
    with surreal_client(SurrealClientConfig()) as db:
        query_result = db.query(query_mapping)
    mapping = dict()
    for res in query_result:
        mapping[res["experiment"].id] = res["interviewerId"]
    return mapping

@dataclass(slots=True)
class TurnPart:
    """One original non-empty database turn retained inside a cleaned turn."""

    text: str
    source_turn_ids: list[Any]
    source_chunk_ids: list[Any]


@dataclass(slots=True)
class Turn:
    """A non-empty turn after removal of empty-turn speaker switches."""

    role: str
    text: str
    source_turn_ids: list[Any]
    source_chunk_ids: list[Any]
    start_time: Any
    end_time: Any
    chunk_count: int
    parts: list[TurnPart]


@dataclass(slots=True)
class InterviewSentences:
    interview_id: Any
    sentences_by_role: dict[str, list[dict[str, Any]]]
    raw_turn_count: int
    empty_turn_count: int
    clean_turn_count: int
    technical_artifact_count: int
    sentence_quality_exclusion_counts: dict[str, int] = field(default_factory=dict)
    interviewer_id: Any = None
    max_interruption_tokens: int = MAX_INTERRUPTION_TOKENS
    turn_count_by_role: dict[str, int] = field(default_factory=dict)
    dialogue_turns: list[Turn] = field(default_factory=list)

    @property
    def total_available(self) -> int:
        return sum(len(self.sentences_by_role[role]) for role in SPEAKER_ROLES)


@dataclass(slots=True)
class _SimilarityIndex:
    cluster_by_sentence_object: dict[int, str]
    cluster_sizes: dict[str, int]
    cluster_interview_counts: dict[str, int]
    scripted_recurrence_clusters: set[str]
    summary: dict[str, Any]


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalise_sentence_for_similarity(text: str) -> str:
    """Case/punctuation-insensitive form used only for duplicate control."""

    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(_WORD_RE.findall(text))


def _token_trigrams(tokens: Sequence[str]) -> set[tuple[str, str, str]]:
    return set(zip(tokens, tokens[1:], tokens[2:]))


def _forms_are_near_duplicates(
    left_form: str,
    left_tokens: Sequence[str],
    right_form: str,
    right_tokens: Sequence[str],
    *,
    threshold: float,
    min_tokens: int,
) -> bool:
    if left_form == right_form:
        return True
    if min(len(left_tokens), len(right_tokens)) < min_tokens:
        return False
    if not (_token_trigrams(left_tokens) & _token_trigrams(right_tokens)):
        return False
    shorter, longer = sorted((len(left_form), len(right_form)))
    if not longer:
        return False
    maximum_possible_ratio = 2.0 * shorter / (shorter + longer)
    if maximum_possible_ratio < threshold:
        return False
    return (
        SequenceMatcher(
            None,
            left_form,
            right_form,
            autojunk=False,
        ).ratio()
        >= threshold
    )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_sentence_similarity_index(
    interviews: Sequence[InterviewSentences],
    *,
    threshold: float = NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
    min_tokens: int = NEAR_DUPLICATE_MIN_TOKENS,
) -> _SimilarityIndex:
    """Cluster exact and conservative surface-form near-duplicates globally."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("Near-duplicate threshold must be in (0, 1]")
    if min_tokens < 1:
        raise ValueError("Near-duplicate minimum tokens must be positive")

    items: list[tuple[Any, str, dict[str, Any], str, list[str]]] = []
    for interview in interviews:
        for role in SPEAKER_ROLES:
            for sentence in interview.sentences_by_role[role]:
                form = normalise_sentence_for_similarity(sentence["text"])
                items.append((interview.interview_id, role, sentence, form, form.split()))
    if not items:
        raise ValueError("Cannot deduplicate an empty sentence corpus")

    union_find = _UnionFind(len(items))
    indices_by_form: dict[str, list[int]] = defaultdict(list)
    for index, (_, _, _, form, _) in enumerate(items):
        indices_by_form[form].append(index)
    exact_form_interview_counts = {
        form: len({_identity_key(items[index][0]) for index in indices})
        for form, indices in indices_by_form.items()
    }
    exact_duplicate_forms = 0
    exact_redundant_candidates = 0
    for indices in indices_by_form.values():
        if len(indices) > 1:
            exact_duplicate_forms += 1
            exact_redundant_candidates += len(indices) - 1
            for index in indices[1:]:
                union_find.union(indices[0], index)

    unique_forms = list(indices_by_form)
    unique_tokens = [form.split() for form in unique_forms]
    representative_index = [indices_by_form[form][0] for form in unique_forms]
    trigram_postings: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for form_index, tokens in enumerate(unique_tokens):
        if len(tokens) < min_tokens:
            continue
        for trigram in _token_trigrams(tokens):
            trigram_postings[trigram].append(form_index)

    compared_pairs: set[int] = set()
    near_duplicate_links = 0
    n_unique_forms = len(unique_forms)
    for posting in trigram_postings.values():
        for position, left in enumerate(posting):
            for right in posting[position + 1 :]:
                first, second = sorted((left, right))
                pair_key = first * n_unique_forms + second
                if pair_key in compared_pairs:
                    continue
                compared_pairs.add(pair_key)
                if _forms_are_near_duplicates(
                    unique_forms[first],
                    unique_tokens[first],
                    unique_forms[second],
                    unique_tokens[second],
                    threshold=threshold,
                    min_tokens=min_tokens,
                ):
                    union_find.union(
                        representative_index[first],
                        representative_index[second],
                    )
                    near_duplicate_links += 1

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(items)):
        members_by_root[union_find.find(index)].append(index)

    cluster_by_sentence_object: dict[int, str] = {}
    cluster_sizes: dict[str, int] = {}
    cluster_interview_counts: dict[str, int] = {}
    scripted_recurrence_clusters: set[str] = set()
    for members in members_by_root.values():
        descriptors = sorted(
            (
                _identity_key(items[index][0]),
                items[index][1],
                int(items[index][2]["sentence_index_within_role"]),
                items[index][3],
            )
            for index in members
        )
        payload = json.dumps(descriptors, ensure_ascii=False, separators=(",", ":"))
        cluster_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        cluster_sizes[cluster_id] = len(members)
        cluster_interview_counts[cluster_id] = len(
            {_identity_key(items[index][0]) for index in members}
        )
        if any(
            exact_form_interview_counts[items[index][3]] >= 3
            for index in members
        ):
            scripted_recurrence_clusters.add(cluster_id)
        for index in members:
            cluster_by_sentence_object[id(items[index][2])] = cluster_id

    duplicate_cluster_sizes = [size for size in cluster_sizes.values() if size > 1]
    return _SimilarityIndex(
        cluster_by_sentence_object=cluster_by_sentence_object,
        cluster_sizes=cluster_sizes,
        cluster_interview_counts=cluster_interview_counts,
        scripted_recurrence_clusters=scripted_recurrence_clusters,
        summary={
            "deduplication_scope": "complete_corpus_across_interviews_and_roles",
            "deduplication_normalisation": (
                "Unicode NFKC, case-folding, punctuation removal, whitespace collapse"
            ),
            "near_duplicate_similarity": "difflib.SequenceMatcher_ratio",
            "near_duplicate_similarity_threshold": threshold,
            "near_duplicate_min_tokens": min_tokens,
            "near_duplicate_requires_shared_token_trigram": True,
            "source_sentence_candidates": len(items),
            "similarity_clusters": len(cluster_sizes),
            "duplicate_similarity_clusters": len(duplicate_cluster_sizes),
            "sentences_in_duplicate_similarity_clusters": sum(
                duplicate_cluster_sizes
            ),
            "largest_similarity_cluster": max(cluster_sizes.values()),
            "exact_duplicate_normalised_forms": exact_duplicate_forms,
            "exact_redundant_candidates": exact_redundant_candidates,
            "near_duplicate_links": near_duplicate_links,
            "scripted_recurrence_definition": (
                "an exact normalised form observed in at least 3 interviews"
            ),
            "scripted_recurrence_clusters": len(scripted_recurrence_clusters),
            "scripted_recurrence_excluded": False,
        },
    )


def _jsonable(value: Any) -> Any:
    """Convert SurrealDB and datetime values to stable output values."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "table_name") and hasattr(value, "id"):
        return _jsonable(value.id)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _identity_key(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _interview_sort_key(value: Any) -> tuple[int, Any]:
    value = _jsonable(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, value)
    return (1, _identity_key(value))


def _append_unique(target: list[Any], values: Sequence[Any]) -> None:
    existing = {_identity_key(value) for value in target}
    for value in values:
        value = _jsonable(value)
        key = _identity_key(value)
        if key not in existing:
            target.append(value)
            existing.add(key)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _join_fragments(left: str, right: str) -> str:
    left = _normalise_text(left)
    right = _normalise_text(right)
    if not left:
        return right
    if not right:
        return left
    separator = "" if right[0] in ".,!?;:%)]}»”" else " "
    return f"{left}{separator}{right}"


def _query_records(query_result: Any) -> list[Mapping[str, Any]]:
    """Accept direct SurrealDB results and older wrapped response shapes."""

    result = query_result
    while isinstance(result, Mapping) and "result" in result:
        result = result["result"]
    if (
        isinstance(result, Sequence)
        and not isinstance(result, (str, bytes, bytearray))
        and len(result) == 1
        and isinstance(result[0], Sequence)
        and not isinstance(result[0], (str, bytes, bytearray, Mapping))
    ):
        result = result[0]
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        raise TypeError("The turn query did not return a sequence of interviews")
    records = list(result)
    if not all(isinstance(record, Mapping) for record in records):
        raise TypeError("Every turn-query result must be an interview mapping")
    return records


def clean_and_group_turns(
    interview_record: Mapping[str, Any],
) -> tuple[list[Turn], int, int]:
    """Remove empty turns, then merge newly adjacent turns of the same role."""

    raw_turns = interview_record.get("turns", [])
    if not isinstance(raw_turns, Sequence) or isinstance(
        raw_turns, (str, bytes, bytearray)
    ):
        raise TypeError("An interview's 'turns' value must be a sequence")

    cleaned: list[Turn] = []
    empty_count = 0
    for position, raw_turn in enumerate(raw_turns, start=1):
        if not isinstance(raw_turn, Mapping):
            raise TypeError("Every turn must be a mapping")
        role = raw_turn.get("role")
        if role not in SPEAKER_ROLES:
            continue
        text = _normalise_text(raw_turn.get("text"))
        if not text:
            empty_count += 1
            continue

        source_turn_ids = _as_list(raw_turn.get("source_turn_ids"))
        if not source_turn_ids:
            source_turn_ids = [raw_turn.get("turn_id", position)]
        source_chunk_ids = _as_list(raw_turn.get("chunk_ids"))
        if not source_chunk_ids:
            source_chunk_ids = _as_list(raw_turn.get("chunk_id"))
        source_turn_ids = [_jsonable(value) for value in source_turn_ids]
        source_chunk_ids = [_jsonable(value) for value in source_chunk_ids]
        part = TurnPart(
            text=text,
            source_turn_ids=list(source_turn_ids),
            source_chunk_ids=list(source_chunk_ids),
        )

        try:
            chunk_count = int(raw_turn.get("chunk_count", len(source_chunk_ids) or 1))
        except (TypeError, ValueError):
            chunk_count = len(source_chunk_ids) or 1

        if cleaned and cleaned[-1].role == role:
            previous = cleaned[-1]
            previous.text = _join_fragments(previous.text, text)
            _append_unique(previous.source_turn_ids, source_turn_ids)
            _append_unique(previous.source_chunk_ids, source_chunk_ids)
            previous.end_time = raw_turn.get("endTime", previous.end_time)
            previous.chunk_count += chunk_count
            previous.parts.append(part)
            continue

        cleaned.append(
            Turn(
                role=role,
                text=text,
                source_turn_ids=source_turn_ids,
                source_chunk_ids=source_chunk_ids,
                start_time=raw_turn.get("startTime"),
                end_time=raw_turn.get("endTime"),
                chunk_count=chunk_count,
                parts=[part],
            )
        )

    return cleaned, len(raw_turns), empty_count


def ends_with_sentence_boundary(text: str) -> bool:
    return bool(_SENTENCE_FINAL_RE.search(text.rstrip()))


def starts_with_sentence_capital(text: str) -> bool:
    """Whether the candidate visibly opens a sentence with a capital letter."""

    opening = text.strip().lstrip(_SENTENCE_OPENING_CHARACTERS)
    return bool(opening) and opening[0].isalpha() and opening[0].isupper()


def _words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


def _is_short_responsive(text: str) -> bool:
    words = _words(text)
    return 0 < len(words) <= 2 and words[0] in _SHORT_RESPONSIVE_WORDS


def _has_finite_verb(tokens: Any) -> bool:
    for token in tokens:
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        verb_forms = token.morph.get("VerbForm")
        if not verb_forms or "Fin" in verb_forms:
            return True
    return False


def _has_explicit_finite_verb(tokens: Any) -> bool:
    return any(
        token.pos_ in {"VERB", "AUX"}
        and "Fin" in token.morph.get("VerbForm")
        for token in tokens
    )


def _has_unresolved_dependent_clause(tokens: Any) -> bool:
    for index, token in enumerate(tokens):
        if token.text.casefold() not in _DEPENDENT_INITIAL_WORDS | {"ob"}:
            continue
        if not _has_finite_verb(tokens[index + 1 :]):
            return True
    return False


def _strip_technical_artifact_prefix(text: str) -> str:
    """Remove repeated subtitle-credit noise without changing spoken wording."""

    previous = None
    text = _normalise_text(text)
    while text and text != previous:
        previous = text
        text = _normalise_text(_TECHNICAL_ARTIFACT_PREFIX_RE.sub("", text, count=1))
    return text


def _has_balanced_ordered_brackets(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in text:
        if character in pairs.values():
            stack.append(character)
        elif character in pairs:
            if not stack or stack.pop() != pairs[character]:
                return False
    return not stack


def _ends_with_incomplete_syntax(tokens: Any, text: str) -> bool:
    words = _words(text)
    if not words:
        return True
    if words[-1] in _UNAMBIGUOUS_INCOMPLETE_FINAL_WORDS:
        return True
    lexical_tokens = [
        token for token in tokens if not token.is_space and not token.is_punct
    ]
    if not lexical_tokens:
        return True
    return (
        lexical_tokens[-1].pos_ in {"ADP", "CCONJ", "DET", "SCONJ"}
        or (
            len(lexical_tokens) >= 2
            and lexical_tokens[-2].pos_ == "DET"
            and lexical_tokens[-1].tag_ == "ADJA"
            and not ends_with_sentence_boundary(text)
        )
    )


def _is_likert_response(text: str) -> bool:
    return bool(
        re.search(
            r"\btr\w{1,12}\s+(?:(?:er|es)\s+)?"
            r"(?:gar\s+|eher\s+|überwiegend\s+)?"
            r"(?:nicht\s+)?zu\b",
            " ".join(_words(text)),
        )
    )


def _quality_exclusion_reason(tokens: Any, text: str) -> str | None:
    """Return a surface/syntax quality reason, independent of negation."""

    words = _words(text)
    n_tokens = len(words)
    if not _has_balanced_ordered_brackets(text):
        return "unbalanced_brackets"
    if n_tokens < MIN_SENTENCE_TOKENS:
        return "too_few_words"
    if not starts_with_sentence_capital(text):
        return "lowercase_sentence_start"
    if text.rstrip().endswith(("...", "…")):
        return "truncated_ellipsis"
    if not ends_with_sentence_boundary(text):
        return "missing_terminal_punctuation"
    if n_tokens > MAX_SENTENCE_TOKENS:
        return "overlong_sentence"
    if _ends_with_incomplete_syntax(tokens, text) and not _is_likert_response(text):
        return "incomplete_final_syntax"
    if words[-1] in {"also", "denn"}:
        return "incomplete_final_syntax"
    if words[0] == "sie" and not _has_explicit_finite_verb(tokens[1:4]):
        return "leading_question_fragment"
    return None


def _has_explicit_boundary_before(doc: Any, token_index: int) -> bool:
    return ends_with_sentence_boundary(doc.text[: doc[token_index].idx])


def _looks_like_abbreviation_before(doc: Any, token_index: int) -> bool:
    prefix = doc.text[: doc[token_index].idx].rstrip()
    return bool(
        re.search(
            r"(?:\b(?:bzw|ca|dr|prof|usw)\.|\b(?:d|z)\."
            r"|\b(?:d|z)\s*\.\s*[bh]\.)$",
            prefix,
            re.IGNORECASE,
        )
        or re.search(r"\d\.\d$", prefix)
    )


def _is_high_confidence_unpunctuated_start(
    doc: Any,
    start_index: int,
    token_index: int,
) -> bool:
    token = doc[token_index]
    capitalised_start = token.text in _HIGH_CONFIDENCE_UNPUNCTUATED_STARTS
    next_tokens = doc[token_index + 1 : min(len(doc), token_index + 5)]
    nearby_finite_verb = any(
        _has_explicit_finite_verb(doc[index : index + 1])
        for index in range(token_index + 1, min(len(doc), token_index + 5))
    )
    if (
        token.text in {"Du", "Er", "Es", "Man", "Sie", "Und"}
        and not nearby_finite_verb
    ):
        capitalised_start = False
    lowercase_question_restart = (
        token.text == "wie"
        and nearby_finite_verb
        and len(next_tokens)
        and next_tokens[0].text.casefold()
        in {"auch", "häufig", "ist", "oft", "sind", "viel"}
    ) or (
        token.text in {"warum", "wieso"}
        and nearby_finite_verb
    ) or (
        token.text in {"wann", "was", "wer", "wo"}
        and len(next_tokens)
        and _has_explicit_finite_verb(next_tokens[:1])
    )
    lowercase_subject_restart = (
        token.text in {"du", "er", "es", "ich", "man", "sie", "wir"}
        and nearby_finite_verb
    )
    lowercase_discourse_restart = (
        token.text in {"also", "danach", "dann", "jetzt"}
        and nearby_finite_verb
    )
    if (
        not capitalised_start
        and not lowercase_question_restart
        and not lowercase_subject_restart
        and not lowercase_discourse_restart
    ):
        return False
    left = doc[start_index:token_index]
    left_word_count = len(_words(left.text))
    if left_word_count < 4:
        return False
    if not _has_finite_verb(left) and not (
        capitalised_start and nearby_finite_verb and left_word_count >= 6
    ):
        return False
    if _words(left.text)[-1] in _PARSER_INCOMPLETE_FINAL_WORDS:
        return False
    if _ends_with_incomplete_syntax(left, left.text) and not _is_likert_response(
        left.text
    ):
        return False
    if _has_unresolved_dependent_clause(left):
        return False
    prefix = doc.text[: token.idx].rstrip()
    return bool(prefix) and prefix[-1] not in ",:;/-–—("


def _parser_boundary_has_complete_left(
    doc: Any,
    start_index: int,
    token_index: int,
) -> bool:
    left = doc[start_index:token_index]
    text = _normalise_text(left.text)
    if (
        not text
        or not _has_balanced_ordered_brackets(text)
        or (
            text.casefold().startswith(("untertitel", "©"))
            and not ends_with_sentence_boundary(text)
        )
        or _ends_with_incomplete_syntax(left, text)
    ):
        return False
    words = _words(text)
    if _has_unresolved_dependent_clause(left):
        return False
    if words and words[-1] in _PARSER_INCOMPLETE_FINAL_WORDS:
        return False
    interrogative_stub = (
        words
        and words[0]
        in {"wann", "warum", "was", "wer", "wie", "wieso", "wo"}
        and len(words) <= 6
        and (
            left[-1].pos_ in {"AUX", "VERB"}
            or any(
                word in {"bin", "bist", "ist", "sind", "war", "waren"}
                for word in words
            )
        )
        and doc[token_index].pos_ in {"ADJ", "DET", "NOUN", "PRON", "PROPN"}
    )
    return not interrogative_stub


def _sentence_spans(doc: Any) -> list[Any]:
    """Reconcile parser starts with transcript punctuation and safe run-on cues.

    The German parser sometimes inserts a boundary inside a single question or
    noun phrase. A parser-only boundary is accepted only when its left side is
    demonstrably complete. Conversely, explicit punctuation and a small set of
    unambiguous question, subject, and discourse starts can restore boundaries
    the parser missed in noisy ASR text.
    """

    if not len(doc):
        return []
    parser_starts = {sentence.start for sentence in doc.sents}
    starts = [0]
    for token_index in range(1, len(doc)):
        explicit_boundary = _has_explicit_boundary_before(doc, token_index)
        abbreviation = _looks_like_abbreviation_before(doc, token_index)
        if explicit_boundary and not abbreviation:
            starts.append(token_index)
            continue
        if abbreviation:
            continue
        if (
            token_index in parser_starts
            and _parser_boundary_has_complete_left(
                doc, starts[-1], token_index
            )
        ):
            starts.append(token_index)
            continue
        if _is_high_confidence_unpunctuated_start(
            doc, starts[-1], token_index
        ):
            starts.append(token_index)
    starts = sorted(set(starts))
    return [
        doc[start:end]
        for start, end in zip(starts, starts[1:] + [len(doc)])
        if start < end
    ]


def _is_backchannel(text: str) -> bool:
    return " ".join(_words(text)) in _BACKCHANNEL_PHRASES


def _looks_like_continuation(
    first_fragment: str,
    interruption: str,
    continuation: str,
) -> bool:
    first_words = _words(first_fragment)
    tail = re.split(r"[.!?…]+[\"'»”’)]*\s*", first_fragment)[-1]
    tail_words = _words(tail)
    next_character = next((char for char in continuation if char.isalpha()), "")
    starts_lowercase = bool(next_character and next_character.islower())
    ends_with_joining_punctuation = first_fragment.rstrip().endswith((",", "-", "–", "—"))
    ends_with_incomplete_word_after_backchannel = bool(
        first_words and first_words[-1] in _INCOMPLETE_FINAL_WORDS
    )
    ends_with_unambiguous_incomplete_word = bool(
        tail_words
        and tail_words[-1] in _UNAMBIGUOUS_INCOMPLETE_FINAL_WORDS
    )
    return (
        starts_lowercase
        or ends_with_joining_punctuation
        or ends_with_unambiguous_incomplete_word
        or (
            _is_backchannel(interruption)
            and ends_with_incomplete_word_after_backchannel
        )
    )


def detect_interruption_bridges(
    turns: Sequence[Turn],
    nlp: Any,
    *,
    max_interruption_tokens: int = MAX_INTERRUPTION_TOKENS,
) -> dict[tuple[int, int], list[Any]]:
    """Find conservative A-unfinished / B-short / A-continuation bridges."""

    bridges: dict[tuple[int, int], list[Any]] = {}
    for index in range(len(turns) - 2):
        first, interruption, continuation = turns[index : index + 3]
        if first.role != continuation.role or first.role == interruption.role:
            continue
        if ends_with_sentence_boundary(first.text):
            continue
        interruption_tokens = sum(
            not token.is_space and not token.is_punct
            for token in nlp.make_doc(interruption.text)
        )
        if interruption_tokens > max_interruption_tokens:
            continue
        if not _looks_like_continuation(
            first.text,
            interruption.text,
            continuation.text,
        ):
            continue
        bridges[(index, index + 2)] = list(interruption.source_turn_ids)
    return bridges


def _is_technical_artifact(text: str) -> bool:
    text = _normalise_text(text)
    if not text or not any(char.isalnum() for char in text):
        return True
    return any(pattern.fullmatch(text) for pattern in _TECHNICAL_ARTIFACT_RES)


def _make_role_blocks(
    turns: Sequence[Turn],
    role: str,
    bridges: Mapping[tuple[int, int], list[Any]],
) -> list[tuple[list[tuple[int, Turn]], list[list[Any]]]]:
    blocks: list[tuple[list[tuple[int, Turn]], list[list[Any]]]] = []
    for turn_index, turn in enumerate(turns):
        if turn.role != role:
            continue
        if blocks:
            previous_index = blocks[-1][0][-1][0]
            interruption_ids = bridges.get((previous_index, turn_index))
            if interruption_ids is not None:
                blocks[-1][0].append((turn_index, turn))
                blocks[-1][1].append(interruption_ids)
                continue
        blocks.append(([(turn_index, turn)], []))
    return blocks


def _segment_block(
    interview_id: Any,
    role: str,
    block_turns: Sequence[tuple[int, Turn]],
    interruptions: Sequence[Sequence[Any]],
    role_turn_index_by_global: Mapping[int, int],
    nlp: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    block_text = ""
    piece_spans: list[tuple[int, int, TurnPart, int]] = []
    turn_spans: list[tuple[int, int]] = []
    for global_turn_index, turn in block_turns:
        turn_start: int | None = None
        for part in turn.parts:
            if block_text:
                separator = "" if part.text[0] in ".,!?;:%)]}»”" else " "
                block_text += separator
            start = len(block_text)
            if turn_start is None:
                turn_start = start
            block_text += part.text
            piece_spans.append(
                (start, len(block_text), part, global_turn_index)
            )
        if turn_start is None:
            raise AssertionError("A cleaned turn must contain at least one source part")
        turn_spans.append((turn_start, len(block_text)))

    sentences: list[dict[str, Any]] = []
    exclusion_counts: defaultdict[str, int] = defaultdict(int)
    for span in _sentence_spans(nlp(block_text)):
        raw_text = _normalise_text(span.text)
        text = _strip_technical_artifact_prefix(raw_text)
        if _is_technical_artifact(text):
            exclusion_counts["technical_artifact"] += 1
            continue

        quality_tokens = span if text == raw_text else nlp(text)
        exclusion_reason = _quality_exclusion_reason(quality_tokens, text)
        if exclusion_reason is not None:
            exclusion_counts[exclusion_reason] += 1
            continue

        overlapping = [
            (piece, global_turn_index)
            for start, end, piece, global_turn_index in piece_spans
            if start < span.end_char and end > span.start_char
        ]
        source_turn_indices_within_role: list[Any] = []
        source_turn_indices_within_interview: list[Any] = []
        source_turn_ids: list[Any] = []
        source_chunk_ids: list[Any] = []
        for piece, global_turn_index in overlapping:
            _append_unique(
                source_turn_indices_within_interview,
                [global_turn_index],
            )
            _append_unique(
                source_turn_indices_within_role,
                [role_turn_index_by_global[global_turn_index]],
            )
            _append_unique(source_turn_ids, piece.source_turn_ids)
            _append_unique(source_chunk_ids, piece.source_chunk_ids)

        interrupted_by: list[Any] = []
        for boundary_index, interruption_ids in enumerate(interruptions):
            left_end = turn_spans[boundary_index][1]
            right_start = turn_spans[boundary_index + 1][0]
            if span.start_char < left_end and span.end_char > right_start:
                _append_unique(interrupted_by, list(interruption_ids))

        cross_turn = bool(interrupted_by)
        if not source_turn_indices_within_role:
            raise AssertionError("A sentence must overlap at least one source turn")
        finite_verb = _has_finite_verb(quality_tokens)
        n_tokens = len(_words(text))
        sentences.append(
            {
                "interview_id": _jsonable(interview_id),
                "speaker_role": role,
                "text": text,
                "n_tokens": n_tokens,
                "short_responsive": _is_short_responsive(text),
                "finite_verb": finite_verb,
                "fragment": not finite_verb,
                "primary_turn_index_within_role": source_turn_indices_within_role[0],
                "source_turn_indices_within_role": source_turn_indices_within_role,
                "source_turn_indices_within_interview": (
                    source_turn_indices_within_interview
                ),
                "source_turn_ids": source_turn_ids,
                "source_chunk_ids": source_chunk_ids,
                "interrupted_by_turn_ids": interrupted_by,
                "cross_turn_sentence": cross_turn,
                "reconstruction": "automatic" if cross_turn else "none",
                "review_required": (
                    cross_turn
                    or (not finite_verb and n_tokens >= MIN_FRAGMENT_REVIEW_TOKENS)
                ),
            }
        )
    return sentences, dict(exclusion_counts)


def segment_interview(
    interview_id: Any,
    turns: Sequence[Turn],
    nlp: Any,
    *,
    max_interruption_tokens: int = MAX_INTERRUPTION_TOKENS,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Segment each role, reconnecting only detected short interruptions."""

    bridges = detect_interruption_bridges(
        turns,
        nlp,
        max_interruption_tokens=max_interruption_tokens,
    )
    by_role: dict[str, list[dict[str, Any]]] = {}
    exclusion_counts: defaultdict[str, int] = defaultdict(int)
    for role in SPEAKER_ROLES:
        role_sentences: list[dict[str, Any]] = []
        role_global_turn_indices = [
            index for index, turn in enumerate(turns) if turn.role == role
        ]
        role_turn_index_by_global = {
            global_index: role_index
            for role_index, global_index in enumerate(role_global_turn_indices)
        }
        for block_turns, interruptions in _make_role_blocks(turns, role, bridges):
            block_sentences, removed = _segment_block(
                interview_id,
                role,
                block_turns,
                interruptions,
                role_turn_index_by_global,
                nlp,
            )
            role_sentences.extend(block_sentences)
            for reason, count in removed.items():
                exclusion_counts[reason] += count
        for sentence_index, sentence in enumerate(role_sentences):
            sentence["sentence_index_within_role"] = sentence_index
        by_role[role] = role_sentences
    return by_role, dict(exclusion_counts)


def prepare_interviews(
    query_result: Any,
    nlp: Any,
    *,
    interviewer_mapping: Mapping[Any, Any] | None = None,
    max_interruption_tokens: int = MAX_INTERRUPTION_TOKENS,
) -> list[InterviewSentences]:
    """Turn a query result into deterministic, role-specific sentence streams."""

    if max_interruption_tokens < 0:
        raise ValueError("max_interruption_tokens must be non-negative")
    normalised_mapping = (
        {
            _identity_key(_jsonable(experiment_id)): _jsonable(interviewer_id)
            for experiment_id, interviewer_id in interviewer_mapping.items()
        }
        if interviewer_mapping is not None
        else None
    )
    prepared: list[InterviewSentences] = []
    seen_interviews: set[str] = set()
    for record in _query_records(query_result):
        raw_interview_id = record.get("experiment", record.get("interview_id"))
        if raw_interview_id is None:
            raise ValueError("An interview record is missing its experiment/interview ID")
        interview_id = _jsonable(raw_interview_id)
        key = _identity_key(interview_id)
        if key in seen_interviews:
            raise ValueError(f"Duplicate interview ID returned by query: {interview_id!r}")
        seen_interviews.add(key)
        interviewer_id = (
            normalised_mapping.get(key) if normalised_mapping is not None else None
        )
        if normalised_mapping is not None and interviewer_id is None:
            raise ValueError(
                f"No interviewer assignment found for interview {interview_id!r}"
            )

        turns, raw_count, empty_count = clean_and_group_turns(record)
        sentences_by_role, exclusion_counts = segment_interview(
            interview_id,
            turns,
            nlp,
            max_interruption_tokens=max_interruption_tokens,
        )
        for role in SPEAKER_ROLES:
            speaker_id = interviewer_id if role == "Interviewer" else interview_id
            for sentence in sentences_by_role[role]:
                sentence["speaker_id"] = speaker_id
        interview = InterviewSentences(
            interview_id=interview_id,
            sentences_by_role=sentences_by_role,
            raw_turn_count=raw_count,
            empty_turn_count=empty_count,
            clean_turn_count=len(turns),
            technical_artifact_count=exclusion_counts.get("technical_artifact", 0),
            sentence_quality_exclusion_counts=exclusion_counts,
            interviewer_id=interviewer_id,
            max_interruption_tokens=max_interruption_tokens,
            turn_count_by_role={
                role: sum(turn.role == role for turn in turns)
                for role in SPEAKER_ROLES
            },
            dialogue_turns=list(turns),
        )
        if interview.total_available == 0:
            raise ValueError(f"Interview {interview_id!r} has no eligible sentences")
        prepared.append(interview)

    if not prepared:
        raise ValueError("The query returned no interviews")
    prepared.sort(key=lambda interview: _interview_sort_key(interview.interview_id))
    return prepared


def allocate_equal_quotas(
    capacities: Sequence[int],
    total: int,
    rng: np.random.Generator,
) -> list[int]:
    """Allocate equally, cap shortages, and fairly redistribute their deficit."""

    if not capacities:
        if total:
            raise ValueError("Cannot allocate a positive quota to no strata")
        return []
    if any(capacity < 0 for capacity in capacities):
        raise ValueError("Capacities must be non-negative")
    if total < 0 or total > sum(capacities):
        raise ValueError("Quota must be between zero and total capacity")

    base, remainder = divmod(total, len(capacities))
    quotas = [base] * len(capacities)
    if remainder:
        extra_indices = rng.choice(len(capacities), size=remainder, replace=False)
        for index in np.atleast_1d(extra_indices):
            quotas[int(index)] += 1

    quotas = [min(quota, capacity) for quota, capacity in zip(quotas, capacities)]
    deficit = total - sum(quotas)
    while deficit:
        eligible = [
            index for index, capacity in enumerate(capacities) if quotas[index] < capacity
        ]
        if not eligible:
            raise AssertionError("Quota redistribution exhausted all capacity")
        for index in rng.permutation(eligible):
            index = int(index)
            quotas[index] += 1
            deficit -= 1
            if deficit == 0:
                break
    return quotas


def allocate_interview_splits(
    interviews: Sequence[InterviewSentences],
    rng: np.random.Generator,
) -> list[str]:
    """Assign one interview per interviewer to test and validation."""

    held_out_count = N_TEST_INTERVIEWS + N_VAL_INTERVIEWS
    if len(interviews) < held_out_count:
        raise ValueError(
            f"At least {held_out_count} interviews are required for the requested "
            "test/validation split"
        )

    grouped_indices: dict[str, list[int]] = {}
    for index, interview in enumerate(interviews):
        if interview.interviewer_id is None:
            raise ValueError("Every interview needs an interviewer ID before splitting")
        grouped_indices.setdefault(
            _identity_key(interview.interviewer_id),
            [],
        ).append(index)
    if len(grouped_indices) != N_INTERVIEWERS:
        raise ValueError(
            f"Expected {N_INTERVIEWERS} interviewer IDs, found "
            f"{len(grouped_indices)}"
        )
    if any(len(indices) < 2 for indices in grouped_indices.values()):
        raise ValueError(
            "Each interviewer must have at least two interviews for distinct "
            "test and validation representation"
        )

    splits = ["train"] * len(interviews)
    for interviewer_key in sorted(grouped_indices):
        shuffled = [
            int(index) for index in rng.permutation(grouped_indices[interviewer_key])
        ]
        splits[shuffled[0]] = "test"
        splits[shuffled[1]] = "val"
    return splits


def enforce_split_disjoint_similarity(
    interviews: Sequence[InterviewSentences],
    interview_splits: Sequence[str],
    *,
    threshold: float = SPLIT_DISJOINT_SIMILARITY_THRESHOLD,
    min_tokens: int = SPLIT_DISJOINT_MIN_TOKENS,
) -> tuple[list[InterviewSentences], dict[str, Any]]:
    """Give each near-duplicate cluster to one split and drop the other copies.

    Interview-level splitting alone does not stop a recurring scripted question
    from reaching both training and evaluation data, because the same
    instrument is read in every interview. Every cluster formed at ``threshold``
    is therefore assigned to a single split before quotas are computed.
    Contested clusters go to the split with the fewest clusters per interview so
    far, which keeps per-interview capacity comparable across splits.
    """

    if len(interviews) != len(interview_splits):
        raise ValueError("Every interview needs exactly one split assignment")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("split_disjoint_similarity_threshold must be in (0, 1]")
    if threshold == 1.0:
        return list(interviews), {
            "split_disjoint_similarity_enforced": False,
            "split_disjoint_similarity_threshold": threshold,
        }

    index = build_sentence_similarity_index(
        interviews,
        threshold=threshold,
        min_tokens=min_tokens,
    )
    split_by_interview = {
        _identity_key(interview.interview_id): split
        for interview, split in zip(interviews, interview_splits)
    }
    splits_by_cluster: dict[str, set[str]] = defaultdict(set)
    for interview in interviews:
        split = split_by_interview[_identity_key(interview.interview_id)]
        for role in SPEAKER_ROLES:
            for sentence in interview.sentences_by_role[role]:
                cluster = index.cluster_by_sentence_object[id(sentence)]
                splits_by_cluster[cluster].add(split)

    interviews_per_split = {
        split: max(1, interview_splits.count(split)) for split in DATASET_SPLITS
    }
    owner_by_cluster: dict[str, str] = {}
    owned_counts = {split: 0 for split in DATASET_SPLITS}
    contested: list[str] = []
    for cluster in sorted(splits_by_cluster):
        splits = splits_by_cluster[cluster]
        if len(splits) == 1:
            owner = next(iter(splits))
            owner_by_cluster[cluster] = owner
            owned_counts[owner] += 1
        else:
            contested.append(cluster)
    for cluster in contested:
        owner = min(
            sorted(splits_by_cluster[cluster]),
            key=lambda split: (
                owned_counts[split] / interviews_per_split[split],
                DATASET_SPLITS.index(split),
            ),
        )
        owner_by_cluster[cluster] = owner
        owned_counts[owner] += 1

    filtered: list[InterviewSentences] = []
    removed = 0
    kept = 0
    for interview in interviews:
        split = split_by_interview[_identity_key(interview.interview_id)]
        sentences_by_role: dict[str, list[dict[str, Any]]] = {}
        for role in SPEAKER_ROLES:
            retained = []
            for sentence in interview.sentences_by_role[role]:
                cluster = index.cluster_by_sentence_object[id(sentence)]
                if owner_by_cluster[cluster] == split:
                    retained.append(sentence)
                else:
                    removed += 1
            kept += len(retained)
            sentences_by_role[role] = retained
        replacement = InterviewSentences(
            interview_id=interview.interview_id,
            sentences_by_role=sentences_by_role,
            raw_turn_count=interview.raw_turn_count,
            empty_turn_count=interview.empty_turn_count,
            clean_turn_count=interview.clean_turn_count,
            technical_artifact_count=interview.technical_artifact_count,
            sentence_quality_exclusion_counts=dict(
                interview.sentence_quality_exclusion_counts
            ),
            interviewer_id=interview.interviewer_id,
            max_interruption_tokens=interview.max_interruption_tokens,
            turn_count_by_role=dict(interview.turn_count_by_role),
            dialogue_turns=list(interview.dialogue_turns),
        )
        if replacement.total_available == 0:
            raise ValueError(
                f"Interview {interview.interview_id!r} has no candidates left "
                "after split-disjoint duplicate control"
            )
        filtered.append(replacement)

    summary = {
        "split_disjoint_similarity_enforced": True,
        "split_disjoint_similarity_threshold": threshold,
        "split_disjoint_min_tokens": min_tokens,
        "split_disjoint_rule": (
            "every near-duplicate cluster belongs to exactly one dataset split"
        ),
        "split_disjoint_clusters_total": len(splits_by_cluster),
        "split_disjoint_clusters_contested": len(contested),
        "split_disjoint_clusters_by_owner": dict(owned_counts),
        "split_disjoint_candidates_removed": removed,
        "split_disjoint_candidates_retained": kept,
    }
    return filtered, summary


def allocate_role_quotas(
    available_by_role: Mapping[str, int],
    interview_quota: int,
    *,
    min_role_sample: int = MIN_ROLE_SAMPLE,
) -> dict[str, int]:
    """Allocate proportionally while enforcing feasible per-role coverage."""

    if min_role_sample < 0:
        raise ValueError("min_role_sample must be non-negative")
    available = {role: int(available_by_role.get(role, 0)) for role in SPEAKER_ROLES}
    if any(value < 0 for value in available.values()):
        raise ValueError("Role capacities must be non-negative")
    total_available = sum(available.values())
    if not 0 <= interview_quota <= total_available:
        raise ValueError("Interview quota exceeds its sentence capacity")
    if interview_quota == 0:
        return {role: 0 for role in SPEAKER_ROLES}

    nonempty_roles = [role for role in SPEAKER_ROLES if available[role]]
    if interview_quota < len(nonempty_roles):
        raise ValueError("Interview quota is too small to represent every available role")

    interviewer_target = int(
        round(interview_quota * available["Interviewer"] / total_available)
    )
    quotas = {
        "Interviewer": min(interviewer_target, available["Interviewer"]),
        "Interviewee": min(
            interview_quota - interviewer_target,
            available["Interviewee"],
        ),
    }

    deficit = interview_quota - sum(quotas.values())
    for role in SPEAKER_ROLES:
        addition = min(deficit, available[role] - quotas[role])
        quotas[role] += addition
        deficit -= addition
    if deficit:
        raise AssertionError("Role quota redistribution exhausted all capacity")

    preferred_minimum = {
        role: min(min_role_sample, available[role]) for role in SPEAKER_ROLES
    }
    if interview_quota >= sum(preferred_minimum.values()):
        lower_bounds = preferred_minimum
    else:
        lower_bounds = {role: int(available[role] > 0) for role in SPEAKER_ROLES}

    for receiver in SPEAKER_ROLES:
        needed = max(0, lower_bounds[receiver] - quotas[receiver])
        if not needed:
            continue
        for donor in SPEAKER_ROLES:
            if donor == receiver:
                continue
            transferable = max(0, quotas[donor] - lower_bounds[donor])
            amount = min(needed, transferable)
            quotas[donor] -= amount
            quotas[receiver] += amount
            needed -= amount
        if needed:
            raise AssertionError("Could not enforce feasible role lower bounds")

    assert sum(quotas.values()) == interview_quota
    assert all(quotas[role] <= available[role] for role in SPEAKER_ROLES)
    return quotas


def _interviewer_quota_bounds(
    available_by_role: Mapping[str, int],
    interview_quota: int,
    min_role_sample: int,
    *,
    enforce_preferred_minimum: bool,
) -> tuple[int, int]:
    available = {role: int(available_by_role.get(role, 0)) for role in SPEAKER_ROLES}
    preferred_minimum = {
        role: min(min_role_sample, available[role]) for role in SPEAKER_ROLES
    }
    if (
        enforce_preferred_minimum
        and interview_quota >= sum(preferred_minimum.values())
    ):
        lower_by_role = preferred_minimum
    else:
        lower_by_role = {
            role: int(available[role] > 0) for role in SPEAKER_ROLES
        }

    interviewer_lower = max(
        lower_by_role["Interviewer"],
        interview_quota - available["Interviewee"],
    )
    interviewer_upper = min(
        available["Interviewer"],
        interview_quota - lower_by_role["Interviewee"],
    )
    if interviewer_lower > interviewer_upper:
        raise ValueError("No feasible interviewer quota exists for an interview")
    return interviewer_lower, interviewer_upper


def _adjust_quotas_to_total(
    preferred: Sequence[int],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    target: int,
    rng: np.random.Generator,
) -> list[int]:
    quotas = [
        min(max(int(value), int(lower)), int(upper))
        for value, lower, upper in zip(preferred, lower_bounds, upper_bounds)
    ]
    if not sum(lower_bounds) <= target <= sum(upper_bounds):
        raise ValueError("Requested group quota is outside feasible bounds")

    while sum(quotas) != target:
        direction = 1 if sum(quotas) < target else -1
        candidates = [
            index
            for index, quota in enumerate(quotas)
            if (
                direction > 0 and quota < upper_bounds[index]
            ) or (
                direction < 0 and quota > lower_bounds[index]
            )
        ]
        if not candidates:
            raise AssertionError("Speaker quota adjustment exhausted feasible bounds")
        costs = {
            index: (
                abs((quotas[index] + direction) - preferred[index])
                - abs(quotas[index] - preferred[index])
            )
            for index in candidates
        }
        best_cost = min(costs.values())
        tied = sorted(index for index, cost in costs.items() if cost == best_cost)
        chosen = int(rng.choice(tied))
        quotas[chosen] += direction
    return quotas


def allocate_speaker_balanced_role_quotas(
    interviews: Sequence[InterviewSentences],
    interview_quotas: Sequence[int],
    interview_splits: Sequence[str],
    rng: np.random.Generator,
    *,
    min_role_sample: int = MIN_ROLE_SAMPLE,
    role_capacities: Sequence[Mapping[str, int]] | None = None,
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    """Prefer equal recurring-interviewer totals and protect held-out roles."""

    if not (
        len(interviews) == len(interview_quotas) == len(interview_splits)
    ):
        raise ValueError(
            "Interviews, interview quotas, and interview splits must have "
            "equal length"
        )
    if any(split not in DATASET_SPLITS for split in interview_splits):
        raise ValueError("Every interview split must be train, val, or test")
    if any(interview.interviewer_id is None for interview in interviews):
        raise ValueError("Every interview requires an interviewer ID before sampling")

    if role_capacities is not None and len(role_capacities) != len(interviews):
        raise ValueError("role_capacities must cover every interview")
    preferred_role_quotas: list[dict[str, int]] = []
    available_by_interview: list[dict[str, int]] = []
    for index, (interview, interview_quota) in enumerate(
        zip(interviews, interview_quotas)
    ):
        # Capacity is the number of distinct similarity clusters the role can
        # supply, never its raw sentence count: only one sentence per cluster
        # may be sampled.
        available = {
            role: (
                int(role_capacities[index][role])
                if role_capacities is not None
                else len(interview.sentences_by_role[role])
            )
            for role in SPEAKER_ROLES
        }
        available_by_interview.append(available)
        preferred_role_quotas.append(
            allocate_role_quotas(
                available,
                interview_quota,
                min_role_sample=min_role_sample,
            )
        )

    group_indices: dict[str, list[int]] = {}
    group_values: dict[str, Any] = {}
    for index, interview in enumerate(interviews):
        key = _identity_key(interview.interviewer_id)
        group_indices.setdefault(key, []).append(index)
        group_values[key] = _jsonable(interview.interviewer_id)
    group_keys = sorted(group_indices)
    if len(group_keys) != N_INTERVIEWERS:
        raise ValueError(
            f"Expected {N_INTERVIEWERS} interviewer IDs, found {len(group_keys)}"
        )

    def bounds_for_all(
        *,
        relax_training_minimum: bool,
    ) -> tuple[list[int], list[int]]:
        lower_bounds: list[int] = []
        upper_bounds: list[int] = []
        for available, interview_quota, dataset_split in zip(
            available_by_interview,
            interview_quotas,
            interview_splits,
        ):
            lower, upper = _interviewer_quota_bounds(
                available,
                interview_quota,
                min_role_sample,
                enforce_preferred_minimum=(
                    not relax_training_minimum or dataset_split != "train"
                ),
            )
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        return lower_bounds, upper_bounds

    minimum_relaxed = False
    lower_bounds, upper_bounds = bounds_for_all(relax_training_minimum=False)

    def common_range(
        lowers: Sequence[int],
        uppers: Sequence[int],
    ) -> tuple[int, int]:
        return (
            max(sum(lowers[index] for index in group_indices[key]) for key in group_keys),
            min(sum(uppers[index] for index in group_indices[key]) for key in group_keys),
        )

    common_lower, common_upper = common_range(lower_bounds, upper_bounds)
    if common_lower > common_upper:
        minimum_relaxed = True
        lower_bounds, upper_bounds = bounds_for_all(relax_training_minimum=True)
        common_lower, common_upper = common_range(lower_bounds, upper_bounds)

    preferred_interviewer_total = sum(
        quotas["Interviewer"] for quotas in preferred_role_quotas
    )
    desired_common_quota = int(round(preferred_interviewer_total / len(group_keys)))
    exact_balance_feasible = common_lower <= common_upper
    if exact_balance_feasible:
        balance_anchor = min(
            max(desired_common_quota, common_lower), common_upper
        )
    else:
        # The feasible group intervals do not overlap. The midpoint of the
        # narrowest gap minimizes imbalance before the later duplicate-safe
        # rebalance, while every hard per-interview bound remains enforced.
        balance_anchor = int(round((common_upper + common_lower) / 2.0))

    group_targets = {
        key: min(
            max(
                balance_anchor,
                sum(lower_bounds[index] for index in group_indices[key]),
            ),
            sum(upper_bounds[index] for index in group_indices[key]),
        )
        for key in group_keys
    }

    interviewer_quotas = [0] * len(interviews)
    for key in group_keys:
        indices = group_indices[key]
        adjusted = _adjust_quotas_to_total(
            [preferred_role_quotas[index]["Interviewer"] for index in indices],
            [lower_bounds[index] for index in indices],
            [upper_bounds[index] for index in indices],
            group_targets[key],
            rng,
        )
        for index, quota in zip(indices, adjusted):
            interviewer_quotas[index] = quota

    role_quotas = [
        {
            "Interviewer": interviewer_quota,
            "Interviewee": interview_quota - interviewer_quota,
        }
        for interviewer_quota, interview_quota in zip(
            interviewer_quotas,
            interview_quotas,
        )
    ]
    totals_by_interviewer = {
        key: sum(role_quotas[index]["Interviewer"] for index in group_indices[key])
        for key in group_keys
    }
    assert totals_by_interviewer == group_targets
    common_quota = (
        next(iter(totals_by_interviewer.values()))
        if len(set(totals_by_interviewer.values())) == 1
        else None
    )
    return role_quotas, {
        "interviewer_sentence_quota_each": common_quota,
        "preferred_interviewer_sentence_quota_each": desired_common_quota,
        "interviewer_balance_exact_at_allocation": common_quota is not None,
        "interviewer_sentence_quotas_at_allocation": [
            {
                "interviewer_id": group_values[key],
                "sentence_quota": totals_by_interviewer[key],
            }
            for key in group_keys
        ],
        "number_of_interviewer_ids": len(group_keys),
        "minimum_role_sample_relaxed_for_interviewer_balance": minimum_relaxed,
        "minimum_role_sample_relaxation_scope": (
            "training_interviews_only" if minimum_relaxed else "none"
        ),
        "held_out_minimum_role_sample_enforced": True,
        "interviewer_ids": [group_values[key] for key in group_keys],
    }


def split_into_position_strata(
    sentences: Sequence[dict[str, Any]],
    n_turns: int,
    n_strata: int = N_POSITION_STRATA,
) -> list[list[dict[str, Any]]]:
    """Assign sentences using the position of their primary source turn."""

    if n_turns < 0:
        raise ValueError("n_turns must be non-negative")
    turn_index_strata = np.array_split(np.arange(n_turns), n_strata)
    turn_to_stratum = {
        int(turn_index): stratum_index
        for stratum_index, turn_indices in enumerate(turn_index_strata)
        for turn_index in turn_indices
    }
    strata: list[list[dict[str, Any]]] = [[] for _ in range(n_strata)]
    for sentence in sentences:
        primary_turn = int(sentence["primary_turn_index_within_role"])
        if primary_turn not in turn_to_stratum:
            raise ValueError(
                f"Sentence refers to missing role turn index {primary_turn}"
            )
        strata[turn_to_stratum[primary_turn]].append(sentence)
    return strata


def _sample_strata(
    strata: Sequence[Sequence[dict[str, Any]]],
    quota: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Allocate by position, then balance each quota across source turns."""

    stratum_quotas = allocate_equal_quotas(
        [len(stratum) for stratum in strata],
        quota,
        rng,
    )
    sampled: list[dict[str, Any]] = []
    represented_turn_counts: list[int] = []
    for stratum_number, (stratum, stratum_quota) in enumerate(
        zip(strata, stratum_quotas), start=1
    ):
        if not stratum_quota:
            represented_turn_counts.append(0)
            continue

        sentences_by_turn: dict[int, list[dict[str, Any]]] = {}
        for sentence in stratum:
            primary_turn = int(sentence["primary_turn_index_within_role"])
            sentences_by_turn.setdefault(primary_turn, []).append(sentence)
        ordered_turns = sorted(sentences_by_turn)
        turn_quotas = allocate_equal_quotas(
            [len(sentences_by_turn[turn]) for turn in ordered_turns],
            stratum_quota,
            rng,
        )
        represented_turn_counts.append(sum(turn_quota > 0 for turn_quota in turn_quotas))
        for turn, turn_quota in zip(ordered_turns, turn_quotas):
            if not turn_quota:
                continue
            turn_sentences = sentences_by_turn[turn]
            chosen = rng.choice(len(turn_sentences), size=turn_quota, replace=False)
            for index in np.atleast_1d(chosen):
                row = dict(turn_sentences[int(index)])
                row.update(
                    {
                        "position_stratum": stratum_number,
                        "sampling_method": SAMPLING_METHOD,
                    }
                )
                sampled.append(row)
    return sampled, stratum_quotas, represented_turn_counts


def _share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _role_turn_count(interview: InterviewSentences, role: str) -> int:
    if role in interview.turn_count_by_role:
        return int(interview.turn_count_by_role[role])
    sentences = interview.sentences_by_role[role]
    if not sentences:
        return 0
    return 1 + max(
        int(sentence["primary_turn_index_within_role"])
        for sentence in sentences
    )


def _stable_item_uuid(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        (
            _jsonable(row["interview_id"]),
            str(row["speaker_role"]),
            int(row["sentence_index_within_role"]),
            normalise_sentence_for_similarity(str(row["text"])),
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
    return str(uuid.UUID(digest))


def _turn_context(
    interview: InterviewSentences,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    turns = interview.dialogue_turns
    source_indices = sorted(
        {
            int(index)
            for index in row.get("source_turn_indices_within_interview", [])
        }
    )
    if not turns or not source_indices:
        return {
            "previous_turn_role": "",
            "previous_turn_text": "",
            "source_turn_text": "",
            "intervening_turns": [],
            "following_turn_role": "",
            "following_turn_text": "",
        }
    if source_indices[0] < 0 or source_indices[-1] >= len(turns):
        raise ValueError("Sentence context refers to an unavailable dialogue turn")

    previous_index = source_indices[0] - 1
    following_index = source_indices[-1] + 1
    previous = turns[previous_index] if previous_index >= 0 else None
    following = turns[following_index] if following_index < len(turns) else None
    source_set = set(source_indices)
    intervening = [
        {
            "turn_index_within_interview": index,
            "role": turns[index].role,
            "text": turns[index].text,
            "source_turn_ids": turns[index].source_turn_ids,
        }
        for index in range(source_indices[0] + 1, source_indices[-1])
        if index not in source_set
    ]
    return {
        "previous_turn_role": previous.role if previous else "",
        "previous_turn_text": previous.text if previous else "",
        "source_turn_text": " ".join(turns[index].text for index in source_indices),
        "intervening_turns": intervening,
        "following_turn_role": following.role if following else "",
        "following_turn_text": following.text if following else "",
    }


def assign_annotation_metadata(
    sample: Sequence[dict[str, Any]],
    interviews: Sequence[InterviewSentences],
    rng: np.random.Generator,
    *,
    overlap_fraction: float = DOUBLE_ANNOTATION_FRACTION,
) -> dict[str, Any]:
    """Add stable IDs, real turn context, randomized order, and overlap flags."""

    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("Double-annotation fraction must be in [0, 1]")
    interview_by_key = {
        _identity_key(interview.interview_id): interview
        for interview in interviews
    }
    for row in sample:
        row["item_uuid"] = _stable_item_uuid(row)
        row["n_tokens"] = int(row.get("n_tokens", len(_words(row["text"]))))
        row["short_responsive"] = bool(
            row.get("short_responsive", _is_short_responsive(row["text"]))
        )
        row.setdefault("finite_verb", None)
        row.setdefault("fragment", None)
        row.update(
            _turn_context(
                interview_by_key[_identity_key(row["interview_id"])],
                row,
            )
        )

    presentation_permutation = rng.permutation(len(sample))
    for presentation_order, sample_index in enumerate(presentation_permutation):
        sample[int(sample_index)]["presentation_order"] = presentation_order
        sample[int(sample_index)]["double_annotate"] = False

    overlap_count = int(round(overlap_fraction * len(sample)))
    overlap_positions: set[int] = set()
    if overlap_count:
        step = len(sample) / overlap_count
        offset = float(rng.random()) * step
        overlap_positions = {
            min(len(sample) - 1, int(np.floor(offset + index * step)))
            for index in range(overlap_count)
        }
        row_by_position = {
            int(row["presentation_order"]): row for row in sample
        }
        for position in overlap_positions:
            row_by_position[position]["double_annotate"] = True

    item_ids = [str(row["item_uuid"]) for row in sample]
    if len(item_ids) != len(set(item_ids)):
        raise AssertionError("Stable annotation item UUIDs are not unique")
    return {
        "annotation_presentation_randomized": True,
        "annotation_presentation_order_scope": "complete_sample",
        "double_annotation_fraction": overlap_fraction,
        "double_annotation_items": len(overlap_positions),
        "double_annotation_selection": (
            "systematic_across_seeded_random_presentation_order"
        ),
        "annotation_context_source": "cleaned_chronological_query2_turns",
    }


@dataclass(slots=True)
class _SamplingCell:
    key: tuple[Any, ...]
    quota: int
    sentences: list[dict[str, Any]]


@dataclass(slots=True)
class _FlowEdge:
    to: int
    reverse_index: int
    residual_capacity: int
    original_capacity: int


class _DinicFlow:
    def __init__(self, node_count: int) -> None:
        self.graph: list[list[_FlowEdge]] = [[] for _ in range(node_count)]

    def add_edge(self, source: int, target: int, capacity: int) -> _FlowEdge:
        forward = _FlowEdge(
            to=target,
            reverse_index=len(self.graph[target]),
            residual_capacity=capacity,
            original_capacity=capacity,
        )
        reverse = _FlowEdge(
            to=source,
            reverse_index=len(self.graph[source]),
            residual_capacity=0,
            original_capacity=0,
        )
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return forward

    def maximum_flow(self, source: int, sink: int, target: int) -> int:
        total = 0
        while total < target:
            levels = [-1] * len(self.graph)
            levels[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge.residual_capacity > 0 and levels[edge.to] < 0:
                        levels[edge.to] = levels[node] + 1
                        queue.append(edge.to)
            if levels[sink] < 0:
                break

            next_edge = [0] * len(self.graph)

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while next_edge[node] < len(self.graph[node]):
                    edge = self.graph[node][next_edge[node]]
                    if (
                        edge.residual_capacity > 0
                        and levels[edge.to] == levels[node] + 1
                    ):
                        sent = send(
                            edge.to,
                            min(amount, edge.residual_capacity),
                        )
                        if sent:
                            edge.residual_capacity -= sent
                            reverse = self.graph[edge.to][edge.reverse_index]
                            reverse.residual_capacity += sent
                            return sent
                    next_edge[node] += 1
                return 0

            while total < target:
                sent = send(source, target - total)
                if not sent:
                    break
                total += sent
        return total


def _build_deduplicated_sampling_cells(
    interviews: Sequence[InterviewSentences],
    role_quotas_by_interview: Sequence[Mapping[str, int]],
    rng: np.random.Generator,
    *,
    min_role_sample: int,
) -> tuple[
    dict[str, list[_SamplingCell]],
    dict[tuple[str, str], list[list[dict[str, Any]]]],
    dict[int, int],
]:
    """Create successively relaxable turn/position/role quota cells."""

    cells_by_level: dict[str, list[_SamplingCell]] = {
        "turn_and_position_fixed": [],
        "position_fixed_turns_redistributed": [],
        "role_fixed_position_and_turns_redistributed": [],
        "interview_fixed_role_minima_only": [],
    }
    strata_by_interview_role: dict[
        tuple[str, str], list[list[dict[str, Any]]]
    ] = {}
    position_by_sentence_object: dict[int, int] = {}

    for interview, role_quotas in zip(interviews, role_quotas_by_interview):
        interview_key = _identity_key(interview.interview_id)
        interview_quota = sum(int(role_quotas[role]) for role in SPEAKER_ROLES)
        available_by_role = {
            role: len(interview.sentences_by_role[role]) for role in SPEAKER_ROLES
        }
        preferred_minimums = {
            role: min(min_role_sample, available_by_role[role])
            for role in SPEAKER_ROLES
        }
        if interview_quota >= sum(preferred_minimums.values()):
            hard_minimums = preferred_minimums
        else:
            hard_minimums = {
                role: int(available_by_role[role] > 0) for role in SPEAKER_ROLES
            }
        flexible_quota = interview_quota - sum(hard_minimums.values())
        if flexible_quota:
            cells_by_level["interview_fixed_role_minima_only"].append(
                _SamplingCell(
                    key=(interview_key, "flexible_remainder"),
                    quota=flexible_quota,
                    sentences=[
                        sentence
                        for role in SPEAKER_ROLES
                        for sentence in interview.sentences_by_role[role]
                    ],
                )
            )
        for role in SPEAKER_ROLES:
            sentences = interview.sentences_by_role[role]
            role_quota = int(role_quotas[role])
            strata = split_into_position_strata(
                sentences,
                _role_turn_count(interview, role),
            )
            strata_by_interview_role[(interview_key, role)] = strata
            for stratum_number, stratum in enumerate(strata, start=1):
                for sentence in stratum:
                    position_by_sentence_object[id(sentence)] = stratum_number

            hard_minimum = hard_minimums[role]
            if hard_minimum:
                hard_position_quotas = allocate_equal_quotas(
                    [len(stratum) for stratum in strata],
                    hard_minimum,
                    rng,
                )
                for stratum_number, (stratum, position_quota) in enumerate(
                    zip(strata, hard_position_quotas),
                    start=1,
                ):
                    if position_quota:
                        cells_by_level[
                            "interview_fixed_role_minima_only"
                        ].append(
                            _SamplingCell(
                                key=(
                                    interview_key,
                                    role,
                                    "hard_minimum_position",
                                    stratum_number,
                                ),
                                quota=position_quota,
                                sentences=list(stratum),
                            )
                        )

            if not role_quota:
                continue
            cells_by_level[
                "role_fixed_position_and_turns_redistributed"
            ].append(
                _SamplingCell(
                    key=(interview_key, role),
                    quota=role_quota,
                    sentences=list(sentences),
                )
            )
            stratum_quotas = allocate_equal_quotas(
                [len(stratum) for stratum in strata],
                role_quota,
                rng,
            )
            for stratum_number, (stratum, stratum_quota) in enumerate(
                zip(strata, stratum_quotas),
                start=1,
            ):
                if not stratum_quota:
                    continue
                cells_by_level["position_fixed_turns_redistributed"].append(
                    _SamplingCell(
                        key=(interview_key, role, stratum_number),
                        quota=stratum_quota,
                        sentences=list(stratum),
                    )
                )
                sentences_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(
                    list
                )
                for sentence in stratum:
                    turn_index = int(
                        sentence["primary_turn_index_within_role"]
                    )
                    sentences_by_turn[turn_index].append(sentence)
                ordered_turns = sorted(sentences_by_turn)
                turn_quotas = allocate_equal_quotas(
                    [len(sentences_by_turn[turn]) for turn in ordered_turns],
                    stratum_quota,
                    rng,
                )
                for turn, turn_quota in zip(ordered_turns, turn_quotas):
                    if not turn_quota:
                        continue
                    cells_by_level["turn_and_position_fixed"].append(
                        _SamplingCell(
                            key=(interview_key, role, stratum_number, turn),
                            quota=turn_quota,
                            sentences=list(sentences_by_turn[turn]),
                        )
                    )

    return (
        cells_by_level,
        strata_by_interview_role,
        position_by_sentence_object,
    )


def _solve_similarity_constrained_cells(
    cells: Sequence[_SamplingCell],
    similarity_index: _SimilarityIndex,
    target: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]] | None, int]:
    """Select cell quotas with at most one sentence per similarity cluster."""

    clusters_by_cell: list[dict[str, list[dict[str, Any]]]] = []
    all_clusters: set[str] = set()
    for cell in cells:
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sentence in cell.sentences:
            cluster_id = similarity_index.cluster_by_sentence_object[id(sentence)]
            clusters[cluster_id].append(sentence)
            all_clusters.add(cluster_id)
        clusters_by_cell.append(clusters)

    cluster_ids = sorted(all_clusters)
    cluster_number = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    source = 0
    cell_offset = 1
    cluster_offset = cell_offset + len(cells)
    sink = cluster_offset + len(cluster_ids)
    flow = _DinicFlow(sink + 1)
    edge_references: list[
        dict[str, tuple[_FlowEdge, list[dict[str, Any]]]]
    ] = [dict() for _ in cells]

    for cell_index in rng.permutation(len(cells)):
        cell_index = int(cell_index)
        cell = cells[cell_index]
        flow.add_edge(source, cell_offset + cell_index, cell.quota)
        candidate_clusters = list(clusters_by_cell[cell_index])
        rng.shuffle(candidate_clusters)
        for cluster_id in candidate_clusters:
            edge = flow.add_edge(
                cell_offset + cell_index,
                cluster_offset + cluster_number[cluster_id],
                1,
            )
            edge_references[cell_index][cluster_id] = (
                edge,
                clusters_by_cell[cell_index][cluster_id],
            )
    shuffled_clusters = list(cluster_ids)
    rng.shuffle(shuffled_clusters)
    for cluster_id in shuffled_clusters:
        flow.add_edge(
            cluster_offset + cluster_number[cluster_id],
            sink,
            1,
        )

    achieved = flow.maximum_flow(source, sink, target)
    if achieved != target:
        return None, achieved

    selected: list[dict[str, Any]] = []
    for references in edge_references:
        for cluster_id, (edge, candidates) in references.items():
            if edge.original_capacity == 1 and edge.residual_capacity == 0:
                chosen = candidates[int(rng.integers(len(candidates)))]
                row = dict(chosen)
                row["_source_sentence_object_id"] = id(chosen)
                row["similarity_cluster_id"] = cluster_id
                row["similarity_cluster_size"] = (
                    similarity_index.cluster_sizes[cluster_id]
                )
                row["similarity_cluster_interview_count"] = (
                    similarity_index.cluster_interview_counts[cluster_id]
                )
                row["scripted_recurrence"] = (
                    cluster_id in similarity_index.scripted_recurrence_clusters
                )
                selected.append(row)
    if len(selected) != target:
        raise AssertionError(
            f"Similarity-constrained flow selected {len(selected)}, expected {target}"
        )
    return selected, achieved


def _rebalance_recurring_interviewers(
    selected: list[dict[str, Any]],
    interviews: Sequence[InterviewSentences],
    interview_splits: Sequence[str],
    similarity_index: _SimilarityIndex,
    position_by_sentence_object: Mapping[int, int],
    rng: np.random.Generator,
    *,
    min_role_sample: int,
) -> dict[str, Any]:
    """Improve interviewer equality through duplicate-safe within-interview swaps."""

    interview_by_key = {
        _identity_key(interview.interview_id): interview
        for interview in interviews
    }
    if len(interview_splits) != len(interviews):
        raise ValueError("Every interview must have one split before rebalancing")
    split_by_interview = {
        _identity_key(interview.interview_id): split
        for interview, split in zip(interviews, interview_splits)
    }
    selected_clusters = {
        str(row["similarity_cluster_id"]) for row in selected
    }
    selected_source_objects = {
        int(row["_source_sentence_object_id"]) for row in selected
    }
    selected_role_counts: dict[tuple[str, str], int] = defaultdict(int)
    selected_position_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    interviewer_counts: dict[str, int] = defaultdict(int)
    interviewer_values: dict[str, Any] = {}
    for row in selected:
        interview_key = _identity_key(row["interview_id"])
        role = str(row["speaker_role"])
        selected_role_counts[(interview_key, role)] += 1
        source_object = int(row["_source_sentence_object_id"])
        selected_position_counts[
            (interview_key, role, int(position_by_sentence_object[source_object]))
        ] += 1
        if role == "Interviewer":
            interviewer_key = _identity_key(row["speaker_id"])
            interviewer_counts[interviewer_key] += 1
            interviewer_values[interviewer_key] = _jsonable(row["speaker_id"])

    interviewer_interviews: dict[str, list[str]] = defaultdict(list)
    for interview in interviews:
        interviewer_key = _identity_key(interview.interviewer_id)
        interviewer_values[interviewer_key] = _jsonable(interview.interviewer_id)
        interviewer_interviews[interviewer_key].append(
            _identity_key(interview.interview_id)
        )

    before = dict(interviewer_counts)

    def imbalance(counts: Mapping[str, int]) -> float:
        values = list(counts.values())
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values)

    def proposed_improvement(interviewer_key: str, direction: int) -> float:
        proposed = dict(interviewer_counts)
        proposed[interviewer_key] += direction
        return imbalance(interviewer_counts) - imbalance(proposed)

    def find_swap(
        interviewer_key: str,
        direction: int,
    ) -> tuple[int, dict[str, Any]] | None:
        new_role = "Interviewer" if direction > 0 else "Interviewee"
        old_role = "Interviewee" if direction > 0 else "Interviewer"
        options: list[
            tuple[tuple[int, int, int, float], int, dict[str, Any]]
        ] = []
        represented_turns = {
            (
                _identity_key(row["interview_id"]),
                str(row["speaker_role"]),
                int(row["primary_turn_index_within_role"]),
            )
            for row in selected
        }
        for interview_key in interviewer_interviews[interviewer_key]:
            interview = interview_by_key[interview_key]
            # Interviewees are unique speakers, so their per-interview count is
            # also their corpus-level speaker count. Held-out role coverage is
            # protected as well. A recurring interviewer in a training
            # interview may fall below the role-preferred minimum because the
            # hard requirement applies to that interviewer identity globally.
            protected_minimum = (
                min_role_sample
                if old_role == "Interviewee"
                or split_by_interview[interview_key] in {"val", "test"}
                else 1
            )
            old_minimum = min(
                protected_minimum,
                len(interview.sentences_by_role[old_role]),
            )
            if selected_role_counts[(interview_key, old_role)] <= old_minimum:
                continue
            removable_indices = [
                index
                for index, row in enumerate(selected)
                if _identity_key(row["interview_id"]) == interview_key
                and row["speaker_role"] == old_role
                and selected_position_counts[
                    (
                        interview_key,
                        old_role,
                        int(
                            position_by_sentence_object[
                                int(row["_source_sentence_object_id"])
                            ]
                        ),
                    )
                ]
                > 1
            ]
            if not removable_indices:
                continue
            candidate_sentences = []
            for sentence in interview.sentences_by_role[new_role]:
                sentence_object = id(sentence)
                cluster_id = similarity_index.cluster_by_sentence_object[
                    sentence_object
                ]
                if sentence_object in selected_source_objects:
                    continue
                if cluster_id in selected_clusters:
                    continue
                candidate_sentences.append((sentence, sentence_object, cluster_id))
            if not candidate_sentences:
                continue

            removable_index = max(
                removable_indices,
                key=lambda index: (
                    int(selected[index]["similarity_cluster_size"]),
                    float(rng.random()),
                ),
            )
            removed_position = int(
                position_by_sentence_object[
                    int(selected[removable_index]["_source_sentence_object_id"])
                ]
            )
            for sentence, sentence_object, cluster_id in candidate_sentences:
                candidate_position = int(
                    position_by_sentence_object[sentence_object]
                )
                turn_key = (
                    interview_key,
                    new_role,
                    int(sentence["primary_turn_index_within_role"]),
                )
                score = (
                    int(candidate_position != removed_position),
                    int(turn_key in represented_turns),
                    similarity_index.cluster_sizes[cluster_id],
                    float(rng.random()),
                )
                options.append((score, removable_index, sentence))
        if not options:
            return None
        _, removable_index, sentence = min(options, key=lambda option: option[0])
        return removable_index, sentence

    operations = 0
    increases = 0
    decreases = 0
    while True:
        moves: list[
            tuple[float, float, str, int, int, dict[str, Any]]
        ] = []
        for interviewer_key in sorted(interviewer_counts):
            for direction in (-1, 1):
                improvement = proposed_improvement(interviewer_key, direction)
                if improvement <= 0:
                    continue
                swap = find_swap(interviewer_key, direction)
                if swap is None:
                    continue
                removable_index, sentence = swap
                moves.append(
                    (
                        -improvement,
                        float(rng.random()),
                        interviewer_key,
                        direction,
                        removable_index,
                        sentence,
                    )
                )
        if not moves:
            break
        (
            _,
            _,
            interviewer_key,
            direction,
            removable_index,
            replacement,
        ) = min(moves, key=lambda move: (move[0], move[1]))
        removed = selected[removable_index]
        removed_interview_key = _identity_key(removed["interview_id"])
        removed_role = str(removed["speaker_role"])
        removed_cluster = str(removed["similarity_cluster_id"])
        removed_source_object = int(removed["_source_sentence_object_id"])
        removed_position = int(position_by_sentence_object[removed_source_object])

        replacement_object = id(replacement)
        replacement_cluster = similarity_index.cluster_by_sentence_object[
            replacement_object
        ]
        replacement_position = int(
            position_by_sentence_object[replacement_object]
        )
        replacement_row = dict(replacement)
        replacement_row["_source_sentence_object_id"] = replacement_object
        replacement_row["similarity_cluster_id"] = replacement_cluster
        replacement_row["similarity_cluster_size"] = (
            similarity_index.cluster_sizes[replacement_cluster]
        )
        replacement_row["similarity_cluster_interview_count"] = (
            similarity_index.cluster_interview_counts[replacement_cluster]
        )
        replacement_row["scripted_recurrence"] = (
            replacement_cluster
            in similarity_index.scripted_recurrence_clusters
        )
        selected[removable_index] = replacement_row

        selected_clusters.remove(removed_cluster)
        selected_clusters.add(replacement_cluster)
        selected_source_objects.remove(removed_source_object)
        selected_source_objects.add(replacement_object)
        selected_role_counts[(removed_interview_key, removed_role)] -= 1
        selected_role_counts[
            (removed_interview_key, str(replacement["speaker_role"]))
        ] += 1
        selected_position_counts[
            (removed_interview_key, removed_role, removed_position)
        ] -= 1
        selected_position_counts[
            (
                removed_interview_key,
                str(replacement["speaker_role"]),
                replacement_position,
            )
        ] += 1
        interviewer_counts[interviewer_key] += direction
        operations += 1
        increases += int(direction > 0)
        decreases += int(direction < 0)

    serialise_counts = lambda counts: {
        str(interviewer_values[key]): counts[key]
        for key in sorted(counts)
    }
    return {
        "interviewer_rebalancing_method": (
            "duplicate_safe_within_interview_role_swaps_minimising_squared_imbalance"
        ),
        "interviewer_rebalancing_protected_minima": (
            "10 for unique interviewees and both held-out roles; recurring "
            "interviewers are protected at speaker level across training interviews"
        ),
        "interviewer_counts_before_similarity_rebalancing": serialise_counts(before),
        "interviewer_counts_after_similarity_rebalancing": serialise_counts(
            interviewer_counts
        ),
        "interviewer_rebalancing_swaps": operations,
        "interviewer_rebalancing_increases": increases,
        "interviewer_rebalancing_decreases": decreases,
        "interviewer_imbalance_squared_before": imbalance(before),
        "interviewer_imbalance_squared_after": imbalance(interviewer_counts),
    }


def sample_globally_without_similar_sentences(
    interviews: Sequence[InterviewSentences],
    role_quotas_by_interview: Sequence[Mapping[str, int]],
    interview_splits: Sequence[str],
    rng: np.random.Generator,
    *,
    min_role_sample: int = MIN_ROLE_SAMPLE,
    similarity_threshold: float = NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
    near_duplicate_min_tokens: int = NEAR_DUPLICATE_MIN_TOKENS,
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[tuple[str, str], list[list[dict[str, Any]]]],
    dict[str, Any],
]:
    """Draw all quotas jointly while forbidding global similarity duplicates."""

    target = sum(
        int(role_quotas[role])
        for role_quotas in role_quotas_by_interview
        for role in SPEAKER_ROLES
    )
    similarity_index = build_sentence_similarity_index(
        interviews,
        threshold=similarity_threshold,
        min_tokens=near_duplicate_min_tokens,
    )
    (
        cells_by_level,
        strata_by_interview_role,
        position_by_sentence_object,
    ) = _build_deduplicated_sampling_cells(
        interviews,
        role_quotas_by_interview,
        rng,
        min_role_sample=min_role_sample,
    )

    selected: list[dict[str, Any]] | None = None
    selected_level = ""
    achieved_by_level: dict[str, int] = {}
    for level, cells in cells_by_level.items():
        selected, achieved = _solve_similarity_constrained_cells(
            cells,
            similarity_index,
            target,
            rng,
        )
        achieved_by_level[level] = achieved
        if selected is not None:
            selected_level = level
            break
    if selected is None:
        interview_capacity_diagnostics = []
        for interview, role_quotas in zip(
            interviews, role_quotas_by_interview
        ):
            cluster_ids = {
                similarity_index.cluster_by_sentence_object[id(sentence)]
                for role in SPEAKER_ROLES
                for sentence in interview.sentences_by_role[role]
            }
            quota = sum(int(role_quotas[role]) for role in SPEAKER_ROLES)
            if len(cluster_ids) <= quota + 10:
                interview_capacity_diagnostics.append(
                    {
                        "interview_id": interview.interview_id,
                        "quota": quota,
                        "candidates": interview.total_available,
                        "unique_similarity_clusters": len(cluster_ids),
                        "exclusions": interview.sentence_quality_exclusion_counts,
                    }
                )
        raise ValueError(
            "Cannot select the requested sample without exact/near-duplicate "
            "sentences under any configured stratification relaxation level. "
            f"Maximum feasible counts by relaxation level: {achieved_by_level}; "
            "low-capacity interviews: "
            f"{interview_capacity_diagnostics!r}"
        )

    interviewer_rebalancing_summary = _rebalance_recurring_interviewers(
        selected,
        interviews,
        interview_splits,
        similarity_index,
        position_by_sentence_object,
        rng,
        min_role_sample=min_role_sample,
    )

    selected_by_interview_role: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    selected_cluster_ids: set[str] = set()
    for row in selected:
        sentence_object_id = int(row.pop("_source_sentence_object_id"))
        row["position_stratum"] = position_by_sentence_object[sentence_object_id]
        row["sampling_method"] = SAMPLING_METHOD
        cluster_id = str(row["similarity_cluster_id"])
        if cluster_id in selected_cluster_ids:
            raise AssertionError("A similarity cluster was selected twice")
        selected_cluster_ids.add(cluster_id)
        selected_by_interview_role[
            (_identity_key(row["interview_id"]), row["speaker_role"])
        ].append(row)

    selected_duplicate_clusters = [
        cluster_id
        for cluster_id in selected_cluster_ids
        if similarity_index.cluster_sizes[cluster_id] > 1
    ]
    deduplication_summary = dict(similarity_index.summary)
    deduplication_summary.update(
        {
            "deduplication_enabled": True,
            "selected_similarity_clusters": len(selected_cluster_ids),
            "sample_similarity_cluster_duplicates": 0,
            "selected_clusters_with_alternative_duplicates": len(
                selected_duplicate_clusters
            ),
            "duplicate_alternatives_suppressed_for_selected_clusters": sum(
                similarity_index.cluster_sizes[cluster_id] - 1
                for cluster_id in selected_duplicate_clusters
            ),
            "deduplication_quota_relaxation_level": selected_level,
            "deduplication_flow_achieved_by_level": achieved_by_level,
            **interviewer_rebalancing_summary,
        }
    )
    return (
        dict(selected_by_interview_role),
        strata_by_interview_role,
        deduplication_summary,
    )


def validate_sample(
    interviews: Sequence[InterviewSentences],
    sample: Sequence[Mapping[str, Any]],
    interview_quotas: Sequence[int],
    interview_splits: Sequence[str],
    requested_sample_size: int,
    min_role_sample: int,
    available_capacity: int | None = None,
) -> None:
    # Availability is measured in distinct similarity clusters, because only
    # one sentence per cluster may be sampled; the raw sentence count would
    # overstate what the corpus can deliver.
    total_available = (
        sum(interview.total_available for interview in interviews)
        if available_capacity is None
        else available_capacity
    )
    expected_size = min(requested_sample_size, total_available)
    assert len(sample) == expected_size
    if total_available >= requested_sample_size:
        assert len(sample) == requested_sample_size

    for row in sample:
        text = str(row["text"])
        assert len(_words(text)) >= MIN_SENTENCE_TOKENS, (
            f"Sampled sentence has fewer than {MIN_SENTENCE_TOKENS} words: {text!r}"
        )
        assert starts_with_sentence_capital(text), (
            f"Sampled sentence does not start with a capital letter: {text!r}"
        )
        assert ends_with_sentence_boundary(text), (
            f"Sampled sentence lacks terminal punctuation: {text!r}"
        )
        assert not text.rstrip().endswith(("...", "…")), (
            f"Sampled sentence ends in a truncating ellipsis: {text!r}"
        )

    source_keys = [
        (
            _identity_key(row["interview_id"]),
            row["speaker_role"],
            int(row["sentence_index_within_role"]),
        )
        for row in sample
    ]
    assert len(source_keys) == len(set(source_keys)), "A source sentence was sampled twice"
    similarity_cluster_ids = [
        str(row["similarity_cluster_id"]) for row in sample
    ]
    assert all(similarity_cluster_ids), "A sampled sentence lacks a similarity cluster"
    assert all(int(row["similarity_cluster_size"]) >= 1 for row in sample)
    assert all(
        int(row["similarity_cluster_interview_count"]) >= 1 for row in sample
    )
    assert len(similarity_cluster_ids) == len(set(similarity_cluster_ids)), (
        "Exact or near-duplicate sentence text was sampled more than once"
    )
    item_uuids = [str(row["item_uuid"]) for row in sample]
    assert len(item_uuids) == len(set(item_uuids)), "Annotation item UUID collision"
    assert sorted(int(row["presentation_order"]) for row in sample) == list(
        range(len(sample))
    ), "Presentation order is not a complete permutation"

    assert len(interview_splits) == len(interviews)
    assert interview_splits.count("test") == N_TEST_INTERVIEWS
    assert interview_splits.count("val") == N_VAL_INTERVIEWS
    assert interview_splits.count("train") == (
        len(interviews) - N_TEST_INTERVIEWS - N_VAL_INTERVIEWS
    )

    sampled_by_interview_role: dict[tuple[str, str], int] = {}
    observed_splits: dict[str, set[str]] = {}
    interview_by_key = {
        _identity_key(interview.interview_id): interview
        for interview in interviews
    }
    for row in sample:
        interview_key = _identity_key(row["interview_id"])
        split = str(row["split"])
        assert split in DATASET_SPLITS
        role = str(row["speaker_role"])
        expected_speaker_id = (
            interview_by_key[interview_key].interviewer_id
            if role == "Interviewer"
            else interview_by_key[interview_key].interview_id
        )
        assert _identity_key(row["speaker_id"]) == _identity_key(
            expected_speaker_id
        )
        source_role_turns = [
            int(index) for index in row["source_turn_indices_within_role"]
        ]
        primary_turn = int(row["primary_turn_index_within_role"])
        assert source_role_turns and primary_turn == source_role_turns[0]
        assert 0 <= primary_turn < _role_turn_count(
            interview_by_key[interview_key],
            role,
        )
        observed_splits.setdefault(interview_key, set()).add(split)
        key = (interview_key, role)
        sampled_by_interview_role[key] = sampled_by_interview_role.get(key, 0) + 1

    for interview, quota, expected_split in zip(
        interviews,
        interview_quotas,
        interview_splits,
    ):
        interview_key = _identity_key(interview.interview_id)
        assert observed_splits.get(interview_key) == {expected_split}, (
            f"Interview {interview.interview_id!r} crossed dataset splits"
        )
        represented = sum(
            sampled_by_interview_role.get((interview_key, role), 0)
            for role in SPEAKER_ROLES
        )
        assert represented == quota and represented > 0
        available_roles = [
            role for role in SPEAKER_ROLES if interview.sentences_by_role[role]
        ]
        if len(available_roles) == 2:
            assert all(
                sampled_by_interview_role.get((interview_key, role), 0) > 0
                for role in available_roles
            )
        if expected_split in {"val", "test"}:
            preferred_minimums = {
                role: min(
                    min_role_sample,
                    len(interview.sentences_by_role[role]),
                )
                for role in SPEAKER_ROLES
            }
            if quota >= sum(preferred_minimums.values()):
                assert all(
                    sampled_by_interview_role.get((interview_key, role), 0)
                    >= preferred_minimums[role]
                    for role in available_roles
                ), (
                    f"Held-out interview {interview.interview_id!r} violated "
                    "the role sampling minimum"
                )

    available_by_speaker: dict[tuple[str, str], int] = {}
    for interview in interviews:
        for role in SPEAKER_ROLES:
            speaker_id = (
                interview.interviewer_id
                if role == "Interviewer"
                else interview.interview_id
            )
            speaker_key = (role, _identity_key(speaker_id))
            available_by_speaker[speaker_key] = (
                available_by_speaker.get(speaker_key, 0)
                + len(interview.sentences_by_role[role])
            )
    sampled_by_speaker: dict[tuple[str, str], int] = {}
    for row in sample:
        speaker_key = (
            str(row["speaker_role"]),
            _identity_key(row["speaker_id"]),
        )
        sampled_by_speaker[speaker_key] = sampled_by_speaker.get(speaker_key, 0) + 1
    assert set(sampled_by_speaker) == set(available_by_speaker)
    assert all(
        sampled_by_speaker[speaker_key]
        >= min(min_role_sample, available_count)
        for speaker_key, available_count in available_by_speaker.items()
    ), "At least one speaker received fewer than the required minimum"

    for split in ("val", "test"):
        split_interviewer_ids = {
            _identity_key(row["speaker_id"])
            for row in sample
            if row["split"] == split and row["speaker_role"] == "Interviewer"
        }
        assert len(split_interviewer_ids) == N_INTERVIEWERS


def stratified_sample(
    interviews: Sequence[InterviewSentences],
    *,
    total_sample_size: int = TOTAL_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED,
    min_role_sample: int = MIN_ROLE_SAMPLE,
    near_duplicate_similarity_threshold: float = (
        NEAR_DUPLICATE_SIMILARITY_THRESHOLD
    ),
    near_duplicate_min_tokens: int = NEAR_DUPLICATE_MIN_TOKENS,
    double_annotation_fraction: float = DOUBLE_ANNOTATION_FRACTION,
    split_disjoint_similarity_threshold: float = (
        SPLIT_DISJOINT_SIMILARITY_THRESHOLD
    ),
    split_disjoint_min_tokens: int = SPLIT_DISJOINT_MIN_TOKENS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Sample interviews, roles, turn-position strata, turns, then sentences."""

    if total_sample_size <= 0:
        raise ValueError("total_sample_size must be positive")
    if not 0.0 < near_duplicate_similarity_threshold <= 1.0:
        raise ValueError("near_duplicate_similarity_threshold must be in (0, 1]")
    if near_duplicate_min_tokens < 1:
        raise ValueError("near_duplicate_min_tokens must be positive")
    if not 0.0 <= double_annotation_fraction <= 1.0:
        raise ValueError("double_annotation_fraction must be in [0, 1]")
    if not 0.0 < split_disjoint_similarity_threshold <= 1.0:
        raise ValueError("split_disjoint_similarity_threshold must be in (0, 1]")
    if split_disjoint_min_tokens < 1:
        raise ValueError("split_disjoint_min_tokens must be positive")
    ordered_interviews = sorted(
        interviews, key=lambda interview: _interview_sort_key(interview.interview_id)
    )
    if not ordered_interviews:
        raise ValueError("At least one interview is required")

    rng = np.random.default_rng(random_seed)
    interview_splits = allocate_interview_splits(ordered_interviews, rng)
    candidates_before_split_control = sum(
        interview.total_available for interview in ordered_interviews
    )
    ordered_interviews, split_disjoint_summary = enforce_split_disjoint_similarity(
        ordered_interviews,
        interview_splits,
        threshold=split_disjoint_similarity_threshold,
        min_tokens=split_disjoint_min_tokens,
    )
    split_disjoint_summary["candidates_before_split_disjoint_control"] = (
        candidates_before_split_control
    )
    # An interview can only contribute as many rows as it has distinct
    # similarity clusters: cluster uniqueness, not the raw sentence count, is
    # its real capacity. Allocating against sentence counts would hand a
    # duplicate-heavy interview a quota it can never fill, leaving the whole
    # sample permanently short by that difference.
    capacity_index = build_sentence_similarity_index(
        ordered_interviews,
        threshold=near_duplicate_similarity_threshold,
        min_tokens=near_duplicate_min_tokens,
    )
    role_capacities: list[dict[str, int]] = []
    interview_capacities: list[int] = []
    for interview in ordered_interviews:
        clusters_by_role = {
            role: {
                capacity_index.cluster_by_sentence_object[id(sentence)]
                for sentence in interview.sentences_by_role[role]
            }
            for role in SPEAKER_ROLES
        }
        # A cluster shared by both roles can still yield only one row, so the
        # interview capacity is the union, not the sum of the role capacities.
        interview_capacities.append(
            len(set().union(*clusters_by_role.values()))
        )
        role_capacities.append(
            {role: len(clusters_by_role[role]) for role in SPEAKER_ROLES}
        )
    total_cluster_capacity = sum(interview_capacities)
    total_available = sum(
        interview.total_available for interview in ordered_interviews
    )
    effective_sample_size = min(total_sample_size, total_cluster_capacity)
    if effective_sample_size < len(ordered_interviews):
        raise ValueError("Sample size is too small to represent every interview")

    interview_quotas = allocate_equal_quotas(
        interview_capacities,
        effective_sample_size,
        rng,
    )
    role_quotas_by_interview, interviewer_balance = (
        allocate_speaker_balanced_role_quotas(
            ordered_interviews,
            interview_quotas,
            interview_splits,
            rng,
            min_role_sample=min_role_sample,
            role_capacities=role_capacities,
        )
    )
    (
        selected_by_interview_role,
        planned_strata_by_interview_role,
        deduplication_summary,
    ) = sample_globally_without_similar_sentences(
        ordered_interviews,
        role_quotas_by_interview,
        interview_splits,
        rng,
        min_role_sample=min_role_sample,
        similarity_threshold=near_duplicate_similarity_threshold,
        near_duplicate_min_tokens=near_duplicate_min_tokens,
    )

    sample: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    interview_order = {
        _identity_key(interview.interview_id): index
        for index, interview in enumerate(ordered_interviews)
    }
    role_order = {role: index for index, role in enumerate(SPEAKER_ROLES)}
    split_order = {split: index for index, split in enumerate(DATASET_SPLITS)}

    for interview, interview_quota, dataset_split, role_quotas in zip(
        ordered_interviews,
        interview_quotas,
        interview_splits,
        role_quotas_by_interview,
    ):
        available = {
            role: len(interview.sentences_by_role[role]) for role in SPEAKER_ROLES
        }
        role_strata: dict[str, list[list[dict[str, Any]]]] = {}
        sampled_by_role: dict[str, list[dict[str, Any]]] = {}
        stratum_quotas_by_role: dict[str, list[int]] = {}
        represented_turns_by_role: dict[str, list[int]] = {}
        turn_counts_by_role = {
            role: _role_turn_count(interview, role) for role in SPEAKER_ROLES
        }

        for role in SPEAKER_ROLES:
            strata = planned_strata_by_interview_role[
                (_identity_key(interview.interview_id), role)
            ]
            role_sample = selected_by_interview_role.get(
                (_identity_key(interview.interview_id), role),
                [],
            )
            for row in role_sample:
                row["random_seed"] = random_seed
                row["split"] = dataset_split
            stratum_quotas = [
                sum(
                    int(row["position_stratum"]) == stratum_number
                    for row in role_sample
                )
                for stratum_number in range(1, N_POSITION_STRATA + 1)
            ]
            represented_turn_counts = [
                len(
                    {
                        int(row["primary_turn_index_within_role"])
                        for row in role_sample
                        if int(row["position_stratum"]) == stratum_number
                    }
                )
                for stratum_number in range(1, N_POSITION_STRATA + 1)
            ]
            role_strata[role] = strata
            sampled_by_role[role] = role_sample
            stratum_quotas_by_role[role] = stratum_quotas
            represented_turns_by_role[role] = represented_turn_counts
            sample.extend(role_sample)

        total_sampled = sum(len(rows) for rows in sampled_by_role.values())
        report_row: dict[str, Any] = {
            "interview_id": interview.interview_id,
            "interviewer_id": interview.interviewer_id,
            "interviewee_id": interview.interview_id,
            "split": dataset_split,
            "interviewer_available": available["Interviewer"],
            "interviewee_available": available["Interviewee"],
            "total_available": sum(available.values()),
            "interview_quota": interview_quota,
            "interviewer_sampled": len(sampled_by_role["Interviewer"]),
            "interviewee_sampled": len(sampled_by_role["Interviewee"]),
            "interviewer_turns_total": turn_counts_by_role["Interviewer"],
            "interviewee_turns_total": turn_counts_by_role["Interviewee"],
            "interviewer_turns_eligible": len(
                {
                    sentence["primary_turn_index_within_role"]
                    for sentence in interview.sentences_by_role["Interviewer"]
                }
            ),
            "interviewee_turns_eligible": len(
                {
                    sentence["primary_turn_index_within_role"]
                    for sentence in interview.sentences_by_role["Interviewee"]
                }
            ),
            "interviewer_turns_sampled": len(
                {
                    row["primary_turn_index_within_role"]
                    for row in sampled_by_role["Interviewer"]
                }
            ),
            "interviewee_turns_sampled": len(
                {
                    row["primary_turn_index_within_role"]
                    for row in sampled_by_role["Interviewee"]
                }
            ),
            "interviewer_share_available": _share(
                available["Interviewer"], sum(available.values())
            ),
            "interviewer_share_sampled": _share(
                len(sampled_by_role["Interviewer"]), total_sampled
            ),
            "interviewee_share_available": _share(
                available["Interviewee"], sum(available.values())
            ),
            "interviewee_share_sampled": _share(
                len(sampled_by_role["Interviewee"]), total_sampled
            ),
        }
        for role in SPEAKER_ROLES:
            role_label = role.casefold()
            for stratum_index in range(N_POSITION_STRATA):
                prefix = f"{role_label}_s{stratum_index + 1}"
                report_row[f"{prefix}_available"] = len(
                    role_strata[role][stratum_index]
                )
                report_row[f"{prefix}_sampled"] = stratum_quotas_by_role[role][
                    stratum_index
                ]
                report_row[f"{prefix}_turns_eligible"] = len(
                    {
                        sentence["primary_turn_index_within_role"]
                        for sentence in role_strata[role][stratum_index]
                    }
                )
                report_row[f"{prefix}_turns_sampled"] = represented_turns_by_role[
                    role
                ][stratum_index]
        report.append(report_row)

    sample.sort(
        key=lambda row: (
            split_order[row["split"]],
            interview_order[_identity_key(row["interview_id"])],
            role_order[row["speaker_role"]],
            row["sentence_index_within_role"],
        )
    )
    annotation_summary = assign_annotation_metadata(
        sample,
        ordered_interviews,
        rng,
        overlap_fraction=double_annotation_fraction,
    )
    validate_sample(
        ordered_interviews,
        sample,
        interview_quotas,
        interview_splits,
        total_sample_size,
        min_role_sample,
        available_capacity=total_cluster_capacity,
    )

    interviewer_sampled = sum(
        row["speaker_role"] == "Interviewer" for row in sample
    )
    interviewee_sampled = len(sample) - interviewer_sampled
    sampled_per_interview = [row["interview_quota"] for row in report]
    interviews_by_split = {
        split: interview_splits.count(split) for split in DATASET_SPLITS
    }
    available_by_split = {
        split: sum(
            interview.total_available
            for interview, interview_split in zip(
                ordered_interviews,
                interview_splits,
            )
            if interview_split == split
        )
        for split in DATASET_SPLITS
    }
    sampled_by_split = {
        split: sum(row["split"] == split for row in sample)
        for split in DATASET_SPLITS
    }
    interviewer_sentence_counts: dict[str, int] = {}
    interviewer_values: dict[str, Any] = {}
    interviewer_interview_counts: dict[str, int] = {}
    interviewer_available_counts: dict[str, int] = {}
    interviewer_split_counts: dict[str, dict[str, int]] = {}
    for interview in ordered_interviews:
        interviewer_key = _identity_key(interview.interviewer_id)
        interviewer_values[interviewer_key] = _jsonable(interview.interviewer_id)
        interviewer_interview_counts[interviewer_key] = (
            interviewer_interview_counts.get(interviewer_key, 0) + 1
        )
        interviewer_available_counts[interviewer_key] = (
            interviewer_available_counts.get(interviewer_key, 0)
            + len(interview.sentences_by_role["Interviewer"])
        )
    for row in sample:
        if row["speaker_role"] != "Interviewer":
            continue
        interviewer_key = _identity_key(row["speaker_id"])
        interviewer_values[interviewer_key] = _jsonable(row["speaker_id"])
        interviewer_sentence_counts[interviewer_key] = (
            interviewer_sentence_counts.get(interviewer_key, 0) + 1
        )
        split_counts = interviewer_split_counts.setdefault(
            interviewer_key,
            {split: 0 for split in DATASET_SPLITS},
        )
        split_counts[row["split"]] += 1
    interviewer_sampling_report = [
        {
            "interviewer_id": interviewer_values[key],
            "interviews_conducted": interviewer_interview_counts[key],
            "sentences_available": interviewer_available_counts[key],
            "sentences_sampled": interviewer_sentence_counts.get(key, 0),
            "train_sentences_sampled": interviewer_split_counts.get(key, {}).get(
                "train", 0
            ),
            "val_sentences_sampled": interviewer_split_counts.get(key, {}).get(
                "val", 0
            ),
            "test_sentences_sampled": interviewer_split_counts.get(key, {}).get(
                "test", 0
            ),
        }
        for key in sorted(interviewer_values)
    ]
    speaker_sampling_report = [
        {
            "speaker_role": "Interviewer",
            "speaker_id": row["interviewer_id"],
            "interviews_represented": row["interviews_conducted"],
            "sentences_available": row["sentences_available"],
            "sentences_sampled": row["sentences_sampled"],
        }
        for row in interviewer_sampling_report
    ] + [
        {
            "speaker_role": "Interviewee",
            "speaker_id": _jsonable(interview.interview_id),
            "interviews_represented": 1,
            "sentences_available": len(
                interview.sentences_by_role["Interviewee"]
            ),
            "sentences_sampled": next(
                int(row["interviewee_sampled"])
                for row in report
                if _identity_key(row["interview_id"])
                == _identity_key(interview.interview_id)
            ),
        }
        for interview in ordered_interviews
    ]
    interviewer_sample_totals = [
        row["sentences_sampled"] for row in interviewer_sampling_report
    ]
    interviewer_balance_exact = len(set(interviewer_sample_totals)) == 1
    sampled_turn_keys = {
        (
            _identity_key(row["interview_id"]),
            row["speaker_role"],
            int(row["primary_turn_index_within_role"]),
        )
        for row in sample
    }
    eligible_turn_keys = {
        (
            _identity_key(interview.interview_id),
            role,
            int(sentence["primary_turn_index_within_role"]),
        )
        for interview in ordered_interviews
        for role in SPEAKER_ROLES
        for sentence in interview.sentences_by_role[role]
    }
    interviewer_turns_sampled = sum(
        role == "Interviewer" for _, role, _ in sampled_turn_keys
    )
    interviewee_turns_sampled = len(sampled_turn_keys) - interviewer_turns_sampled
    summary = {
        "requested_sample_size": total_sample_size,
        "total_sentences_available": total_available,
        "total_unique_cluster_capacity": total_cluster_capacity,
        "interview_quota_capacity_basis": (
            "distinct similarity clusters per interview, not raw sentence counts"
        ),
        "total_sentences_sampled": len(sample),
        "total_interviewer_sentences_available": sum(
            len(interview.sentences_by_role["Interviewer"])
            for interview in ordered_interviews
        ),
        "total_interviewee_sentences_available": sum(
            len(interview.sentences_by_role["Interviewee"])
            for interview in ordered_interviews
        ),
        "total_interviewer_sentences_sampled": interviewer_sampled,
        "total_interviewee_sentences_sampled": interviewee_sampled,
        "total_interviewer_turns": sum(
            _role_turn_count(interview, "Interviewer")
            for interview in ordered_interviews
        ),
        "total_interviewee_turns": sum(
            _role_turn_count(interview, "Interviewee")
            for interview in ordered_interviews
        ),
        "total_interviewer_turns_eligible": sum(
            role == "Interviewer" for _, role, _ in eligible_turn_keys
        ),
        "total_interviewee_turns_eligible": sum(
            role == "Interviewee" for _, role, _ in eligible_turn_keys
        ),
        "total_interviewer_turns_sampled": interviewer_turns_sampled,
        "total_interviewee_turns_sampled": interviewee_turns_sampled,
        "total_turns_sampled": len(sampled_turn_keys),
        "percentage_interviewer": 100.0 * _share(interviewer_sampled, len(sample)),
        "percentage_interviewee": 100.0 * _share(interviewee_sampled, len(sample)),
        "minimum_sentences_per_interview": min(sampled_per_interview),
        "maximum_sentences_per_interview": max(sampled_per_interview),
        "number_of_interviews_represented": len(ordered_interviews),
        "number_of_unique_interviewees": len(ordered_interviews),
        "number_of_unique_interviewers": len(interviewer_sampling_report),
        "interviewee_id_source": "interview_id",
        "interviewer_id_source": "get_mapping()",
        "interviewer_balance_scope": "complete_corpus",
        "interviewer_balance_exact": interviewer_balance_exact,
        "interviewer_sentence_quota_each": (
            interviewer_sample_totals[0] if interviewer_balance_exact else None
        ),
        "preferred_interviewer_sentence_quota_each": interviewer_balance[
            "preferred_interviewer_sentence_quota_each"
        ],
        "interviewer_balance_exact_at_allocation": interviewer_balance[
            "interviewer_balance_exact_at_allocation"
        ],
        "interviewer_sentence_quotas_at_allocation": interviewer_balance[
            "interviewer_sentence_quotas_at_allocation"
        ],
        "interviewer_balance_relaxed_for_candidate_capacity": (
            not interviewer_balance["interviewer_balance_exact_at_allocation"]
        ),
        "interviewer_balance_relaxed_for_similarity_deduplication": (
            interviewer_balance["interviewer_balance_exact_at_allocation"]
            and not interviewer_balance_exact
        ),
        "interviewer_balance_not_exact_after_all_constraints": (
            not interviewer_balance_exact
        ),
        "interviewer_sampling": interviewer_sampling_report,
        "speaker_sampling": speaker_sampling_report,
        "minimum_samples_for_any_speaker": min(
            row["sentences_sampled"] for row in speaker_sampling_report
        ),
        "minimum_role_sample_relaxed_for_interviewer_balance": (
            interviewer_balance[
                "minimum_role_sample_relaxed_for_interviewer_balance"
            ]
        ),
        "minimum_role_sample_relaxation_scope": interviewer_balance[
            "minimum_role_sample_relaxation_scope"
        ],
        "held_out_minimum_role_sample_enforced": interviewer_balance[
            "held_out_minimum_role_sample_enforced"
        ],
        "minimum_per_speaker_enforced": True,
        "final_minimum_role_sample_enforced_for_every_interview": all(
            int(row[f"{role.casefold()}_sampled"])
            >= min(min_role_sample, int(row[f"{role.casefold()}_available"]))
            for row in report
            for role in SPEAKER_ROLES
        ),
        "number_of_train_interviews": interviews_by_split["train"],
        "number_of_val_interviews": interviews_by_split["val"],
        "number_of_test_interviews": interviews_by_split["test"],
        "train_sentences_available": available_by_split["train"],
        "val_sentences_available": available_by_split["val"],
        "test_sentences_available": available_by_split["test"],
        "train_sentences_sampled": sampled_by_split["train"],
        "val_sentences_sampled": sampled_by_split["val"],
        "test_sentences_sampled": sampled_by_split["test"],
        "random_seed": random_seed,
        "minimum_role_sample": min_role_sample,
        "number_of_position_strata": N_POSITION_STRATA,
        "maximum_interruption_tokens": ordered_interviews[
            0
        ].max_interruption_tokens,
        "sampling_method": SAMPLING_METHOD,
        "split_method": SPLIT_METHOD,
        "split_unit": "interview",
        "interview_split_overlap": False,
        "sentence_index_base": 0,
        "position_stratum_base": 1,
        "cross_turn_stratum_assignment": "initiating_primary_turn",
        "raw_turns": sum(interview.raw_turn_count for interview in ordered_interviews),
        "empty_turns_removed": sum(
            interview.empty_turn_count for interview in ordered_interviews
        ),
        "clean_turns": sum(
            interview.clean_turn_count for interview in ordered_interviews
        ),
        "technical_artifact_sentences_removed": sum(
            interview.technical_artifact_count for interview in ordered_interviews
        ),
        "sentence_candidates_excluded_by_quality": sum(
            count
            for interview in ordered_interviews
            for reason, count in interview.sentence_quality_exclusion_counts.items()
            if reason != "technical_artifact"
        ),
        "sentence_quality_exclusion_counts": {
            reason: sum(
                interview.sentence_quality_exclusion_counts.get(reason, 0)
                for interview in ordered_interviews
            )
            for reason in sorted(
                {
                    reason
                    for interview in ordered_interviews
                    for reason in interview.sentence_quality_exclusion_counts
                }
            )
        },
        "maximum_sentence_tokens": MAX_SENTENCE_TOKENS,
        "minimum_sentence_tokens": MIN_SENTENCE_TOKENS,
        "mandatory_sentence_form": (
            "at least 4 words, a capital-letter start, and terminal '.', '!' "
            "or '?' without a truncating ellipsis"
        ),
        "minimum_fragment_tokens_for_review": MIN_FRAGMENT_REVIEW_TOKENS,
        "sentence_boundary_policy": (
            "German spaCy boundaries reconciled with transcript punctuation; "
            "unsupported parser-only boundaries are merged and conservative "
            "capitalised question/pronoun starts may split ASR run-ons."
        ),
        "cross_turn_sentences_for_review": sum(
            sentence["cross_turn_sentence"]
            for interview in ordered_interviews
            for role in SPEAKER_ROLES
            for sentence in interview.sentences_by_role[role]
        ),
        "dialogue_order_source": "chronological database turns",
        "turn_provenance": (
            "Uses query-provided turn IDs when present; otherwise source_turn_ids "
            "are stable one-based positions in each interview's query2 turns array."
        ),
        "chunk_provenance": (
            "source_chunk_ids is retained when the query result provides chunk IDs; "
            "query2's current result shape does not, so the field is otherwise empty."
        ),
        "reconstruction_policy": (
            "Conservative A-B-A bridging: unfinished A, a short B below the "
            "configured token limit, and a likely A continuation; all "
            "reconstructed sentences require review."
        ),
        "sampling_uses_negation_information": False,
        **annotation_summary,
        **deduplication_summary,
        **split_disjoint_summary,
    }
    return sample, report, summary


def _csv_value(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def write_outputs(
    sample: Sequence[Mapping[str, Any]],
    report: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    sample_path: Path = DEFAULT_SAMPLE_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    annotation_path: Path = DEFAULT_ANNOTATION_PATH,
) -> None:
    write_csv(sample_path, sample, SAMPLE_FIELDNAMES)
    sample_suffix = sample_path.suffix or ".csv"
    sample_stem = sample_path.stem if sample_path.suffix else sample_path.name
    for dataset_split in DATASET_SPLITS:
        split_path = sample_path.with_name(
            f"{sample_stem}_{dataset_split}{sample_suffix}"
        )
        split_rows = [row for row in sample if row.get("split") == dataset_split]
        write_csv(split_path, split_rows, SAMPLE_FIELDNAMES)
    report_fieldnames = list(report[0]) if report else []
    write_csv(report_path, report, report_fieldnames)
    annotation_rows = sorted(
        sample,
        key=lambda row: int(row["presentation_order"]),
    )
    write_csv(annotation_path, annotation_rows, ANNOTATION_FIELDNAMES)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump(_jsonable(summary), output, ensure_ascii=False, indent=2)
        output.write("\n")


def load_sentence_segmenter(model_name: str = "de_core_news_sm") -> Any:
    import spacy

    return spacy.load(model_name, disable=["lemmatizer", "ner"])


def build_provenance(
    turn_query_result: Any,
    interviewer_mapping: Mapping[Any, Any],
    nlp: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash the two authorized query inputs, corpus result, model, and config."""

    stable_corpus = json.dumps(
        _jsonable(turn_query_result),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    stable_mapping = json.dumps(
        _jsonable(interviewer_mapping),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    stable_config = json.dumps(
        _jsonable(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    model_meta = getattr(nlp, "meta", {})
    return {
        "corpus_query2_result_sha256": hashlib.sha256(stable_corpus).hexdigest(),
        "interviewer_mapping_result_sha256": hashlib.sha256(
            stable_mapping
        ).hexdigest(),
        "query2_sha256": hashlib.sha256(query2.encode("utf-8")).hexdigest(),
        "query_mapping_sha256": hashlib.sha256(
            query_mapping.encode("utf-8")
        ).hexdigest(),
        "sampling_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(stable_config).hexdigest(),
        "config": _jsonable(config),
        "spacy_model": model_meta.get("name"),
        "spacy_model_version": model_meta.get("version"),
        "spacy_language": model_meta.get("lang"),
        "spacy_pipeline": list(getattr(nlp, "pipe_names", [])),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "authorized_database_queries": ["query_mapping", "query2"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=TOTAL_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--min-role-sample", type=int, default=MIN_ROLE_SAMPLE)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
    )
    parser.add_argument(
        "--near-duplicate-min-tokens",
        type=int,
        default=NEAR_DUPLICATE_MIN_TOKENS,
    )
    parser.add_argument(
        "--split-disjoint-threshold",
        type=float,
        default=SPLIT_DISJOINT_SIMILARITY_THRESHOLD,
        help=(
            "similarity threshold at which a near-duplicate cluster is confined "
            "to one dataset split; 1.0 disables the rule"
        ),
    )
    parser.add_argument(
        "--split-disjoint-min-tokens",
        type=int,
        default=SPLIT_DISJOINT_MIN_TOKENS,
    )
    parser.add_argument(
        "--double-annotation-fraction",
        type=float,
        default=DOUBLE_ANNOTATION_FRACTION,
    )
    parser.add_argument(
        "--max-interruption-tokens",
        type=int,
        default=MAX_INTERRUPTION_TOKENS,
    )
    parser.add_argument("--spacy-model", default="de_core_news_sm")
    parser.add_argument("--sample-output", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--annotation-output",
        type=Path,
        default=DEFAULT_ANNOTATION_PATH,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_role_sample < 0:
        raise ValueError("--min-role-sample must be non-negative")
    if args.max_interruption_tokens < 0:
        raise ValueError("--max-interruption-tokens must be non-negative")
    if not 0.0 < args.near_duplicate_threshold <= 1.0:
        raise ValueError("--near-duplicate-threshold must be in (0, 1]")
    if args.near_duplicate_min_tokens < 1:
        raise ValueError("--near-duplicate-min-tokens must be positive")
    if not 0.0 <= args.double_annotation_fraction <= 1.0:
        raise ValueError("--double-annotation-fraction must be in [0, 1]")
    if not 0.0 < args.split_disjoint_threshold <= 1.0:
        raise ValueError("--split-disjoint-threshold must be in (0, 1]")
    if args.split_disjoint_min_tokens < 1:
        raise ValueError("--split-disjoint-min-tokens must be positive")

    nlp = load_sentence_segmenter(args.spacy_model)
    interviewer_mapping = get_mapping()
    with surreal_client(SurrealClientConfig()) as db:
        query_result = db.query(query2)
    interviews = prepare_interviews(
        query_result,
        nlp,
        interviewer_mapping=interviewer_mapping,
        max_interruption_tokens=args.max_interruption_tokens,
    )
    sample, report, summary = stratified_sample(
        interviews,
        total_sample_size=args.sample_size,
        random_seed=args.seed,
        min_role_sample=args.min_role_sample,
        near_duplicate_similarity_threshold=args.near_duplicate_threshold,
        near_duplicate_min_tokens=args.near_duplicate_min_tokens,
        double_annotation_fraction=args.double_annotation_fraction,
        split_disjoint_similarity_threshold=args.split_disjoint_threshold,
        split_disjoint_min_tokens=args.split_disjoint_min_tokens,
    )
    config = {
        "sample_size": args.sample_size,
        "random_seed": args.seed,
        "minimum_role_or_speaker_sample": args.min_role_sample,
        "maximum_interruption_tokens": args.max_interruption_tokens,
        "maximum_sentence_tokens": MAX_SENTENCE_TOKENS,
        "minimum_sentence_tokens": MIN_SENTENCE_TOKENS,
        "minimum_fragment_tokens_for_review": MIN_FRAGMENT_REVIEW_TOKENS,
        "spacy_model_requested": args.spacy_model,
        "near_duplicate_similarity_threshold": args.near_duplicate_threshold,
        "near_duplicate_min_tokens": args.near_duplicate_min_tokens,
        "split_disjoint_similarity_threshold": args.split_disjoint_threshold,
        "split_disjoint_min_tokens": args.split_disjoint_min_tokens,
        "double_annotation_fraction": args.double_annotation_fraction,
        "number_of_position_strata": N_POSITION_STRATA,
        "number_of_test_interviews": N_TEST_INTERVIEWS,
        "number_of_validation_interviews": N_VAL_INTERVIEWS,
    }
    summary["provenance"] = build_provenance(
        query_result,
        interviewer_mapping,
        nlp,
        config,
    )
    summary["output_files"] = {
        "complete_sample": str(args.sample_output),
        "sampling_report": str(args.report_output),
        "sampling_summary": str(args.summary_output),
        "annotation_items_blinded_to_split_and_interview": str(
            args.annotation_output
        ),
    }
    write_outputs(
        sample,
        report,
        summary,
        sample_path=args.sample_output,
        report_path=args.report_output,
        summary_path=args.summary_output,
        annotation_path=args.annotation_output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    # print(*get_mapping().items(), sep="\n")
