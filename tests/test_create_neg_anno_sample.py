from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "psrc"))

from main_utils import create_neg_anno_sample as sampling  # noqa: E402


class SentenceCleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp = sampling.load_sentence_segmenter()

    def spans(self, text: str) -> list[str]:
        return [
            " ".join(span.text.split())
            for span in sampling._sentence_spans(self.nlp(text))
        ]

    def exclusion_reason(self, text: str) -> str | None:
        doc = self.nlp(text)
        return sampling._quality_exclusion_reason(doc, text)

    def test_merges_parser_only_question_stub(self) -> None:
        text = (
            "Wie hoch ist denn Ihr monatlicher Netto -Arbeitsverdienst für "
            "Ihre Tätigkeiten? Bitte schätzen Sie Ihren Gewinn."
        )
        self.assertEqual(
            self.spans(text),
            [
                "Wie hoch ist denn Ihr monatlicher Netto -Arbeitsverdienst "
                "für Ihre Tätigkeiten?",
                "Bitte schätzen Sie Ihren Gewinn.",
            ],
        )

    def test_does_not_split_common_abbreviation(self) -> None:
        text = "Nutzen Sie Angebote, z. B. Wikis, Podcasts oder YouTube?"
        self.assertEqual(self.spans(text), [text])

    def test_splits_high_confidence_unpunctuated_restarts(self) -> None:
        text = "Ich bin jetzt fertig dann beginnen wir mit der Befragung."
        self.assertEqual(
            self.spans(text),
            ["Ich bin jetzt fertig", "dann beginnen wir mit der Befragung."],
        )

    def test_splits_run_on_when_left_verb_is_mistagged(self) -> None:
        text = (
            "das gerade ja passt ja auch zur olympiade Jetzt sind ja Ihr "
            "aktueller Beruf und der Wunschberuf nicht der gleiche."
        )
        self.assertEqual(
            self.spans(text),
            [
                "das gerade ja passt ja auch zur olympiade",
                "Jetzt sind ja Ihr aktueller Beruf und der Wunschberuf nicht "
                "der gleiche.",
            ],
        )

    def test_splits_concatenated_likert_answers(self) -> None:
        text = "trifft er nicht zu Trifft gar nicht zu."
        self.assertEqual(
            self.spans(text),
            ["trifft er nicht zu", "Trifft gar nicht zu."],
        )

    def test_merges_parser_boundary_after_incomplete_also(self) -> None:
        text = (
            "Also ich versuche ihr, also sie hat ein sehr viel ausgeprägteres "
            "Ordnungsbedürfnis als ich."
        )
        self.assertEqual(self.spans(text), [text])

    def test_merges_parser_boundary_after_complement_taking_verb(self) -> None:
        text = (
            "Ich habe 2024 meinen Bachelor abgeschlossen, deswegen glaube "
            "ich zählt es nicht."
        )
        self.assertEqual(self.spans(text), [text])

    def test_merges_parser_boundary_after_modal(self) -> None:
        text = (
            "Ich denke, wo die Berufswahl recht offen ist, sollte man "
            "möglichst viel Raum geben für Praxiserfahrung."
        )
        self.assertEqual(self.spans(text), [text])

    def test_does_not_split_inside_unfinished_dependent_clause(self) -> None:
        text = (
            "Sie können sich gerne umsehen, und wenn Sie dann soweit sind, "
            "können wir starten."
        )
        self.assertEqual(self.spans(text), [text])

    def test_rejects_clear_incomplete_tail(self) -> None:
        self.assertEqual(
            self.exclusion_reason("Wir würden im Rahmen der"),
            "incomplete_final_syntax",
        )

    def test_retains_self_contained_dialogue_responses(self) -> None:
        for text in ("Nein.", "Eher nicht.", "Eine 9.", "Warum nicht?"):
            with self.subTest(text=text):
                self.assertIsNone(self.exclusion_reason(text))

    def test_strips_repeated_subtitle_credit_prefixes(self) -> None:
        text = (
            "Untertitelung des ZDF, 2020 Untertitelung des ZDF, 2020 "
            "Hm, trifft überwiegend nicht zu."
        )
        self.assertEqual(
            sampling._strip_technical_artifact_prefix(text),
            "Hm, trifft überwiegend nicht zu.",
        )
        self.assertTrue(sampling._is_technical_artifact("2020"))

    def test_rejects_out_of_order_brackets(self) -> None:
        self.assertFalse(sampling._has_balanced_ordered_brackets("UH] text ["))
        self.assertTrue(sampling._has_balanced_ordered_brackets("[UH] text"))

    def test_rejects_leading_question_complement(self) -> None:
        for text in (
            "Sie Zeit in einer virtuellen Welt, wo man Charaktere steuert?",
            "Sie Ihr Urteil abstufen.",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    self.exclusion_reason(text),
                    "leading_question_fragment",
                )

    def test_rejects_dangling_relative_tail(self) -> None:
        self.assertEqual(
            self.exclusion_reason(
                "Dort befindet sich auch ein Bildschirm, in welchem"
            ),
            "incomplete_final_syntax",
        )


class QuotaAllocationTests(unittest.TestCase):
    def test_uses_closest_feasible_interviewer_totals_when_exact_is_impossible(
        self,
    ) -> None:
        def interview(
            interview_id: int,
            interviewer_id: int,
            interviewer_available: int,
            interviewee_available: int,
        ) -> sampling.InterviewSentences:
            return sampling.InterviewSentences(
                interview_id=interview_id,
                interviewer_id=interviewer_id,
                sentences_by_role={
                    "Interviewer": [{} for _ in range(interviewer_available)],
                    "Interviewee": [{} for _ in range(interviewee_available)],
                },
                raw_turn_count=0,
                empty_turn_count=0,
                clean_turn_count=0,
                technical_artifact_count=0,
            )

        interviews = [
            interview(1, 1, 5, 20),
            interview(2, 2, 20, 1),
            interview(3, 3, 20, 20),
        ]
        role_quotas, report = sampling.allocate_speaker_balanced_role_quotas(
            interviews,
            [10, 10, 10],
            ["train", "train", "train"],
            np.random.default_rng(42),
        )

        self.assertEqual(
            [quota["Interviewer"] for quota in role_quotas],
            [5, 9, 7],
        )
        self.assertFalse(report["interviewer_balance_exact_at_allocation"])


if __name__ == "__main__":
    unittest.main()
