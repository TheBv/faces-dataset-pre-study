from typing import Any, List, Optional, Dict
import os

import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer
from datasets import Dataset, DatasetDict
from functools import partial
from transformers import DataCollatorForTokenClassification

from .spacy_utils import process_sent_spacy, dep_dict, upos_dict


BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


def with_word_spaces(tokenizer: PreTrainedTokenizer, batch_words: List[List[str]]) -> List[List[str]]:
    """Byte-level BPE tokenizers (EuroBERT, Llama-style) encode pre-split words without their
    leading space ("ist" instead of "Ġist"), unlike the running text they were pretrained on.
    Prepend a space to every non-initial, non-special word; WordPiece/SentencePiece tokenizers
    strip it, so their encodings are unchanged. Word ids are unaffected."""
    special = set(tokenizer.all_special_tokens)
    return [[w if i == 0 or w in special else f" {w}" for i, w in enumerate(words)] for words in batch_words]


class PreprocessorUtility:
    @staticmethod
    def tokenize_and_align_labels(examples: Any, tokenizer: PreTrainedTokenizer, label_mask: str, max_length: int = 128) -> Any:
        # Tokenize inputs with padding to max_length and truncation
        tokenized_inputs = tokenizer(
            with_word_spaces(tokenizer, examples["tokens"]),
            truncation=True,
            is_split_into_words=True,
            padding="max_length",  # Pad to max_length
            max_length=max_length,  # Specify maximum sequence length
            return_tensors=None,  # Return Python lists, compatible with datasets
        )

        labels = []
        for i, label in enumerate(examples[label_mask]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set special tokens and padding to -100
                if word_idx is None:  # Special tokens (e.g., [CLS], [SEP])
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Label first token of a word
                    label_ids.append(label[word_idx])
                else:  # Subword tokens: ignored, so metrics are word level like the GCN models
                    label_ids.append(-100)
                previous_word_idx = word_idx
            # Pad labels to match max_length if necessary
            while len(label_ids) < max_length:
                label_ids.append(-100)  # Add -100 for padding tokens
            label_ids = label_ids[:max_length]  # Ensure labels match max_length
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    @staticmethod
    def tokenize_deps_pos_align_labels(examples: Any, tokenizer: PreTrainedTokenizer, spacy_model: Any, max_length: int = 128, special_token: Optional[str] = None) -> Any:
        pos_tags = []
        dep_tags = []
        edge_indices = []
        for sent in examples["tokens"]:
            # print(sent)
            upos, deps, edge_index = process_sent_spacy(sent, spacy_model)
            pos_tags.append(upos)
            dep_tags.append(deps)
            edge_indices.append(edge_index)

        if special_token is not None:
            if "cue" in special_token.lower():
                target_mask = examples["cue_masks"]
            else:
                raise Exception("Provide valid special token: [CUE]")
            for j in range(len(examples["tokens"])):
                for i in range(len(examples["tokens"][j])):
                    if target_mask[j][i] == 1:
                        examples["tokens"][j][i] = special_token


        result = PreprocessorUtility.retokenize_with_pos(tokens=examples["tokens"],
                                                pos_tags=pos_tags,
                                                dep_tags=dep_tags,
                                                pos_tag_to_id=upos_dict,
                                                dep_tag_to_id=dep_dict[f'{spacy_model.meta.get("lang", "xx")}_{spacy_model.meta.get("name", "unknown")}'],
                                                tokenizer=tokenizer,
                                                max_seq_length=max_length
                                                )
        """result["cue_labels"] = [lab for cue_mask in examples["cue_masks"] for lab in cue_mask]
        result["scope_labels"] = [lab for scope_mask in examples["scope_masks"] for lab in scope_mask]
        result["event_labels"] = [lab for event_mask in examples["event_masks"] for lab in event_mask]
        result["focus_labels"] = [lab for focus_mask in examples["focus_masks"] for lab in focus_mask]"""
        result["cue_labels"] = examples["cue_masks"]
        result["scope_labels"] = examples["scope_masks"]
        # result["event_labels"] = examples["event_masks"]
        # result["focus_labels"] = examples["focus_masks"]
        result["edge_indices"] = edge_indices
        return result


    @staticmethod
    def retokenize_with_pos(
            tokens: List[List[str]],
            pos_tags: Optional[List[List[str]]],
            dep_tags: Optional[List[List[str]]],
            pos_tag_to_id: Optional[Dict[str, int]],
            dep_tag_to_id: Optional[Dict[str, int]],
            tokenizer: PreTrainedTokenizer,
            max_seq_length: int = 128,
            default_id: int = 0) -> dict:

        # Ensure tokens and pos_tags have the same length
        assert len(tokens) == len(pos_tags), "Tokens and POS tags must have the same length"
        batch_size = len(tokens)
        # print(100*"=")
        # Convert POS tags to indices
        pos_indices = [[pos_tag_to_id.get(tag, default_id) for tag in sub_lst] for sub_lst in pos_tags]
        dep_indices = [[dep_tag_to_id.get(tag, default_id) for tag in sub_lst] for sub_lst in dep_tags]

        # Tokenize the sentence
        encoding = tokenizer(
            with_word_spaces(tokenizer, tokens),
            is_split_into_words=True,  # Input is pre-tokenized
            return_tensors=None,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True
        )

        input_ids = encoding['input_ids']  # Shape: [max_seq_length]
        attention_mask = encoding['attention_mask']  # Shape: [max_seq_length]

        # Get word IDs to align subwords with original tokens
        word_id_list = [encoding.word_ids(i) for i in range(len(encoding['input_ids']))]

        # Assign POS indices to subword tokens
        pos_ids = torch.full((batch_size, max_seq_length), default_id, dtype=torch.long).tolist()
        dep_ids = torch.full((batch_size, max_seq_length), default_id, dtype=torch.long).tolist()
        for j, word_ids in enumerate(word_id_list):
            for i, word_id in enumerate(word_ids):
                if word_id is not None:  # Non-special token

                    pos_ids[j][i] = pos_indices[j][word_id]
                    dep_ids[j][i] = dep_indices[j][word_id]


                # Special tokens ([CLS], [SEP], [PAD]) keep default_pos_id
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pos_ids": pos_ids,
            "dep_ids": dep_ids,
            "word_ids": word_id_list,
        }

    """
    @staticmethod
    def tokenize_and_align_labels_scope_detection(examples: Any,
                                                  tokenizer: PreTrainedTokenizer,
                                                  label_mask: str,
                                                  max_length: int = 128) -> Any:
        # Tokenize inputs with padding to max_length and truncation
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",  # Pad to max_length
            max_length=max_length,  # Specify maximum sequence length
            return_tensors=None,  # Return Python lists, compatible with datasets
        )

        labels = []
        for i, label in enumerate(examples[label_mask]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set special tokens and padding to -100
                if word_idx is None:  # Special tokens (e.g., [CLS], [SEP])
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            # Pad labels to match max_length if necessary
            while len(label_ids) < max_length:
                label_ids.append(-100)  # Add -100 for padding tokens
            label_ids = label_ids[:max_length]  # Ensure labels match max_length
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    """

if __name__ == "__main__":
    from .data import load_hf_dataset

    ds = load_hf_dataset("cue")["train"]
    ds = ds.map(partial(PreprocessorUtility.tokenize_and_align_labels,
                        tokenizer=AutoTokenizer.from_pretrained("google-bert/bert-base-german-cased"),
                        label_mask="cue_masks"), batched=True)
    print(ds[0]["tokens"])
    print(ds[0]["cue_masks"])
    print(ds[0]["input_ids"])
    print(ds[0]["labels"])
