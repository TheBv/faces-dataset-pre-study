import copy
import json
from functools import partial

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
import spacy
from safetensors.torch import load_file
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizer, \
    DataCollatorForTokenClassification, PreTrainedTokenizerBase
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os

from .preprocessing import PreprocessorUtility, with_word_spaces
from .train_gcn import DataCollatorForTokenClassificationGCN, \
    DataCollatorForTokenClassificationGCNInference
from .spacy_utils import process_sent_spacy, upos_dict, dep_dict, get_spacy_model
from .gcn_bert_nn_module import BERTResidualGATv2ContextGatedFusion

BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


class BasicInference:
    special_tokens = ...

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        ...

    @classmethod
    def init_component(cls, model_path: str, device: Any, max_len: Any, **kwargs):
        ...

    def run(self, batch_tokens: List[List[str]], original_input: Optional[List[List[str]]] = None):
        ...

    @staticmethod
    def pretty_print(result: List[List[Dict[str, str]]]) -> None:
        for sent in result:
            for tok in sent:
                print(f"{str(tok['token']):<{15}} {str(tok['label']):<{15}}")
            print("\n")


class NegBertInference(BasicInference):

    @staticmethod
    def load_model_and_tokenizer(model_path: str, bert_model: str = "prajjwal1/bert-tiny") -> tuple:
        """Load the fine-tuned model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(bert_model, trust_remote_code=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer

    @staticmethod
    def preprocess_input(tokens: List[List[str]], tokenizer: PreTrainedTokenizer, max_length: int = 128) -> tuple:
        """Tokenize batched pre-split input tokens and return word IDs for merging subtokens."""
        # Tokenize with is_split_into_words=True to match training
        tokenized_inputs = tokenizer(
            with_word_spaces(tokenizer, tokens),
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"  # Return PyTorch tensors for inference
        )
        # Get word IDs for each sequence in the batch
        word_ids = [tokenized_inputs.word_ids(batch_index=i) for i in range(len(tokens))]
        return tokenized_inputs, word_ids

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        ...

    @staticmethod
    def predict(model: torch.nn.Module,
                tokenizer: AutoTokenizer,
                input_tokens,
                tokenized_inputs,
                word_ids: List[int],
                max_length: int = 128,
                device: str = "cuda"):
        ...

    @staticmethod
    def evaluate(model: torch.nn.Module,
                 tokenizer: AutoTokenizer,
                 dataset: Dataset,
                 device: str = "cuda"):
        ...


class NegBertInferenceGAT(BasicInference):

    @staticmethod
    def load_model_and_tokenizer(model_path: str,
                                 model_architecture: Any = BERTResidualGATv2ContextGatedFusion) -> tuple:
        """Load the fine-tuned model and tokenizer."""
        if os.path.isdir(model_path):
            # TODO: retrain models and save correct tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path
                                                      #, trust_remote_code=True
            )
            # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", trust_remote_code=True)
            with open(f"{model_path}/config.json", "r") as f:
                config = json.load(f)
            model = model_architecture(
                bert_id=config.get("bert_id"),
                id2label=config.get("id2label"),
                label2id=config.get("label2id"),
                num_labels=config.get("lcount"),
                pos_vocab_size=config.get("pos_vocab_size"),  # Update as needed
                dep_vocab_size=config.get("dep_vocab_size")  # Update as needed
            )
            if config.get("vocab_size") != model.bert.config.vocab_size:
                model.bert.resize_token_embeddings(config.get("vocab_size"))
            # 3. Load the safetensors file
            state_dict = load_file(f"{model_path}/model.safetensors")

            # 4. Load the weights into your model
            missing, unexpected = model.load_state_dict(state_dict, strict=True)

            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            weights_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
            config_path = hf_hub_download(repo_id=model_path, filename="config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            model = model_architecture(
                bert_id=config.get("bert_id"),
                id2label=config.get("id2label"),
                label2id=config.get("label2id"),
                num_labels=config.get("lcount"),
                pos_vocab_size=config.get("pos_vocab_size"),  # Update as needed
                dep_vocab_size=config.get("dep_vocab_size")  # Update as needed
            )
            if config.get("vocab_size") != model.bert.config.vocab_size:
                model.bert.resize_token_embeddings(config.get("vocab_size"))
            state_dict = load_file(weights_path)
            # 4. Load the weights into your model
            missing, unexpected = model.load_state_dict(state_dict, strict=True)

            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        max_length = config.get("max_length")
        # TODO: verify
        """spacy_model = spacy.load(config.get("spacy_id"))
        spacy_model.tokenizer = PreTokenizedTokenizer(spacy_model.vocab)"""
        spacy_model = get_spacy_model(lang=config.get("spacy_id").split("_")[0])
        print("spacymodel:", spacy_model, "lang", config.get("spacy_id").split("_")[0])

        return model, tokenizer, spacy_model, max_length, config.get("id2label")

    """@staticmethod
    def preprocess_input(tokens: List[List[str]], tokenizer: PreTrainedTokenizer, spacy_model: Any,
                               max_length: int = 256) -> dict:
        pos_tags = []
        dep_tags = []
        edge_indices = []
        for sent in tokens:
            upos, deps, edge_index = process_sent_spacy(sent, spacy_model)
            pos_tags.append(upos)
            dep_tags.append(deps)
            edge_indices.append(edge_index)

        # Ensure tokens and pos_tags have the same length
        assert len(tokens) == len(pos_tags), "Tokens and POS tags must have the same length"
        batch_size = len(tokens)
        # print(100*"=")
        # Convert POS tags to indices
        pos_indices = [[upos_dict.get(tag, 0) for tag in sub_lst] for sub_lst in pos_tags]
        dep_indices = [
            [dep_dict[f'{spacy_model.meta.get("lang", "xx")}_{spacy_model.meta.get("name", "unknown")}'].get(tag, 0) for
             tag in sub_lst] for sub_lst in dep_tags]

        # Tokenize the sentence
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,  # Input is pre-tokenized
            return_tensors=None,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )

        input_ids = encoding['input_ids']  # Shape: [max_seq_length]
        attention_mask = encoding['attention_mask']
        # Get word IDs to align subwords with original tokens
        word_id_list = [encoding.word_ids(i) for i in range(len(encoding['input_ids']))]

        # Assign POS indices to subword tokens
        pos_ids = torch.full((batch_size, max_length), 0, dtype=torch.long).tolist()
        dep_ids = torch.full((batch_size, max_length), 0, dtype=torch.long).tolist()
        for j, word_ids in enumerate(word_id_list):
            for i, word_id in enumerate(word_ids):
                if word_id is not None:  # Non-special token

                    pos_ids[j][i] = pos_indices[j][word_id]
                    dep_ids[j][i] = dep_indices[j][word_id]

        return {"input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "word_ids": word_id_list,
                "edge_index": [torch.tensor(edge_index, dtype=torch.int64) for edge_index in edge_indices],
                "pos_ids": torch.tensor(pos_ids),
                "dep_ids": torch.tensor(dep_ids),
                "word_count": [len(sent) for sent in tokens] }"""

    @staticmethod
    def preprocess_input(tokens: List[List[str]],
                         original_tokens: List[List[str]],
                         tokenizer: PreTrainedTokenizer,
                         spacy_model: Any,
                         max_length: int = 128):
        pos_tags = []
        dep_tags = []
        edge_indices = []
        for sent in original_tokens:
            upos, deps, edge_index = process_sent_spacy(sent, spacy_model)
            pos_tags.append(upos)
            dep_tags.append(deps)
            edge_indices.append(edge_index)

        result = PreprocessorUtility.retokenize_with_pos(tokens=tokens,
                                                         pos_tags=pos_tags,
                                                         dep_tags=dep_tags,
                                                         pos_tag_to_id=upos_dict,
                                                         dep_tag_to_id=dep_dict[
                                                             f'{spacy_model.meta.get("lang", "xx")}_{spacy_model.meta.get("name", "unknown")}'],
                                                         tokenizer=tokenizer,
                                                         max_seq_length=max_length
                                                         )


        # result["event_labels"] = examples["event_masks"]
        # result["focus_labels"] = examples["focus_masks"]
        result["word_count"] = [len(sent) for sent in tokens]
        for key in result:
            try:
                result[key] = torch.tensor(result[key])
            except:
                pass
        result["edge_index"] = [torch.tensor(edge_index, dtype=torch.int64) for edge_index in edge_indices]

        return result

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        ...

    @staticmethod
    def predict(model: AutoModelForTokenClassification,
                tokenizer: AutoTokenizer,
                input_tokens,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                pos_ids: torch.Tensor,
                dep_ids: torch.Tensor,
                edge_index: List[torch.Tensor],
                word_ids: List[int],
                word_count: List[int],
                id2label: dict,
                max_length: int = 128,
                device: str = "cuda"):
        ...


class CueBertInference(NegBertInference):
    special_tokens = {"C": "[CUE]"}

    def __init__(self,
                 model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 max_length: Optional[int] = None,
                 device: Optional[str] = None
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        self.model.to(device)
        self.model.eval()

    @classmethod
    def init_component(cls, model_path: str, device: Any, max_len: Optional[int] = None, **kwargs):
        return cls(*CueBertInference.load_model_and_tokenizer(model_path, model_path), device=device, max_length=max_len)

    def run(self,
            batch_tokens: List[List[str]], original_input: Optional[List[List[str]]] = None):
        # Preprocess input
        tokenized_inputs, word_ids = CueBertInference.preprocess_input(batch_tokens, self.tokenizer, self.max_length)

        # Perform inference
        batch_predictions = CueBertInference.predict(self.model,
                                                     self.tokenizer,
                                                     batch_tokens,
                                                     tokenized_inputs,
                                                     word_ids,
                                                     self.max_length,
                                                     self.device
                                                     )
        return batch_predictions

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        """Return the majority label from a list of labels ('X' or 'C')."""
        if not labels:
            return "X"  # Default to "X" if no labels
        count_c = labels.count("C")
        count_x = labels.count("X")
        return "C" if count_c >= count_x else "X"

    @staticmethod
    def predict(model: AutoModelForTokenClassification,
                tokenizer: AutoTokenizer,
                input_tokens: List[List[str]],
                tokenized_inputs,
                word_ids: List[List[int]],
                max_length: int = 128,
                device: str = "cuda:0") -> List[List[Dict]]:
        """Perform inference on batched inputs, merging subtoken predictions by majority vote."""
        # Move inputs to the same device as the model
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()  # Shape: (batch_size, sequence_length)

        # Process each sequence in the batch
        batch_results = []
        for seq_idx, (seq_tokens, seq_word_ids, seq_predictions) in enumerate(zip(input_tokens, word_ids, predictions)):
            pred_labels = [model.config.id2label[pred] for pred in seq_predictions]

            # Merge subtoken predictions to original tokens
            results = []
            current_word_id = None
            current_subtoken_labels = []
            for token_idx, (word_id, label) in enumerate(zip(seq_word_ids, pred_labels)):
                if word_id is None:  # Skip special tokens ([CLS], [SEP], [PAD])
                    continue
                if word_id != current_word_id:
                    # Process previous word
                    if current_subtoken_labels and current_word_id is not None:
                        merged_label = CueBertInference.majority_label(current_subtoken_labels)
                        original_token = seq_tokens[current_word_id]
                        results.append({"token": original_token, "label": merged_label})
                    # Start new word
                    current_word_id = word_id
                    current_subtoken_labels = [label]
                else:
                    # models are trained on the first subword only (see preprocessing), ignore the rest
                    pass

            # Process the last word
            if current_subtoken_labels and current_word_id is not None:
                merged_label = CueBertInference.majority_label(current_subtoken_labels)
                original_token = seq_tokens[current_word_id]
                results.append({"token": original_token, "label": merged_label})

            batch_results.append(results)

        return batch_results

    @staticmethod
    def evaluate(model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizerBase,
                 dataset_path: str,
                 device: str = "cuda:0"):
        # f"{BP}/data/HF-DATASETS/NEG-cue-(cleaned){exp_name}"
        ds = DatasetDict.load_from_disk(dataset_path)["test"]
        ds = ds.map(partial(PreprocessorUtility.tokenize_and_align_labels,
                                      tokenizer=tokenizer,
                                      label_mask="cue_masks"),
                              batched=True)
        ds = ds.select_columns(["input_ids", "attention_mask", "labels"])
        print(ds)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        dataloader = DataLoader(ds, batch_size=16, collate_fn=data_collator)
        y_pred = []
        y_true = []
        for batch in tqdm(dataloader, desc="Testing"):
            for key in batch:
                try:
                    batch[key] = batch[key].to(device)
                except:
                    pass
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

            # Convert logits to predictions
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()  # Shape: (batch_size, sequence_length)
            labels = batch["labels"]
            label_list = [0, 1]
            true_predictions = [

                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]

                for prediction, label in zip(predictions, labels)

            ]

            true_labels = [

                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]

                for prediction, label in zip(predictions, labels)

            ]
            y_pred.extend([i for j in true_predictions for i in j])
            y_true.extend([i for j in true_labels for i in j])

        report = classification_report(y_true=y_true, y_pred=y_pred)
        print(report)
        import evaluate
        f1_metric = evaluate.load("f1")
        micro = f1_metric.compute(predictions=y_pred, references=y_true, average='micro')
        macro = f1_metric.compute(predictions=y_pred, references=y_true, average='macro')
        weighted = f1_metric.compute(predictions=y_pred, references=y_true, average='weighted')
        binary = f1_metric.compute(predictions=y_pred, references=y_true, average='binary')
        print("macro: ", macro)
        print("micro: ", micro)
        print("weighted: ", weighted)
        print("binary: ", binary)
        return report, micro, macro, weighted, binary


    @staticmethod
    def main(model_path: str, tok_path: str, device: str = "cuda:0", max_length: int = 128) -> None:
        # Example batched input
        batch_tokens = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]
        original_input = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]

        cb_inf = CueBertInference.init_component(model_path, device, max_length)
        res = cb_inf.run(batch_tokens)

        cb_inf.pretty_print(res)
        """# Load model and tokenizer
        model, tokenizer = CueBertInference.load_model_and_tokenizer(model_path, tok_path)
        model = model.to(device)
        model.eval()  # Set model to evaluation mode

        # Preprocess input
        tokenized_inputs, word_ids = CueBertInference.preprocess_input(batch_tokens, tokenizer, max_length)

        # Perform inference
        batch_predictions = CueBertInference.predict(model, tokenizer, batch_tokens, tokenized_inputs, word_ids, max_length, device)

        # Print results
        print("Inference Results:")
        for seq_idx, predictions in enumerate(batch_predictions):
            print(f"\nSequence {seq_idx + 1}:")
            for result in predictions:
                print(f"Token: {result['token']}\nLabel: {result['label']}")"""


class ScopeBertInference(NegBertInference):
    special_tokens = {"S": "[SCO]"}

    def __init__(self,
                 model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 max_length: Optional[int] = None,
                 device: Optional[str] = None
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        self.model.to(device)
        self.model.eval()

    @classmethod
    def init_component(cls, model_path: str, device: Any, max_len: Optional[int] = None, **kwargs):
        return cls(*ScopeBertInference.load_model_and_tokenizer(model_path, model_path), device=device, max_length=max_len)

    def run(self,
            batch_tokens: List[List[str]], original_input: Optional[List[List[str]]] = None):
        # Preprocess input
        tokenized_inputs, word_ids = ScopeBertInference.preprocess_input(batch_tokens, self.tokenizer, self.max_length)

        # Perform inference
        batch_predictions = ScopeBertInference.predict(self.model,
                                                     self.tokenizer,
                                                     batch_tokens,
                                                     tokenized_inputs,
                                                     word_ids,
                                                     self.max_length,
                                                     self.device
                                                     )
        return batch_predictions

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        """Return the majority label from a list of labels ('X' or 'C')."""
        if not labels:
            return "X"  # Default to "X" if no labels
        count_c = labels.count("S")
        count_x = labels.count("X")
        return "S" if count_c >= count_x else "X"

    @staticmethod
    def predict(model: AutoModelForTokenClassification,
                tokenizer: AutoTokenizer,
                input_tokens: List[List[str]],
                tokenized_inputs,
                word_ids: List[List[int]],
                max_length: int = 128,
                device: str = "cuda:0") -> List[List[Dict]]:
        """Perform inference on batched inputs, merging subtoken predictions by majority vote."""
        # Move inputs to the same device as the model
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()  # Shape: (batch_size, sequence_length)
        # Process each sequence in the batch
        batch_results = []
        for seq_idx, (seq_tokens, seq_word_ids, seq_predictions) in enumerate(zip(input_tokens, word_ids, predictions)):
            pred_labels = [model.config.id2label[pred] for pred in seq_predictions]

            # Merge subtoken predictions to original tokens
            results = []
            current_word_id = None
            current_subtoken_labels = []
            for token_idx, (word_id, label) in enumerate(zip(seq_word_ids, pred_labels)):
                if word_id is None:  # Skip special tokens ([CLS], [SEP], [PAD])
                    continue
                if word_id != current_word_id:
                    # Process previous word
                    if current_subtoken_labels and current_word_id is not None:
                        merged_label = ScopeBertInference.majority_label(current_subtoken_labels)
                        original_token = seq_tokens[current_word_id]
                        results.append({"token": original_token, "label": merged_label})
                    # Start new word
                    current_word_id = word_id
                    current_subtoken_labels = [label]
                else:
                    # models are trained on the first subword only (see preprocessing), ignore the rest
                    pass

            # Process the last word
            if current_subtoken_labels and current_word_id is not None:
                merged_label = ScopeBertInference.majority_label(current_subtoken_labels)
                original_token = seq_tokens[current_word_id]
                results.append({"token": original_token, "label": merged_label})

            batch_results.append(results)

        return batch_results

    @staticmethod
    def evaluate(model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizerBase,
                 dataset_path: str,
                 device: str = "cuda:0"):
        # f"{BP}/data/HF-DATASETS/NEG-cue-(cleaned){exp_name}"
        ds = DatasetDict.load_from_disk(dataset_path)["test"]
        ds = ds.map(partial(PreprocessorUtility.tokenize_and_align_labels,
                                  tokenizer=tokenizer,
                                  label_mask="scope_masks"),
                          batched=True)
        ds = ds.select_columns(["input_ids", "attention_mask", "labels"])
        print(ds)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        dataloader = DataLoader(ds, batch_size=16, collate_fn=data_collator)
        y_pred = []
        y_true = []
        for batch in tqdm(dataloader, desc="Testing"):
            for key in batch:
                try:
                    batch[key] = batch[key].to(device)
                except:
                    pass
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

            # Convert logits to predictions
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()  # Shape: (batch_size, sequence_length)
            labels = batch["labels"]
            label_list = [0, 1]
            true_predictions = [

                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]

                for prediction, label in zip(predictions, labels)

            ]

            true_labels = [

                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]

                for prediction, label in zip(predictions, labels)

            ]
            y_pred.extend([i for j in true_predictions for i in j])
            y_true.extend([i for j in true_labels for i in j])

        report = classification_report(y_true=y_true, y_pred=y_pred)
        print(report)
        import evaluate
        f1_metric = evaluate.load("f1")
        micro = f1_metric.compute(predictions=y_pred, references=y_true, average='micro')
        macro = f1_metric.compute(predictions=y_pred, references=y_true, average='macro')
        weighted = f1_metric.compute(predictions=y_pred, references=y_true, average='weighted')
        binary = f1_metric.compute(predictions=y_pred, references=y_true, average='binary')
        print("macro: ", macro)
        print("micro: ", micro)
        print("weighted: ", weighted)
        print("binary: ", binary)
        return report, micro, macro, weighted, binary

    @staticmethod
    def main(model_path: str, tok_path: str, device: str = "cuda:0", max_length: int = 128) -> None:
        # Example batched input
        batch_tokens = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', '[CUE]', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]
        original_input = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]

        cb_inf = ScopeBertInference.init_component(model_path, device, max_length)
        res = cb_inf.run(batch_tokens)
        cb_inf.pretty_print(res)


class CueBertInferenceGAT(NegBertInferenceGAT):
    special_tokens = {"C": "[CUE]"}

    def __init__(self,
                 model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 spacy_model: Optional[Any] = None,
                 max_length: Optional[int] = None,
                 id2label: Optional[dict] = None,
                 device: Optional[str] = None
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.spacy_model = spacy_model
        self.max_length = max_length
        self.id2label = id2label
        self.device = device

        self.model.to(device)
        self.model.eval()

    @classmethod
    def init_component(cls,
                       model_path: str,
                       device: Any,
                       max_len: Optional[int] = None,
                       model_architecture: Any = BERTResidualGATv2ContextGatedFusion,
                       **kwargs):
        return cls(*CueBertInferenceGAT.load_model_and_tokenizer(model_path, model_architecture), device=device)

    def run(self, batch_tokens: List[List[str]], original_input: Optional[List[List[str]]] = None):
        # Preprocess input
        inputs = CueBertInferenceGAT.preprocess_input(batch_tokens, original_input, self.tokenizer, self.spacy_model, self.max_length)

        # Perform inference
        batch_predictions = CueBertInferenceGAT.predict(self.model,
                                                        self.tokenizer,
                                                        batch_tokens,
                                                        inputs["input_ids"],
                                                        inputs["attention_mask"],
                                                        inputs["pos_ids"],
                                                        inputs["dep_ids"],
                                                        inputs["edge_index"],
                                                        inputs["word_ids"],
                                                        inputs["word_count"],
                                                        self.id2label,
                                                        self.max_length,
                                                        self.device)
        return batch_predictions

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        """Return the majority label from a list of labels ('X' or 'C')."""
        if not labels:
            return "X"  # Default to "X" if no labels
        count_c = labels.count("C")
        count_x = labels.count("X")
        return "C" if count_c >= count_x else "X"

    @staticmethod
    def predict(model: torch.nn.Module,
                tokenizer: AutoTokenizer,
                input_tokens,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                pos_ids: torch.Tensor,
                dep_ids: torch.Tensor,
                edge_index: List[torch.Tensor],
                word_ids: List[int],
                word_count: List[int],
                id2label: dict,
                max_length: int = 128,
                device: str = "cuda:0") -> List[List[Dict]]:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pos_ids = pos_ids.to(device)
        dep_ids = dep_ids.to(device)
        for idx, edge in enumerate(edge_index):
            edge_index[idx] = edge.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pos_ids=pos_ids,
                            dep_ids=dep_ids,
                            edge_index=edge_index,
                            word_ids=word_ids,
                            word_count=word_count,
                            )
            logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()  # Shape: (batch_size, sequence_length)

        # Process each sequence in the batch
        batch_results = []
        idx = 0
        for sent in input_tokens:
            # Merge subtoken predictions to original tokens
            results = []
            for token in sent:
                results.append({"token": token, "label": id2label[f"{int(predictions[idx])}"]})
                idx += 1
            batch_results.append(results)

        return batch_results

    @staticmethod
    def evaluate(model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizerBase,
                 dataset_path: str,
                 device: str = "cuda:0"):
        ds = DatasetDict.load_from_disk(dataset_path)["test"]
        ds = ds.map(
            lambda x: {**x, "labels": x["cue_labels"]},
            batched=True,
            batch_size=100
        )
        print(ds)
        ds = ds.select_columns(["input_ids", "attention_mask", "labels", "word_ids", "pos_ids", "dep_ids", "edge_indices", "tokens"])

        data_collator = DataCollatorForTokenClassificationGCNInference()
        dataloader = DataLoader(ds, batch_size=16, collate_fn=data_collator)
        y_pred = []
        y_true = []
        total_tokens, total_pos_ids, total_dep_ids = [], [], []
        for batch in tqdm(dataloader, desc="Testing"):
            for key in batch:
                try:
                    batch[key] = batch[key].to(device)
                except:
                    pass
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

            pos_ids = batch["pos_ids"].cpu().numpy().tolist()
            dep_ids = batch["dep_ids"].cpu().numpy().tolist()
            tokens = batch["tokens"]
            word_pos_ids = []
            word_dep_ids = []
            for i in range(len(batch["tokens"])):
                word_pos_id = []
                word_dep_id = []
                for j in range(batch["word_count"][i]):
                    if j in batch["word_ids"][i]:
                        word_pos_id.append(pos_ids[i][batch["word_ids"][i].index(j)])
                        word_dep_id.append(dep_ids[i][batch["word_ids"][i].index(j)])
                    else:
                        word_pos_id.append(pos_ids[i][0])
                        word_dep_id.append(dep_ids[i][0])

                word_pos_ids.append(word_pos_id)
                word_dep_ids.append(word_dep_id)
            total_tokens.extend([i for j in tokens for i in j])
            total_pos_ids.extend([i for j in word_pos_ids for i in j])
            total_dep_ids.extend([i for j in word_dep_ids for i in j])

            # Convert logits to predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()  # Shape: (batch_size, sequence_length)
            labels = batch["labels"]
            label_list = [0, 1]
            true_predictions = [

                [label_list[p] for (p, l) in zip(predictions, labels) if l != -100]

            ]

            true_labels = [

                [label_list[l] for (p, l) in zip(predictions, labels) if l != -100]

            ]
            y_pred.extend([i for j in true_predictions for i in j])
            y_true.extend([i for j in true_labels for i in j])

        report = classification_report(y_true=y_true, y_pred=y_pred)
        print(report)
        print(len(y_pred), len(y_true), len(total_pos_ids), len(total_dep_ids), len(total_tokens))
        import evaluate
        f1_metric = evaluate.load("f1")
        micro = f1_metric.compute(predictions=y_pred, references=y_true, average='micro')
        macro = f1_metric.compute(predictions=y_pred, references=y_true, average='macro')
        weighted = f1_metric.compute(predictions=y_pred, references=y_true, average='weighted')
        binary = f1_metric.compute(predictions=y_pred, references=y_true, average='binary')
        print("macro: ", macro)
        print("micro: ", micro)
        print("weighted: ", weighted)
        print("binary: ", binary)
        return report, micro, macro, weighted, binary, {"tokens": total_tokens,
                                                        "pos_ids": total_pos_ids,
                                                        "dep_ids": total_dep_ids,
                                                        "y_true": y_true,
                                                        "y_pred": y_pred}


    @staticmethod
    def main(model_path: str, device: str = "cuda:0", max_length: int = 128) -> None:
        # Example batched input
        batch_tokens = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]
        original_input = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]

        cb_inf = CueBertInferenceGAT.init_component(model_path, device, max_length)
        batch_predictions = cb_inf.run(batch_tokens, original_input)

        cb_inf.pretty_print(batch_predictions)


class ScopeBertInferenceGAT(NegBertInferenceGAT):
    special_tokens = {"S": "[SCO]"}

    def __init__(self,
                 model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 spacy_model: Optional[Any] = None,
                 max_length: Optional[int] = None,
                 id2label: Optional[dict] = None,
                 device: Optional[str] = None
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.spacy_model = spacy_model
        self.max_length = max_length
        self.id2label = id2label
        self.device = device
        self.model.eval()
        self.model.to(device)


    @classmethod
    def init_component(cls,
                       model_path: str,
                       device: Any,
                       max_len: Optional[int] = None,
                       model_architecture: Any = BERTResidualGATv2ContextGatedFusion,
                       **kwargs):
        return cls(*ScopeBertInferenceGAT.load_model_and_tokenizer(model_path, model_architecture), device=device)

    def run(self, batch_tokens: List[List[str]], original_input: Optional[List[List[str]]] = None):
        # Preprocess input
        inputs = ScopeBertInferenceGAT.preprocess_input(batch_tokens, original_input, self.tokenizer, self.spacy_model, self.max_length)
        # print(batch_tokens)
        # Perform inference
        batch_predictions = ScopeBertInferenceGAT.predict(model=self.model,
                                                        tokenizer=self.tokenizer,
                                                        input_tokens=batch_tokens,
                                                        input_ids=inputs["input_ids"],
                                                        attention_mask=inputs["attention_mask"],
                                                        pos_ids=inputs["pos_ids"],
                                                        dep_ids=inputs["dep_ids"],
                                                        edge_index=inputs["edge_index"],
                                                        word_ids=inputs["word_ids"],
                                                        word_count=inputs["word_count"],
                                                        id2label=self.id2label,
                                                        max_length=self.max_length,
                                                        device=self.device)
        return batch_predictions

    @staticmethod
    def predict(model: torch.nn.Module,
                tokenizer: AutoTokenizer,
                input_tokens,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                pos_ids: torch.Tensor,
                dep_ids: torch.Tensor,
                edge_index: List[torch.Tensor],
                word_ids: List[int],
                word_count: List[int],
                id2label: dict,
                max_length: int = 128,
                device: str = "cuda:0") -> List[List[Dict]]:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pos_ids = pos_ids.to(device)
        dep_ids = dep_ids.to(device)
        for idx, edge in enumerate(edge_index):
            edge_index[idx] = edge.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pos_ids=pos_ids,
                            dep_ids=dep_ids,
                            edge_index=edge_index,
                            word_ids=word_ids,
                            word_count=word_count,
                            )
            logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=1).cpu().numpy()  # Shape: (batch_size, sequence_length)
        # Process each sequence in the batch
        batch_results = []
        idx = 0
        for sent in input_tokens:
            # Merge subtoken predictions to original tokens
            results = []
            for token in sent:
                results.append({"token": token, "label": id2label[f"{int(predictions[idx])}"]})
                idx += 1
            batch_results.append(results)

        return batch_results

    @staticmethod
    def evaluate(model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizerBase,
                 dataset_path: str,
                 device: str = "cuda:0",
                 batchsize: int = 16):
        ds = DatasetDict.load_from_disk(dataset_path)["test"]
        ds = ds.map(
            lambda x: {**x, "labels": x["scope_labels"]},
            batched=True,
            batch_size=100
        )
        print(ds)
        ds = ds.select_columns(
            ["input_ids", "attention_mask", "labels", "word_ids", "pos_ids", "dep_ids", "edge_indices", "tokens"])

        data_collator = DataCollatorForTokenClassificationGCNInference()
        dataloader = DataLoader(ds, batch_size=batchsize, collate_fn=data_collator)
        y_pred = []
        y_true = []
        total_pos_ids = []
        total_dep_ids = []
        total_tokens = []
        total_cue_count_per_sentence = []
        total_dep_indices = ds["edge_indices"]
        for batch in tqdm(dataloader, desc="Testing"):
            for key in batch:
                try:
                    batch[key] = batch[key].to(device)
                except:
                    pass
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

            # Convert logits to predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()  # Shape: (batch_size, sequence_length)
            # print(predictions)
            labels = batch["labels"]

            pos_ids = batch["pos_ids"].cpu().numpy().tolist()
            dep_ids = batch["dep_ids"].cpu().numpy().tolist()
            tokens = batch["tokens"]
            word_pos_ids = []
            word_dep_ids = []
            for i in range(len(batch["tokens"])):
                word_pos_id = []
                word_dep_id = []
                for j in range(batch["word_count"][i]):
                    if j in batch["word_ids"][i]:
                        word_pos_id.append(pos_ids[i][batch["word_ids"][i].index(j)])
                        word_dep_id.append(dep_ids[i][batch["word_ids"][i].index(j)])
                    else:
                        word_pos_id.append(pos_ids[i][0])
                        word_dep_id.append(dep_ids[i][0])

                word_pos_ids.append(word_pos_id)
                word_dep_ids.append(word_dep_id)
            total_tokens.extend(tokens)

            for sent in tokens:
                n_cues_in_sent = sent.count("[CUE]")
                total_cue_count_per_sentence.extend([n_cues_in_sent for _ in range(len(sent))])

            total_pos_ids.extend([i for j in word_pos_ids for i in j])
            total_dep_ids.extend([i for j in word_dep_ids for i in j])

            label_list = [0, 1]
            true_predictions = [

                [label_list[p] for (p, l) in zip(predictions, labels) if l != -100]

            ]

            true_labels = [

                [label_list[l] for (p, l) in zip(predictions, labels) if l != -100]

            ]

            y_pred.extend([i for j in true_predictions for i in j])
            y_true.extend([i for j in true_labels for i in j])

        report = classification_report(y_true=y_true, y_pred=y_pred)
        print(report)
        # print(len(y_pred), len(y_true), len(total_pos_ids), len(total_tokens))
        # print(total_tokens)
        # print(total_pos_ids)
        import evaluate
        f1_metric = evaluate.load("f1")
        micro = f1_metric.compute(predictions=y_pred, references=y_true, average='micro')
        macro = f1_metric.compute(predictions=y_pred, references=y_true, average='macro')
        weighted = f1_metric.compute(predictions=y_pred, references=y_true, average='weighted')
        binary = f1_metric.compute(predictions=y_pred, references=y_true, average='binary')
        print("macro: ", macro)
        print("micro: ", micro)
        print("weighted: ", weighted)
        print("binary: ", binary)
        return report, micro, macro, weighted, binary, {"tokens": total_tokens,
                                                        "pos_ids": total_pos_ids,
                                                        "dep_ids": total_dep_ids,
                                                        "dep_indices": total_dep_indices,
                                                        "y_true": y_true,
                                                        "y_pred": y_pred,
                                                        "cue_per_sent": total_cue_count_per_sentence}

    @staticmethod
    def main(model_path: str, device: str = "cuda:0", max_length: int = 128) -> None:
        # Example batched input
        batch_tokens = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', '[CUE]', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]
        original_input = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]

        cb_inf = ScopeBertInferenceGAT.init_component(model_path, device)
        batch_predictions = cb_inf.run(batch_tokens, original_input)

        cb_inf.pretty_print(batch_predictions)


class Pipeline:
    def __init__(self,
                 components: List[BasicInference],
                 model_paths: List[str],
                 device: str = "cuda:0",
                 max_length: int = 128,
                 replace: bool = True,
                 model_architecture: Any = BERTResidualGATv2ContextGatedFusion):
        self.replace = replace
        self.components = []
        for component, model_path in zip(components, model_paths):
            self.components.append(component.init_component(model_path=model_path, device=device, max_len=max_length, model_architecture=model_architecture))

        self.special_tokens = {value[1]:value[0] for comp in self.components for value in comp.special_tokens.items()}

    def run(self, batch_tokens: List[List[str]]) -> list[
        Tuple[list[str], list[str]]]:
        back_up_seq = copy.deepcopy(batch_tokens)
        replacements = {}
        for idx, component in enumerate(self.components):
            batch_predictions = component.run(batch_tokens, back_up_seq)
            batch_tokens = []
            for seq_idx, predictions in enumerate(batch_predictions):
                batch_seq = []
                for word_idx, result in enumerate(predictions):
                    if self.replace:
                        if result['label'] in component.special_tokens:
                            batch_seq.append(component.special_tokens[result['label']])
                            replacements[(seq_idx, word_idx)] = back_up_seq[seq_idx][word_idx]
                        else:
                            batch_seq.append(result['token'])
                    else:
                        if result['label'] in component.special_tokens:
                            batch_seq.append(component.special_tokens[result['label']])
                        batch_seq.append(result['token'])
                batch_tokens.append(batch_seq)

        result = []

        if self.replace:
            for seq_idx, predictions in enumerate(batch_tokens):
                clean_seq = []
                labels = []
                for word_idx, token in enumerate(predictions):
                    if token in self.special_tokens:
                        clean_seq.append(replacements[(seq_idx, word_idx)])
                        labels.append(self.special_tokens[token])
                    else:
                        clean_seq.append(token)
                        labels.append("X")
                result.append((clean_seq, labels))
        else:
            for seq_idx, predictions in enumerate(batch_tokens):
                clean_seq = []
                labels = []
                next_label = "X"
                for token in predictions:
                    if token in self.special_tokens:
                        next_label = self.special_tokens[token]
                    else:
                        clean_seq.append(token)
                        labels.append(next_label)
                        next_label = "X"

                result.append((clean_seq, labels))
        return result

    @staticmethod
    def pretty_print(result: List[Tuple[List[str], List[str]]]) -> None:
        for res in result:
            for item1, item2 in zip(res[0], res[1]):
                print(f"{str(item1):<{15}} {str(item2):<{15}}")
            print()

    @staticmethod
    def main_english():
        print("English Inference Baseline")
        mcue_path = f"{BP}/data/MODELS/CUE/gcn_cue_english_deberta_v2_pb_foc"
        mscope_path = f"{BP}/data/MODELS/SCOPE/gcn_scope_english_deberta_replace_v2_bioscope_abstracts"
        pipe = Pipeline(components=[CueBertInference, ScopeBertInference],
                        model_paths=[mcue_path, mscope_path],
                        device="cuda:0",
                        max_length=128)
        """batch_tokens = [
            "Your sample input does n't go here , i live in the prestreetlondon .".split(" "),
            "This is not another test sentence .".split(" "),
            "Virus isolated from IFNA-producing cells was able to replicate in the U937 cells but did not replicate efficiently in U937 cells transduced with the IFNA gene .".split(" "),
            ['Second', ',', 'T', 'cells', ',', 'which', 'lack', 'CD45', 'and', 'can', 'not', 'signal', 'via', 'the', 'TCR', ',', 'supported', 'higher', 'levels', 'of', 'viral', 'replication', 'and', 'gene', 'expression', '.'],
            ['Our', 'results', 'indicate', 'that', 'I', 'kappa', 'b', 'beta', ',', 'but', 'not', 'I', 'kappa', 'B', 'alpha', ',', 'is', 'required', 'for', 'the', 'signal', '-', 'dependent', 'activation', 'of', 'NF', '-', 'kappa', 'B', 'in', 'fibroblasts', '.']
        ]"""
        batch_tokens = [
            ['In', 'contrast', 'to', 'anti-CD3/IL-2-activated', 'LN', 'cells', ',', 'adoptive', 'transfer', 'of',
             'freshly', 'isolated', 'tumor-draining', 'LN', 'T', 'cells', 'has', 'no', 'therapeutic', 'activity',
             '.'],
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]
        res = pipe.run(batch_tokens)
        Pipeline.pretty_print(res)

        mcue_path = f"{BP}/data/MODELS/CUE_GAT/gcn_cue_english_debertabase_residual_gatv2_attentiongate_pb_foc"
        mscope_path = f"{BP}/data/MODELS/SCOPE_GAT/gcn_scope_english_debertabase_replace_residual_gatv2_attentiongate_bioscope_abstracts"
        pipe = Pipeline(components=[CueBertInferenceGAT, ScopeBertInferenceGAT],
                        model_paths=[mcue_path, mscope_path],
                        device="cuda:0",
                        max_length=128,
                        model_architecture=BERTResidualGATv2ContextGatedFusion)

        res = pipe.run(batch_tokens)
        Pipeline.pretty_print(res)

        return res

    @staticmethod
    def main_german():
        mcue_path = f"{BP}/data/MODELS/CUEDE/gcn_cue_de_EuroBERT_EuroBERT-210m_v2_pb_foc"
        mscope_path = f"{BP}/data/MODELS/SCOPEDE/gcn_scope_de_EuroBERT_EuroBERT-210m_v2_dt_neg"
        pipe = Pipeline(components=[CueBertInference, ScopeBertInference],
                        model_paths=[mcue_path, mscope_path],
                        device="cuda:0",
                        max_length=128)

        batch_tokens = [
            "Ich werde heute nicht mehr nach Hause fahren .".split(" "),
            "Ich sage dir nicht , dass du nett bist , aber ich umarme dich .".split(" "),
            "Oskar findet du bist nicht sehr nett , aber ich denke du bist intelligent .".split(" ")
        ]

        res = pipe.run(batch_tokens)
        Pipeline.pretty_print(res)

        mcue_path = f"{BP}/data/MODELS/CUE_GATDE/gcn_cue_de_EuroBERT_EuroBERT-210m_residual_gatv2_attentiongate_pb_foc"
        mscope_path = f"{BP}/data/MODELS/SCOPE_GATDE/gcn_scope_de_EuroBERT_EuroBERT-210m_residual_gatv2_attentiongate_dt_neg"
        pipe = Pipeline(components=[CueBertInferenceGAT, ScopeBertInferenceGAT],
                        model_paths=[mcue_path, mscope_path],
                        device="cuda:0",
                        max_length=128,
                        model_architecture=BERTResidualGATv2ContextGatedFusion)

        res = pipe.run(batch_tokens)
        Pipeline.pretty_print(res)
        return res


if __name__ == "__main__":
    # mpath = f"{BP}/data/MODELS/base_full"
    # ScopeBertInference.main(mpath, mpath)
    #r1 = Pipeline.main_english()
    r2 = Pipeline.main_german()
    # print(r1 == r2)
    # res = CueBertInference.preprocess_input(["This is a testio .".split(" "), "This is a testio .".split(" ")], AutoTokenizer.from_pretrained("prajjwal1/bert-tiny"))
    # (res)