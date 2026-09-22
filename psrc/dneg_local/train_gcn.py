"""Dependency-graph (GATv2) token classifiers for negation cue and scope detection.

Each sentence is parsed with spaCy on the given whitespace tokens (see
``spacy_utils.parser_form``); the transformer states of each word's first
subword, fused with UPOS/deprel embeddings, become graph nodes and the
dependency arcs (child -> head) become edges. Predictions are per word.
Same data, splits, cleaning, model selection and metrics as ``train.py``.
"""
import json
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from datasets import DatasetDict
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizer, Trainer
from transformers.data.data_collator import DataCollatorMixin

from .data import load_hf_dataset, remove_cue_from_scope
from .gcn_bert_nn_module import BERTResidualGATv2ContextGatedFusion
from .metrics import compute_prf
from .preprocessing import PreprocessorUtility
from .spacy_utils import dep_dict, get_spacy_model, lang_dict, upos_dict
from .telegram_callback import TelegramLoggingCallback, send_telegram_message
from .train import (CUE_ID2LABEL, CUE_SPECIAL_TOKEN, DEFAULT_BERT, MODEL_DIR, SCOPE_ID2LABEL, count_too_long,
                    evaluate_and_report, make_training_args)


def prepare_dataset_gcn(task: str,
                        variant: str,
                        tokenizer: PreTrainedTokenizer,
                        spacy_model: Any,
                        max_length: int = 256) -> DatasetDict:
    ds_dict = load_hf_dataset(task, variant)
    if task == "scope":
        ds_dict = ds_dict.map(remove_cue_from_scope)
    too_long = count_too_long(ds_dict, tokenizer, max_length)
    # parsing uses the original tokens; for scope the cue tokens are replaced by [CUE] afterwards
    ds_dict = ds_dict.map(partial(PreprocessorUtility.tokenize_deps_pos_align_labels,
                                  tokenizer=tokenizer,
                                  spacy_model=spacy_model,
                                  max_length=max_length,
                                  special_token=CUE_SPECIAL_TOKEN if task == "scope" else None),
                          batched=True,
                          batch_size=10)
    # truncated sentences lose words (they fall back to the [CLS] state in the model):
    # drop them from training, but keep eval/test complete so scores stay comparable
    if too_long["train"]:
        ds_dict["train"] = ds_dict["train"].filter(lambda x: len(x["attention_mask"]) != sum(x["attention_mask"]))
    if too_long["eval"] or too_long["test"]:
        print(f"WARNING: eval/test examples longer than max_length={max_length}: {too_long}")
    ds_dict = ds_dict.map(lambda x: {"labels": x[f"{task}_labels"]}, batched=True)
    print(ds_dict)
    return ds_dict


def inspect_model_modules(model) -> dict:
    params = {}
    for name, module in model.named_modules():
        # Embedding layer
        if isinstance(module, torch.nn.Embedding):
            if "position" in name:
                params["max_position_embeddings"] = module.weight.size(0)
            elif "token_type" in name:
                params["type_vocab_size"] = module.weight.size(0)
            elif "word_embeddings" in name:
                params["vocab_size"] = module.weight.size(0)
                params["hidden_size"] = module.weight.size(1)

        # Transformer layers
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            # params["hidden_size"] = module.d_model
            params["num_attention_heads"] = module.nhead
            params["intermediate_size"] = module.dim_feedforward
            # Count layers by checking parent ModuleList later

        # Classifier
        if isinstance(module, torch.nn.Linear) and "classifier" in name:
            params["num_labels"] = module.out_features

    # Count transformer layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and "transformer" in name.lower():
            params["num_hidden_layers"] = len(module)

    return params


def extract_config(tokenizer: Any,
                   model: Any,
                   id2label: Any,
                   label2id: Any,
                   num_labels: Any,
                   pos_vocab_size: int,
                   dep_vocab_size: int,
                   bert_model: str,
                   spacy_id: str,
                   max_length: int
                   ) -> PretrainedConfig:

    # Extract model parameters
    model_params = inspect_model_modules(model)  # Or use extract_bert_params_from_state_dict

    # Merge and validate
    config_params = {
        **model_params,
        "max_position_embeddings": min(
            model_params.get("max_position_embeddings", tokenizer.model_max_length),
            tokenizer.model_max_length
        ),
        "vocab_size": model_params.get("vocab_size", len(tokenizer)),
        "model_type": "custom_bert",
        "hidden_act": "gelu",  # BERT default
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "custom_pytorch_model": True,
        "is_encoder_decoder": False,
        "id2label": id2label,
        "label2id": label2id,
        "num_labels": num_labels,
        "lcount": num_labels,
        "bert_id": bert_model,
        "pos_vocab_size": pos_vocab_size,
        "dep_vocab_size": dep_vocab_size,
        "spacy_id": spacy_id,
        "max_length": max_length
    }
    # Ensure vocab_size matches
    if config_params["vocab_size"] != len(tokenizer):
        print(
            f"Warning: Model vocab_size ({config_params['vocab_size']}) differs from tokenizer ({tokenizer.vocab_size})")
    return PretrainedConfig(**config_params)


@dataclass
class DataCollatorForTokenClassificationGCN(DataCollatorMixin):
    return_tensors: str = "pt"

    def torch_call(self, features):

        batch = dict()
        """for feature in features:
            assert len(feature["tokens"]) == len(feature["cue_masks"]) == len(list(set(feature["word_ids"]))) - 1, feature"""
        label_name = "label" if "label" in features[0].keys() else "labels"
        edge_idx_name = "edge_indices" if "edge_indices" in features[0].keys() else "edge_index"
        batch["labels"] = torch.cat([torch.tensor(feature[label_name], dtype=torch.int64) for feature in features])
        batch["input_ids"] = torch.tensor([feature["input_ids"] for feature in features])
        batch["attention_mask"] = torch.tensor([feature["attention_mask"] for feature in features])
        batch["pos_ids"] = torch.tensor([feature["pos_ids"] for feature in features])
        batch["dep_ids"] = torch.tensor([feature["dep_ids"] for feature in features])
        batch["word_ids"] = [feature["word_ids"] for feature in features]
        batch["edge_index"] = [torch.tensor(feature[edge_idx_name], dtype=torch.int64) for feature in features]
        batch["word_count"] = [len(feature[label_name]) for feature in features]

        return batch

@dataclass
class DataCollatorForTokenClassificationGCNInference(DataCollatorMixin):
    return_tensors: str = "pt"

    def torch_call(self, features):

        batch = dict()
        """for feature in features:
            assert len(feature["tokens"]) == len(feature["cue_masks"]) == len(list(set(feature["word_ids"]))) - 1, feature"""
        label_name = "label" if "label" in features[0].keys() else "labels"
        edge_idx_name = "edge_indices" if "edge_indices" in features[0].keys() else "edge_index"
        batch["labels"] = torch.cat([torch.tensor(feature[label_name], dtype=torch.int64) for feature in features])
        batch["input_ids"] = torch.tensor([feature["input_ids"] for feature in features])
        batch["attention_mask"] = torch.tensor([feature["attention_mask"] for feature in features])
        batch["pos_ids"] = torch.tensor([feature["pos_ids"] for feature in features])
        batch["dep_ids"] = torch.tensor([feature["dep_ids"] for feature in features])
        batch["word_ids"] = [feature["word_ids"] for feature in features]
        batch["edge_index"] = [torch.tensor(feature[edge_idx_name], dtype=torch.int64) for feature in features]
        batch["word_count"] = [len(feature[label_name]) for feature in features]
        batch["tokens"] = [feature["tokens"] for feature in features]

        return batch


class GCNTrainer(Trainer):
    """The GCN models return flat word-level outputs (total_words, labels). Accelerate's
    gather_for_metrics treats dim 0 as samples and cuts the last eval batch down to the
    remaining number of *sentences*, silently dropping words from the scores. Re-pad the
    outputs to (batch, max_words) so every evaluated word is kept (padding label = -100)."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        word_count = list(inputs["word_count"])
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if logits is not None:
            logits = pad_sequence(torch.split(logits, word_count), batch_first=True, padding_value=-100)
        if labels is not None:
            labels = pad_sequence(torch.split(labels, word_count), batch_first=True, padding_value=-100)
        return loss, logits, labels


def _train_gcn(task: str,
               exp_name: str,
               variant: str,
               model_architecture: Any,
               bert_model: str,
               epochs: int,
               batchsize: int,
               learning_rate: float,
               max_length: int,
               seed: int,
               lang: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(bert_model, trust_remote_code=True)
    if task == "scope":
        tokenizer.add_special_tokens({"additional_special_tokens": [CUE_SPECIAL_TOKEN]})
    spacy_id = lang_dict[lang]
    ds_dict = prepare_dataset_gcn(task, variant, tokenizer, get_spacy_model(lang=lang), max_length)

    id2label = CUE_ID2LABEL if task == "cue" else SCOPE_ID2LABEL
    label2id = {v: k for k, v in id2label.items()}
    pos_vocab_size, dep_vocab_size = len(upos_dict), len(dep_dict[spacy_id])
    model = model_architecture(bert_model,
                               num_labels=len(id2label),
                               pos_vocab_size=pos_vocab_size,
                               dep_vocab_size=dep_vocab_size,
                               id2label=id2label,
                               label2id=label2id)
    if len(tokenizer) != model.bert.get_input_embeddings().weight.size(0):
        print(f"resizing token embeddings to {len(tokenizer)}")
        model.bert.resize_token_embeddings(len(tokenizer))

    model_dir = f"{MODEL_DIR}/{task.upper()}_GAT/{exp_name}"
    trainer = GCNTrainer(
        model=model,
        args=make_training_args(f"{model_dir}/checkpoints", epochs, batchsize, learning_rate, seed,
                                remove_unused_columns=False, label_names=["labels"]),
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["eval"],
        data_collator=DataCollatorForTokenClassificationGCN(),
        compute_metrics=compute_prf,
        callbacks=[TelegramLoggingCallback()],
    )
    send_telegram_message(f"*START TRAINING*\n{exp_name}")
    trainer.train()

    # SAVING (layout expected by inference.NegBertInferenceGAT.load_model_and_tokenizer)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    extract_config(tokenizer, model.bert, id2label, label2id, len(id2label), pos_vocab_size, dep_vocab_size,
                   bert_model, spacy_id, max_length).save_pretrained(model_dir)
    # newer transformers drop generation keys such as max_length when saving a config
    with open(f"{model_dir}/config.json", "r") as f:
        saved_config = json.load(f)
    saved_config["max_length"] = max_length
    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(saved_config, f, indent=2)

    return evaluate_and_report(trainer, ds_dict, exp_name, model_dir,
                               {"task": task, "variant": variant, "architecture": model_architecture.__name__,
                                "bert_model": bert_model, "spacy_model": spacy_id, "epochs": epochs,
                                "batchsize": batchsize, "learning_rate": learning_rate,
                                "max_length": max_length, "seed": seed})


def train_cue_gcn(exp_name: str,
                  variant: str = "reference",
                  model_architecture: Any = BERTResidualGATv2ContextGatedFusion,
                  bert_model: str = DEFAULT_BERT,
                  epochs: int = 10,
                  batchsize: int = 16,
                  learning_rate: float = 2e-5,
                  max_length: int = 256,
                  seed: int = 42,
                  lang: str = "de") -> dict:
    return _train_gcn("cue", exp_name, variant, model_architecture, bert_model, epochs, batchsize,
                      learning_rate, max_length, seed, lang)


def train_scope_gcn(exp_name: str,
                    variant: str = "reference",
                    model_architecture: Any = BERTResidualGATv2ContextGatedFusion,
                    bert_model: str = DEFAULT_BERT,
                    epochs: int = 10,
                    batchsize: int = 16,
                    learning_rate: float = 2e-5,
                    max_length: int = 256,
                    seed: int = 42,
                    lang: str = "de") -> dict:
    return _train_gcn("scope", exp_name, variant, model_architecture, bert_model, epochs, batchsize,
                      learning_rate, max_length, seed, lang)
