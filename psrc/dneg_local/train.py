"""Plain transformer token classifiers for negation cue and scope detection.

Data: the aggregated CONLL exports (see ``data.py``), splits train/eval/test.
- cue:   every sentence (incl. negatives), label 1 = cue token
- scope: one example per negation group, cue tokens replaced by ``[CUE]``,
         label 1 = scope token (cue tokens themselves are never scope targets)
Labels are placed on the first subword of each word, so all reported metrics
are word level and directly comparable with the dependency-graph models.
"""
import json
import os
import shutil
from functools import partial

from datasets import DatasetDict
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          PreTrainedTokenizer, Trainer, TrainingArguments)

from .data import load_hf_dataset, remove_cue_from_scope, replace_cue_special_token
from .metrics import compute_prf
from .preprocessing import PreprocessorUtility, with_word_spaces
from .telegram_callback import TelegramLoggingCallback, send_telegram_message

BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))
MODEL_DIR = f"{BP}/data/MODELS"
RESULT_DIR = f"{BP}/data/RESULTS"
DEFAULT_BERT = "EuroBERT/EuroBERT-210m"
CUE_SPECIAL_TOKEN = "[CUE]"

CUE_ID2LABEL = {0: "X", 1: "C"}  # label names as expected by inference.py
SCOPE_ID2LABEL = {0: "X", 1: "S"}


def count_too_long(ds_dict: DatasetDict, tokenizer: PreTrainedTokenizer, max_length: int) -> dict:
    """Number of examples per split whose subword sequence does not fit into max_length."""
    return {split: sum(len(ids) > max_length for ids in
                       tokenizer(with_word_spaces(tokenizer, ds["tokens"]), is_split_into_words=True)["input_ids"])
            for split, ds in ds_dict.items()}


def make_training_args(out_dir: str, epochs: int, batchsize: int, learning_rate: float, seed: int,
                       **kwargs) -> TrainingArguments:
    return TrainingArguments(
        output_dir=out_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batchsize,
        per_device_eval_batch_size=batchsize,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # model selection on the eval split only
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=seed,
        **kwargs,
    )


def evaluate_and_report(trainer: Trainer, ds_dict: DatasetDict, exp_name: str, model_dir: str, config: dict) -> dict:
    """Evaluate the (best) model on eval and test, store results next to all other runs."""
    results = {"config": config,
               "eval": trainer.evaluate(ds_dict["eval"], metric_key_prefix="eval"),
               "test": trainer.evaluate(ds_dict["test"], metric_key_prefix="test")}
    shutil.rmtree(f"{model_dir}/checkpoints", ignore_errors=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f"{RESULT_DIR}/{exp_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{model_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    message = (f"*Results {exp_name}*\n"
               f"EVAL  F1: {results['eval']['eval_f1']:.4f}\n"
               f"TEST  F1: {results['test']['test_f1']:.4f}")
    send_telegram_message(message)
    print(message)
    return results


def _train_plain(task: str,
                 exp_name: str,
                 ds_dict: DatasetDict,
                 tokenizer: PreTrainedTokenizer,
                 id2label: dict,
                 bert_model: str,
                 epochs: int,
                 batchsize: int,
                 learning_rate: float,
                 max_length: int,
                 seed: int,
                 config: dict) -> dict:
    too_long = count_too_long(ds_dict, tokenizer, max_length)
    if any(too_long.values()):
        print(f"WARNING: examples longer than max_length={max_length} are truncated "
              f"(words past the limit are not evaluated): {too_long}")

    label_mask = f"{task}_masks"
    ds_dict = ds_dict.map(partial(PreprocessorUtility.tokenize_and_align_labels,
                                  tokenizer=tokenizer,
                                  label_mask=label_mask,
                                  max_length=max_length),
                          batched=True)
    print(ds_dict)

    model = AutoModelForTokenClassification.from_pretrained(
        bert_model, num_labels=len(id2label), id2label=id2label,
        label2id={v: k for k, v in id2label.items()}, trust_remote_code=True)
    if len(tokenizer) != model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))

    model_dir = f"{MODEL_DIR}/{task.upper()}_PLAIN/{exp_name}"
    trainer = Trainer(
        model=model,
        args=make_training_args(f"{model_dir}/checkpoints", epochs, batchsize, learning_rate, seed),
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["eval"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_prf,
        callbacks=[TelegramLoggingCallback()],
    )
    send_telegram_message(f"*START TRAINING*\n{exp_name}")
    trainer.train()
    trainer.save_model(model_dir)
    return evaluate_and_report(trainer, ds_dict, exp_name, model_dir,
                               {**config, "task": task, "architecture": "plain", "bert_model": bert_model,
                                "epochs": epochs, "batchsize": batchsize, "learning_rate": learning_rate,
                                "max_length": max_length, "seed": seed, "too_long": too_long})


def train_cue_plain(exp_name: str,
                    variant: str = "reference",
                    bert_model: str = DEFAULT_BERT,
                    epochs: int = 10,
                    batchsize: int = 16,
                    learning_rate: float = 2e-5,
                    max_length: int = 256,
                    seed: int = 42) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(bert_model, trust_remote_code=True)
    ds_dict = load_hf_dataset("cue", variant)
    return _train_plain("cue", exp_name, ds_dict, tokenizer, CUE_ID2LABEL, bert_model, epochs, batchsize,
                        learning_rate, max_length, seed, {"variant": variant})


def train_scope_plain(exp_name: str,
                      variant: str = "reference",
                      bert_model: str = DEFAULT_BERT,
                      epochs: int = 10,
                      batchsize: int = 16,
                      learning_rate: float = 2e-5,
                      max_length: int = 256,
                      seed: int = 42) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(bert_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [CUE_SPECIAL_TOKEN]})
    ds_dict = load_hf_dataset("scope", variant)
    ds_dict = ds_dict.map(remove_cue_from_scope)
    ds_dict = ds_dict.map(partial(replace_cue_special_token, cue_special_token=CUE_SPECIAL_TOKEN))
    return _train_plain("scope", exp_name, ds_dict, tokenizer, SCOPE_ID2LABEL, bert_model, epochs, batchsize,
                        learning_rate, max_length, seed, {"variant": variant})
