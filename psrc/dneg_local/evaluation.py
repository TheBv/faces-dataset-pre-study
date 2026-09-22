"""Evaluate a trained cue or scope model on a split of the aggregated CONLL data.

Models run through their inference components, so this also checks the inference
path. The components come from ``dneg_local.inference`` for models trained here, or
from the public ``dneg`` package (https://github.com/texttechnologylab/dneg) for its
Hub models. Both have the same interface: ``init_component(...)`` and
``run(tokens, original_tokens)``, returning one ``{"token", "label"}`` dict per word.

Scope is evaluated with the gold cues, like during training: cue words are replaced
by ``[CUE]`` and are never scope targets. Metrics are per word. P/R/F1 are for the
positive class (cue or scope). ``exact_match`` is the share of blocks where every
word is right; for scope blocks this is PCS (percentage of correct scopes).
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from .data import load_hf_dataset
from .metrics import aggregate_runs, binary_prf

BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))
EVAL_DIR = f"{BP}/data/RESULTS/EVAL"
CUE_SPECIAL_TOKEN = "[CUE]"
POSITIVE_LABEL = {"cue": "C", "scope": "S"}


def detect_task(model_path: str) -> Optional[str]:
    """Local models live in .../{CUE,SCOPE}_{PLAIN,GAT}/<exp>; dneg Hub repos are named {cue,scope}-..."""
    local_group = os.path.basename(os.path.dirname(os.path.normpath(model_path))).lower()
    name = os.path.basename(os.path.normpath(model_path)).lower()
    for task in ("cue", "scope"):
        if local_group.startswith(task) or name.startswith(task):
            return task
    return None


def detect_arch(model_path: str) -> str:
    """GAT models store a custom config (``bert_id``); dneg Hub GAT repos contain ``-gat-``."""
    config_path = f"{model_path}/config.json"
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            return "gat" if "bert_id" in json.load(f) else "plain"
    return "gat" if "-gat-" in model_path.lower() else "plain"


def load_component(task: str, model_path: str, source: str = "local", arch: Optional[str] = None,
                   device: Optional[str] = None, max_length: int = 256) -> Any:
    if source == "local":
        from . import inference as module
    elif source == "dneg":
        import dneg as module  # pip install dneg
    else:
        raise ValueError(f"Unknown source {source!r}")
    arch = arch or detect_arch(model_path)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cls = {("cue", "plain"): module.CueBertInference, ("scope", "plain"): module.ScopeBertInference,
           ("cue", "gat"): module.CueBertInferenceGAT, ("scope", "gat"): module.ScopeBertInferenceGAT}[(task, arch)]
    return cls.init_component(model_path=model_path, device=device, max_len=max_length)


def predict(task: str, component: Any, tokens: List[List[str]], cue_masks: List[List[int]],
            batch_size: int = 16) -> Tuple[List[List[int]], int]:
    """Word-level 0/1 predictions per block. Words cut off by truncation count as negative."""
    positive = POSITIVE_LABEL[task]
    predictions, n_truncated = [], 0
    for start in range(0, len(tokens), batch_size):
        originals = tokens[start:start + batch_size]
        if task == "scope":
            inputs = [[CUE_SPECIAL_TOKEN if c else t for t, c in zip(sent, cues)]
                      for sent, cues in zip(originals, cue_masks[start:start + batch_size])]
        else:
            inputs = originals
        for sent, result in zip(originals, component.run(inputs, originals)):
            labels = [int(r["label"] == positive) for r in result[:len(sent)]]
            n_truncated += int(len(labels) < len(sent))
            predictions.append(labels + [0] * (len(sent) - len(labels)))
    return predictions, n_truncated


def score(gold: List[List[int]], predictions: List[List[int]]) -> Dict[str, float]:
    metrics = binary_prf([p for sent in predictions for p in sent], [g for sent in gold for g in sent])
    metrics["exact_match"] = sum(p == g for p, g in zip(predictions, gold)) / len(gold) if gold else 0.0
    metrics["blocks"] = len(gold)
    return metrics


def evaluate_model(model_path: str,
                   task: Optional[str] = None,
                   source: str = "local",
                   arch: Optional[str] = None,
                   variant: str = "reference",
                   split: str = "test",
                   batch_size: int = 16,
                   max_length: int = 256,
                   device: Optional[str] = None,
                   save: bool = True) -> dict:
    task = task or detect_task(model_path)
    if task not in ("cue", "scope"):
        raise ValueError(f"Cannot infer the task from {model_path!r}; pass task='cue' or 'scope'")
    arch = arch or detect_arch(model_path)
    ds = load_hf_dataset(task, variant)[split]
    tokens, cue_masks = ds["tokens"], ds["cue_masks"]
    # the gold labels for scope match training: cue words are never scope targets
    gold = cue_masks if task == "cue" else [[0 if c else s for c, s in zip(cues, scopes)]
                                             for cues, scopes in zip(cue_masks, ds["scope_masks"])]

    component = load_component(task, model_path, source, arch, device, max_length)
    predictions, n_truncated = predict(task, component, tokens, cue_masks, batch_size)
    result = {"model": model_path, "source": source, "task": task, "architecture": arch,
              "variant": variant, "split": split, "max_length": max_length,
              "truncated_blocks": n_truncated, "metrics": score(gold, predictions)}
    if n_truncated:
        print(f"WARNING: {n_truncated} blocks were truncated at max_length={max_length}; the missing words count as negative")

    if save:
        os.makedirs(EVAL_DIR, exist_ok=True)
        model_name = os.path.basename(os.path.normpath(model_path)) if source == "local" else model_path.replace("/", "_")
        name = f"{source}__{model_name}__{variant}_{split}"
        result["predictions_file"] = f"{EVAL_DIR}/{name}.pred.conll"
        with open(result["predictions_file"], "w", encoding="utf-8") as f:
            f.write(f"# columns: token, {'gold cue, ' if task == 'scope' else ''}gold {task}, predicted {task}\n\n")
            for sent, sid, cues, g_sent, p_sent in zip(tokens, ds["sentence_id"], cue_masks, gold, predictions):
                f.write(f"# sentence_id = {sid}\n")
                for tok, c, g, p in zip(sent, cues, g_sent, p_sent):
                    cue_col = f"{'X' if c else 'O'}\t" if task == "scope" else ""
                    f.write(f"{tok}\t{cue_col}{'X' if g else 'O'}\t{'X' if p else 'O'}\n")
                f.write("\n")
        with open(f"{EVAL_DIR}/{name}.json", "w") as f:
            json.dump(result, f, indent=2)
    del component
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_models(model_paths: List[str], save_summary: Optional[str] = None, **kwargs) -> dict:
    """Evaluate several models (e.g. one per seed) and report per-model and averaged metrics."""
    runs = [evaluate_model(path, **kwargs) for path in model_paths]
    summary = {"runs": runs, "aggregate": aggregate_runs([run["metrics"] for run in runs])}
    if save_summary:
        os.makedirs(os.path.dirname(save_summary), exist_ok=True)
        with open(save_summary, "w") as f:
            json.dump(summary, f, indent=2)
    return summary
