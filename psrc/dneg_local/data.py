"""Load the aggregated negation CONLL exports as Hugging Face datasets.

Input layout (see psrc/main_utils/NEGATION_CONLL.md):
    data/neg_annotations/aggregated/{train,val,test}/<variant>.{cue,scope}.conll

- cue files:   ``token<TAB>cue``, one block per sentence (negatives included)
- scope files: ``token<TAB>cue<TAB>scope``, one block per negation group

Output: a ``DatasetDict`` with splits ``train``/``eval``/``test`` and the columns
``tokens``, ``cue_masks``, ``scope_masks`` (0/1 per token) and ``sentence_id``.
The reading part only needs the standard library.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Literal, Optional

BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))
CONLL_DIR = f"{BP}/data/neg_annotations/aggregated"
HF_DIR = f"{BP}/data/HF-DATASETS"

# CONLL split directory -> DatasetDict split name used by the training code
SPLITS = {"train": "train", "val": "eval", "test": "test"}
CUE_VARIANTS = ("reference", "baseline_cue_unanimous")
SCOPE_VARIANTS = ("reference", "baseline_scope_union", "baseline_scope_intersection")

Task = Literal["cue", "scope"]


def read_conll(path: str, task: Task) -> Dict[str, list]:
    """Parse one CONLL file into column lists (masks as 0/1 ints)."""
    n_cols = 2 if task == "cue" else 3
    with open(path, "r", encoding="utf-8") as f:
        blocks = [b for b in f.read().split("\n\n") if b.strip()]

    columns = {"tokens": [], "cue_masks": [], "scope_masks": []}
    for b_idx, block in enumerate(blocks):
        rows = [line.split("\t") for line in block.strip("\n").split("\n")]
        for row in rows:
            if len(row) != n_cols or any(label not in ("X", "O") for label in row[1:]):
                raise ValueError(f"{path}: malformed row in block {b_idx}: {row!r}")
        columns["tokens"].append([row[0] for row in rows])
        columns["cue_masks"].append([int(row[1] == "X") for row in rows])
        columns["scope_masks"].append([int(n_cols == 3 and row[2] == "X") for row in rows])

    # sentence ids come from the alignment sidecar written by the exporter
    sidecar = f"{path}.json"
    if os.path.exists(sidecar):
        with open(sidecar, "r", encoding="utf-8") as f:
            meta = json.load(f)["blocks"]
        if len(meta) != len(blocks):
            raise ValueError(f"{sidecar}: {len(meta)} sidecar blocks vs {len(blocks)} CONLL blocks")
        columns["sentence_id"] = [m["sentence_id"] for m in meta]
    else:
        columns["sentence_id"] = [""] * len(blocks)
    return columns


def load_conll_splits(task: Task, variant: str = "reference", conll_dir: str = CONLL_DIR) -> Dict[str, Dict[str, list]]:
    """Read train/val/test of one variant; returns ``{"train"|"eval"|"test": columns}``."""
    allowed = CUE_VARIANTS if task == "cue" else SCOPE_VARIANTS
    if variant not in allowed:
        raise ValueError(f"Unknown {task} variant {variant!r}; choose one of {allowed}")
    splits = {}
    for conll_split, hf_split in SPLITS.items():
        columns = read_conll(f"{conll_dir}/{conll_split}/{variant}.{task}.conll", task)
        if task == "scope":
            # a scope block without a cue cannot be conditioned on anything
            keep = [i for i, cue in enumerate(columns["cue_masks"]) if any(cue)]
            columns = {k: [v[i] for i in keep] for k, v in columns.items()}
        splits[hf_split] = columns
    return splits


def convert_conll_to_hf(task: Task,
                        variant: str = "reference",
                        conll_dir: str = CONLL_DIR,
                        save: bool = True,
                        out_dir: Optional[str] = None):
    """Build (and optionally save) the ``DatasetDict`` for one task/variant."""
    from datasets import Dataset, DatasetDict

    ds_dict = DatasetDict({split: Dataset.from_dict(columns)
                           for split, columns in load_conll_splits(task, variant, conll_dir).items()})
    if save:
        ds_dict.save_to_disk(out_dir or f"{HF_DIR}/NEG-{task}-{variant}")
    print(ds_dict)
    return ds_dict


def load_hf_dataset(task: Task, variant: str = "reference", conll_dir: str = CONLL_DIR):
    """Always rebuild from CONLL so a re-aggregation can never leave a stale cache."""
    return convert_conll_to_hf(task, variant, conll_dir, save=False)


"""
CLEANING FUNCTIONS (applied per example via Dataset.map)
"""


def remove_cue_from_scope(example: dict) -> dict:
    """Cue tokens are marked by the cue input, so they are not scope targets."""
    example["scope_masks"] = [0 if c else s for c, s in zip(example["cue_masks"], example["scope_masks"])]
    return example


def replace_cue_special_token(example: dict, cue_special_token: str = "[CUE]") -> dict:
    """Replace every cue token by the special token (input marking for scope models)."""
    example["tokens"] = [cue_special_token if c else t for t, c in zip(example["tokens"], example["cue_masks"])]
    return example


def dataset_statistics(splits: Dict[str, Dict[str, list]]) -> Dict[str, dict]:
    return {split: {"blocks": len(cols["tokens"]),
                    "tokens": sum(map(len, cols["tokens"])),
                    "cue_tokens": sum(map(sum, cols["cue_masks"])),
                    "scope_tokens": sum(map(sum, cols["scope_masks"])),
                    "max_len": max(map(len, cols["tokens"]), default=0)}
            for split, cols in splits.items()}


if __name__ == "__main__":
    for t, v in (("cue", "reference"), ("scope", "reference")):
        print(t, v, json.dumps(dataset_statistics(load_conll_splits(t, v)), indent=2))
