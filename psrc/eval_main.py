"""Test trained negation cue/scope models on the aggregated CONLL test split and compare them.

Local models are grouped by configuration: directories that differ only in the ``_s<seed>``
suffix form one group, which is reported as the mean ± std over its seeds. Each dneg Hub
model is its own group. Cue and scope models can be mixed; there is one table per task.

Examples (run from the repository root):
    # everything trained with train_main.py
    python psrc/eval_main.py --local data/MODELS/*/*

    # ... compared with published dneg models (pip install dneg), same data and metrics
    python psrc/eval_main.py --local data/MODELS/*/* --dneg-ds sfu conan dt_neg
    python psrc/eval_main.py --local data/MODELS/*/* --dneg D-NEG/cue-de-sfu D-NEG/scope-gat-de-sfu

    # one configuration only
    python psrc/eval_main.py --local data/MODELS/SCOPE_GAT/scope_gat_reference_EuroBERT_EuroBERT-210m_s*

Results: data/RESULTS/EVAL/<source>__<model>__<variant>_<split>.json for each model, with word-level
predictions in .pred.conll; the comparison goes to comparison__<variant>_<split>.{json,csv}.
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../.."))
# Set global Hugging Face cache before imports
os.environ["HF_HOME"] = f"{BP}/data/HF_CACHE"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

from dneg_local.data import CUE_VARIANTS, SCOPE_VARIANTS
from dneg_local.evaluation import EVAL_DIR, detect_arch, detect_task, evaluate_models

METRICS = ("precision", "recall", "f1", "exact_match")
DNEG_DATASETS = ("bioscope_abstracts", "bioscope_full", "conan", "dt_neg", "sfu", "socc")


def dneg_repos(datasets, modes, tasks, lang: str = "de"):
    """Hub ids of the published dneg models: D-NEG/{cue,scope}[-gat]-<lang>-<dataset>."""
    return [f"D-NEG/{task}{'-gat' if mode == 'gat' else ''}-{lang}-{ds}"
            for task in tasks for mode in modes for ds in datasets]


def group_models(local_paths, dneg_paths, task=None, arch=None):
    """{(source, group_name): [paths]}, local seed runs of one configuration share a group."""
    groups = defaultdict(list)
    for path in local_paths:
        path = os.path.normpath(path)
        if not os.path.isfile(f"{path}/config.json"):
            print(f"skipping {path}: no trained model (config.json) found")
            continue
        groups[("local", re.sub(r"_s\d+$", "", os.path.basename(path)))].append(path)
    for repo in dneg_paths:
        groups[("dneg", repo)].append(repo)
    return {key: {"paths": paths,
                  "task": task or detect_task(paths[0]),
                  "arch": arch or detect_arch(paths[0])}
            for key, paths in groups.items()}


def print_tables(rows):
    for task in ("cue", "scope"):
        task_rows = sorted((r for r in rows if r["task"] == task), key=lambda r: -r["f1_mean"])
        if not task_rows:
            continue
        width = max(len(r["model"]) for r in task_rows)
        print(f"\n=== {task.upper()} ({task_rows[0]['split']}, {task_rows[0]['variant']}) — mean ± std over seeds ===")
        print(f"{'model':<{width}}  {'source':<6} {'arch':<5} {'n':>2}  "
              + "  ".join(f"{m:>15}" for m in METRICS))
        for r in task_rows:
            print(f"{r['model']:<{width}}  {r['source']:<6} {r['arch']:<5} {r['n']:>2}  "
                  + "  ".join(f"{r[m + '_mean']:.4f} ± {r[m + '_std']:.4f}" for m in METRICS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", nargs="+", default=[], metavar="MODEL_DIR",
                        help="model directories trained with train_main.py (seed runs are averaged)")
    parser.add_argument("--dneg", nargs="+", default=[], metavar="HUB_REPO",
                        help="dneg models, e.g. D-NEG/cue-de-sfu or D-NEG/scope-gat-de-sfu")
    parser.add_argument("--dneg-ds", nargs="+", default=[], choices=DNEG_DATASETS,
                        help="shortcut: the German dneg models trained on these datasets")
    parser.add_argument("--dneg-mode", nargs="+", default=["plain", "gat"], choices=["plain", "gat"],
                        help="which --dneg-ds models: plain and/or gat (default both)")
    parser.add_argument("--dneg-task", nargs="+", default=["cue", "scope"], choices=["cue", "scope"],
                        help="which --dneg-ds models: cue and/or scope (default both)")
    parser.add_argument("--task", choices=["cue", "scope"], help="force the task (default: detected per model)")
    parser.add_argument("--arch", choices=["plain", "gat"], help="force plain/GAT (default: detected per model)")
    parser.add_argument("--variant", choices=sorted(set(CUE_VARIANTS + SCOPE_VARIANTS)), default="reference",
                        help="gold annotation variant to test against")
    parser.add_argument("--split", choices=["test", "eval", "train"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", help="default: cuda:0 if available, else cpu")
    parser.add_argument("--out", help="output prefix for the comparison (default: data/RESULTS/EVAL/comparison__<variant>_<split>)")
    args = parser.parse_args()

    dneg_paths = args.dneg + dneg_repos(args.dneg_ds, args.dneg_mode, args.dneg_task)
    groups = group_models(args.local, dneg_paths, args.task, args.arch)
    if not groups:
        parser.error("no models given: use --local and/or --dneg / --dneg-ds")

    rows, summaries = [], {}
    for (source, name), group in groups.items():
        print(f"\n### {source}: {name} ({group['task']}, {group['arch']}, {len(group['paths'])} model(s))")
        summary = evaluate_models(group["paths"], task=group["task"], source=source, arch=group["arch"],
                                  variant=args.variant, split=args.split, batch_size=args.batch_size,
                                  max_length=args.max_length, device=args.device)
        summaries[f"{source}:{name}"] = summary
        agg = summary["aggregate"]
        row = {"model": name, "source": source, "task": group["task"], "arch": group["arch"],
               "variant": args.variant, "split": args.split, "n": agg["f1"]["n"]}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = agg[m]["mean"], agg[m]["std"]
        row["truncated_blocks"] = sum(run["truncated_blocks"] for run in summary["runs"])
        rows.append(row)

    print_tables(rows)

    out = args.out or f"{EVAL_DIR}/comparison__{args.variant}_{args.split}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(f"{out}.json", "w") as f:
        json.dump({"table": rows, "groups": summaries}, f, indent=2)
    with open(f"{out}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\ncomparison: {out}.json / {out}.csv")
