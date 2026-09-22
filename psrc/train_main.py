"""Train negation cue/scope models on the aggregated CONLL annotations.

Examples (run from the repository root):
    python psrc/train_main.py convert                        # write HF datasets to data/HF-DATASETS
    python psrc/train_main.py train --task cue --arch plain
    python psrc/train_main.py train --task scope --arch gat --epochs 15
    python psrc/train_main.py train --task both --arch both  # all four models
    python psrc/train_main.py train --task scope --arch both --scope-variant baseline_scope_union
    python psrc/train_main.py train --task both --arch both --seeds 1 2 3 4 5

One model per seed goes to data/MODELS/{CUE,SCOPE}_{PLAIN,GAT}/<group>_s<seed>, its metrics
(eval + test) to data/RESULTS/<group>_s<seed>.json, and the seed-averaged scores (mean, std,
min, max) to data/RESULTS/<group>_seeds-<seeds>.json. Test models afterwards with eval_main.py.
"""
import argparse
import json
import os
import time

BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../.."))
# Set global Hugging Face cache before imports
os.environ["HF_HOME"] = f"{BP}/data/HF_CACHE"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

from dneg_local.data import CUE_VARIANTS, SCOPE_VARIANTS, convert_conll_to_hf


def convert_main():
    for variant in CUE_VARIANTS:
        convert_conll_to_hf("cue", variant)
    for variant in SCOPE_VARIANTS:
        convert_conll_to_hf("scope", variant)


def train_main(tasks, archs, bert_model: str, cue_variant: str, scope_variant: str, epochs: int,
               batchsize: int, learning_rate: float, max_length: int, seeds):
    from dneg_local import train_cue_plain, train_scope_plain, train_cue_gcn, train_scope_gcn
    from dneg_local.metrics import aggregate_runs
    from dneg_local.train import RESULT_DIR

    trainers = {("cue", "plain"): train_cue_plain, ("scope", "plain"): train_scope_plain,
                ("cue", "gat"): train_cue_gcn, ("scope", "gat"): train_scope_gcn}
    summaries = []
    for task in tasks:
        variant = cue_variant if task == "cue" else scope_variant
        for arch in archs:
            group = f"{task}_{arch}_{variant}_{bert_model.replace('/', '_')}"
            runs = {}
            for seed in seeds:
                # one model per seed: data/MODELS/{TASK}_{ARCH}/<group>_s<seed>
                exp_name = f"{group}_s{seed}"
                print(f"=== {exp_name} ===")
                start = time.time()
                res = trainers[(task, arch)](exp_name=exp_name, variant=variant, bert_model=bert_model,
                                             epochs=epochs, batchsize=batchsize, learning_rate=learning_rate,
                                             max_length=max_length, seed=seed)
                res["train_seconds"] = time.time() - start
                runs[seed] = res

            summary = {"group": group,
                       "seeds": list(seeds),
                       "runs": {str(seed): res for seed, res in runs.items()},
                       "aggregate": {split: aggregate_runs([res[split] for res in runs.values()])
                                     for split in ("eval", "test")}}
            os.makedirs(RESULT_DIR, exist_ok=True)
            with open(f"{RESULT_DIR}/{group}_seeds-{'-'.join(map(str, seeds))}.json", "w") as f:
                json.dump(summary, f, indent=2)
            summaries.append(summary)

    print("\ngroup :: split :: precision :: recall :: f1 (mean ± std over seeds)")
    for summary in summaries:
        for split in ("eval", "test"):
            agg = summary["aggregate"][split]
            print(f"{summary['group']} :: {split} :: "
                  + " :: ".join(f"{agg[f'{split}_{m}']['mean']:.4f} ± {agg[f'{split}_{m}']['std']:.4f}"
                                for m in ("precision", "recall", "f1"))
                  + f"  (n={agg[f'{split}_f1']['n']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # sub = parser.add_subparsers(dest="command", required=True)
    # sub.add_parser("convert", help="convert the aggregated CONLL files into HF DatasetDicts")
    # tr = sub.add_parser("train", help="train cue/scope models")
    parser.add_argument("--task", choices=["cue", "scope", "both"], default="both")
    parser.add_argument("--arch", choices=["plain", "gat", "both"], default="both",
                    help="plain transformer or transformer + dependency graph (GATv2)")
    parser.add_argument("--bert-model", default="EuroBERT/EuroBERT-210m")
    parser.add_argument("--cue-variant", choices=CUE_VARIANTS, default="reference")
    parser.add_argument("--scope-variant", choices=SCOPE_VARIANTS, default="reference")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    train_main(tasks=["cue", "scope"] if args.task == "both" else [args.task],
               archs=["plain", "gat"] if args.arch == "both" else [args.arch],
               bert_model=args.bert_model,
               cue_variant=args.cue_variant,
               scope_variant=args.scope_variant,
               epochs=args.epochs,
               batchsize=args.batchsize,
               learning_rate=args.lr,
               max_length=args.max_length,
               seeds=args.seeds)
