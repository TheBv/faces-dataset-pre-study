"""Error analysis and significance tests for the negation results, plus the appendix table.

Reads the word-level predictions written by eval_main.py (data/RESULTS/EVAL/*.pred.conll)
and the comparison table (comparison__<variant>_<split>.csv), then reports:

- cue recall per gold cue type and the most frequent false-positive words,
- a scope error decomposition (exact / too short / too long / other, scope lengths),
- paired bootstrap confidence intervals for F1 differences (resampling test blocks,
  our models averaged over seeds),
- a LaTeX appendix table with precision, recall, F1, and exact match for every model.

Run from the repository root after eval_main.py:
    python psrc/analyze_neg_results.py
Outputs: data/RESULTS/EVAL/analysis__<variant>_<split>.json and neg_results_appendix.tex
"""
import argparse
import csv
import glob
import json
import os
import random
from collections import Counter, defaultdict

BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../.."))
EVAL_DIR = f"{BP}/data/RESULTS/EVAL"

# D-NEG training corpora in table order, with their display names
DNEG_CORPORA = {"bioscope_abstracts": "BioScope (abstracts)", "bioscope_full": "BioScope (full)",
                "conan": "ConanDoyle-neg", "dt_neg": "DT-Neg", "sfu": "SFU Review", "socc": "SOCC"}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0


def read_predictions(path):
    """Blocks of (token, is_gold_cue or None, gold, pred) from a .pred.conll file."""
    blocks, current = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if not line:
                if current:
                    blocks.append(current)
                    current = []
                continue
            parts = line.split("\t")
            cue = parts[1] == "X" if len(parts) == 4 else None
            current.append((parts[0], cue, parts[-2] == "X", parts[-1] == "X"))
    if current:
        blocks.append(current)
    return blocks


def f1(blocks):
    tp = fp = fn = 0
    for block in blocks:
        for _, _, gold, pred in block:
            tp += gold and pred
            fp += pred and not gold
            fn += gold and not pred
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def normalise(token):
    return token.lower().strip(".,;:!?\"'")


def pred_file(source, name, variant, split):
    return f"{EVAL_DIR}/{source}__{name.replace('/', '_')}__{variant}_{split}.pred.conll"


def local_files(task, arch, variant, split):
    return sorted(glob.glob(f"{EVAL_DIR}/local__{task}_{arch}_*_s*__{variant}_{split}.pred.conll"))


def dneg_file(task, arch, corpus, variant, split):
    return pred_file("dneg", f"D-NEG/{task}{'-gat' if arch == 'gat' else ''}-de-{corpus}", variant, split)


def cue_type_analysis(files):
    """Average (over runs) recall per gold cue type and false positives per word."""
    recall, false_pos = defaultdict(lambda: [0, 0]), Counter()
    for path in files:
        for block in read_predictions(path):
            for token, _, gold, pred in block:
                if gold:
                    recall[normalise(token)][0] += pred
                    recall[normalise(token)][1] += 1
                elif pred:
                    false_pos[normalise(token)] += 1
    n = len(files)
    return {"recall": {tok: {"found": hit / n, "gold": total // n}
                       for tok, (hit, total) in sorted(recall.items(), key=lambda x: -x[1][1])},
            "false_positives": {tok: count / n for tok, count in false_pos.most_common(10)}}


def scope_error_analysis(files):
    """Share of exact / too short / too long / other scopes and mean scope lengths."""
    shapes, gold_len, pred_len, blocks_total = Counter(), 0, 0, 0
    for path in files:
        for block in read_predictions(path):
            gold = [b[2] for b in block]
            pred = [b[3] for b in block]
            blocks_total += 1
            gold_len += sum(gold)
            pred_len += sum(pred)
            if gold == pred:
                shapes["exact"] += 1
            elif all(g or not p for g, p in zip(gold, pred)):
                shapes["too_short"] += 1  # prediction is a strict subset of the reference
            elif all(p or not g for g, p in zip(gold, pred)):
                shapes["too_long"] += 1  # prediction is a strict superset of the reference
            else:
                shapes["other"] += 1
    return {"gold_scope_length": gold_len / blocks_total, "pred_scope_length": pred_len / blocks_total,
            **{shape: shapes[shape] / blocks_total for shape in ("exact", "too_short", "too_long", "other")}}


def paired_bootstrap(files_a, files_b, rng):
    """95% CI of F1(a) - F1(b), resampling test blocks; each side is averaged over its runs (seeds)."""
    runs_a = [read_predictions(p) for p in files_a]
    runs_b = [read_predictions(p) for p in files_b]
    n = len(runs_a[0])
    assert all(len(r) == n for r in runs_a + runs_b), "prediction files cover different blocks"

    def mean_f1(runs, idx):
        return sum(f1([run[i] for i in idx]) for run in runs) / len(runs)

    observed = mean_f1(runs_a, range(n)) - mean_f1(runs_b, range(n))
    deltas = sorted(mean_f1(runs_a, idx) - mean_f1(runs_b, idx)
                    for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(BOOTSTRAP_SAMPLES)))
    return {"delta_f1": observed,
            "ci95": [deltas[int(0.025 * BOOTSTRAP_SAMPLES)], deltas[int(0.975 * BOOTSTRAP_SAMPLES) - 1]]}


def load_comparison(variant, split):
    with open(f"{EVAL_DIR}/comparison__{variant}_{split}.csv", newline="") as f:
        return list(csv.DictReader(f))


def fmt(row, metric, with_std):
    mean = float(row[f"{metric}_mean"])
    if with_std:
        return f"{mean:.2f}\\,{{\\scriptsize$\\pm${float(row[f'{metric}_std']):.2f}}}"
    return f"{mean:.2f}"


def appendix_table(rows, bert_model="EuroBERT_EuroBERT-210m"):
    """LaTeX table (one block per task) with P/R/F1/EM for every model."""
    by_key = {}
    for row in rows:
        if row["source"] == "dneg":
            corpus = row["model"].split("-de-")[-1]
            by_key[(row["task"], row["arch"], corpus)] = row
        elif bert_model in row["model"]:
            by_key[(row["task"], row["arch"], "ours")] = row

    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\begin{tabular}{llcccccccc}", r"\toprule",
             r"& & \multicolumn{4}{c}{Cue} & \multicolumn{4}{c}{Scope} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             r"Training data & Model & P & R & F1 & EM & P & R & F1 & EM \\", r"\midrule"]
    for corpus, name in list(DNEG_CORPORA.items()) + [("ours", "Ours (3 seeds)")]:
        if corpus == "ours":
            lines.append(r"\midrule")
        for i, arch in enumerate(("plain", "gat")):
            cells = [name if i == 0 else "", "Plain" if arch == "plain" else "GAT"]
            for task in ("cue", "scope"):
                row = by_key.get((task, arch, corpus))
                cells += [fmt(row, m, corpus == "ours") if row else "--"
                          for m in ("precision", "recall", "f1", "exact_match")]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Negation results on our test split: word-level precision (P), recall (R), and F1 "
              r"for the positive class, and the share of exactly correct sentences (cue) or scopes (scope; EM). "
              r"D-NEG models are applied as released; our models are averaged over three seeds "
              r"(mean $\pm$ standard deviation). Scope detection uses gold cues.}",
              r"\label{tab:neg-results-full}", r"\end{table*}"]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="reference")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    v, s = args.variant, args.split
    rng = random.Random(BOOTSTRAP_SEED)

    ours = {(t, a): local_files(t, a, v, s) for t in ("cue", "scope") for a in ("plain", "gat")}
    missing = [k for k, files in ours.items() if not files]
    if missing:
        raise SystemExit(f"no local prediction files for {missing}; run eval_main.py first")

    rows = load_comparison(v, s)
    best_dneg = {}
    for task in ("cue", "scope"):
        for arch in ("plain", "gat"):
            candidates = [r for r in rows if r["source"] == "dneg" and r["task"] == task and r["arch"] == arch]
            best_dneg[(task, arch)] = max(candidates, key=lambda r: float(r["f1_mean"]))["model"]

    analysis = {
        "cue_types": {"ours_gat": cue_type_analysis(ours[("cue", "gat")]),
                      "ours_plain": cue_type_analysis(ours[("cue", "plain")]),
                      **{f"dneg_{arch}_best": cue_type_analysis([pred_file("dneg", best_dneg[("cue", arch)], v, s)])
                         for arch in ("plain", "gat")}},
        "scope_errors": {"ours_gat": scope_error_analysis(ours[("scope", "gat")]),
                         "ours_plain": scope_error_analysis(ours[("scope", "plain")]),
                         **{f"dneg_{arch}_{corpus}": scope_error_analysis([dneg_file("scope", arch, corpus, v, s)])
                            for arch in ("plain", "gat") for corpus in DNEG_CORPORA
                            if os.path.exists(dneg_file("scope", arch, corpus, v, s))}},
        "bootstrap": {},
        "best_dneg": {f"{t}_{a}": m for (t, a), m in best_dneg.items()},
    }
    for task in ("cue", "scope"):
        best_overall = max((r for r in rows if r["source"] == "dneg" and r["task"] == task),
                           key=lambda r: float(r["f1_mean"]))["model"]
        for arch in ("plain", "gat"):
            analysis["bootstrap"][f"{task}_ours_{arch}_vs_{best_overall}"] = paired_bootstrap(
                ours[(task, arch)], [pred_file("dneg", best_overall, v, s)], rng)
        analysis["bootstrap"][f"{task}_ours_gat_vs_ours_plain"] = paired_bootstrap(
            ours[(task, "gat")], ours[(task, "plain")], rng)

    with open(f"{EVAL_DIR}/analysis__{v}_{s}.json", "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    table = appendix_table(rows)
    with open(f"{EVAL_DIR}/neg_results_appendix.tex", "w") as f:
        f.write(table + "\n")

    print("=== bootstrap (ΔF1, 95% CI)")
    for key, res in analysis["bootstrap"].items():
        print(f"{key}: {res['delta_f1']:+.3f} [{res['ci95'][0]:+.3f}, {res['ci95'][1]:+.3f}]")
    print("\n=== cue recall per type (found/gold)")
    for name, res in analysis["cue_types"].items():
        print(name, {t: f"{r['found']:.1f}/{r['gold']}" for t, r in res["recall"].items()})
        print("   false positives per run:", res["false_positives"])
    print("\n=== scope errors")
    for name, res in analysis["scope_errors"].items():
        print(name, {k: round(val, 2) for k, val in res.items()})
    print(f"\nwritten: {EVAL_DIR}/analysis__{v}_{s}.json, {EVAL_DIR}/neg_results_appendix.tex")
