"""Quality of the corrected reference transcripts, per annotator (base env; after windows.py, align.py, ref_issues.py).

    python refquality.py  -> results/reference_quality.md, results/alignment_flags.csv,
                             results/double_correction_plan.md

Per annotator (pooled over their recordings):
  edit rate       verbatim WER of the CrisperWhisper prefill against the final text
                  (how much the annotator changed)
  model WER       pooled normalized WER of the six independent systems on those recordings
                  (how much good ASR disagrees with the final text; mixes difficulty and residual errors)
  anchor/1k       anchoring suspects per 1,000 reference tokens (ref_issues.py)
  missed speech   windows with speech no reference line contains (windows.py)
  low-align %     heard lines whose forced-alignment score is below LOW (align.py)
Annotators are anonymised as T1..T5 for the paper (mapping only in this local report).

Double-correction plan: a seeded selection of recordings for a second, independent
post-edit (inter-transcriber WER, the human ceiling) and of excerpts to transcribe
from scratch (direct measure of anchoring on the prefill).
"""
import csv
import json
import random
from collections import defaultdict
from statistics import median

import ref_issues
import score

DATA, OUT = score.DATA, score.OUT
INDEPENDENT = ["whisperx-large-v3", "qwen3-asr-1.7b", "voxtral-mini-3b", "parakeet-tdt-0.6b-v3",
               "phi-4-multimodal", "canary-1b-v2"]
# Mean character probability below which a line's text barely matches the audio. Lines no ASR
# system hears are placed heuristically (windows.py) and almost all fall below it, so only
# *heard* lines below LOW are informative about the text itself.
LOW = 0.05


def line_scores(doc):
    """line -> (mean score, n words, heard by an ASR system)."""
    p = DATA / "align" / f"{doc}.json"
    if not p.exists():
        return {}
    heard = [l["heard"] for l in json.loads((DATA / "align" / "windows" / p.name).read_text())["lines"]]
    res = {}
    for l in json.loads(p.read_text())["lines"]:
        s = [w["score"] for w in l["words"] if w["score"] is not None]
        res[l["line"]] = (sum(s) / len(s) if s else 0.0, len(l["words"]), heard[l["line"]])
    return res


def main():
    score.learn_from_data()
    info = ref_issues.line_info()
    docs = sorted(p.stem for p in (DATA / "ref").glob("*.txt"))
    issues = list(csv.DictReader(open(OUT / "reference_issues.csv", encoding="utf-8")))
    coverage = list(csv.DictReader(open(OUT / "coverage.csv", encoding="utf-8")))
    inserted = {(r["doc"], int(r["line_no"]) - 1) for r in issues if r["category"] == "inserted_missed_by_all"}

    agg = defaultdict(lambda: defaultdict(float))
    flags, heard_scores, unheard_scores, ins_scores = [], [], [], []
    for doc in docs:
        who = info[doc][0]
        ref = (DATA / "ref" / f"{doc}.txt").read_text(encoding="utf-8")
        a = agg[who]
        a["docs"] += 1
        a[doc.split("_")[1]] += 1
        pre = score.score_doc(ref, score.hyp_text(DATA / "hyp" / "crisperwhisper-prefill" / f"{doc}.json"))
        a["edit_E"] += pre["verbatim_E"]
        a["edit_N"] += pre["verbatim_N"]
        a["tokens"] += pre["normalized_N"]
        for s in INDEPENDENT:
            r = score.score_doc(ref, score.hyp_text(DATA / "hyp" / s / f"{doc}.json"))
            a["model_E"] += r["normalized_E"]
            a["model_N"] += r["normalized_N"]
        a["anchor"] += sum(r["doc"] == doc and r["category"] == "anchoring_suspect" for r in issues)
        a["missed_min"] += sum(float(r["seconds"]) for r in coverage if r["doc"] == doc) / 60
        texts = [l for l in ref.splitlines() if l.strip()]
        for i, (sc, n, heard) in line_scores(doc).items():
            if n == 0:
                continue
            (heard_scores if heard else unheard_scores).append(sc)
            if (doc, i) in inserted:
                ins_scores.append(sc)
            if heard:
                a["lines"] += 1
                a["low"] += sc < LOW
            if sc < LOW:
                flags.append({"doc": doc, "annotator": who, "line_no": i + 1, "score": round(sc, 3), "heard": heard,
                              "inserted_missed_by_all": (doc, i) in inserted, "text": texts[i]})

    names = sorted(agg, key=lambda w: -agg[w]["docs"])
    anon = {w: f"T{k + 1}" for k, w in enumerate(names)}
    words = [w for p in (DATA / "align").glob("*.json") for l in json.loads(p.read_text())["lines"] for w in l["words"]]
    share = lambda x: f"n={len(x)}, median score {median(x):.2f}, {100 * sum(s < LOW for s in x) / len(x):.0f}% below {LOW}"
    md = ["# Reference quality", "",
          f"Forced alignment: {100 * sum(w['start'] is not None for w in words) / len(words):.1f}% of "
          f"{len(words)} reference words get timestamps.", "",
          f"- lines heard by >= 1 ASR system: {share(heard_scores)}",
          f"- lines no ASR system hears: {share(unheard_scores)}",
          f"- of which added by annotators and missed by all six strong systems: {share(ins_scores)}", "",
          f"Missed speech: {len(coverage)} windows, {sum(float(r['seconds']) for r in coverage) / 60:.1f} min "
          f"(see coverage.csv).", "",
          "| annotator | anon | docs (IR/IE) | tokens | edit rate % | model WER % | anchor / 1k | missed speech min | low-align heard lines % |",
          "|---|---|---|---|---|---|---|---|---|"]
    for w in names:
        a = agg[w]
        md.append(f"| {w} | {anon[w]} | {a['docs']:.0f} ({a['Interviewer']:.0f}/{a['Interviewee']:.0f}) | "
                  f"{a['tokens']:.0f} | {100 * a['edit_E'] / a['edit_N']:.1f} | {100 * a['model_E'] / a['model_N']:.1f} | "
                  f"{1000 * a['anchor'] / a['tokens']:.1f} | {a['missed_min']:.1f} | "
                  f"{100 * a['low'] / max(a['lines'], 1):.1f} |")
    (OUT / "reference_quality.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    flags.sort(key=lambda r: r["score"])
    with open(OUT / "alignment_flags.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["doc", "annotator", "line_no", "score", "heard", "inserted_missed_by_all", "text"])
        wr.writeheader()
        wr.writerows(flags)
    print("\n".join(md))
    plan(info, names)


def plan(info, annotators, seed=42):
    """2 interviewer + 2 interviewee recordings (distinct interviews and first annotators) for a
    second post-edit, and 3 ten-minute excerpts of further recordings to transcribe from scratch."""
    rng = random.Random(seed)
    docs = sorted(info)
    rng.shuffle(docs)
    chosen, used_eid, used_who = [], set(), set()
    for role in ("Interviewer", "Interviewee"):
        for d in docs:
            eid, r = d.split("_")
            if r == role and eid not in used_eid and info[d][0] not in used_who and sum(c.endswith(role) for c in chosen) < 2:
                chosen.append(d)
                used_eid.add(eid)
                used_who.add(info[d][0])
    second = {d: rng.choice([w for w in annotators if w != info[d][0]]) for d in chosen}
    scratch = [d for d in docs if d.split("_")[0] not in used_eid][:3]
    rows = ["# Double-correction plan (seed 42)", "",
            "## Second independent post-edit (start from the same CrisperWhisper prefill, not from the first correction)", "",
            "| recording | audio min | first corrector | second corrector |", "|---|---|---|---|"]
    rows += [f"| {d} | {60 * score.duration_h(d):.0f} | {info[d][0]} | {second[d]} |" for d in chosen]
    rows += ["", f"Total audio: {sum(60 * score.duration_h(d) for d in chosen):.0f} min.", "",
             "## From scratch (empty editor, no prefill), 10-minute excerpt starting at minute 5", "",
             "| recording | excerpt | first corrector | transcriber |", "|---|---|---|---|"]
    rows += [f"| {d} | 05:00-15:00 | {info[d][0]} | {rng.choice([w for w in annotators if w != info[d][0]])} |"
             for d in scratch]
    rows += ["", "Needs in the review tool: an extra (document, annotator) assignment list and an option to start "
             "a document without prefill. Score with score.py by treating each second version as a hypothesis "
             "against the first (and vice versa)."]
    (OUT / "double_correction_plan.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
