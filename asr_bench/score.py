"""Score every system in data/hyp against data/ref (base env: jiwer, num2words).

    python score.py            -> results/per_doc.csv, results/summary.csv, stdout table
    python score.py --selftest

Three WER variants (plus CER on the normalized variant), all lower-cased,
punctuation-free, numbers spelled out:
  verbatim   every filler form (äh, ähm, mhm, CrisperWhisper's [UM]/[UH]) -> one <fil> token;
             non-speech tags ([laughter], [breath], ...) are dropped
  normalized fillers and short word fragments ("hau-") removed on both sides (headline number)
  norep      normalized + immediate word repetitions collapsed on both sides
Reference [unk] markers (unintelligible) are dropped.

Error-analysis columns (normalized variant, from the WER alignment):
  cue/num/yn recall   share of reference negation cues / number words / ja-nein
                      tokens transcribed exactly
  cue_ins             hypothesis cue tokens with no matching reference cue
  drop_runs, ins_runs deletions / insertions of >= RUN consecutive words
                      (dropped passages, hallucinated passages), per hour of audio
  en_rate             share of hypothesis tokens that are English function words
                      (language drift into translation)
Filler precision/recall come from the verbatim alignment.
"""
import csv
import json
import os
import random
import re
import sys
import wave
from collections import defaultdict
from pathlib import Path

import jiwer
from num2words import num2words

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "results"
# Corrected transcripts from the review tool (read-only here); override with ANNOTATIONS_DB=path.
ANNOTATIONS_DB = Path(os.environ.get("ANNOTATIONS_DB", Path(__file__).resolve().parent.parent
                                     / "review" / "state" / "annotations_complete.sqlite"))
RUN = 5
FIL = "<fil>"

FILLER = re.compile(r"^(ä+h+m*|ö+h+m*|e+h+m+|h+m+|m+h+m*|m+|u+h+m*)$")  # not "um": German preposition
UNK = re.compile(r"\[(unk\*?|unknown)\]", re.I)
FILLER_TAG = re.compile(r"\[(?:um+|uh+|hm+|mhm+|äh+m*|eh+m*)\]", re.I)  # verbatim-model filler tags ([UM], [UH])
TAG = re.compile(r"\[[^\]]*\]")  # any other tag ([laughter], [breath], ...) is non-speech: dropped
# Word fragments a verbatim model marks with a trailing hyphen ("hau- hauptsächlich", "z-"). The
# reference never transcribes them, so the normalized variants drop them like fillers; the <= 4
# letter limit keeps suspended compounds ("Offline- und Online-").
FRAGMENT = re.compile(r"(?<![\w-])[^\W\d_]{1,4}-(?=[\s,.;:?!]|$)")
NEG_CUES = {"nicht", "nichts", "nie", "niemals", "niemand", "nirgends", "nirgendwo", "kein", "keine",
            "keinen", "keinem", "keiner", "keines", "nein", "nee", "weder", "ohne"}
YES_NO = {"ja", "nein", "nee", "jein", "jo", "jep", "doch"}
EN = {"the", "and", "you", "we", "is", "are", "of", "to", "this", "that", "it", "have", "what", "with",
      "would", "be", "for", "my", "your", "yes", "yeah", "yep", "i", "do", "can", "they", "there", "about", "like"}
ABBREV = [(r"\bz\s?\.\s?b\.", "zum beispiel"), (r"\bd\.\s?h\.", "das heißt"), (r"\bu\.\s?a\.", "unter anderem"),
          (r"\bbzw\.?", "beziehungsweise"), (r"\busw\.?", "und so weiter"), (r"\bca\.", "circa")]
# ponytail: hand list of spelling variants seen in the error dump, not a full German orthography normaliser
SPELLING = {"okay": "ok", "nochmal": "noch mal", "erstmal": "erst mal", "dankeschön": "danke schön",
            "bitteschön": "bitte schön", "zurzeit": "zur zeit", "irgendwas": "irgend was",
            "tschau": "ciao", "tschüs": "tschüss", "gern": "gerne", "achso": "ach so"}
# Colloquial elisions the verbatim reference keeps; expanded for the normalized variants only.
ELISION = {"würd": "würde", "hab": "habe", "geh": "gehe", "glaub": "glaube", "hör": "höre", "les": "lese",
           "komm": "komme", "mach": "mache", "sag": "sage", "n": "ein", "nen": "einen", "nem": "einem",
           "ner": "einer", "s": "es", "gibts": "gibt es", "gehts": "geht es", "wars": "war es",
           # ponytail: "grad" is also the unit (zwanzig Grad); rare in these interviews, so mapped to gerade
           "wär": "wäre", "hätt": "hätte", "find": "finde", "grad": "gerade", "grade": "gerade"}
COMPOUNDS = {}  # closed compound -> its two parts; filled by learn_compounds()


def learn_compounds(texts):
    """Split closed compounds (Zoomcall) wherever the data also writes them open (Zoom Call / Zoom-Call).

    ponytail: data-driven, only >= 8 letters so short words like "wieder" never split into "wie der".
    """
    toks = [_words(t) for t in texts]
    bigrams = {pair for ts in toks for pair in zip(ts, ts[1:])}
    for w in {w for ts in toks for w in ts if len(w) >= 8}:
        split = next(((w[:i], w[i:]) for i in range(2, len(w) - 1) if (w[:i], w[i:]) in bigrams), None)
        if split:
            COMPOUNDS[w] = list(split)


def number(n):
    w = num2words(n, lang="de")
    return re.sub(r"^ein(?=hundert|tausend)", "", w)  # "tausendsechshundert", as transcribers write it


NUM_WORDS = {number(i) for i in range(10000)} | {"eins", "hundert", "tausend", "million", "millionen"}


def _words(text):
    text = UNK.sub(" ", text)
    text = TAG.sub(" ", FILLER_TAG.sub(f" {FIL} ", text)).lower()
    text = re.sub(r"(\d)\.(\d{3})\b", r"\1\2", text)          # 1.600
    text = re.sub(r"(\d),(\d)", r"\1 komma \2", text)
    text = re.sub(r"(\d)\s*%", r"\1 prozent", text).replace("€", " euro ")
    for pat, rep in ABBREV:
        text = re.sub(pat, rep, text)
    toks = [number(int(t)) if t.isdigit() else t for t in re.findall(rf"{FIL}|\d+|[^\W\d_]+", text)]
    return [w for t in toks for w in SPELLING.get(t, t).split()]


def tokens(text, variant):
    if variant != "verbatim":
        text = FRAGMENT.sub(" ", text)
    toks = [w for t in _words(text) for w in COMPOUNDS.get(t, [t])]
    toks = [FIL if FILLER.match(t) else t for t in toks]
    if variant == "verbatim":
        return toks
    toks = [w for t in toks if t != FIL for w in ELISION.get(t, t).split()]
    if variant == "norep":
        toks = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
    return toks


def hyp_text(path):
    return json.loads(path.read_text())["transcription"] or ""


def duration_h(doc):
    with wave.open(str(DATA / "audio" / f"{doc}.wav")) as w:
        return w.getnframes() / w.getframerate() / 3600


def align(ref, hyp):
    out = jiwer.process_words(" ".join(ref) or FIL, " ".join(hyp) or FIL)
    return out, out.alignments[0]


def recall(chunks, ref, hyp, vocab):
    hit = total = ins = 0
    for c in chunks:
        r, h = ref[c.ref_start_idx:c.ref_end_idx], hyp[c.hyp_start_idx:c.hyp_end_idx]
        total += sum(t in vocab for t in r)
        if c.type == "equal":
            hit += sum(t in vocab for t in r)
        else:
            ins += sum(t in vocab for t in h)
    return hit, total, ins


def score_doc(ref_text, hyp_text_):
    row = {}
    for v in ("verbatim", "normalized", "norep"):
        ref, hyp = tokens(ref_text, v), tokens(hyp_text_, v)
        out, chunks = align(ref, hyp)
        row[f"{v}_N"] = len(ref)
        row[f"{v}_E"] = out.substitutions + out.deletions + out.insertions
        if v == "verbatim":
            row["fil_hit"], row["fil_ref"], _ = recall(chunks, ref, hyp, {FIL})
            row["fil_hyp"] = hyp.count(FIL)
        if v == "normalized":
            row.update(S=out.substitutions, D=out.deletions, I=out.insertions, hyp_N=len(hyp))
            chars = jiwer.process_characters(" ".join(ref) or FIL, " ".join(hyp) or FIL)
            row["cer_E"] = chars.substitutions + chars.deletions + chars.insertions
            row["cer_N"] = len(" ".join(ref))
            for name, vocab in (("cue", NEG_CUES), ("num", NUM_WORDS), ("yn", YES_NO)):
                row[f"{name}_hit"], row[f"{name}_ref"], row[f"{name}_ins"] = recall(chunks, ref, hyp, vocab)
            lens = lambda kind, n: [c.ref_end_idx - c.ref_start_idx if kind == "delete"
                                    else c.hyp_end_idx - c.hyp_start_idx
                                    for c in chunks if c.type == kind and
                                    (c.ref_end_idx - c.ref_start_idx if kind == "delete"
                                     else c.hyp_end_idx - c.hyp_start_idx) >= n]
            row["drop_runs"], row["drop_words"] = len(lens("delete", RUN)), sum(lens("delete", RUN))
            row["ins_runs"], row["ins_words"] = len(lens("insert", RUN)), sum(lens("insert", RUN))
            row["en"] = sum(t in EN for t in hyp)
    return row


def ratio(rows, num, den):
    d = sum(r[den] for r in rows)
    return sum(r[num] for r in rows) / d if d else float("nan")


def bootstrap_ci(rows, num, den, n=1000, seed=42):
    """95% CI resampling interviews (both roles of an interview stay together)."""
    by_eid = defaultdict(list)
    for r in rows:
        by_eid[r["eid"]].append(r)
    eids, rng, stats = sorted(by_eid), random.Random(seed), []
    for _ in range(n):
        sample = [r for e in rng.choices(eids, k=len(eids)) for r in by_eid[e]]
        stats.append(ratio(sample, num, den))
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def summarize(system, rows):
    hours = sum(r["hours"] for r in rows)
    lo, hi = bootstrap_ci(rows, "normalized_E", "normalized_N")
    s = {"system": system, "docs": len(rows),
         "wer": ratio(rows, "normalized_E", "normalized_N"), "wer_lo": lo, "wer_hi": hi,
         "wer_verbatim": ratio(rows, "verbatim_E", "verbatim_N"),
         "wer_norep": ratio(rows, "norep_E", "norep_N"),
         "cer": ratio(rows, "cer_E", "cer_N")}
    for k in "SDI":
        s[k] = ratio(rows, k, "normalized_N")
    for role in ("Interviewer", "Interviewee"):
        s[f"wer_{role.lower()}"] = ratio([r for r in rows if r["role"] == role], "normalized_E", "normalized_N")
    s["fil_P"], s["fil_R"] = ratio(rows, "fil_hit", "fil_hyp"), ratio(rows, "fil_hit", "fil_ref")
    for name in ("cue", "num", "yn"):
        s[f"{name}_recall"] = ratio(rows, f"{name}_hit", f"{name}_ref")
    s["cue_ins"] = sum(r["cue_ins"] for r in rows)
    s["drop_runs_h"], s["ins_runs_h"] = sum(r["drop_runs"] for r in rows) / hours, sum(r["ins_runs"] for r in rows) / hours
    s["drop_share"] = ratio(rows, "drop_words", "normalized_N")
    s["en_rate"] = ratio(rows, "en", "hyp_N")
    return s


def learn_from_data():
    learn_compounds([p.read_text(encoding="utf-8") for p in (DATA / "ref").glob("*.txt")]
                    + [hyp_text(p) for p in (DATA / "hyp").glob("*/*.json")])


def main():
    OUT.mkdir(exist_ok=True)
    learn_from_data()
    refs = {p.stem: p.read_text(encoding="utf-8") for p in (DATA / "ref").glob("*.txt")}
    per_doc, summary = [], []
    for sys_dir in sorted(p for p in (DATA / "hyp").iterdir() if p.is_dir()):
        rows = []
        for doc, ref in sorted(refs.items()):
            hyp_path = sys_dir / f"{doc}.json"
            if not hyp_path.exists():
                continue
            eid, role = doc.split("_")
            row = {"system": sys_dir.name, "doc": doc, "eid": eid, "role": role, "hours": duration_h(doc)}
            row.update(score_doc(ref, hyp_text(hyp_path)))
            rows.append(row)
        if len(rows) < len(refs):
            print(f"skip {sys_dir.name}: {len(rows)}/{len(refs)} documents transcribed", file=sys.stderr)
            continue
        per_doc += rows
        summary.append(summarize(sys_dir.name, rows))
    summary.sort(key=lambda s: s["wer"])
    for name, table in (("per_doc", per_doc), ("summary", summary)):
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    cols = ["wer", "wer_lo", "wer_hi", "cer", "wer_verbatim", "wer_norep", "S", "D", "I", "wer_interviewer",
            "wer_interviewee", "fil_P", "fil_R", "cue_recall", "num_recall", "yn_recall", "drop_share", "en_rate"]
    print("| system | " + " | ".join(cols) + " | cue_ins | drop/h | ins/h |")
    print("|---" * (len(cols) + 4) + "|")
    for s in summary:
        print(f"| {s['system']} | " + " | ".join(f"{100 * s[c]:.1f}" for c in cols)
              + f" | {s['cue_ins']} | {s['drop_runs_h']:.1f} | {s['ins_runs_h']:.1f} |")


def selftest():
    assert tokens("Ähm, ich bin zur zur Schule [unk] gegangen.", "verbatim") == [FIL, "ich", "bin", "zur", "zur", "schule", "gegangen"]
    assert tokens("Ähm, ich bin zur zur Schule gegangen.", "normalized") == ["ich", "bin", "zur", "zur", "schule", "gegangen"]
    assert tokens("ich bin zur zur Schule", "norep") == ["ich", "bin", "zur", "schule"]
    assert tokens("Ja [laughter] [UH] gut [breath]", "verbatim") == ["ja", FIL, "gut"]
    assert tokens("[UM] um 1.600 Euro, 20 % mhm", "normalized") == ["um", "tausendsechshundert", "euro", "zwanzig", "prozent"]
    assert tokens("FACES-Studie 3.", "normalized") == ["faces", "studie", "drei"]
    learn_compounds(["im Zoom-Call", "Zoomcall"])
    assert tokens("Zoomcall", "normalized") == ["zoom", "call"] and tokens("wieder", "normalized") == ["wieder"]
    assert tokens("gibts n Test", "normalized") == ["gibt", "es", "ein", "test"]
    assert tokens("als hau- hau- hauptsächlich, Offline- und", "normalized") == ["als", "hauptsächlich", "offline", "und"]
    assert tokens("als hau- hauptsächlich", "verbatim") == ["als", "hau", "hauptsächlich"]
    assert tokens("Das wär grad gern achso", "normalized") == ["das", "wäre", "gerade", "gerne", "ach", "so"]
    assert tokens("gibts n Test", "verbatim") == ["gibts", "n", "test"]
    assert tokens("Okay, z. B. nochmal bzw. Offline- und", "normalized") == \
        ["ok", "zum", "beispiel", "noch", "mal", "beziehungsweise", "offline", "und"]
    assert "tausendsechshundert" in NUM_WORDS and "zwanzig" in NUM_WORDS
    r = score_doc("Ich war nicht da, ähm, zwanzig Jahre.", "Ich war da zwanzig Jahre")
    assert (r["cue_hit"], r["cue_ref"], r["num_hit"], r["normalized_E"]) == (0, 1, 1, 1), r
    assert (r["fil_ref"], r["fil_hit"], r["fil_hyp"]) == (1, 0, 0), r
    print("selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
