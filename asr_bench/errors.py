"""Error analysis on top of score.py (base env).

    python errors.py  -> results/errors.md

1. Missed utterances: a reference line is "missed" by a system when >= MISS of
   its (normalized) words are deleted. Reported by line length and role, and
   by how many of the STRONG systems miss the same line (missed by all =
   audio/reference problem, missed by one = that model's problem).
2. Hard words: content-word types whose recall across the STRONG systems is
   lowest, with what the systems wrote instead.
3. Inserted passages (>= score.RUN words): repetition loops (inserted text
   that repeats the preceding hypothesis text) vs other insertions, plus the
   most common inserted phrases.
4. Anchoring suspects: reference words that only the prefill (the text the
   correctors edited) reproduces, while >= 4 of the other STRONG systems
   disagree and >= 3 of them agree on the same alternative. Candidates for
   prefill errors the correctors kept (to be checked by listening).
"""
from collections import Counter, defaultdict
from pathlib import Path

import score

DATA, OUT = score.DATA, score.OUT
MISS = 0.8
STRONG = ["crisperwhisper-prefill", "whisperx-large-v3", "voxtral-mini-3b",
          "parakeet-tdt-0.6b-v3", "phi-4-multimodal", "canary-1b-v2"]
STOP = set("""der die das den dem des ein eine einen einem einer eines und oder aber ich du er sie es wir ihr
mir mich dir dich ihm ihn uns euch ist sind war waren bin bist hat habe hab haben hatte hatten wird werden würde
würden kann können muss müssen soll sollen will wollen mal noch schon auch nur so ja nein nee ne doch also dann da
hier wie was wo wer wenn weil dass ob zu zum zur im in an am auf aus bei mit nach von vor für über um bis durch
nicht kein keine gar sehr mehr eher genau gut ok okay jetzt immer oft sich man sein ihre ihr ihren ihrem seine
meine mein meinen gibt gerne gern alles alle etwas viel""".split())


def ref_lines(doc):
    """Normalized tokens of the whole reference plus each token's line index."""
    toks, line_of, lines = [], [], []
    for i, line in enumerate(l for l in (DATA / "ref" / f"{doc}.txt").read_text(encoding="utf-8").splitlines() if l.strip()):
        t = score.tokens(line, "normalized")
        toks += t
        line_of += [i] * len(t)
        lines.append((line, len(t)))
    return toks, line_of, lines


def analyse(system, doc, ref, line_of):
    hyp = score.tokens(score.hyp_text(DATA / "hyp" / system / f"{doc}.json"), "normalized")
    _, chunks = score.align(ref, hyp)
    deleted = [False] * len(ref)
    word_hits = []  # (ref_word, hit, hyp_replacement)
    inserts = []
    at = [None] * len(ref)  # per ref index: "=" or the 1:1 substituted hyp word
    for c in chunks:
        r, h = ref[c.ref_start_idx:c.ref_end_idx], hyp[c.hyp_start_idx:c.hyp_end_idx]
        if c.type == "delete":
            deleted[c.ref_start_idx:c.ref_end_idx] = [True] * len(r)
        if c.type == "insert" and len(h) >= score.RUN:
            before = " ".join(hyp[max(0, c.hyp_start_idx - 4 * len(h)):c.hyp_start_idx])
            inserts.append((" ".join(h), " ".join(h[:len(h) // 2 or 1]) in before))
        if c.type == "equal":
            at[c.ref_start_idx:c.ref_end_idx] = ["="] * len(r)
        elif c.type == "substitute" and len(r) == len(h):
            at[c.ref_start_idx:c.ref_end_idx] = h
        for j, w in enumerate(r):
            word_hits.append((w, c.type == "equal", " ".join(h) if c.type == "substitute" else ""))
    missed = defaultdict(lambda: [0, 0])
    for i, d in zip(line_of, deleted):
        missed[i][0] += d
        missed[i][1] += 1
    return {i: d / n >= MISS for i, (d, n) in missed.items()}, word_hits, inserts, at


def main():
    score.learn_from_data()
    systems = sorted(p.name for p in (DATA / "hyp").iterdir() if p.is_dir())
    docs = sorted(p.stem for p in (DATA / "ref").glob("*.txt"))
    miss_by = defaultdict(dict)          # (doc, line) -> {system: missed}
    lines_meta = {}
    word = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # word -> system -> [hit, n]
    subst = defaultdict(Counter)
    ins = defaultdict(list)
    suspects = Counter()   # (ref word, alternative) -> count
    suspect_ctx = {}
    others = [s for s in STRONG if s != "crisperwhisper-prefill"]
    for doc in docs:
        ref, line_of, lines = ref_lines(doc)
        for i, (text, n) in enumerate(lines):
            lines_meta[(doc, i)] = (text, n)
        at = {}
        for s in systems:
            missed, hits, inserts, at[s] = analyse(s, doc, ref, line_of)
            for i, m in missed.items():
                miss_by[(doc, i)][s] = m
            for w, hit, rep in hits:
                word[w][s][0] += hit
                word[w][s][1] += 1
                if rep and s in STRONG:
                    subst[w][rep] += 1
            ins[s] += inserts
        for i, w in enumerate(ref):
            if at["crisperwhisper-prefill"][i] != "=":
                continue
            alts = [at[s][i] for s in others if at[s][i] != "="]
            top = Counter(a for a in alts if a).most_common(1)
            if len(alts) >= 4 and top and top[0][1] >= 3:
                suspects[(w, top[0][0])] += 1
                suspect_ctx.setdefault((w, top[0][0]), f"{doc}: {' '.join(ref[max(0, i - 5):i + 5])}")

    md = ["# ASR error analysis", "", f"Missed utterance = >= {MISS:.0%} of its words deleted. "
          f"STRONG = {', '.join(STRONG)}.", ""]

    buckets = [("1 word", 1, 1), ("2-3", 2, 3), ("4-9", 4, 9), ("10+", 10, 10**6)]
    md += ["## Missed utterances (% of reference lines)", "",
           "| system | all | " + " | ".join(b[0] for b in buckets) + " | interviewer | interviewee |",
           "|---" * (len(buckets) + 4) + "|"]
    for s in systems:
        cells = []
        for lo, hi in [(1, 10**6)] + [(b[1], b[2]) for b in buckets]:
            keys = [k for k in miss_by if lo <= lines_meta[k][1] <= hi]
            cells.append(100 * sum(miss_by[k][s] for k in keys) / len(keys))
        for role in ("Interviewer", "Interviewee"):
            keys = [k for k in miss_by if k[0].endswith(role)]
            cells.append(100 * sum(miss_by[k][s] for k in keys) / len(keys))
        md.append(f"| {s} | " + " | ".join(f"{c:.1f}" for c in cells) + " |")

    agree = Counter(sum(miss_by[k][s] for s in STRONG) for k in miss_by)
    md += ["", "### How many STRONG systems miss the same line", "",
           "| # systems | lines | example |", "|---|---|---|"]
    for n in range(1, len(STRONG) + 1):
        ex = next((lines_meta[k][0] for k in miss_by if sum(miss_by[k][s] for s in STRONG) == n
                   and lines_meta[k][1] >= 4), "")
        md.append(f"| {n} | {agree[n]} | {ex[:90]} |")
    md += ["", "Lines missed by every STRONG system:", ""]
    md += [f"- `{k[0]}` {lines_meta[k][0][:120]}" for k in sorted(miss_by)
           if all(miss_by[k][s] for s in STRONG) and lines_meta[k][1] >= 3][:40]

    md += ["", "## Hard words (content words, >= 3 reference occurrences, lowest mean recall over STRONG)", "",
           "| word | n | mean recall | " + " | ".join(STRONG) + " | most common replacements |",
           "|---" * (len(STRONG) + 4) + "|"]
    rows = []
    for w, per in word.items():
        n = per[STRONG[0]][1]
        if w in STOP or n < 3 or w in score.NUM_WORDS:
            continue
        rec = [per[s][0] / per[s][1] for s in STRONG]
        rows.append((sum(rec) / len(rec), w, n, rec))
    for mean, w, n, rec in sorted(rows)[:40]:
        reps = ", ".join(f"{r} ({c})" for r, c in subst[w].most_common(4))
        md.append(f"| {w} | {n} | {100 * mean:.0f} | " + " | ".join(f"{100 * r:.0f}" for r in rec) + f" | {reps} |")

    md += ["", "## Inserted passages (>= %d words)" % score.RUN, "",
           "| system | passages | repetition loops | top inserted passages |", "|---|---|---|---|"]
    for s in systems:
        top = Counter(t for t, _ in ins[s]).most_common(3)
        md.append(f"| {s} | {len(ins[s])} | {sum(l for _, l in ins[s])} | "
                  + "; ".join(f"“{t[:60]}” ×{c}" for t, c in top) + " |")

    md += ["", f"## Anchoring suspects: {sum(suspects.values())} reference tokens "
           f"({len(suspects)} types) that only the prefill reproduces", "",
           "| reference | other systems | n | example |", "|---|---|---|---|"]
    md += [f"| {w} | {a} | {n} | {suspect_ctx[(w, a)]} |" for (w, a), n in suspects.most_common(40)]

    OUT.mkdir(exist_ok=True)
    (OUT / "errors.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
