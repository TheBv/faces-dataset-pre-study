"""Semantic distance per reference utterance, and a negation blind-spot test (base env, GPU).

    python semantic.py   -> results/semantic.csv, results/semantic.md

SemDist (Kim et al., 2021) = 1 - cosine(emb(reference), emb(hypothesis)), per
reference line, with the multilingual E5 model (its recommended "query: " prefix,
normalized embeddings). Both sides are the normalized token strings of score.py,
so SemDist sees exactly what WER sees. The hypothesis span of a line is read
off the WER alignment (insertions go to the line of the preceding reference
word); a line a system missed entirely is compared with the empty string.

Blind-spot test (Anschütz et al., 2023, show embedding metrics under-weight
negation): for every distinct reference line with a negation cue and >= 4 words, delete
(a) the first cue and (b) a random other word. Both edits cost exactly one WER
error; SemDist should rank (a) as the bigger meaning change. Real-error check:
per system, SemDist of lines where a cue was lost vs. lines with a same-sized
error that kept all cues.
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import errors
import score

MODEL = "intfloat/multilingual-e5-large-instruct"
DATA, OUT = score.DATA, score.OUT


def hyp_by_line(ref, line_of, hyp, chunks, n_lines):
    """Hypothesis tokens assigned to each reference line via the alignment."""
    out = [[] for _ in range(n_lines)]
    for c in chunks:
        h = hyp[c.hyp_start_idx:c.hyp_end_idx]
        if c.type == "insert":
            out[line_of[c.ref_start_idx - 1] if c.ref_start_idx else 0] += h
        elif c.type in ("equal", "substitute") and c.ref_end_idx - c.ref_start_idx == len(h):
            for k, t in zip(range(c.ref_start_idx, c.ref_end_idx), h):
                out[line_of[k]].append(t)
        elif c.type == "substitute":  # unequal spans: give the whole span to its first line
            out[line_of[c.ref_start_idx]] += h
    return out


class Embedder:
    def __init__(self):
        self.model, self.cache = SentenceTransformer(MODEL, device="cuda"), {}

    def __call__(self, texts):
        new = sorted({t for t in texts if t not in self.cache})
        if new:
            embs = self.model.encode(["query: " + t for t in new], batch_size=128,
                                     normalize_embeddings=True, show_progress_bar=False)
            self.cache.update(zip(new, embs))
        return np.stack([self.cache[t] for t in texts])


def semdist(emb, refs, hyps):
    return 1 - (emb(refs) * emb(hyps)).sum(axis=1)


def main():
    score.learn_from_data()
    emb = Embedder()
    systems = sorted(p.name for p in (DATA / "hyp").iterdir() if p.is_dir())
    docs = sorted(p.stem for p in (DATA / "ref").glob("*.txt"))

    per_line = defaultdict(list)  # system -> [(role, semdist, n_err, cue_lost, has_cue)]
    blind_ref = []                # (ref line tokens) with a cue and >= 4 words
    for doc in docs:
        ref, line_of, lines = errors.ref_lines(doc)
        starts = np.cumsum([0] + [n for _, n in lines])
        ref_lines = [ref[starts[i]:starts[i + 1]] for i in range(len(lines))]
        keep = [i for i, t in enumerate(ref_lines) if t]
        blind_ref += [ref_lines[i] for i in keep if len(ref_lines[i]) >= 4
                      and any(w in score.NEG_CUES for w in ref_lines[i])]
        for s in systems:
            hyp = score.tokens(score.hyp_text(DATA / "hyp" / s / f"{doc}.json"), "normalized")
            _, chunks = score.align(ref, hyp)
            by_line = hyp_by_line(ref, line_of, hyp, chunks, len(lines))
            d = semdist(emb, [" ".join(ref_lines[i]) for i in keep], [" ".join(by_line[i]) for i in keep])
            for i, dist in zip(keep, d):
                r, h = ref_lines[i], by_line[i]
                n_err = score.align(r, h)[0].wer * len(r) if h else len(r)
                cues_r = sum(w in score.NEG_CUES for w in r)
                cues_kept = min(cues_r, sum(w in score.NEG_CUES for w in h))
                per_line[s].append((doc.split("_")[1], float(dist), round(n_err), cues_kept < cues_r, cues_r > 0))

    rows = []
    for s, L in per_line.items():
        d = np.array([x[1] for x in L])
        role = {r: np.mean([x[1] for x in L if x[0] == r]) for r in ("Interviewer", "Interviewee")}
        # cue lost vs. same number of word errors without cue loss (1-2 errors, lines that contain a cue)
        lost = [x[1] for x in L if x[3] and 1 <= x[2] <= 2]
        kept = [x[1] for x in L if x[4] and not x[3] and 1 <= x[2] <= 2]
        rows.append({"system": s, "semdist": d.mean(), "semdist_interviewer": role["Interviewer"],
                     "semdist_interviewee": role["Interviewee"], "exact_lines": np.mean(d < 1e-6),
                     "cue_lost_n": len(lost), "semdist_cue_lost": np.mean(lost) if lost else float("nan"),
                     "semdist_cue_kept": np.mean(kept) if kept else float("nan")})
    rows.sort(key=lambda r: r["semdist"])

    rng = random.Random(42)
    blind_ref = [list(t) for t in sorted({tuple(t) for t in blind_ref})]  # scripted lines repeat across interviews
    a, b = [], []
    for toks in blind_ref:
        cue = next(i for i, w in enumerate(toks) if w in score.NEG_CUES)
        other = rng.choice([i for i, w in enumerate(toks) if w not in score.NEG_CUES])
        a.append(" ".join(toks[:cue] + toks[cue + 1:]))
        b.append(" ".join(toks[:other] + toks[other + 1:]))
    full = [" ".join(t) for t in blind_ref]
    da, db = semdist(emb, full, a), semdist(emb, full, b)

    with open(OUT / "semantic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["# Semantic distance (SemDist x 100, lower = closer; per reference line)", "",
          "| system | SemDist | interviewer | interviewee | exact lines % | lines w/ lost cue (1-2 errors) "
          "| SemDist cue lost | SemDist same errors, cues kept |", "|---" * 8 + "|"]
    md += [f"| {r['system']} | {100 * r['semdist']:.2f} | {100 * r['semdist_interviewer']:.2f} | "
           f"{100 * r['semdist_interviewee']:.2f} | {100 * r['exact_lines']:.1f} | {r['cue_lost_n']} | "
           f"{100 * r['semdist_cue_lost']:.2f} | {100 * r['semdist_cue_kept']:.2f} |" for r in rows]
    md += ["", f"## Blind-spot test on {len(full)} reference lines with a negation cue (>= 4 words)", "",
           "Each edit deletes exactly one word (identical WER).", "",
           f"- delete the negation cue: mean SemDist x100 = {100 * da.mean():.2f} (median {100 * np.median(da):.2f})",
           f"- delete a random other word: mean SemDist x100 = {100 * db.mean():.2f} (median {100 * np.median(db):.2f})",
           f"- cue deletion judged the *smaller* change in {100 * np.mean(da < db):.1f}% of lines", ""]
    ex = sorted(zip(da - db, full, a, b))[:5]
    md += ["Examples where deleting the cue looks least harmful:", ""]
    md += [f"- `{f}` -> cue deleted `{x}` vs. random word deleted `{y}`" for _, f, x, y in ex]
    (OUT / "semantic.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
