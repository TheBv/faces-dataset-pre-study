"""Place every reference line in a VAD window, and find speech the reference misses (base env).

    python windows.py  -> data/align/windows/<doc>.json, results/coverage.csv

Reference lines carry no reliable times (inserted lines have none, 07/14 none
at all), so each line is located through the segment-level models, whose
output is stored per VAD window: every reference word aligned (WER alignment)
to a hypothesis word votes for that word's window, over four models; a line
takes its majority window, kept monotone in time. Lines no model heard go to
an empty window between their neighbours if there is one, else to the
previous line's window. A long line spans every window holding >= 25% of its
votes; windows joined by such a line form one alignment unit, which align.py
force-aligns against that unit's lines.

Coverage: a window where >= 3 of the 4 models output >= 2 words, most of which
match no reference word (insertions in the whole-document alignment), is
speech the annotators may have missed (or cross-talk they chose to omit).
"""
import csv
import json
from collections import Counter

import errors
import score

DATA, OUT = score.DATA, score.OUT
SEG_SYSTEMS = ["qwen3-asr-1.7b", "parakeet-tdt-0.6b-v3", "canary-1b-v2", "voxtral-mini-3b"]


def place(votes, n_windows):
    """Window per line from vote Counters (None = unheard), monotone, unheard lines into gaps."""
    win, prev = [], 0
    for v in votes:
        w = max(v.most_common(1)[0][0], prev) if v else None
        win.append(w)
        prev = w if w is not None else prev
    i = 0
    while i < len(win):
        if win[i] is not None:
            i += 1
            continue
        j = i
        while j < len(win) and win[j] is None:
            j += 1
        a = win[i - 1] if i else -1
        b = win[j] if j < len(win) else n_windows
        used = set(w for w in win if w is not None)
        gaps = [w for w in range(a + 1, b) if w not in used]
        for k in range(i, j):
            win[k] = gaps[min(k - i, len(gaps) - 1)] if gaps else max(a, 0)
        i = j
    return win


def spans(votes, win):
    """(first, last) window per line: its placed window widened to strongly voted neighbours, monotone."""
    out, prev_hi = [], 0
    for v, w in zip(votes, win):
        strong = [k for k, n in v.items() if n >= 0.25 * sum(v.values())] + [w]
        lo, hi = max(min(strong), prev_hi if out else 0), max(max(strong), w)
        out.append((min(lo, w), hi))
        prev_hi = max(prev_hi, lo)
    return out


def units(line_spans, n_windows):
    """Merge windows covered by one line into alignment units: [(first_win, last_win, [lines])]."""
    res = []
    for i, (lo, hi) in enumerate(line_spans):
        if res and lo <= res[-1][1]:
            res[-1] = (res[-1][0], max(res[-1][1], hi), res[-1][2] + [i])
        else:
            res.append((lo, hi, [i]))
    return res


def main():
    score.learn_from_data()
    (DATA / "align" / "windows").mkdir(parents=True, exist_ok=True)
    rows = []
    for doc in sorted(p.stem for p in (DATA / "ref").glob("*.txt")):
        ref, line_of, lines = errors.ref_lines(doc)
        vad = json.loads((DATA / "vad" / f"{doc}.json").read_text())
        votes = [Counter() for _ in lines]
        heard = [Counter() for _ in vad]  # window -> models with >= 2 words there
        for s in SEG_SYSTEMS:
            segs = json.loads((DATA / "hyp" / s / f"{doc}.json").read_text())["segments"]
            assert len(segs) == len(vad), (s, doc)
            hyp, win_of = [], []
            for w, seg in enumerate(segs):
                t = score.tokens(seg["text"], "normalized")
                hyp += t
                win_of += [w] * len(t)
            chunks = score.align(ref, hyp)[1]
            inserted = [False] * len(hyp)
            for c in chunks:
                if c.type == "insert":
                    inserted[c.hyp_start_idx:c.hyp_end_idx] = [True] * (c.hyp_end_idx - c.hyp_start_idx)
            per_win = Counter(win_of)
            ins_win = Counter(w for w, x in zip(win_of, inserted) if x)
            for w, n in per_win.items():  # the model hears >= 2 words here that match nothing in the reference
                heard[w][s] = n >= 2 and ins_win[w] >= 0.5 * n
            for c in chunks:
                if c.type in ("equal", "substitute") and c.ref_end_idx - c.ref_start_idx == c.hyp_end_idx - c.hyp_start_idx:
                    for k, j in zip(range(c.ref_start_idx, c.ref_end_idx), range(c.hyp_start_idx, c.hyp_end_idx)):
                        votes[line_of[k]][win_of[j]] += 1
        win = place(votes, len(vad))
        line_spans = spans(votes, win)
        texts = [t for t, _ in lines]
        (DATA / "align" / "windows" / f"{doc}.json").write_text(json.dumps({
            "units": [{"start": vad[lo][0], "end": vad[hi][1], "windows": [lo, hi], "lines": ls}
                      for lo, hi, ls in units(line_spans, len(vad))],
            "lines": [{"text": t, "windows": list(sp), "heard": bool(v)} for t, sp, v in zip(texts, line_spans, votes)],
        }, ensure_ascii=False))
        for k, (s, e) in enumerate(vad):
            if sum(heard[k].values()) >= 3:
                seg_text = json.loads((DATA / "hyp" / "qwen3-asr-1.7b" / f"{doc}.json").read_text())["segments"][k]["text"]
                rows.append({"doc": doc, "window": k, "start": s, "end": e, "seconds": round(e - s, 1),
                             "models_hearing": sum(heard[k].values()), "qwen3_text": seg_text})
    with open(OUT / "coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc", "window", "start", "end", "seconds", "models_hearing", "qwen3_text"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} windows with speech the reference does not contain, "
          f"{sum(r['seconds'] for r in rows) / 60:.1f} min -> {OUT / 'coverage.csv'}")


if __name__ == "__main__":
    assert place([Counter({0: 3}), Counter(), Counter({3: 2})], 5) == [0, 1, 3]
    assert place([Counter({2: 1}), Counter({1: 5})], 4) == [2, 2]          # monotone
    assert place([Counter(), Counter({1: 1})], 3) == [0, 1]                # leading unheard line
    assert spans([Counter({6: 5, 7: 1, 8: 4}), Counter({9: 3})], [6, 9]) == [(6, 8), (9, 9)]
    assert units([(0, 0), (0, 2), (3, 3)], 4) == [(0, 2, [0, 1]), (3, 3, [2])]
    main()
