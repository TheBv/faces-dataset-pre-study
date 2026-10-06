"""WER of every system under alternative text normalisers (base env).

    python norms.py   -> results/norms.csv, results/norms.md

raw            lower-case, whitespace split (punctuation stays attached)
whisper        Whisper's BasicTextNormalizer (multilingual)
openasr        Open ASR Leaderboard German path (huggingface/open_asr_leaderboard, normalizer/data_utils.py
               MultilingualNormalizer(remove_diacritics=False)(t, lang="de") + eval_utils.normalize_compound_pairs):
               basic normaliser keeping umlauts, digits -> num2words(lang="de"), split/joined compounds merged
openasr-nofil  openasr + filled pauses dropped (our filler decision, nothing else)
ours           score.tokens(..., "normalized"), the paper's main WER
ours-X         ours with one step switched off (ablation)
"""
import contextlib
import csv
import re
import unicodedata
from difflib import SequenceMatcher

import num2words
import regex

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import score

DATA, OUT = score.DATA, score.OUT
basic = BasicTextNormalizer()


@contextlib.contextmanager
def without(**patch):
    """Temporarily replace score.py globals to switch one normalisation step off."""
    old = {k: getattr(score, k) for k in patch}
    for k, v in patch.items():
        setattr(score, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(score, k, v)


def openasr_norm(s):
    """Re-implementation of the leaderboard's MultilingualNormalizer(remove_diacritics=False)(s, lang="de")."""
    s = s.lower()
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
    s = re.sub(r"\(([^)]+?)\)", "", s)
    s = "".join(" " if unicodedata.category(c)[0] in "SP" else c for c in unicodedata.normalize("NFKC", s)).lower()
    s = regex.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", s)
    return re.sub(r"\d+", lambda m: num2words.num2words(int(m.group()), lang="de"), s)


def merge_compounds(ref, hyp):
    """Leaderboard's normalize_compound_pairs: where ref/hyp differ only by spaces, use the joined form.
    autojunk off: the leaderboard aligns short utterances, we align whole recordings."""
    r, h = [], []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        rc, hc = "".join(ref[i1:i2]), "".join(hyp[j1:j2])
        if tag != "equal" and rc == hc:
            r.append(rc), h.append(hc)
        else:
            r += ref[i1:i2]
            h += hyp[j1:j2]
    return r, h


NOTHING = re.compile(r"(?!x)x")
ABLATIONS = {
    "ours-numbers": dict(number=str),
    "ours-compounds": dict(COMPOUNDS={}),
    "ours-elision/spelling": dict(ELISION={}, SPELLING={}),
    "ours-fragments": dict(FRAGMENT=NOTHING),
    "ours-tags": dict(TAG=NOTHING, FILLER_TAG=NOTHING),
}
NORMS = {
    "raw": lambda t: t.lower().split(),
    "whisper": lambda t: basic(t).split(),
    "openasr": lambda t: openasr_norm(t).split(),
    "openasr-nofil": lambda t: [w for w in openasr_norm(t).split() if not score.FILLER.match(w)],
    "ours": lambda t: score.tokens(t, "normalized"),
    **{k: (lambda t: score.tokens(t, "normalized")) for k in ABLATIONS},
}


def wer(pairs, norm):
    e = n = 0
    for ref, hyp in pairs:
        r, h = norm(ref), norm(hyp)
        if norm in (NORMS["openasr"], NORMS["openasr-nofil"]):
            r, h = merge_compounds(r, h)
        out, _ = score.align(r, h)
        e += out.substitutions + out.deletions + out.insertions
        n += len(r)
    return e / n


def kendall(a, b):
    """Kendall tau between two {system: value} rankings."""
    ks = list(a)
    pairs = [(x, y) for i, x in enumerate(ks) for y in ks[i + 1:]]
    s = sum((a[x] - a[y]) * (b[x] - b[y]) > 0 for x, y in pairs) - sum((a[x] - a[y]) * (b[x] - b[y]) < 0 for x, y in pairs)
    return s / len(pairs)


def main():
    score.learn_from_data()
    refs = {p.stem: p.read_text(encoding="utf-8") for p in (DATA / "ref").glob("*.txt")}
    systems = {}
    for d in sorted(p for p in (DATA / "hyp").iterdir() if p.is_dir()):
        if all((d / f"{doc}.json").exists() for doc in refs):
            systems[d.name] = [(refs[doc], score.hyp_text(d / f"{doc}.json")) for doc in sorted(refs)]
    res = {}
    for name, norm in NORMS.items():
        with without(**ABLATIONS.get(name, {})):
            res[name] = {s: wer(pairs, norm) for s, pairs in systems.items()}
    order = sorted(systems, key=res["ours"].get)
    with open(OUT / "norms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", *NORMS])
        w.writerows([s, *(f"{res[n][s]:.4f}" for n in NORMS)] for s in order)
    lines = ["| system | " + " | ".join(NORMS) + " |", "|---" * (len(NORMS) + 1) + "|"]
    lines += [f"| {s} | " + " | ".join(f"{100 * res[n][s]:.1f}" for n in NORMS) + " |" for s in order]
    lines.append("| Kendall tau vs ours | " + " | ".join(f"{kendall(res[n], res['ours']):.2f}" for n in NORMS) + " |")
    (OUT / "norms.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def selftest():
    assert basic("Ähm, ich hab's [unknown] gesagt (leise).") == "ähm ich hab s gesagt "
    assert openasr_norm("Ähm, 1.600 Euro [unknown] Zoom-Call") == "ähm eintausendsechshundert euro zoom call"
    assert merge_compounds("im zoom call".split(), "im zoomcall".split()) == (["im", "zoomcall"], ["im", "zoomcall"])
    assert kendall({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 3, "c": 2}) == 1 / 3
    with without(**ABLATIONS["ours-numbers"]):
        assert score.tokens("3 Jahre", "normalized") == ["3", "jahre"]
    assert score.tokens("3 Jahre", "normalized") == ["drei", "jahre"]
    print("selftest ok")


if __name__ == "__main__":
    import sys
    selftest() if "--selftest" in sys.argv else main()
