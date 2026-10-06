"""List problematic occurrences in the corrected references, and convert the rule-based ones (base env).

    python ref_issues.py  -> results/reference_issues.csv (+ counts per annotator)

Detection runs on the references as the annotators wrote them (review DB,
data/ref_raw for 07/14); prepare.py uses convert() to write data/ref.
One row per occurrence, with the annotator, line number (1-based, non-empty
lines as shown in the review tool) and WAV time (the line's own time, or
"~" + the previous line's end for lines the corrector inserted).

Categories
  digit_1_13              number 1-13 written as digits (convention: spell out up to dreizehn)
  elision                 colloquial contraction not written out (würd -> würde, geht's -> geht es)
  unk_marker              [unk] / [unk*] instead of the convention's [unknown]
  video_garble            Video...-garble kept from the prefill (Videotilfonat, Videotip, ...)
  anchoring_suspect       word only the prefill reproduces; >= 4 of the other STRONG systems
                          disagree and >= 3 agree on `models_say` (check by listening)
  inserted_missed_by_all  line added by the corrector that no STRONG system hears
                          (quiet echo of the answer, or the other speaker bleeding in?)
"""
import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from num2words import num2words

import errors
import score

HERE = Path(__file__).resolve().parent
EXTERNAL_ANNOTATOR = {"07": "Patrick", "14": "Leon"}  # corrected outside the tool (review/config.py)
ELISION = dict(score.ELISION, habs="habe es", ists="ist es", wirds="wird es")
SUFFIX = {"s": "es", "m": "dem"}  # bin's -> bin es, auf'm -> auf dem
VIDEO_GARBLE = re.compile(r"\bvideo-?\s?t(?!elefonat)\w*|\btilfonat\w*", re.I)
WORD = re.compile(r"\d+(?:[.,:]\d+)+|\w+(?:'\w*)?|'\w+|\[[^\]]*\]")
# Verb stems that are also correct imperatives ("Sag mal", "Komm", "Hör mal"): only an elision next to "ich".
IMPERATIVE_OK = {"sag", "geh", "glaub", "hör", "les", "komm", "mach"}
ENGLISH_APOS = {"it's", "that's", "what's", "let's", "there's", "here's"}
AUTO = {"digit_1_13", "unk_marker", "elision"}  # rule-based: converted by convert(); the rest need listening


def expand(low, near_ich):
    """Written-out form of an elided (lower-case) token, or None."""
    stem, apos, suf = low.partition("'")
    if stem in IMPERATIVE_OK and not near_ich or low in ENGLISH_APOS:
        return None
    if not apos:
        return ELISION.get(stem)
    if suf not in SUFFIX and not (suf == "" and stem in ELISION):  # Girls', It's, That's
        return None
    parts = [ELISION.get(stem, stem)] if stem else []
    return " ".join(parts + ([SUFFIX[suf]] if suf else []))


def line_info():
    """doc -> (annotator, [(text, time)]) for every reference document."""
    ann = sqlite3.connect(f"file:{score.ANNOTATIONS_DB}?mode=ro", uri=True)
    out = {}
    for doc, who, lines in ann.execute("SELECT doc_id, annotator, lines_json FROM transcripts WHERE done = 1"):
        rows, prev_end = [], None
        for l in json.loads(lines):
            if not l["text"].strip():
                continue
            if "start" in l:
                rows.append((l["text"].strip(), f"{l['start']:.1f}-{l['end']:.1f}"))
                prev_end = l["end"]
            else:
                rows.append((l["text"].strip(), f"~{prev_end:.1f}" if prev_end is not None else "~start"))
        out[doc] = (who, rows)
    for p in (score.DATA / "ref_raw").glob("*.txt"):
        if p.stem not in out:
            out[p.stem] = (EXTERNAL_ANNOTATOR.get(p.stem[:2], "?"),
                           [(l.strip(), "") for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])
    return out


def issues(text):
    """Yield (category, regex match, replacement) for every surface issue in one line."""
    ms = list(WORD.finditer(text))
    for j, m in enumerate(ms):
        tok, low = m.group(0), m.group(0).lower()
        if tok.isdigit() and not tok.startswith("0") and 1 <= int(tok) <= 13:
            word = num2words(int(tok), lang="de")
            before = text[:m.start()].rstrip()
            yield "digit_1_13", m, word.capitalize() if not before or before[-1] in ".?!" else word
        elif low in ("[unk]", "[unk*]"):
            yield "unk_marker", m, "[unknown]"
        else:
            near_ich = "ich" in (ms[j - 1].group(0).lower() if j else "",
                                 ms[j + 1].group(0).lower() if j + 1 < len(ms) else "")
            full = expand(low, near_ich)
            if full:
                yield "elision", m, full[0].upper() + full[1:] if tok[0].isupper() else full
    for m in VIDEO_GARBLE.finditer(text):
        yield "video_garble", m, "Videotelefonat?"


def convert(text):
    """Apply the rule-based (AUTO) fixes to one line."""
    for _, m, rep in sorted((i for i in issues(text) if i[0] in AUTO), key=lambda i: -i[1].start()):
        text = text[:m.start()] + rep + text[m.end():]
    return text


def main():
    score.learn_from_data()
    info = line_info()
    others = [s for s in errors.STRONG if s != "crisperwhisper-prefill"]
    rows = []
    for doc in sorted(info):
        who, lines = info[doc]
        eid, role = doc.split("_")
        base = lambda i, cat, tok, sugg: {"doc": doc, "eid": eid, "role": role, "annotator": who,
                                          "line_no": i + 1, "time_s": lines[i][1], "category": cat,
                                          "token": tok, "suggestion": sugg,
                                          "auto_converted": "yes" if cat in AUTO else "no",
                                          "line_text": lines[i][0]}
        for i, (text, _) in enumerate(lines):
            rows += [base(i, cat, m.group(0), rep) for cat, m, rep in issues(text)]

        # Alignment-based checks run on the converted reference (same lines, same order).
        ref, line_of, ref_lines = errors.ref_lines(doc)
        assert len(ref_lines) == len(lines), doc
        at, missed = {}, {}
        for s in errors.STRONG:
            missed[s], _, _, at[s] = errors.analyse(s, doc, ref, line_of)
        for k, w in enumerate(ref):
            if at["crisperwhisper-prefill"][k] != "=":
                continue
            alts = [at[s][k] for s in others if at[s][k] != "="]
            top = Counter(a for a in alts if a).most_common(1)
            if len(alts) >= 4 and top and top[0][1] >= 3:
                rows.append(base(line_of[k], "anchoring_suspect", w, f"{top[0][0]} ({top[0][1]}/{len(others)} models)"))
        for i, (text, n) in enumerate(ref_lines):
            if n and lines[i][1] in ("",) or (n and lines[i][1].startswith("~")):
                if all(missed[s].get(i) for s in errors.STRONG):
                    rows.append(base(i, "inserted_missed_by_all", text, "listen: echo or other speaker?"))

    rows.sort(key=lambda r: (r["doc"], r["line_no"], r["category"]))
    out = score.OUT / "reference_issues.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cats = sorted({r["category"] for r in rows})
    by = Counter((r["annotator"], r["category"]) for r in rows)
    print(f"{len(rows)} rows -> {out}\n")
    print("| annotator | docs | " + " | ".join(cats) + " |")
    print("|---" * (len(cats) + 2) + "|")
    for who in sorted({r["annotator"] for r in rows}):
        docs = sum(1 for d in info.values() if d[0] == who)
        print(f"| {who} | {docs} | " + " | ".join(str(by[(who, c)]) for c in cats) + " |")


def selftest():
    got = [(c, m.group(0)) for c, m, _ in issues("Sag mal, ich sag mal 1.600 und 3 würd auf'm It's")]
    assert got == [("elision", "sag"), ("digit_1_13", "3"), ("elision", "würd"), ("elision", "auf'm")], got
    assert convert("3. Hab's n' Plan, [unk*] Sag mal 12 Girls' Day, 06 Uhr, nix.") == \
        "Drei. Habe es ein Plan, [unknown] Sag mal zwölf Girls' Day, 06 Uhr, nix."
    assert convert("Gibts das? Ja, 3.") == "Gibt es das? Ja, drei."


if __name__ == "__main__":
    selftest()
    main()
