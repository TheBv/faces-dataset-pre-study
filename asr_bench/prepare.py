"""Export the ASR benchmark inputs (no GPU, base env).

    data/ref_raw/<eid>_<Role>.txt          corrected reference as the annotators wrote it
    data/ref/<eid>_<Role>.txt              same, with the rule-based convention fixes applied
                                           (ref_issues.convert: [unk] -> [unknown], digits 1-13
                                           spelled out, elisions written out); used for scoring
    data/hyp/crisperwhisper-prefill/*.json  the CrisperWhisper v1 transcript the
                                            correctors started from (neo/transcribe)
    data/audio/<eid>_<Role>.wav             symlinks into transcription_output/

References: every `done` document from the review tool, plus 07/14 which were
corrected the same way outside the tool (plain .txt in the transcription repo).
A document with no finished reference still gets its audio link, so the
runners transcribe all 54 recordings and newly finished references can be
scored without re-running any model.
"""
import json
import sqlite3
from pathlib import Path

from ref_issues import convert
from score import ANNOTATIONS_DB

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent / "review"
REPO_ROOT = HERE.parents[3]
WAV_DIR = REPO_ROOT / "transcription_output"
EXTERNAL_REFS = Path("/mnt/d/repos/transcription/datasets-transcription/custom/transcripts")
DATA = HERE / "data"


def main():
    for d in ("ref", "ref_raw", "hyp/crisperwhisper-prefill", "audio"):
        (DATA / d).mkdir(parents=True, exist_ok=True)

    pred = sqlite3.connect(REVIEW / "data" / "predicted.sqlite")
    docs = pred.execute("SELECT id, eid, role, words_json FROM documents").fetchall()
    for doc_id, eid, role, words in docs:
        link = DATA / "audio" / f"{doc_id}.wav"
        if not link.exists():
            link.symlink_to(WAV_DIR / f"experiment_{eid:02d}" / f"{role.lower()}.wav")
        text = " ".join(w["text"] for w in json.loads(words))
        (DATA / "hyp/crisperwhisper-prefill" / f"{doc_id}.json").write_text(
            json.dumps({"filename": doc_id, "status": "success", "transcription": text}, ensure_ascii=False))

    refs = {}
    ann = sqlite3.connect(f"file:{ANNOTATIONS_DB}?mode=ro", uri=True)
    for doc_id, lines in ann.execute("SELECT doc_id, lines_json FROM transcripts WHERE done = 1"):
        refs[doc_id] = [l["text"] for l in json.loads(lines)]
    for f in EXTERNAL_REFS.glob("*/*.txt"):
        refs[f"{f.parent.name}_{f.stem.capitalize()}"] = f.read_text(encoding="utf-8").splitlines()
    changed = 0
    for doc_id, lines in refs.items():
        raw = [l.strip() for l in lines if l.strip()]
        conv = [convert(l) for l in raw]
        changed += sum(a != b for a, b in zip(raw, conv))
        (DATA / "ref_raw" / f"{doc_id}.txt").write_text("\n".join(raw) + "\n", encoding="utf-8")
        (DATA / "ref" / f"{doc_id}.txt").write_text("\n".join(conv) + "\n", encoding="utf-8")

    missing = sorted({d[0] for d in docs} - set(refs))
    print(f"{ANNOTATIONS_DB.name}: {len(docs)} recordings, {len(refs)} references ({changed} lines converted); no reference yet: {missing}")


if __name__ == "__main__":
    main()
