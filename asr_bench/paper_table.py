"""Print the LaTeX rows of the paper's ASR table from results/summary.csv + semantic.csv (base env).

    python paper_table.py  -> results/paper_table.tex (also printed)
"""
import csv

import score

OUT = score.OUT
GROUPS = [
    (None, [("crisperwhisper-prefill", r"CrisperWhisper$^\dagger$")]),
    ("VAD segments", [
        ("whisperx-large-v3", "WhisperX (large-v3)"),
        ("qwen3-asr-1.7b", "Qwen3-ASR 1.7B"),
        ("voxtral-mini-3b", "Voxtral Mini 3B"),
        ("parakeet-tdt-0.6b-v3", "Parakeet-TDT 0.6B v3"),
        ("phi-4-multimodal", "Phi-4-multimodal"),
        ("canary-1b-v2", "Canary 1B v2"),
        ("crisperwhisper2-large-verbatim-vad", "CrisperWhisper 2 (verbatim)"),
        ("crisperwhisper2-large-intended-vad", "CrisperWhisper 2 (intended)"),
        ("granite-speech-4.1-2b", "Granite-Speech 4.1 2B"),
    ]),
    ("Full recordings (long-form decoding)", [
        ("whisper-large-v3-turbo-nocond", r"Whisper large-v3-turbo$^\ddagger$"),
        ("crisperwhisper2-large-verbatim", "CrisperWhisper 2 (verbatim)"),
        ("whisper-large-v3-turbo", "Whisper large-v3-turbo"),
        ("whisper-large-v3", "Whisper large-v3"),
    ]),
]


def main():
    summ = {r["system"]: r for r in csv.DictReader(open(OUT / "summary.csv"))}
    sem = {r["system"]: r for r in csv.DictReader(open(OUT / "semantic.csv"))}
    pct = lambda r, k: f"{100 * float(r[k]):.1f}"
    lines = []
    for title, rows in GROUPS:
        lines.append(r"\midrule")
        if title:
            lines.append(rf"\multicolumn{{12}}{{l}}{{\emph{{{title}}}}} \\")
        for key, name in rows:
            if key not in summ:
                lines.append(f"% {key}: no results")
                continue
            r, s = summ[key], sem.get(key)
            sd = f"{100 * float(s['semdist']):.2f}" if s else "--"
            lines.append(" & ".join([name, f"{pct(r, 'wer')} [{pct(r, 'wer_lo')}--{pct(r, 'wer_hi')}]",
                                     pct(r, "cer"), sd, pct(r, "wer_interviewer"), pct(r, "wer_interviewee"),
                                     pct(r, "wer_verbatim"), pct(r, "D"), pct(r, "I"), pct(r, "cue_recall"),
                                     pct(r, "num_recall"), f"{float(r['ins_runs_h']):.1f}"]) + r" \\")
    tex = "\n".join(lines[1:])  # the table's own \midrule after the header replaces the first one
    (OUT / "paper_table.tex").write_text(tex + "\n", encoding="utf-8")
    print(tex)


if __name__ == "__main__":
    main()
