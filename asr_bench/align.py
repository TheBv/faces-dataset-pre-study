"""Force-align the reference transcripts to the audio (transcription-whisperx env, GPU).

    ~/miniforge3/envs/transcription-whisperx/bin/python align.py   (after windows.py)

For each alignment unit from windows.py (one or more VAD windows plus the
reference lines placed there), the unit's reference text is CTC-aligned to its
audio with torchaudio's German wav2vec2 model (VoxPopuli, the WhisperX default
for German) and torchaudio's C++ forced_align; WhisperX's own Python trellis
takes ~30 min per recording on our long units. Writes data/align/<doc>.json:
per line its words with start/end (seconds in the WAV) and score (mean
character probability; None = unalignable: no known characters, e.g. digits,
or text too long for its audio). These word times are also what the released
Word layer needs.
"""
import json
import re
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F

DATA = Path(__file__).resolve().parent / "data"
TAG = re.compile(r"\[[^\]]*\]")  # [unknown] etc. have no audio to align to
BUNDLE = torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_DE
CHUNK_S = 30  # emissions in 30 s pieces: wav2vec2 attention is quadratic in length


def clean(line):
    return re.sub(r"\s+([.,;:?!])", r"\1", " ".join(TAG.sub(" ", line).split()))


class Aligner:
    def __init__(self):
        self.model = BUNDLE.get_model().to("cuda").eval()
        self.labels = {c: i for i, c in enumerate(BUNDLE.get_labels())}

    @torch.inference_mode()
    def emission(self, wav):
        step = CHUNK_S * BUNDLE.sample_rate
        parts = [self.model(wav[:, i:i + step].cuda())[0] for i in range(0, wav.shape[1], step)
                 if wav.shape[1] - i >= 400]  # skip a sub-frame tail
        return torch.log_softmax(torch.cat(parts, dim=1), dim=-1).cpu()

    def __call__(self, wav, offset, words):
        """[(start, end, score) or None] per word; times in seconds from WAV start."""
        chars = [[self.labels[c] for c in w.lower() if c in self.labels and c not in "-|"] for w in words]
        targets = []
        for k, c in enumerate(x for x in chars if x):
            targets += ([self.labels["|"]] if k else []) + c
        if not targets:
            return [None] * len(words)
        em = self.emission(wav)
        try:
            ali, scores = F.forced_align(em, torch.tensor([targets], dtype=torch.int32), blank=0)
        except RuntimeError:  # more tokens than frames
            return [None] * len(words)
        spans = [s for s in F.merge_tokens(ali[0], scores[0].exp()) if s.token != self.labels["|"]]
        sec = wav.shape[1] / em.shape[1] / BUNDLE.sample_rate
        out, k = [], 0
        for c in chars:
            if not c:
                out.append(None)
                continue
            sp = spans[k:k + len(c)]
            k += len(c)
            out.append((offset + sp[0].start * sec, offset + sp[-1].end * sec,
                        sum(s.score for s in sp) / len(sp)))
        return out


def main():
    aligner = Aligner()
    out_dir = DATA / "align"
    for wf in sorted((out_dir / "windows").glob("*.json")):
        out = out_dir / wf.name
        if out.exists():
            continue
        plan = json.loads(wf.read_text())
        audio, sr = torchaudio.load(str(DATA / "audio" / f"{wf.stem}.wav"))
        assert sr == BUNDLE.sample_rate
        lines = [clean(l["text"]).split() for l in plan["lines"]]
        res = []
        for u in plan["units"]:
            words = [w for i in u["lines"] for w in lines[i]]
            if not words:
                continue
            a, b = int(u["start"] * sr), int(u["end"] * sr)
            timed = aligner(audio[:, a:b], u["start"], words)
            k = 0
            for i in u["lines"]:
                res.append({"line": i, "words": [
                    {"w": w, "start": t and round(t[0], 3), "end": t and round(t[1], 3), "score": t and round(t[2], 4)}
                    for w, t in zip(lines[i], timed[k:k + len(lines[i])])]})
                k += len(lines[i])
        out.write_text(json.dumps({"lines": sorted(res, key=lambda r: r["line"])}, ensure_ascii=False))
        print(wf.stem, len(res), "lines", flush=True)


if __name__ == "__main__":
    assert clean("Genau drei [unknown].") == "Genau drei."
    main()
