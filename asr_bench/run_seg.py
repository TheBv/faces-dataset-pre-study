"""Transcribe every recording with a <=30 s model over the shared VAD windows.

    python run_seg.py <system>     (env per system: see run_all.sh)

Writes data/hyp/<system>/<doc>.json ({"transcription": ..., "segments": [...]}),
skipping documents that already have output. Granite, Phi-4 and Qwen3-ASR reuse
the transcription repo's own processors (prompt, loading, decoding); only their
fixed 30 s / 2 s-overlap chunker is swapped for data/vad windows, because that
chunker concatenates the overlap twice and would inflate insertions.
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np

SR = 16000
TRANSCRIPTION_REPO = "/mnt/d/repos/transcription"
DATA = Path(__file__).resolve().parent / "data"


def load(wav):
    with wave.open(str(wav)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, wav
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def cut(audio, segs):
    return [audio[int(s * SR):int(e * SR)] for s, e in segs]


def _repo(module, cls, *args):
    sys.path.insert(0, TRANSCRIPTION_REPO)
    sys.argv = ["run_seg", "--input", str(DATA / "audio"), "--output", "/dev/null", *args]
    return getattr(__import__(module, fromlist=[cls]), cls)()


def granite():
    proc = _repo("src.models.granite_speech.main", "GraniteSpeechProcessor", "--max_new_tokens", "448")
    return lambda wav, segs: [proc._transcribe_chunk(c) for c in cut(load(wav), segs)]


def qwen3():
    proc = _repo("src.models.qwen3_asr.main", "Qwen3ASRProcessor", "--lang", "de", "--max_new_tokens", "448")
    return lambda wav, segs: [proc._transcribe_chunk(c) for c in cut(load(wav), segs)]


def phi4():
    proc = _repo("src.models.phi_4_multimodal_instruct.main", "Phi4MultimodalInstruct", "--max_new_tokens", "448")

    def run(wav, segs):
        proc.files = [wav]
        proc._chunk_audio = lambda a, sr: [(c, sr) for c in cut(a, segs)]
        [res] = proc.process()
        if res.status != "success":
            raise RuntimeError(f"phi4 failed on {wav}")
        return res.transcription  # the repo processor joins chunks itself
    return run


def nemo(name, **prompt):
    def factory():
        from nemo.collections.asr.models import ASRModel
        model = ASRModel.from_pretrained(name).cuda().eval()

        def run(wav, segs):
            hyps = model.transcribe(cut(load(wav), segs), batch_size=8, verbose=False, **prompt)
            return [h.text if hasattr(h, "text") else h for h in hyps]
        return run
    return factory


def voxtral():
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    rid = "mistralai/Voxtral-Mini-3B-2507"
    proc = AutoProcessor.from_pretrained(rid)
    model = VoxtralForConditionalGeneration.from_pretrained(rid, dtype=torch.bfloat16).to("cuda")

    def run(wav, segs):
        chunks, out = cut(load(wav), segs), []
        for i in range(0, len(chunks), 8):
            batch = chunks[i:i + 8]
            inputs = proc.apply_transcription_request(
                language="de", audio=batch, format=["wav"] * len(batch), model_id=rid, sampling_rate=SR
            ).to("cuda", dtype=torch.bfloat16)
            gen = model.generate(**inputs, max_new_tokens=448, do_sample=False)
            out += proc.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return out
    return run


def crisper2(mode):
    """CrisperWhisper 2.0 on the VAD windows: each <= 30 s, so no long-form context to loop on."""
    def factory():
        from crisperwhisper import CrisperWhisperModel
        model = CrisperWhisperModel("large", backend="ct2")
        return lambda wav, segs: [model.transcribe(c, sr=SR, language="de", mode=mode).text
                                  for c in cut(load(wav), segs)]
    return factory


SYSTEMS = {
    "crisperwhisper2-large-verbatim-vad": crisper2("verbatim"),
    "crisperwhisper2-large-intended-vad": crisper2("intended"),
    "granite-speech-4.1-2b": granite,
    "phi-4-multimodal": phi4,
    "qwen3-asr-1.7b": qwen3,
    "parakeet-tdt-0.6b-v3": nemo("nvidia/parakeet-tdt-0.6b-v3"),
    "canary-1b-v2": nemo("nvidia/canary-1b-v2", source_lang="de", target_lang="de"),
    "voxtral-mini-3b": voxtral,
}


def main(system):
    out_dir = DATA / "hyp" / system
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [w for w in sorted((DATA / "audio").glob("*.wav")) if not (out_dir / f"{w.stem}.json").exists()]
    if not todo:
        return
    run = SYSTEMS[system]()
    for i, wav in enumerate(todo, 1):
        segs = json.loads((DATA / "vad" / f"{wav.stem}.json").read_text())
        texts = run(wav, segs)
        seg_out = [] if isinstance(texts, str) else [
            {"start": s, "end": e, "text": t.strip()} for (s, e), t in zip(segs, texts)]
        text = texts if isinstance(texts, str) else " ".join(s["text"] for s in seg_out if s["text"])
        (out_dir / f"{wav.stem}.json").write_text(json.dumps(
            {"filename": wav.stem, "status": "success", "transcription": text, "segments": seg_out},
            ensure_ascii=False))
        print(f"[{system}] {i}/{len(todo)} {wav.stem}: {len(text.split())} words", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
