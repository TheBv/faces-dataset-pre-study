"""Shared Silero-VAD segmentation for the models that only take <=30 s input.

Run in the `transcription-whisperx` env (faster-whisper ships Silero as ONNX):

    ~/miniforge3/envs/transcription-whisperx/bin/python vad.py

Writes data/vad/<doc>.json = [[start_s, end_s], ...]. Every segment-level
model sees exactly these windows, so their differences are the model's, not
their chunking's. Adjacent speech regions are packed WhisperX-style into
windows of at most MAX_S, but only across pauses of at most MAX_GAP_S, so a
window never carries a long stretch of silence (the usual trigger for
hallucinated text).
"""
import json
from pathlib import Path

from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

SR = 16000
MAX_S = 30.0
MAX_GAP_S = 2.0
DATA = Path(__file__).resolve().parent / "data"


def pack(regions, max_s=MAX_S, max_gap=MAX_GAP_S):
    out = []
    for s, e in regions:
        if out and e - out[-1][0] <= max_s and s - out[-1][1] <= max_gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def main():
    (DATA / "vad").mkdir(exist_ok=True)
    opts = VadOptions(max_speech_duration_s=MAX_S, min_silence_duration_ms=500, speech_pad_ms=300)
    for wav in sorted((DATA / "audio").glob("*.wav")):
        out = DATA / "vad" / f"{wav.stem}.json"
        if out.exists():
            continue
        ts = get_speech_timestamps(decode_audio(str(wav), sampling_rate=SR), opts)
        segs = pack([(t["start"] / SR, t["end"] / SR) for t in ts])
        out.write_text(json.dumps([[round(s, 3), round(e, 3)] for s, e in segs]))
        print(wav.stem, len(segs), "segments", round(sum(e - s for s, e in segs) / 60, 1), "min speech")


if __name__ == "__main__":
    assert pack([(0, 5), (6, 10), (20, 25), (25.5, 40), (41, 55)]) == [[0, 10], [20, 40], [41, 55]]
    main()
