#!/usr/bin/env bash
# Run every ASR system on all 54 recordings (sequential; one 16 GB GPU).
# Prereqs: python3 prepare.py; transcription-whisperx env: python vad.py
# A system whose output dir already holds all 54 files is skipped.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
ENVS=~/miniforge3/envs
REPO=/mnt/d/repos/transcription
N=$(ls data/audio/*.wav | wc -l)

done_() { [ "$(ls "data/hyp/$1" 2>/dev/null | wc -l)" -ge "$N" ]; }

# Models limited to <=30 s input: shared VAD windows (run_seg.py).
seg() { done_ "$2" || "$ENVS/$1/bin/python" run_seg.py "$2"; }
seg transcription-canary  parakeet-tdt-0.6b-v3
seg transcription-canary  canary-1b-v2
seg vllm                  voxtral-mini-3b
seg transcription-granite granite-speech-4.1-2b
seg transcription-phi4    phi-4-multimodal
seg transcription-qwen3-asr qwen3-asr-1.7b
seg transcription-whisper-crisper2 crisperwhisper2-large-verbatim-vad
seg transcription-whisper-crisper2 crisperwhisper2-large-intended-vad

# Long-form models: the transcription repo's runners on the full recordings,
# each with its own long-form strategy.
repo() { # env system module args...
  local env=$1 sys=$2 mod=$3; shift 3
  done_ "$sys" || (cd "$REPO" && "$ENVS/$env/bin/python" -m "$mod" \
    --input "$HERE/data/audio" --output "$HERE/data/hyp/$sys" "$@")
}
repo transcription-whisper  whisper-large-v3        src.models.whisper.main  --model large --lang de
repo transcription-whisper  whisper-large-v3-turbo  src.models.whisper.main  --model turbo --lang de
repo transcription-whisper  whisper-large-v3-turbo-nocond  src.models.whisper.main  --model turbo --lang de --no_condition_on_previous_text
repo transcription-whisperx whisperx-large-v3       src.models.whisper_x.main --model large-v3 --language de
repo transcription-whisper-crisper2 crisperwhisper2-large-verbatim src.models.whisper_crisper2.main \
  --model-id large --language de --mode verbatim
