"""
Smoke test: synthesise one sentence using the Amy base ONNX model.
This proves the entire pipeline (text to phonemes to mel to waveform) works.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
from piper_phonemize import phonemize_espeak, phoneme_ids_espeak

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = PROJECT_ROOT / "checkpoints" / "amy_medium_base.onnx"
CONFIG_PATH = PROJECT_ROOT / "checkpoints" / "en_US-amy-medium.onnx.json"
OUT_PATH = PROJECT_ROOT / "wav_examples" / "smoke_test_amy_base.wav"

SENTENCE = "Hello, this is a test of the Piper text to speech system."

print("Loading config...")
with open(CONFIG_PATH) as f:
    config = json.load(f)

sample_rate = config["audio"]["sample_rate"]
voice = config["espeak"]["voice"]
inference_cfg = config["inference"]
print(f"  sample_rate = {sample_rate} Hz, voice = {voice}")

print("Loading ONNX model...")
session = onnxruntime.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)
print(f"  ONNX session ready. Inputs: {[i.name for i in session.get_inputs()]}")

print(f"\nSynthesising: {SENTENCE!r}")

print("Step 1: text to phonemes via eSpeak NG")
phoneme_lists = phonemize_espeak(SENTENCE, voice)
print(f"  got {sum(len(p) for p in phoneme_lists)} phonemes across {len(phoneme_lists)} sentence(s)")

print("Step 2: phonemes to integer IDs")
# Flatten the per-sentence phoneme lists into one sequence
all_phonemes = []
for sent in phoneme_lists:
    all_phonemes.extend(sent)
phoneme_ids = phoneme_ids_espeak(all_phonemes)
print(f"  got {len(phoneme_ids)} phoneme IDs")

print("Step 3: ONNX inference (phoneme IDs to waveform)")
text_input = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
text_lengths = np.array([text_input.shape[1]], dtype=np.int64)
scales = np.array([
    inference_cfg["noise_scale"],
    inference_cfg["length_scale"],
    inference_cfg["noise_w"],
], dtype=np.float32)

audio = session.run(
    None,
    {"input": text_input, "input_lengths": text_lengths, "scales": scales},
)[0].squeeze()

print(f"  got audio of shape {audio.shape}, dtype {audio.dtype}")
print(f"  audio duration = {len(audio) / sample_rate:.2f} seconds")

print(f"\nStep 4: saving to {OUT_PATH}")
OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
sf.write(str(OUT_PATH), audio, sample_rate)

size_kb = OUT_PATH.stat().st_size / 1024
print(f"\nDone. Saved {size_kb:.1f} KB.")
print(f"Find the WAV at: {OUT_PATH}")
