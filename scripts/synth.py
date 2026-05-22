"""
Single entry point for turning text into a playable WAV file.

This script wraps the ONNX inference pipeline so the rest of the project
(and any future user) can synthesise audio with one clean command. It
supports two modes:

  1. Single sentence mode. Pass --text "Some sentence" and it writes one
     WAV to the path given in --out.

  2. Batch mode. Pass --prompts pointing at a text file with one sentence
     per line, plus --out-dir, and it writes one WAV per line.

In both modes the script also writes a small JSON sidecar next to each
WAV recording the synthesis time, audio duration, and resulting real
time factor. The evaluation step reads those JSON files.

Example usage:
    python scripts/synth.py --text "Hello world." \\
        --model checkpoints/amy_medium_base.onnx \\
        --config checkpoints/en_US-amy-medium.onnx.json \\
        --out wav_examples/test_single.wav

    python scripts/synth.py \\
        --prompts scripts/test_prompts.txt \\
        --model checkpoints/expA/model.onnx \\
        --config checkpoints/expA/config.json \\
        --out-dir wav_examples \\
        --prefix expA_

Author: Group 41 (Rohit Ritesh Maini a1946109, Nikhil Nakade, Sahaj Pal Singh Mahla)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime
import soundfile as sf
from piper_phonemize import phonemize_espeak, phoneme_ids_espeak

# Allow importing common.py from scripts/.
THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))

from common import save_json


class PiperSynthesiser:
    """
    Loads an ONNX Piper voice model and turns text into audio. Once
    constructed it can be called many times without paying the model
    loading cost each time.
    """

    def __init__(self, onnx_path, config_path):
        self.onnx_path = Path(onnx_path)
        self.config_path = Path(config_path)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        # Read the model config so we know the sample rate, the espeak
        # voice code, and the default inference scaling parameters.
        with open(self.config_path) as f:
            self.config = json.load(f)

        self.sample_rate = self.config["audio"]["sample_rate"]
        self.voice = self.config["espeak"]["voice"]
        inf = self.config["inference"]
        self.default_scales = np.array(
            [inf["noise_scale"], inf["length_scale"], inf["noise_w"]],
            dtype=np.float32,
        )

        # Create the ONNX session. CPU is fine for inference at this
        # scale and means we do not have to fight with CUDA versions.
        self.session = onnxruntime.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        print(f"[synth] Loaded model {self.onnx_path.name}, "
              f"sample rate {self.sample_rate} Hz, voice {self.voice}")

    def synthesise(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Takes a single sentence of text and returns a tuple of
        (audio waveform as float32 numpy array, synthesis time in
        seconds). The waveform is at self.sample_rate.
        """
        # Step 1: text into phonemes via eSpeak NG.
        phoneme_lists = phonemize_espeak(text, self.voice)
        # Flatten across sentences in case eSpeak splits at punctuation.
        flat_phonemes = []
        for sent in phoneme_lists:
            flat_phonemes.extend(sent)

        # Step 2: phonemes into integer IDs the model can consume.
        phoneme_ids = phoneme_ids_espeak(flat_phonemes)

        # Step 3: ONNX inference. We time only this step because that
        # is the part the user actually waits for.
        text_in = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        lengths = np.array([text_in.shape[1]], dtype=np.int64)

        start = time.perf_counter()
        audio = self.session.run(
            None,
            {
                "input": text_in,
                "input_lengths": lengths,
                "scales": self.default_scales,
            },
        )[0].squeeze()
        elapsed = time.perf_counter() - start

        return audio.astype(np.float32), elapsed

    def synthesise_to_file(self, text: str, out_path) -> dict:
        """
        Run the full pipeline and save the audio plus a JSON sidecar.
        Returns a dict with timing and duration info.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        audio, synth_seconds = self.synthesise(text)
        sf.write(str(out_path), audio, self.sample_rate)

        duration_seconds = len(audio) / self.sample_rate
        rtf = synth_seconds / duration_seconds if duration_seconds > 0 else float("nan")

        sidecar = {
            "text": text,
            "wav_path": str(out_path),
            "sample_rate": self.sample_rate,
            "audio_duration_seconds": round(duration_seconds, 4),
            "synthesis_seconds": round(synth_seconds, 4),
            "real_time_factor": round(rtf, 4),
            "model": str(self.onnx_path),
        }
        sidecar_path = out_path.with_suffix(".json")
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)

        return sidecar


def load_prompts(prompts_path) -> List[str]:
    """
    Read the prompt file. Blank lines and lines starting with '#' are
    ignored, so we can comment things in the file if we ever need to.
    """
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            prompts.append(stripped)
    return prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesise text into WAV files.")
    p.add_argument("--model", required=True, help="Path to an ONNX voice model.")
    p.add_argument("--config", required=True, help="Path to the corresponding ONNX config JSON.")

    # Mutually exclusive input modes.
    p.add_argument("--text", help="A single sentence to synthesise.")
    p.add_argument("--prompts", help="Path to a text file with one sentence per line.")

    p.add_argument("--out", help="Output WAV path (used with --text).")
    p.add_argument("--out-dir", help="Output folder for batch mode (used with --prompts).")
    p.add_argument("--prefix", default="sample_", help="Filename prefix for batch mode.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not (args.text or args.prompts):
        print("Error: pass either --text or --prompts.")
        return 1
    if args.text and not args.out:
        print("Error: --text mode also needs --out.")
        return 1
    if args.prompts and not args.out_dir:
        print("Error: --prompts mode also needs --out-dir.")
        return 1

    synth = PiperSynthesiser(args.model, args.config)

    if args.text:
        # Single sentence mode.
        print(f"[synth] Synthesising one sentence: {args.text!r}")
        info = synth.synthesise_to_file(args.text, args.out)
        print(f"[synth] Saved {info['wav_path']}, "
              f"duration {info['audio_duration_seconds']:.2f} s, "
              f"RTF {info['real_time_factor']:.4f}")
        return 0

    # Batch mode.
    prompts = load_prompts(args.prompts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[synth] Synthesising {len(prompts)} prompt(s) into {out_dir}")

    all_info = []
    for i, prompt in enumerate(prompts, start=1):
        out_name = f"{args.prefix}{i:02d}.wav"
        out_path = out_dir / out_name
        info = synth.synthesise_to_file(prompt, out_path)
        info["index"] = i
        info["prompt"] = prompt
        all_info.append(info)
        print(f"  [{i:02d}/{len(prompts)}] {out_name}  "
              f"dur {info['audio_duration_seconds']:5.2f} s, "
              f"RTF {info['real_time_factor']:.4f}, "
              f"text {prompt[:60]!r}")

    # Save a combined index so the notebook can pick everything up
    # in one read.
    index_path = out_dir / f"{args.prefix}index.json"
    save_json(index_path, all_info)
    print(f"[synth] Wrote index file: {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())