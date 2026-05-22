"""
Objective evaluation: measure Real Time Factor for each experiment.

Real Time Factor (RTF) is the ratio between how long the model took to
generate audio and how long that audio actually plays for. An RTF of 1.0
means the model runs in real time. Anything below 1.0 is faster than
real time, anything above 1.0 means generation is slower than playback.

To get a number we can defend in the report we do not just measure once.
For every prompt we:

  - Run a couple of warm up syntheses that do not count, so the ONNX
    runtime has cached its kernels and the first slow run does not skew
    the timing.
  - Run the actual measurement several times and record each timing.
  - Report the mean and standard deviation across those runs.

The script can evaluate one model on its own or two models side by side,
which is exactly what we want for comparing expA and expB.

Example usage:
    python scripts/evaluate.py \\
        --models checkpoints/amy_medium_base.onnx \\
        --configs checkpoints/en_US-amy-medium.onnx.json \\
        --labels base \\
        --prompts scripts/test_prompts.txt \\
        --runs 5 --warmups 2

    python scripts/evaluate.py \\
        --models checkpoints/expA/model.onnx checkpoints/expB/model.onnx \\
        --configs checkpoints/expA/config.json checkpoints/expB/config.json \\
        --labels expA expB \\
        --prompts scripts/test_prompts.txt \\
        --runs 5 --warmups 2 \\
        --out results/rtf_results.json

Author: Group 41 (Rohit Ritesh Maini a1946109, Nikhil Nakade, Sahaj Pal Singh Mahla)
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Allow importing common.py and synth.py from scripts/.
THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))

from common import save_json
from synth import PiperSynthesiser, load_prompts


class RTFEvaluator:
    """
    Measures the Real Time Factor of a single synthesiser across a set
    of prompts, with multiple runs per prompt so we can report a mean
    and a spread.
    """

    def __init__(self, synth: PiperSynthesiser, label: str,
                 runs: int = 5, warmups: int = 2):
        self.synth = synth
        self.label = label
        self.runs = runs
        self.warmups = warmups

    def evaluate(self, prompts: List[str]) -> Dict:
        """
        Run the timing protocol across every prompt and return a
        dictionary of per prompt results plus an overall summary.
        """
        print(f"\n[eval] === Measuring RTF for {self.label} ===")
        print(f"[eval] {len(prompts)} prompts, {self.runs} timed runs each, "
              f"{self.warmups} warm up run(s) discarded")

        # Warm up first. We use the first prompt for warm up because it
        # exercises every part of the pipeline that the timed runs will.
        if prompts:
            print(f"[eval] Warming up...")
            for w in range(self.warmups):
                self.synth.synthesise(prompts[0])

        per_prompt_results = []

        for prompt_index, prompt in enumerate(prompts, start=1):
            synth_times = []
            duration_seconds = 0.0  # All runs of the same prompt should match.

            for run_index in range(self.runs):
                audio, synth_seconds = self.synth.synthesise(prompt)
                if run_index == 0:
                    duration_seconds = len(audio) / self.synth.sample_rate
                synth_times.append(synth_seconds)

            rtfs = [t / duration_seconds for t in synth_times]
            mean_rtf = statistics.mean(rtfs)
            stdev_rtf = statistics.stdev(rtfs) if len(rtfs) > 1 else 0.0

            entry = {
                "index": prompt_index,
                "prompt": prompt,
                "audio_duration_seconds": round(duration_seconds, 4),
                "synthesis_times_seconds": [round(t, 4) for t in synth_times],
                "rtf_values": [round(r, 4) for r in rtfs],
                "rtf_mean": round(mean_rtf, 4),
                "rtf_stdev": round(stdev_rtf, 4),
            }
            per_prompt_results.append(entry)

            print(f"  [{prompt_index:02d}/{len(prompts):02d}] "
                  f"dur {duration_seconds:5.2f} s | "
                  f"RTF {mean_rtf:.4f} +/- {stdev_rtf:.4f} | "
                  f"{prompt[:55]!r}")

        # Overall summary across every prompt and every run.
        all_rtfs = [r for entry in per_prompt_results for r in entry["rtf_values"]]
        all_durations = [e["audio_duration_seconds"] for e in per_prompt_results]
        overall = {
            "label": self.label,
            "num_prompts": len(prompts),
            "runs_per_prompt": self.runs,
            "warmups_per_prompt": self.warmups,
            "rtf_overall_mean": round(statistics.mean(all_rtfs), 4) if all_rtfs else 0.0,
            "rtf_overall_stdev": round(statistics.stdev(all_rtfs), 4) if len(all_rtfs) > 1 else 0.0,
            "rtf_overall_min": round(min(all_rtfs), 4) if all_rtfs else 0.0,
            "rtf_overall_max": round(max(all_rtfs), 4) if all_rtfs else 0.0,
            "audio_total_seconds": round(sum(all_durations), 4),
            "per_prompt": per_prompt_results,
        }

        print(f"[eval] {self.label} overall: "
              f"RTF mean {overall['rtf_overall_mean']:.4f}, "
              f"stdev {overall['rtf_overall_stdev']:.4f}, "
              f"range [{overall['rtf_overall_min']:.4f}, "
              f"{overall['rtf_overall_max']:.4f}]")

        return overall


def save_csv_summary(results: List[Dict], csv_path) -> None:
    """
    Flatten the results into a CSV that can be opened in any spreadsheet
    tool. One row per (model, prompt) combination.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_result in results:
        label = model_result["label"]
        for entry in model_result["per_prompt"]:
            rows.append({
                "model": label,
                "prompt_index": entry["index"],
                "prompt": entry["prompt"],
                "audio_duration_seconds": entry["audio_duration_seconds"],
                "rtf_mean": entry["rtf_mean"],
                "rtf_stdev": entry["rtf_stdev"],
            })

    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] Wrote CSV summary to {csv_path}")


def print_comparison_table(results: List[Dict]) -> None:
    """
    Prints a nicely aligned comparison of the headline RTF for each
    model. Useful when running with multiple models.
    """
    print()
    print("=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    header = f"{'Model':<15} {'Mean RTF':>10} {'Stdev':>10} {'Min':>10} {'Max':>10}"
    print(header)
    print("-" * 70)
    for r in results:
        line = (f"{r['label']:<15} "
                f"{r['rtf_overall_mean']:>10.4f} "
                f"{r['rtf_overall_stdev']:>10.4f} "
                f"{r['rtf_overall_min']:>10.4f} "
                f"{r['rtf_overall_max']:>10.4f}")
        print(line)
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Real Time Factor for one or more TTS models.")
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more ONNX model paths.")
    p.add_argument("--configs", nargs="+", required=True,
                   help="Matching ONNX config JSON paths in the same order.")
    p.add_argument("--labels", nargs="+", required=True,
                   help="Short labels for each model, in the same order.")
    p.add_argument("--prompts", required=True,
                   help="Text file with one sentence per line.")
    p.add_argument("--runs", type=int, default=5,
                   help="How many timed runs per prompt (default: 5).")
    p.add_argument("--warmups", type=int, default=2,
                   help="How many discarded warm up runs per model (default: 2).")
    p.add_argument("--out", default="results/rtf_results.json",
                   help="Where to save the full JSON results.")
    p.add_argument("--csv", default="results/rtf_results.csv",
                   help="Where to save the flat CSV summary.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Quick sanity check that the three lists line up.
    n = len(args.models)
    if not (len(args.configs) == n and len(args.labels) == n):
        print("Error: --models, --configs, and --labels must have the same number of entries.")
        return 1

    prompts = load_prompts(args.prompts)
    if not prompts:
        print(f"Error: no prompts found in {args.prompts}")
        return 1

    all_results = []
    for model_path, config_path, label in zip(args.models, args.configs, args.labels):
        synth = PiperSynthesiser(model_path, config_path)
        evaluator = RTFEvaluator(synth, label, runs=args.runs, warmups=args.warmups)
        result = evaluator.evaluate(prompts)
        all_results.append(result)

    save_json(args.out, all_results)
    save_csv_summary(all_results, args.csv)
    print_comparison_table(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())