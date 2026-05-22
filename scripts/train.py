"""
Training wrapper for a single experiment.

This is the script the notebook calls (and that you can also run from
the terminal). All it really does is take one of our experiment YAML
files, set the random seeds, work out whether to resume from a previous
checkpoint, and then launch Piper's actual trainer with the right
arguments.

While the trainer is running, we keep an eye on its output. Any line
that looks like it contains a loss value gets parsed and saved into our
own CSV. That way we end up with two records of training:

  1. Piper's TensorBoard logs (the official source of truth).
  2. Our CSV of step level losses (easy to plot, easy to inspect).

Usage:
    python scripts/train.py --config configs/expA.yaml
    python scripts/train.py --config configs/expB.yaml --no-resume

The --no-resume flag forces a fresh start even if a previous checkpoint
exists, which is occasionally useful for debugging.

Author: Group 41 (Rohit Ritesh Maini a1946109, Nikhil Nakade, Sahaj Pal Singh Mahla)
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make sure we can import common.py from the scripts folder.
THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))

from common import (
    ArtifactPaths,
    ExperimentConfig,
    MetricsRecorder,
    PROJECT_ROOT,
    save_json,
    set_all_seeds,
    training_timer,
)


# This pattern catches lines that look like "loss=3.4521" or "loss: 3.4521".
# Piper's trainer prints lots of these. We pull out the numeric value so we
# can store it.
LOSS_LINE_PATTERN = re.compile(
    r"(?P<name>train_loss|val_loss|loss_gen|loss_disc|loss_kl|loss_mel|loss)"
    r"[\s=:]+(?P<value>-?\d+\.\d+(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)

STEP_PATTERN = re.compile(r"step[\s=:]+(?P<step>\d+)", re.IGNORECASE)
EPOCH_PATTERN = re.compile(r"epoch[\s=:]+(?P<epoch>\d+)", re.IGNORECASE)


class TrainingRun:
    """
    Bundles together everything we need to launch and monitor one
    fine-tuning run. The notebook (or the user from the command line)
    builds one of these and calls .run() on it.
    """

    def __init__(self, cfg: ExperimentConfig, allow_resume: bool = True):
        self.cfg = cfg
        self.allow_resume = allow_resume
        self.paths = ArtifactPaths()
        self.paths.ensure_all_exist()

        # Make sure the experiment specific folders exist too.
        self.exp_ckpt_dir = Path(cfg.output_dir)
        self.exp_log_dir = Path(cfg.log_dir)
        self.exp_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.exp_log_dir.mkdir(parents=True, exist_ok=True)

        self.recorder = MetricsRecorder(cfg.metrics_csv)
        self._was_interrupted = False

    def find_resume_checkpoint(self) -> Optional[Path]:
        """
        Look inside the experiment checkpoint folder for any previous
        checkpoint. If we find one, we will resume from it. If not, we
        start fresh from the Amy base checkpoint.
        """
        if not self.allow_resume:
            return None
        candidates = sorted(self.exp_ckpt_dir.glob("*.ckpt"))
        if not candidates:
            return None
        # Pick the most recently modified checkpoint.
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest

    def _infer_starting_epoch(self, resume_path: Optional[Path]) -> int:
        """
        Look inside the checkpoint we will resume from and read its
        epoch counter. Returns 0 if anything goes wrong, which is the
        safe default for a from scratch run.
        """
        import torch as _torch
        ckpt_path = resume_path if resume_path is not None else Path(self.cfg.base_checkpoint)
        try:
            data = _torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            epoch = int(data.get("epoch", 0))
            print(f"[resume] Detected starting epoch in checkpoint: {epoch}")
            return epoch
        except Exception as exc:
            print(f"[resume] Could not read epoch from checkpoint: {exc}. Assuming 0.")
            return 0

    def build_command(self, resume_path: Optional[Path]) -> list[str]:
        """
        Build the exact command line we will hand to Piper's trainer.
        Returns a list of strings suitable for subprocess.Popen.
        """
        # Work out what max_epochs to ask Lightning for. When Lightning
        # resumes from a checkpoint it preserves the epoch counter, so
        # if the Amy base was trained for 6679 epochs we need to ask for
        # 6679 + our budget. When we resume from our own previous run we
        # do the same relative addition.
        starting_epoch = self._infer_starting_epoch(resume_path)
        target_max_epochs = starting_epoch + self.cfg.max_epochs

        cmd = [
            sys.executable,
            "-m", "piper_train",
            "--dataset-dir", self.cfg.processed_data_dir,
            "--accelerator", "gpu",
            "--devices", "1",
            "--batch-size", str(self.cfg.batch_size),
            "--validation-split", "0.0",
            "--num-test-examples", "0",
            "--quality", self.cfg.quality,
            "--max_epochs", str(target_max_epochs),
            "--checkpoint-epochs", str(self.cfg.checkpoint_every_n_epochs),
            "--precision", str(self.cfg.precision),
            "--default_root_dir", self.cfg.log_dir,
        ]
        if resume_path is not None:
            cmd += ["--resume_from_checkpoint", str(resume_path)]
        else:
            # First time training, start from the Amy base checkpoint.
            cmd += ["--resume_from_checkpoint", self.cfg.base_checkpoint]
        return cmd

    def parse_line(self, line: str, current_epoch: int) -> Optional[dict]:
        """
        Try to extract step number, epoch number, and loss values from a
        single line of training output. Returns a dict of the metrics
        found, or None if the line did not look like a metric line.
        """
        # Find a step number if present in the same line.
        step_match = STEP_PATTERN.search(line)
        epoch_match = EPOCH_PATTERN.search(line)

        metrics_found = {}
        for m in LOSS_LINE_PATTERN.finditer(line):
            name = m.group("name").lower()
            value = float(m.group("value"))
            metrics_found[name] = value

        if not metrics_found:
            return None

        result = {"metrics": metrics_found}
        if step_match:
            result["step"] = int(step_match.group("step"))
        if epoch_match:
            result["epoch"] = int(epoch_match.group("epoch"))
        else:
            result["epoch"] = current_epoch
        return result

    def run(self) -> dict:
        """
        Actually run the training. Returns a small dictionary summarising
        what happened (final losses, elapsed time, whether we were
        interrupted, and so on).
        """
        # First, lock in the seeds. Reproducibility is the priority for
        # this assignment so we make every random source deterministic.
        set_all_seeds(self.cfg.seed)

        # Pretty print the config so the run is self documenting.
        print("=" * 70)
        print(self.cfg.describe())
        print("=" * 70)

        resume_from = self.find_resume_checkpoint()
        if resume_from is not None:
            print(f"[resume] Found previous checkpoint, will resume from: {resume_from}")
        else:
            print(f"[fresh ] No previous checkpoint, starting from base: {self.cfg.base_checkpoint}")

        cmd = self.build_command(resume_from)
        print("\n[command] Running:")
        print("  " + " ".join(cmd))
        print()

        # Open a log file so the full raw output is saved on disk too.
        # The CSV gets the parsed numbers, this file gets the raw text.
        raw_log_path = Path(self.cfg.log_dir) / "training_stdout.log"
        raw_log_path.parent.mkdir(parents=True, exist_ok=True)

        last_metrics: dict = {}
        line_count = 0
        current_epoch = 0

        # Catch Ctrl+C cleanly. If the user interrupts, we mark the run
        # as interrupted and still print a final summary.
        def handle_sigint(signum, frame):
            print("\n[interrupt] Caught Ctrl+C, finishing up...")
            self._was_interrupted = True
        signal.signal(signal.SIGINT, handle_sigint)

        with training_timer(self.cfg.experiment_name) as timer, \
             open(raw_log_path, "w") as raw_log:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                bufsize=1,
                text=True,
            )

            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n")
                    line_count += 1
                    print(line)
                    raw_log.write(raw_line)
                    raw_log.flush()

                    # Watch for epoch changes so we can tag metrics correctly.
                    em = EPOCH_PATTERN.search(line)
                    if em:
                        current_epoch = int(em.group("epoch"))

                    parsed = self.parse_line(line, current_epoch)
                    if parsed and parsed["metrics"]:
                        step = parsed.get("step", line_count)
                        epoch = parsed.get("epoch", current_epoch)
                        self.recorder.record(step=step, epoch=epoch, **parsed["metrics"])
                        last_metrics = parsed["metrics"]

                    if self._was_interrupted:
                        proc.terminate()
                        break

            finally:
                proc.wait()

        # Pull together a nice summary that we can print and save.
        summary = {
            "experiment_name": self.cfg.experiment_name,
            "description": self.cfg.description,
            "max_epochs": self.cfg.max_epochs,
            "batch_size": self.cfg.batch_size,
            "seed": self.cfg.seed,
            "elapsed_seconds": round(timer["elapsed_seconds"], 2),
            "elapsed_minutes": round(timer["elapsed_seconds"] / 60.0, 2),
            "was_interrupted": self._was_interrupted,
            "exit_code": proc.returncode,
            "final_metrics": {k: round(v, 4) for k, v in last_metrics.items()},
            "metrics_csv": self.cfg.metrics_csv,
            "raw_log": str(raw_log_path),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }

        summary_path = Path("results") / f"{self.cfg.experiment_name}_summary.json"
        save_json(summary_path, summary)

        print()
        print("=" * 70)
        print(f"Training summary for {self.cfg.experiment_name}")
        print("=" * 70)
        for key, value in summary.items():
            print(f"  {key:18s}: {value}")
        print("=" * 70)

        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one TTS fine-tuning experiment.")
    parser.add_argument(
        "--config", required=True,
        help="Path to an experiment YAML, for example configs/expA.yaml",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any previous checkpoint and start fresh from the Amy base.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    runner = TrainingRun(cfg, allow_resume=not args.no_resume)
    summary = runner.run()
    return 0 if summary.get("exit_code", 0) == 0 and not summary.get("was_interrupted") else 1


if __name__ == "__main__":
    sys.exit(main())