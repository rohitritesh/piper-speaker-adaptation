"""
Shared helpers for the TTS assignment.

Everything in here is stuff that more than one script needs. If we put
these classes and functions in one place, the rest of the code stays
short and we don't end up with copy-pasted logic floating around.

Author: Group 41 (Rohit Ritesh Maini a1946109, Nikhil Nakade, Sahaj Pal Singh Mahla)
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import yaml


def get_project_root() -> Path:
    """
    Walk up the folder tree until we find the project root. We know we
    are at the root when we can see both a "data" folder and a
    "checkpoints" folder sitting next to each other. This way the
    scripts work no matter which folder we run them from.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        looks_right = (parent / "data").is_dir() and (parent / "checkpoints").is_dir()
        if looks_right:
            return parent
    raise RuntimeError("Could not locate project root from " + str(here))


PROJECT_ROOT = get_project_root()


def set_all_seeds(seed: int) -> None:
    """
    Make every random number generator we touch behave the same way every
    time. If we re-run the project tomorrow with the same seed, we get
    the same splits, the same shuffles, the same initial weights, and so
    on. Without this, results would wobble slightly between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"[seed] All random sources seeded with {seed}")


@dataclass
class ExperimentConfig:
    """
    A typed wrapper around an experiment YAML file. Instead of passing
    dictionaries around the codebase, we load the YAML once into one of
    these objects and then access fields with dot notation, like
    cfg.batch_size or cfg.max_epochs. If a field is missing from the
    YAML we find out immediately when loading, not halfway through
    training.
    """
    experiment_name: str
    description: str
    seed: int
    batch_size: int
    sample_rate: int
    quality: str
    max_epochs: int
    processed_data_dir: str
    base_checkpoint: str
    output_dir: str
    log_dir: str
    metrics_csv: str
    checkpoint_every_n_epochs: int
    log_every_n_steps: int
    precision: int

    @classmethod
    def from_yaml(cls, yaml_path):
        """
        Read a YAML file and build an ExperimentConfig out of it. If the
        file is missing or has extra unknown fields, this raises a
        helpful error instead of failing somewhere deep in the trainer.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        try:
            return cls(**data)
        except TypeError as e:
            raise ValueError(
                f"Config file {yaml_path} is missing or has extra fields: {e}"
            )

    def describe(self) -> str:
        """
        Returns a tidy multi line summary of the configuration. We print
        this at the start of every training run, so anyone scrolling
        through the logs can see exactly what settings were used.
        """
        lines = [
            f"Experiment Configuration: {self.experiment_name}",
            f"  Description       : {self.description}",
            f"  Seed              : {self.seed}",
            f"  Batch size        : {self.batch_size}",
            f"  Sample rate       : {self.sample_rate} Hz",
            f"  Quality           : {self.quality}",
            f"  Max epochs        : {self.max_epochs}",
            f"  Precision         : {self.precision}",
            f"  Base checkpoint   : {self.base_checkpoint}",
            f"  Processed data    : {self.processed_data_dir}",
            f"  Output dir        : {self.output_dir}",
            f"  Log dir           : {self.log_dir}",
            f"  Metrics CSV       : {self.metrics_csv}",
        ]
        return "\n".join(lines)


@dataclass
class ArtifactPaths:
    """
    A single place that knows where every file in the project lives.
    Instead of writing "results/figures/whatever.png" by hand in twenty
    different files, we ask this class. If we ever rename a folder we
    only have to change it here.
    """
    project_root: Path = PROJECT_ROOT
    data_root: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    lj_subset_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "lj_subset")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "lj_subset_processed")

    checkpoints_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints")
    base_ckpt: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints" / "amy_medium_base.ckpt")
    base_onnx: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints" / "amy_medium_base.onnx")
    base_config_json: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints" / "en_US-amy-medium.onnx.json")

    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    figures_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results" / "figures")
    spectrograms_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results" / "spectrograms")
    wav_examples_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "wav_examples")

    test_prompts_file: Path = field(default_factory=lambda: PROJECT_ROOT / "scripts" / "test_prompts.txt")

    def ensure_all_exist(self) -> None:
        """
        Make sure every output folder exists on disk. Call this once at
        the start of any script that will write things, so later code
        does not blow up because a folder is missing.
        """
        for path in [
            self.checkpoints_dir,
            self.logs_dir,
            self.results_dir,
            self.figures_dir,
            self.spectrograms_dir,
            self.wav_examples_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@contextmanager
def training_timer(label: str):
    """
    A little timer you wrap around a block of code to find out how long
    it took. Use it like this:

        with training_timer("expA") as t:
            do_some_work()
        elapsed = t["elapsed_seconds"]

    It also prints "Starting X" and "Finished X in Y seconds" so the
    timing shows up in the logs without any extra work.
    """
    bucket = {"elapsed_seconds": 0.0}
    start = time.perf_counter()
    print(f"[timer] Starting {label}")
    try:
        yield bucket
    finally:
        elapsed = time.perf_counter() - start
        bucket["elapsed_seconds"] = elapsed
        mins = elapsed / 60.0
        print(f"[timer] Finished {label} in {elapsed:.2f} s ({mins:.2f} min)")


class MetricsRecorder:
    """
    Logs training metrics (loss values, learning rate, and so on) to a
    plain CSV file. CSV was chosen because:

      - It is easy for anyone to open in Excel or pandas to inspect.
      - We want real numeric loss values that we can quote in the report.
      - It plays nicely with crashes. Each row is flushed immediately,
        so if training dies we still have everything up to that point.

    Each call to record() appends one row. The first call also writes
    the column headers.
    """

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = self.csv_path.exists() and self.csv_path.stat().st_size > 0

    def record(self, step, epoch, **metrics):
        """
        Append one row with the given step number, epoch number, and any
        metric values passed as keyword arguments. Examples:

            recorder.record(step=42, epoch=2, train_loss=3.21)
        """
        row = {"step": step, "epoch": epoch}
        row.update({k: float(v) for k, v in metrics.items()})
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)

    def read_all(self):
        """
        Reads the whole CSV back as a list of dicts. Useful when we want
        to plot the loss curves afterwards.
        """
        if not self.csv_path.exists():
            return []
        with open(self.csv_path) as f:
            return list(csv.DictReader(f))


def save_json(path, data):
    """
    Save any Python object as nicely formatted JSON. Used for storing
    final metrics, summary tables, and so on.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[save] Wrote {path}")


def load_json(path):
    """
    The other half of save_json. Reads a JSON file back into a Python
    object.
    """
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # Running this file directly does a small self test. Handy if we
    # ever want to confirm the helpers and folder structure are all
    # in order without launching a full experiment.
    print("Sanity checking common.py...")
    print(f"Project root: {PROJECT_ROOT}")
    paths = ArtifactPaths()
    paths.ensure_all_exist()
    print("All required directories present.")
    cfg = ExperimentConfig.from_yaml(PROJECT_ROOT / "configs" / "expA.yaml")
    print()
    print(cfg.describe())
    print()
    print("common.py looks healthy.")