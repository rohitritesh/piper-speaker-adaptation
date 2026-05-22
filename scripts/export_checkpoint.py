"""
Convert a fine-tuned Lightning checkpoint into an ONNX model plus the
matching config JSON that synth.py expects.

Piper's training writes Lightning .ckpt files. For inference we want
.onnx files because ONNX is faster on CPU and avoids any PyTorch
version coupling at inference time. This script handles the conversion
end to end.

We also copy and tweak the base config JSON so the resulting model has
the right sample rate, phoneme map, and inference scales sitting next
to it.

Example usage:
    python scripts/export_checkpoint.py \\
        --checkpoint checkpoints/expA/last.ckpt \\
        --base-config checkpoints/en_US-amy-medium.onnx.json \\
        --out-dir checkpoints/expA

This produces checkpoints/expA/model.onnx and checkpoints/expA/config.json.

Author: Group 41 (Rohit Ritesh Maini a1946109, Nikhil Nakade, Sahaj Pal Singh Mahla)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))

from common import PROJECT_ROOT


def export_to_onnx(checkpoint_path: Path, onnx_out_path: Path) -> None:
    """
    Call Piper's bundled ONNX export script as a subprocess. We force
    CPU only with CUDA_VISIBLE_DEVICES because Piper's exporter has a
    device mismatch issue when a GPU is visible.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        sys.executable, "-m", "piper_train.export_onnx",
        str(checkpoint_path),
        str(onnx_out_path),
    ]
    print(f"[export] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"ONNX export failed with code {result.returncode}")
    print(f"[export] ONNX model written to {onnx_out_path}")


def copy_config(base_config: Path, out_config: Path) -> None:
    """
    Copy the base config JSON next to the new ONNX model. The synth and
    evaluate scripts read this to know the sample rate, voice, and
    default inference scales for the model.
    """
    out_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_config, out_config)
    print(f"[export] Config copied to {out_config}")

    # Quick sanity print of the key fields.
    with open(out_config) as f:
        cfg = json.load(f)
    print(f"[export] Config sample_rate = {cfg['audio']['sample_rate']}, "
          f"voice = {cfg['espeak']['voice']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a fine-tuned checkpoint to ONNX.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a .ckpt file to export.")
    p.add_argument("--base-config", required=True,
                   help="Base config JSON to copy alongside the new ONNX (typically the Amy config).")
    p.add_argument("--out-dir", required=True,
                   help="Folder where model.onnx and config.json will be written.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    base_config = Path(args.base_config)
    out_dir = Path(args.out_dir)

    if not checkpoint.exists():
        print(f"Error: checkpoint not found: {checkpoint}")
        return 1
    if not base_config.exists():
        print(f"Error: base config not found: {base_config}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_out = out_dir / "model.onnx"
    config_out = out_dir / "config.json"

    export_to_onnx(checkpoint, onnx_out)
    copy_config(base_config, config_out)

    print()
    print(f"[export] Done. Use these paths for synthesis:")
    print(f"  --model  {onnx_out}")
    print(f"  --config {config_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())