#!/usr/bin/env bash
#
# setup_piper.sh
#
# This script sets up the Piper TTS training code that this project
# depends on. Because Piper is a third party project and is fairly large,
# we do not ship a copy of it inside this repository. Instead this script
# clones it fresh at the exact commit we developed against, applies our
# two small compatibility patches, compiles the one Cython module Piper
# needs, and installs the package onto the current Python environment.
#
# Run this from the project root, with your virtual environment already
# activated:
#
#     source venv/bin/activate
#     bash setup_piper.sh
#
# The patches themselves are documented in PATCHES.md.

set -e  # Stop immediately if any command fails.

# The exact Piper commit this project was developed and tested against.
# Pinning the commit means the patch always applies cleanly and the
# training behaviour matches what is reported in the notebook.
PIPER_COMMIT="73c04d81d5590ecc46e522de3601ce7fb29fc2be"
PIPER_REPO="https://github.com/rhasspy/piper.git"
PATCH_FILE="piper_patches.patch"

echo "=============================================================="
echo "Piper setup"
echo "=============================================================="

# Make sure we are being run from the project root by checking for the
# patch file and the scripts folder.
if [ ! -f "${PATCH_FILE}" ]; then
    echo "ERROR: ${PATCH_FILE} not found."
    echo "Please run this script from the project root, where ${PATCH_FILE} lives."
    exit 1
fi
if [ ! -d "scripts" ]; then
    echo "ERROR: scripts/ folder not found."
    echo "Please run this script from the project root."
    exit 1
fi

# Step 1: clone Piper if it is not already here.
if [ -d "piper" ]; then
    echo "[1/5] A piper/ folder already exists. Skipping clone."
    echo "      If you want a clean setup, delete the piper/ folder first."
else
    echo "[1/5] Cloning Piper from ${PIPER_REPO} ..."
    git clone "${PIPER_REPO}" piper
fi

# Step 2: check out the exact commit we developed against.
echo "[2/5] Checking out pinned commit ${PIPER_COMMIT} ..."
cd piper
git checkout --quiet "${PIPER_COMMIT}"
cd ..

# Step 3: apply our compatibility patch.
# We first check whether it is already applied so re-running the script
# is safe and does not error out on an already patched tree.
echo "[3/5] Applying compatibility patch ..."
cd piper
if git apply --reverse --check "../${PATCH_FILE}" 2>/dev/null; then
    echo "      Patch already applied, skipping."
else
    git apply "../${PATCH_FILE}"
    echo "      Patch applied successfully."
fi
cd ..

# Step 4: compile the Cython monotonic alignment module.
echo "[4/5] Compiling the Cython monotonic alignment module ..."
cd piper/src/python
bash build_monotonic_align.sh
cd ../../..

# Step 5: install the Piper training package without touching our other
# dependencies. The --no-deps flag is essential because Piper's setup.py
# pins old PyTorch and Lightning versions that would otherwise downgrade
# our modern stack.
echo "[5/5] Installing the piper_train package (pip install -e, no deps) ..."
pip install -e piper/src/python --no-deps

echo "=============================================================="
echo "Piper setup complete."
echo "=============================================================="
echo
echo "Quick verification:"
python -c "from piper_train.vits.monotonic_align import maximum_path; print('  monotonic_align import: OK')"
python -c "from piper_train.vits.lightning import VitsModel; print('  VitsModel import: OK')"
python -c "import pytorch_lightning, torch; print(f'  Lightning {pytorch_lightning.__version__}, Torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
echo
echo "If all three lines above printed OK, Piper is ready for training."
