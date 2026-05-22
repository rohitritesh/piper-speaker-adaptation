"""
Prepare a reproducible 1500-utterance subset of LJSpeech with a fixed
90/5/5 train/val/test split. Output: a Piper-format dataset directory
at data/lj_subset/ ready for piper_train.preprocess.

Usage:  python scripts/prepare_subset.py
"""
import csv
import random
import shutil
from pathlib import Path

# ---- Reproducibility ----
SEED = 42
SUBSET_SIZE = 1500
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.90, 0.05, 0.05

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LJ_ROOT = PROJECT_ROOT / "data" / "LJSpeech-1.1"
OUT_ROOT = PROJECT_ROOT / "data" / "lj_subset"
OUT_WAVS = OUT_ROOT / "wav"

# ---- Sanity checks ----
assert LJ_ROOT.exists(), f"LJSpeech root not found: {LJ_ROOT}"
metadata_path = LJ_ROOT / "metadata.csv"
wavs_path = LJ_ROOT / "wavs"
assert metadata_path.exists()
assert wavs_path.exists()

# ---- Load metadata ----
# Format: ID|raw_text|normalized_text  (pipe-separated, no header)
rows = []
with open(metadata_path, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
    for r in reader:
        if len(r) >= 3:
            rows.append((r[0], r[2]))  # use normalized text

print(f"Loaded {len(rows)} metadata rows.")

# ---- Reproducible sampling ----
rng = random.Random(SEED)
rng.shuffle(rows)
subset = rows[:SUBSET_SIZE]
print(f"Selected {len(subset)} utterances with seed={SEED}.")

# ---- Reproducible split ----
n = len(subset)
n_train = int(n * TRAIN_FRAC)
n_val = int(n * VAL_FRAC)
# remainder goes to test to keep totals exact
n_test = n - n_train - n_val

train = subset[:n_train]
val = subset[n_train:n_train + n_val]
test = subset[n_train + n_val:]
print(f"Split sizes: train={len(train)} val={len(val)} test={len(test)}")

# ---- Write Piper-format dataset ----
# Piper preprocessor expects: a single metadata.csv with format `audio_id|text`
# and a 'wav/' folder of corresponding audio files.
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_WAVS.mkdir(exist_ok=True)

print("Copying wav files...")
all_items = train + val + test
for i, (audio_id, _) in enumerate(all_items):
    src = wavs_path / f"{audio_id}.wav"
    dst = OUT_WAVS / f"{audio_id}.wav"
    if not dst.exists():
        shutil.copy2(src, dst)
    if (i + 1) % 200 == 0:
        print(f"  copied {i+1}/{len(all_items)}")

# Piper expects a single metadata.csv for the full training set;
# we'll keep our val/test as separate CSVs alongside so we can use them
# explicitly for evaluation later (Piper doesn't require this format,
# we use them ourselves).
def write_csv(path, items):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for aid, txt in items:
            w.writerow([aid, txt])

write_csv(OUT_ROOT / "metadata.csv", train + val)   # piper trains on train+val combined; we'll use val for monitoring
write_csv(OUT_ROOT / "train_list.csv", train)
write_csv(OUT_ROOT / "val_list.csv", val)
write_csv(OUT_ROOT / "test_list.csv", test)

print(f"\nDone. Output at: {OUT_ROOT}")
print(f"  metadata.csv:    {n_train + n_val} rows (train+val for Piper)")
print(f"  train_list.csv:  {n_train} rows")
print(f"  val_list.csv:    {n_val} rows")
print(f"  test_list.csv:   {n_test} rows")
print(f"  wav/:            {n_train + n_val + n_test} files")
