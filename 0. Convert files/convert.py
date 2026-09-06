"""
Convert jsonl.gz to parquet
Run this once. All future steps load from the parquet file.
"""

import pandas as pd
import gzip
import os

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.dirname(SCRIPT_DIR)
INPUT_FILE  = os.path.join(DATA_DIR, 'Musical_Instruments.jsonl.gz')
OUTPUT_FILE = os.path.join(DATA_DIR, 'Musical_Instruments.parquet')

# ── Convert ──────────────────────────────────────────────────────────────────
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

print("\nLoading jsonl.gz...")
with gzip.open(INPUT_FILE, 'rt') as f:
    df = pd.read_json(f, lines=True)

print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

print("Saving to parquet...")
df.to_parquet(OUTPUT_FILE, index=False)

size_mb = os.path.getsize(OUTPUT_FILE) / (1024 ** 2)
print(f"Done. Parquet file size: {size_mb:.1f} MB")
