"""
Check which columns to use for joining reviews and metadata.
"""

import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NLP_DIR    = os.path.dirname(SCRIPT_DIR)

# ── Load reviews (just asin columns) ─────────────────────────────────────────
print("Loading reviews...")
reviews = pd.read_parquet(
    os.path.join(NLP_DIR, 'Musical_Instruments.parquet'),
    columns=['asin', 'parent_asin']
)
print(f"Reviews: {len(reviews):,} rows")
print(f"  asin        — unique: {reviews['asin'].nunique():,}  | nulls: {reviews['asin'].isna().sum()}")
print(f"  parent_asin — unique: {reviews['parent_asin'].nunique():,}  | nulls: {reviews['parent_asin'].isna().sum()}")

# ── Load metadata (just asin columns) ────────────────────────────────────────
print("\nLoading metadata...")
meta_a = pd.read_parquet(os.path.join(NLP_DIR, 'full-00000-of-00002.parquet'))
meta_b = pd.read_parquet(os.path.join(NLP_DIR, 'full-00001-of-00002.parquet'))
meta   = pd.concat([meta_a, meta_b], ignore_index=True)
print(f"Metadata: {len(meta):,} rows")
print(f"Metadata columns: {meta.columns.tolist()}")

# Check which asin-like columns exist in metadata
asin_cols = [c for c in meta.columns if 'asin' in c.lower()]
print(f"\nAsin-like columns in metadata: {asin_cols}")
for col in asin_cols:
    print(f"  {col} — unique: {meta[col].nunique():,}  | nulls: {meta[col].isna().sum()}")

# ── Check overlap ─────────────────────────────────────────────────────────────
print("\n── Overlap check ───────────────────────────────────────────────────")
for meta_col in asin_cols:
    meta_ids = set(meta[meta_col].dropna())
    for rev_col in ['asin', 'parent_asin']:
        rev_ids  = set(reviews[rev_col].dropna())
        overlap  = len(rev_ids & meta_ids)
        pct      = overlap / len(rev_ids) * 100
        print(f"  reviews.{rev_col} vs metadata.{meta_col}: {overlap:,} matches ({pct:.1f}% of reviews covered)")

print("\nDone.")
