"""
Join reviews with metadata
Joins on parent_asin to add product category to each review.
Output: Musical_Instruments_joined.parquet
"""

import pandas as pd
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NLP_DIR     = os.path.dirname(SCRIPT_DIR)

REVIEWS     = os.path.join(NLP_DIR, 'Musical_Instruments.parquet')
META_A      = os.path.join(NLP_DIR, 'full-00000-of-00002.parquet')
META_B      = os.path.join(NLP_DIR, 'full-00001-of-00002.parquet')
OUTPUT      = os.path.join(NLP_DIR, 'Musical_Instruments_joined.parquet')

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading reviews...")
reviews = pd.read_parquet(REVIEWS)
print(f"  {len(reviews):,} rows")

print("Loading metadata...")
meta = pd.concat([
    pd.read_parquet(META_A, columns=['parent_asin', 'categories', 'store', 'average_rating', 'price']),
    pd.read_parquet(META_B, columns=['parent_asin', 'categories', 'store', 'average_rating', 'price'])
], ignore_index=True).drop_duplicates(subset='parent_asin')
print(f"  {len(meta):,} unique products")

# ── Extract subcategory ───────────────────────────────────────────────────────
# categories is a list e.g. ['Musical Instruments', 'Drums & Percussion', ...]
# index 1 is the subcategory (guitar, drums, etc.)
meta['subcategory'] = meta['categories'].apply(
    lambda x: x[1] if x is not None and len(x) > 1 else None
)

print("\nSubcategory distribution:")
print(meta['subcategory'].value_counts().head(20).to_string())

# ── Join ─────────────────────────────────────────────────────────────────────
print("\nJoining...")
df = reviews.merge(
    meta[['parent_asin', 'subcategory', 'store', 'average_rating', 'price']],
    on='parent_asin',
    how='left'
)

print(f"Joined shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Nulls in subcategory:    {df['subcategory'].isna().sum():,}")
print(f"Nulls in store:          {df['store'].isna().sum():,}")
print(f"Nulls in average_rating: {df['average_rating'].isna().sum():,}")
print(f"Nulls in price:          {df['price'].isna().sum():,}")

print("\nReview count by subcategory:")
print(df['subcategory'].value_counts().head(20).to_string())

# ── Save ─────────────────────────────────────────────────────────────────────
df.to_parquet(OUTPUT, index=False)
print(f"\nSaved → {os.path.basename(OUTPUT)}")
print("Done.")
