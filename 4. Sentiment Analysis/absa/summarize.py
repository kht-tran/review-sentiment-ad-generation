"""
Generate results.txt from guitars_absa.parquet
Applies a confidence threshold: if winning score < THRESHOLD, reclassify as 'not mentioned'.
Includes a comparison table showing the effect of the threshold vs raw (no threshold).
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE  = os.path.join(SCRIPT_DIR, 'guitars_absa.parquet')
RESULTS_OUT = os.path.join(SCRIPT_DIR, 'results.txt')

THRESHOLD = 0.70  # if winning label scores below this, assign 'not mentioned'

TOPICS = {
    't01_customer_service': 'Customer service / returns',
    't04_beginner':         'Beginner learning',
    't05_accessories':      'Accessories',
    't06_electronics':      'Electronics / controls',
    't07_setup':            'Setup / action',
    't08_size':             'Guitar size',
    't09_fret_neck':        'Fret / neck setup',
    't10_strings':          'String quality',
    't11_shipping':         'Shipping damage',
    't14_appearance':       'Visual appearance',
    't15_pickups':          'Pickups',
    't21_playability':      'Playability / chords',
    't22_acoustic':         'Acoustic tone',
    't24_tuning':           'Tuning stability',
}

print("Loading guitars_absa.parquet...")
df = pd.read_parquet(INPUT_FILE)
print(f"  {len(df):,} reviews")

# Apply threshold — reclassify low-confidence predictions as 'not mentioned'
for key in TOPICS:
    label_col = f'{key}_label'
    score_col = f'{key}_score'
    df[f'{key}_label_filtered'] = df.apply(
        lambda row, lc=label_col, sc=score_col:
            'not mentioned' if row[sc] < THRESHOLD else row[lc],
        axis=1
    )

lines = []
lines.append("ABSA Sentiment Distribution (confidence threshold)\n")
lines.append("=" * 75 + "\n")
lines.append(f"Model: facebook/bart-large-mnli\n")
lines.append(f"Confidence threshold: {THRESHOLD} (below this -> 'not mentioned')\n")
lines.append(f"Total reviews: {len(df):,}\n\n")

# ── Comparison table: raw vs threshold ────────────────────────────────────────
lines.append(f"{'Topic':<30} {'NA% (raw)':>11} {'NA% (thr.)':>11} {'change':>8}\n")
lines.append("-" * 64 + "\n")
for key, label in TOPICS.items():
    na_raw = (df[f'{key}_label']          == 'not mentioned').sum() / len(df) * 100
    na_thr = (df[f'{key}_label_filtered'] == 'not mentioned').sum() / len(df) * 100
    lines.append(f"{label:<30} {na_raw:>10.1f}% {na_thr:>10.1f}% {na_thr-na_raw:>+7.1f}%\n")

lines.append("\n")

# ── Sentiment ranking (with threshold) ────────────────────────────────────────
lines.append("=" * 75 + "\n")
lines.append(f"Sentiment ranking (threshold={THRESHOLD})\n")
lines.append("=" * 75 + "\n")

# Compute overall baseline positive rate (across all topics, excluding NA)
all_pos = sum((df[f'{k}_label_filtered'] == 'positive').sum() for k in TOPICS)
all_neg = sum((df[f'{k}_label_filtered'] == 'negative').sum() for k in TOPICS)
baseline_pos_rate = all_pos / (all_pos + all_neg)
lines.append(f"Overall baseline positive rate (excl. NA): {baseline_pos_rate*100:.1f}%\n")
lines.append("Relative score = topic pos rate - baseline (positive = above average praise)\n\n")

rows = []
for key, label in TOPICS.items():
    col    = f'{key}_label_filtered'
    counts = df[col].value_counts()
    total  = len(df)
    n_pos  = counts.get('positive', 0)
    n_neg  = counts.get('negative', 0)
    n_na   = counts.get('not mentioned', 0)
    p_pos  = n_pos / total * 100
    p_neg  = n_neg / total * 100
    p_na   = n_na  / total * 100
    denom  = n_pos + n_neg if (n_pos + n_neg) > 0 else 1
    topic_pos_rate = n_pos / denom
    score     = (n_pos - n_neg) / denom
    rel_score = topic_pos_rate - baseline_pos_rate
    rows.append((label, total, p_pos, p_neg, p_na, score, rel_score))

rows.sort(key=lambda x: x[6], reverse=True)

lines.append(f"{'Topic':<30} {'Pos%':>7} {'Neg%':>7} {'NA%':>7} {'Score':>7} {'RelScore':>9}\n")
lines.append("-" * 75 + "\n")
for label, total, p_pos, p_neg, p_na, score, rel_score in rows:
    lines.append(f"{label:<30} {p_pos:>6.1f}% {p_neg:>6.1f}% {p_na:>6.1f}% {score:>+.3f} {rel_score:>+.3f}\n")

with open(RESULTS_OUT, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Saved -> {RESULTS_OUT}")
print(f"\nBaseline positive rate: {baseline_pos_rate*100:.1f}%")
print(f"\nTop 3 most praised (by relative score):")
for label, total, p_pos, p_neg, p_na, score, rel_score in rows[:3]:
    print(f"  {label:<30} rel={rel_score:+.3f}  pos={p_pos:.1f}%  neg={p_neg:.1f}%")
print(f"\nTop 3 most criticized (by relative score):")
for label, total, p_pos, p_neg, p_na, score, rel_score in rows[-3:]:
    print(f"  {label:<30} rel={rel_score:+.3f}  pos={p_pos:.1f}%  neg={p_neg:.1f}%")
print("\nDone.")
