"""
Aspect-Based Sentiment Analysis (ABSA) on Guitar Reviews
For each review, score sentiment toward each of the 14 topics independently
using BART zero-shot NLI. No rating labels required.

Output: guitars_absa.parquet
  - One row per review (all 134,068 guitar reviews)
  - 3 columns per topic: {key}_label, {key}_score, {key}_conf
      label: "positive" / "negative" / "not mentioned"
      score: probability of the winning label
      conf:  margin between top two labels (higher = more confident)

Checkpointing: results saved per topic in ckpt/ folder.
Safe to resubmit if job is interrupted.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NLP_DIR    = os.path.dirname(SCRIPT_DIR)
CKPT_DIR   = os.path.join(SCRIPT_DIR, 'ckpt')
os.makedirs(CKPT_DIR, exist_ok=True)

DATA_FILE  = os.path.join(NLP_DIR, 'Musical_Instruments_joined.parquet')
OUTPUT     = os.path.join(SCRIPT_DIR, 'guitars_absa.parquet')

MODEL_NAME       = 'facebook/bart-large-mnli'
BATCH_SIZE       = 64
CANDIDATE_LABELS = ['positive', 'negative', 'not mentioned']

# 14 topics — key is used for column names, label goes into hypothesis
TOPICS = {
    't01_customer_service': 'customer service and returns',
    't04_beginner':         'beginner learning experience',
    't05_accessories':      'accessories such as straps and cases',
    't06_electronics':      'electronics and controls',
    't07_setup':            'setup and action',
    't08_size':             'guitar size and body',
    't09_fret_neck':        'fret and neck setup',
    't10_strings':          'string quality',
    't11_shipping':         'shipping and delivery damage',
    't14_appearance':       'visual appearance and finish',
    't15_pickups':          'pickups and hardware',
    't21_playability':      'playability and chords',
    't22_acoustic':         'acoustic tone and sound',
    't24_tuning':           'tuning stability',
}

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading guitar reviews...")
df      = pd.read_parquet(DATA_FILE)
guitars = df[df['subcategory'] == 'Guitars'].copy()
guitars = guitars[guitars['text'].notna()].copy()
guitars['text'] = guitars['text'].str.strip().replace('', '[no text]')
texts   = guitars['text'].tolist()
print(f"  {len(texts):,} reviews", flush=True)

# ── Load model ─────────────────────────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else -1
print(f"\nGPUs available: {torch.cuda.device_count()}")
print(f"Loading {MODEL_NAME} on {'GPU' if device == 0 else 'CPU'}...", flush=True)
pipe = pipeline(
    'zero-shot-classification',
    model=MODEL_NAME,
    device=device,
)

# ── Run ABSA per topic ─────────────────────────────────────────────────────────
results = {}

for key, topic_label in TOPICS.items():
    ckpt_file = os.path.join(CKPT_DIR, f'{key}.pkl')

    if os.path.exists(ckpt_file):
        print(f"\n[skip] {key} — checkpoint found", flush=True)
        with open(ckpt_file, 'rb') as f:
            results[key] = pickle.load(f)
        continue

    print(f"\nProcessing: {key} ({topic_label})", flush=True)
    hypothesis = f"The reviewer's opinion about {topic_label} is {{}}."

    labels_out = []
    scores_out = []
    confs_out  = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        out   = pipe(
            batch,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template=hypothesis,
            truncation=True,
        )
        for r in out:
            label_scores = dict(zip(r['labels'], r['scores']))
            winner = max(label_scores, key=label_scores.get)
            score  = label_scores[winner]
            sorted_scores = sorted(label_scores.values(), reverse=True)
            conf = sorted_scores[0] - sorted_scores[1]

            labels_out.append(winner)
            scores_out.append(score)
            confs_out.append(conf)

        if (i // BATCH_SIZE + 1) % 200 == 0:
            done = min(i + BATCH_SIZE, len(texts))
            print(f"  {done:,} / {len(texts):,}", flush=True)

    topic_result = {
        'label': labels_out,
        'score': scores_out,
        'conf':  confs_out,
    }

    with open(ckpt_file, 'wb') as f:
        pickle.dump(topic_result, f)
    print(f"  Done. Saved checkpoint: {key}.pkl", flush=True)

    label_series = pd.Series(labels_out)
    counts = label_series.value_counts()
    total  = len(labels_out)
    for lbl, cnt in counts.items():
        print(f"    {lbl:<15} {cnt:>7,}  ({cnt/total*100:.1f}%)", flush=True)

    results[key] = topic_result

# ── Combine into output parquet ────────────────────────────────────────────────
print("\nCombining results...", flush=True)

output_df = guitars.copy()
for key, res in results.items():
    output_df[f'{key}_label'] = res['label']
    output_df[f'{key}_score'] = res['score']
    output_df[f'{key}_conf']  = res['conf']

output_df.to_parquet(OUTPUT, index=True)
print(f"Saved {len(output_df):,} rows -> {OUTPUT}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n\nABSA Summary — Sentiment distribution per topic")
print(f"{'Topic':<30} {'Positive':>10} {'Negative':>10} {'Not mentioned':>14}")
print("-" * 68)

for key, topic_label in TOPICS.items():
    col    = f'{key}_label'
    counts = output_df[col].value_counts()
    total  = len(output_df)
    p_pos  = counts.get('positive', 0) / total * 100
    p_neg  = counts.get('negative', 0) / total * 100
    p_na   = counts.get('not mentioned', 0) / total * 100
    print(f"{topic_label:<30} {p_pos:>9.1f}% {p_neg:>9.1f}% {p_na:>13.1f}%")

print("\nDone.")
