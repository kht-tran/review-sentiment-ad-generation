"""
Topic Assignment Comparison
Runs three configurations to isolate effect of threshold vs. topic merging:
  v1: 13 original topics, threshold=0.15
  v2: merged topics (14), threshold=0.15  — isolates merge effect
  v3: merged topics (14), threshold=0.30  — isolates threshold effect
Saves guitars_with_topics_v1/v2/v3.parquet.
"""

import os
import pickle
import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaMulticore

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NLP_DIR     = os.path.dirname(SCRIPT_DIR)
LDA_DIR     = os.path.join(NLP_DIR, '3. Topic modelling')

DATA_FILE   = os.path.join(NLP_DIR, 'Musical_Instruments_joined.parquet')
TOKENS_FILE = os.path.join(LDA_DIR, 'guitar_tokens.pkl')
DICT_FILE   = os.path.join(LDA_DIR, 'guitar_dict.pkl')
CORPUS_FILE = os.path.join(LDA_DIR, 'guitar_corpus.pkl')
MODEL_DIR   = os.path.join(LDA_DIR, 'guitar_lda_model')

ORIGINAL_TOPICS = {
    1:  'Customer service / returns',
    4:  'Beginner learning',
    6:  'Electronics / controls',
    7:  'Setup / action',
    8:  'Guitar size',
    9:  'Fret / neck setup',
    10: 'String quality',
    11: 'Shipping damage',
    14: 'Visual appearance',
    15: 'Pickups',
    21: 'Playability / chords',
    22: 'Acoustic tone',
    24: 'Tuning stability',
}

MERGED_TOPICS = {
    1:  'Customer service / returns',
    4:  'Beginner learning',
    5:  'Accessories',
    6:  'Electronics / controls',
    7:  'Setup / action',
    8:  'Guitar size',
    9:  'Fret / neck setup',
    10: 'String quality',
    11: 'Shipping damage',
    12: 'Accessories',
    14: 'Visual appearance',
    15: 'Pickups',
    17: 'Fret / neck setup',
    21: 'Playability / chords',
    22: 'Acoustic tone',
    24: 'Tuning stability',
}

CONFIGS = {
    'v1': {'desc': '13 original topics, threshold=0.15', 'topics': ORIGINAL_TOPICS, 'threshold': 0.15, 'output': os.path.join(NLP_DIR, 'guitars_with_topics_v1.parquet')},
    'v2': {'desc': 'merged topics (14),  threshold=0.15', 'topics': MERGED_TOPICS,   'threshold': 0.15, 'output': os.path.join(NLP_DIR, 'guitars_with_topics_v2.parquet')},
    'v3': {'desc': 'merged topics (14),  threshold=0.30', 'topics': MERGED_TOPICS,   'threshold': 0.30, 'output': os.path.join(NLP_DIR, 'guitars_with_topics_v3.parquet')},
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading preprocessed v2 data...")
with open(TOKENS_FILE, 'rb') as f:
    tokens = pickle.load(f)
dictionary = corpora.Dictionary.load(DICT_FILE)
with open(CORPUS_FILE, 'rb') as f:
    corpus = pickle.load(f)
print(f"  {len(tokens):,} reviews | {len(dictionary):,} terms")

print(f"\nLoading LDA model...")
lda = LdaMulticore.load(os.path.join(MODEL_DIR, 'model'))
print(f"  Loaded (25 topics)")

print("\nLoading guitar reviews...")
df = pd.read_parquet(DATA_FILE)
guitars = df[df['subcategory'] == 'Guitars'].copy()
guitars = guitars[guitars['text'].notna()].copy()
print(f"  {len(guitars):,} reviews")
assert len(guitars) == len(tokens)

# ── Pre-compute topic distributions once ──────────────────────────────────────
print("\nPre-computing topic distributions (once for all configs)...")
topic_dists = []
for i, doc_bow in enumerate(corpus):
    topic_dists.append(dict(lda.get_document_topics(doc_bow, minimum_probability=0.0)))
    if (i + 1) % 20_000 == 0:
        print(f"  {i+1:,} / {len(corpus):,}")
print("  Done.")

# ── Assignment function ───────────────────────────────────────────────────────
def assign_topics(topic_dists, final_topics, threshold):
    ids, labels, probs = [], [], []
    for dist in topic_dists:
        sorted_topics = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        assigned_id = assigned_label = assigned_prob = None
        for rank, (tid, tprob) in enumerate(sorted_topics):
            if tid not in final_topics:
                continue
            if rank == 0 or tprob >= threshold:
                assigned_id    = tid
                assigned_label = final_topics[tid]
                assigned_prob  = tprob
            break
        ids.append(assigned_id)
        labels.append(assigned_label)
        probs.append(assigned_prob)
    return ids, labels, probs

# ── Run all configs ───────────────────────────────────────────────────────────
results = {}
for version, cfg in CONFIGS.items():
    print(f"\n── {version}: {cfg['desc']} ──")
    ids, labels, probs = assign_topics(topic_dists, cfg['topics'], cfg['threshold'])

    g = guitars.copy()
    g['topic_id']    = ids
    g['topic_label'] = labels
    g['topic_prob']  = probs

    assigned = g['topic_id'].notna().sum()
    total    = len(g)
    print(f"  Assigned: {assigned:,} ({assigned/total*100:.1f}%)  |  N/A: {total-assigned:,} ({(total-assigned)/total*100:.1f}%)")

    g.to_parquet(cfg['output'], index=True)
    print(f"  Saved → {os.path.basename(cfg['output'])}")
    results[version] = {'df': g, 'assigned': assigned, 'total': total, 'desc': cfg['desc']}

# ── Comparison table ──────────────────────────────────────────────────────────
total = results['v1']['total']
all_labels = sorted(set(
    results['v1']['df']['topic_label'].dropna().tolist() +
    results['v2']['df']['topic_label'].dropna().tolist() +
    results['v3']['df']['topic_label'].dropna().tolist()
))

print("\n" + "=" * 75)
print("COMPARISON SUMMARY")
print("=" * 75)
print(f"  {'':32} {'v1':>12} {'v2':>12} {'v3':>12}")
print(f"  {'':32} {'13t, t=0.15':>12} {'14t, t=0.15':>12} {'14t, t=0.30':>12}")
print("  " + "-" * 70)

for v in ['v1', 'v2', 'v3']:
    a = results[v]['assigned']
    print(f"  {'TOTAL ASSIGNED':<32} {a:>8,} ({a/total*100:.0f}%)" if v == 'v1' else
          f"  {'':32} {a:>8,} ({a/total*100:.0f}%)" if v != 'v1' else "")

# reprint cleanly
print(f"\n  {'':32} {'v1':>12} {'v2':>12} {'v3':>12}")
print("  " + "-" * 70)
print(f"  {'Total assigned':<32} " +
      "  ".join(f"{results[v]['assigned']:>8,} ({results[v]['assigned']/total*100:.0f}%)" for v in ['v1','v2','v3']))
print()

for label in all_labels:
    c1 = (results['v1']['df']['topic_label'] == label).sum()
    c2 = (results['v2']['df']['topic_label'] == label).sum()
    c3 = (results['v3']['df']['topic_label'] == label).sum()
    marker = " *" if c1 == 0 else ""
    print(f"  {label:<32} {c1:>12,} {c2:>12,} {c3:>12,}{marker}")

print("\n  * = new topic (only in v2/v3)")
print("=" * 75)
print("\nDone.")
