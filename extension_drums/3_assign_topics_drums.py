"""
Extension — Assign Topics to Drums & Percussion Reviews
========================================================
Mirrors step08/script.py exactly.
Trains a final LDA with the best params from tune_params_drums.py,
assigns dominant topics, saves drums_with_topics.parquet.

BEFORE RUNNING — fill in the two sections marked TODO below:
  1. Best params   → from tune_results_drums.txt  ("Best:" line)
  2. FINAL_TOPICS  → from tune_results_drums.txt  (topic word lists for best model)
"""

import os
import pickle
import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaMulticore

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NLP_DIR     = os.path.dirname(SCRIPT_DIR)

DATA_FILE   = os.path.join(NLP_DIR, 'Musical_Instruments_joined.parquet')
TOKENS_FILE = os.path.join(SCRIPT_DIR, 'drums_tokens.pkl')
DICT_FILE   = os.path.join(SCRIPT_DIR, 'drums_dict.pkl')
CORPUS_FILE = os.path.join(SCRIPT_DIR, 'drums_corpus.pkl')
MODEL_DIR   = os.path.join(SCRIPT_DIR, 'drums_lda_final')
OUTPUT_FILE = os.path.join(NLP_DIR, 'drums_with_topics.parquet')

SUBCATEGORY    = 'Drums & Percussion'
PROB_THRESHOLD = 0.3
RANDOM_SEED    = 42

# ── Best params from tune_results_drums.txt ───────────────────────────────────
# Best: topics=10 | alpha=asymmetric | eta=auto | passes=10 | coherence=0.6558
NUM_TOPICS = 10
ALPHA      = 'asymmetric'
ETA        = 'auto'
PASSES     = 10

# ── Selected topics from tune_results_drums.txt ───────────────────────────────
# Dropped: Topic 00 (generic praise: quality/beautiful/perfect) — not feature-specific
#          Topic 09 (noise: thing/know/going/think) — uninterpretable
FINAL_TOPICS = {
    1: 'Gift / holiday purchase',
    2: 'Cymbals',
    3: 'Singing bowls / meditation',
    4: 'Beginner / ease of play',
    5: 'Product defects / returns',
    6: 'Small percussion accessories',
    7: 'Drum hardware / shells',
    8: 'Electronic drums / practice pad',
}

# ── Validation ────────────────────────────────────────────────────────────────
if None in (NUM_TOPICS, ALPHA, ETA, PASSES):
    raise ValueError(
        "Fill in NUM_TOPICS, ALPHA, ETA, PASSES from tune_results_drums.txt before running."
    )
if not FINAL_TOPICS:
    raise ValueError(
        "FINAL_TOPICS is empty. Read the topic word lists in tune_results_drums.txt "
        "and fill in the FINAL_TOPICS dict before running."
    )

# ── Load preprocessed data ───────────────────────────────────────────────────
print("Loading preprocessed drums data...")
with open(TOKENS_FILE, 'rb') as f:
    tokens = pickle.load(f)
dictionary = corpora.Dictionary.load(DICT_FILE)
with open(CORPUS_FILE, 'rb') as f:
    corpus = pickle.load(f)
print(f"  {len(tokens):,} reviews | {len(dictionary):,} terms")

# ── Train final LDA (with checkpoint) ────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, 'model')
if os.path.exists(model_path):
    print(f"\nLoading saved final LDA model from {MODEL_DIR}...")
    lda = LdaMulticore.load(model_path)
    print(f"  Loaded ({NUM_TOPICS} topics)")
else:
    print(f"\nTraining final LDA ({NUM_TOPICS} topics, alpha={ALPHA}, eta={ETA}, passes={PASSES})...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    lda = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        alpha=ALPHA,
        eta=ETA,
        passes=PASSES,
        random_state=RANDOM_SEED,
        workers=3,
    )
    lda.save(model_path)
    print(f"  Model saved → {MODEL_DIR}/")

# ── Load reviews (same filter as preprocess_drums.py) ─────────────────────────
print(f"\nLoading {SUBCATEGORY} reviews...")
df = pd.read_parquet(DATA_FILE)
subset = df[df['subcategory'] == SUBCATEGORY].copy()
subset = subset[subset['text'].notna()].copy()
print(f"  {len(subset):,} reviews (matching token list: {len(tokens):,})")
assert len(subset) == len(tokens), (
    f"Row count mismatch: {len(subset)} reviews vs {len(tokens)} tokens. "
    "Ensure preprocess_drums.py and this script use the same data file and filters."
)

# ── Assign dominant topic ─────────────────────────────────────────────────────
print("\nAssigning topics to reviews...")

topic_ids    = []
topic_labels = []
topic_probs  = []

for i, doc_bow in enumerate(corpus):
    topic_dist    = dict(lda.get_document_topics(doc_bow, minimum_probability=0.0))
    sorted_topics = sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)

    assigned_id    = None
    assigned_label = None
    assigned_prob  = None

    for rank, (tid, tprob) in enumerate(sorted_topics):
        if tid not in FINAL_TOPICS:
            continue
        if rank == 0:
            assigned_id    = tid
            assigned_label = FINAL_TOPICS[tid]
            assigned_prob  = tprob
        else:
            if tprob >= PROB_THRESHOLD:
                assigned_id    = tid
                assigned_label = FINAL_TOPICS[tid]
                assigned_prob  = tprob
        break

    topic_ids.append(assigned_id)
    topic_labels.append(assigned_label)
    topic_probs.append(assigned_prob)

    if (i + 1) % 10_000 == 0:
        print(f"  Processed {i+1:,} / {len(corpus):,}")

subset = subset.copy()
subset['topic_id']    = topic_ids
subset['topic_label'] = topic_labels
subset['topic_prob']  = topic_probs

# ── Summary ───────────────────────────────────────────────────────────────────
total      = len(subset)
assigned   = subset['topic_id'].notna().sum()
unassigned = total - assigned

print(f"\nTopic assignment summary:")
print(f"  Total reviews:      {total:,}")
print(f"  Assigned:           {assigned:,} ({assigned/total*100:.1f}%)")
print(f"  N/A (excluded):     {unassigned:,} ({unassigned/total*100:.1f}%)")
print(f"\nPer-topic counts:")
counts = subset[subset['topic_id'].notna()].groupby(['topic_id', 'topic_label']).size()
for (tid, tlabel), count in counts.items():
    print(f"  Topic {int(tid):02d} — {tlabel:<35} {count:,}")

# ── N/A threshold diagnostic (mirrors step08) ─────────────────────────────────
print("\n── N/A Threshold Diagnostic ────────────────────────────────────────")
print("For each N/A review, best probability across the selected final topics:")

na_max_probs = []
for i, doc_bow in enumerate(corpus):
    tid_val = subset.iloc[i]['topic_id']
    if tid_val is not None and not (isinstance(tid_val, float) and pd.isna(tid_val)):
        continue
    topic_dist = dict(lda.get_document_topics(doc_bow, minimum_probability=0.0))
    best_prob  = max((topic_dist.get(tid, 0.0) for tid in FINAL_TOPICS), default=0.0)
    na_max_probs.append(best_prob)

import pandas as pd_inner
na_series   = pd.Series(na_max_probs)
thresholds  = [0.10, 0.15, 0.20, 0.25, 0.30]
print(f"\n  Total N/A reviews: {len(na_series):,}")
print(f"\n  {'Threshold':>10} {'Recovered':>10} {'% of N/A':>10} {'New assign%':>13}")
for t in thresholds:
    recovered    = (na_series >= t).sum()
    new_assigned = assigned + recovered
    print(f"  {t:>10.2f} {recovered:>10,} {recovered/max(len(na_series),1)*100:>9.1f}%"
          f" {new_assigned/total*100:>12.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
subset.to_parquet(OUTPUT_FILE, index=True)
print(f"\nSaved {len(subset):,} rows → {OUTPUT_FILE}")
print("\nDone.")
print("\nNEXT STEPS:")
print("  1. scp drums_with_topics.parquet to your laptop")
print("  2. Run step1_data_prep notebook (change input path, N_PRODUCTS=2, N_TOPICS_PER_PROD=2)")
print("  3. Run step2_ad_generation notebook with your OpenAI API key")
