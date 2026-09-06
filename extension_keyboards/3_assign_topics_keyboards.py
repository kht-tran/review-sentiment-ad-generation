"""
Extension — Assign Topics to Keyboards & MIDI Reviews
======================================================
Mirrors step08/script.py exactly.
Trains a final LDA with the best params from tune_params_keyboards.py,
assigns dominant topics, saves keyboards_with_topics.parquet.

BEFORE RUNNING — fill in the two sections marked TODO below:
  1. Best params   → from tune_results_keyboards.txt  ("Best:" line)
  2. FINAL_TOPICS  → from tune_results_keyboards.txt  (topic word lists for best model)
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
TOKENS_FILE = os.path.join(SCRIPT_DIR, 'keyboards_tokens.pkl')
DICT_FILE   = os.path.join(SCRIPT_DIR, 'keyboards_dict.pkl')
CORPUS_FILE = os.path.join(SCRIPT_DIR, 'keyboards_corpus.pkl')
MODEL_DIR   = os.path.join(SCRIPT_DIR, 'keyboards_lda_final')
OUTPUT_FILE = os.path.join(NLP_DIR, 'keyboards_with_topics.parquet')

SUBCATEGORY    = 'Keyboards & MIDI'
PROB_THRESHOLD = 0.3
RANDOM_SEED    = 42

# ── Best params from tune_results_keyboards.txt ───────────────────────────────
# Best: topics=25 | alpha=asymmetric | eta=auto | passes=10 | coherence=0.5758
NUM_TOPICS = 25
ALPHA      = 'asymmetric'
ETA        = 'auto'
PASSES     = 10

# ── Selected topics from tune_results_keyboards.txt ───────────────────────────
# Dropped: Topic 00 (generic positive praise) — not feature-specific
#          Topic 03 (cute/travel/baby toy) — overlaps with 05, too toy-like
#          Topic 04 (easy/assemble) — overlaps with 05
#          Topic 06 (stand/bench assembly) — too mechanical for ad targeting
#          Topic 10 (generic instrument/youtube) — uninterpretable
#          Topic 13 (melodica/case) — too niche single instrument
#          Topic 15 (gift/christmas) — overlaps with 02
#          Topic 23 (kalimba) — too niche single instrument
#          Topic 24 (tone variety/organ) — overlaps with 09
FINAL_TOPICS = {
    2:  'Gift / holiday purchase',
    5:  'Beginner learning',
    7:  'Appearance / color / display',
    8:  'Power supply & accessories',
    9:  'Sound quality / speaker',
    11: 'Sustain pedal',
    12: 'Returns / broken keys',
    14: 'Brand quality / acoustic feel',
    17: 'Durability / stopped working',
    18: 'Kids music lessons',
    19: 'MIDI controller / live performance',
    20: 'Synthesizer / sound design',
    21: 'Connectivity (USB/MIDI/phone)',
    22: 'Key feel / weighted keys',
}

# ── Validation ────────────────────────────────────────────────────────────────
if None in (NUM_TOPICS, ALPHA, ETA, PASSES):
    raise ValueError(
        "Fill in NUM_TOPICS, ALPHA, ETA, PASSES from tune_results_keyboards.txt before running."
    )
if not FINAL_TOPICS:
    raise ValueError(
        "FINAL_TOPICS is empty. Read the topic word lists in tune_results_keyboards.txt "
        "and fill in the FINAL_TOPICS dict before running."
    )

# ── Load preprocessed data ───────────────────────────────────────────────────
print("Loading preprocessed keyboards data...")
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

# ── Load reviews (same filter as preprocess_keyboards.py) ─────────────────────
print(f"\nLoading {SUBCATEGORY} reviews...")
df = pd.read_parquet(DATA_FILE)
subset = df[df['subcategory'] == SUBCATEGORY].copy()
subset = subset[subset['text'].notna()].copy()
print(f"  {len(subset):,} reviews (matching token list: {len(tokens):,})")
assert len(subset) == len(tokens), (
    f"Row count mismatch: {len(subset)} reviews vs {len(tokens)} tokens. "
    "Ensure preprocess_keyboards.py and this script use the same data file and filters."
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

na_series  = pd.Series(na_max_probs)
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
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
print("  1. scp keyboards_with_topics.parquet to your laptop")
print("  2. Run step1_data_prep notebook (change input path, N_PRODUCTS=2, N_TOPICS_PER_PROD=2)")
print("  3. Run step2_ad_generation notebook with your OpenAI API key")
