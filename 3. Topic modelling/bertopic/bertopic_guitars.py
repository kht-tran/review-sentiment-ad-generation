"""
BERTopic on Guitar Reviews
Uses sentence transformers + UMAP + HDBSCAN for topic modeling.
Generally produces cleaner topics than LDA.
Checkpointed: resubmit safely if job breaks.

Stages:
  1. Load & filter guitar reviews  → guitar_texts.pkl
  2. Encode with sentence transformer → guitar_embeddings.npy
  3. Run BERTopic                  → bertopic_results.txt
"""

import os
import pickle
import numpy as np
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TOPIC_DIR   = os.path.dirname(SCRIPT_DIR)   # 3. Topic modelling/
NLP_DIR     = os.path.dirname(TOPIC_DIR)    # ~/nlp/

JOINED      = os.path.join(NLP_DIR, 'Musical_Instruments_joined.parquet')
TEXTS_OUT   = os.path.join(SCRIPT_DIR, 'guitar_texts.pkl')
EMB_OUT     = os.path.join(SCRIPT_DIR, 'guitar_embeddings.npy')
RESULTS_OUT = os.path.join(SCRIPT_DIR, 'bertopic_results.txt')

EMBED_MODEL = 'all-mpnet-base-v2'


def stage_done(path):
    exists = os.path.exists(path)
    if exists:
        print(f"  [skip] Found: {os.path.basename(path)}")
    return exists


# ════════════════════════════════════════════════════════════════════════════
# Stage 1 — Load & filter guitar reviews
# ════════════════════════════════════════════════════════════════════════════
print("\n── Stage 1: Load guitar reviews ────────────────────────────────────")

if stage_done(TEXTS_OUT):
    with open(TEXTS_OUT, 'rb') as f:
        docs = pickle.load(f)
    print(f"  Loaded {len(docs):,} reviews")
else:
    print("  Loading parquet...")
    df = pd.read_parquet(JOINED, columns=['text', 'subcategory'])
    guitars = df[df['subcategory'] == 'Guitars'].copy()
    guitars = guitars[guitars['text'].notna()]
    guitars = guitars[guitars['text'].str.strip() != '']
    docs = guitars['text'].tolist()

    with open(TEXTS_OUT, 'wb') as f:
        pickle.dump(docs, f)
    print(f"  Saved {len(docs):,} reviews")


# ════════════════════════════════════════════════════════════════════════════
# Stage 2 — Encode with sentence transformer
# ════════════════════════════════════════════════════════════════════════════
print("\n── Stage 2: Encode reviews ─────────────────────────────────────────")

if stage_done(EMB_OUT):
    embeddings = np.load(EMB_OUT)
    print(f"  Loaded embeddings: {embeddings.shape}")
else:
    from sentence_transformers import SentenceTransformer
    print(f"  Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("  Encoding (this may take a few minutes on GPU)...")
    embeddings = model.encode(
        docs,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    np.save(EMB_OUT, embeddings)
    print(f"  Saved embeddings: {embeddings.shape}")


# ════════════════════════════════════════════════════════════════════════════
# Stage 3 — Run BERTopic
# ════════════════════════════════════════════════════════════════════════════
print("\n── Stage 3: Run BERTopic ───────────────────────────────────────────")

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Configure components
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=50,    # small → many granular topics, low outliers
    min_samples=10,
    metric='euclidean',
    prediction_data=True
)

# Remove stopwords from topic word representations
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    nr_topics=15,           # merge 300+ granular topics down to 15
    language='english',
    calculate_probabilities=False,
    verbose=True
)

print("  Fitting BERTopic...")
topics, _ = topic_model.fit_transform(docs, embeddings)

# ── Results ───────────────────────────────────────────────────────────────────
topic_info = topic_model.get_topic_info()
n_topics   = len(topic_info[topic_info['Topic'] != -1])
n_outliers = topic_info[topic_info['Topic'] == -1]['Count'].values[0] if -1 in topic_info['Topic'].values else 0

print(f"\n  Topics found: {n_topics}")
print(f"  Outliers (topic -1): {n_outliers:,} reviews ({n_outliers/len(docs)*100:.1f}%)")

print("\n── Topic Summary ───────────────────────────────────────────────────")
lines = [f"BERTopic Results — Guitars\n",
         f"Topics found: {n_topics} | Outliers: {n_outliers:,} ({n_outliers/len(docs)*100:.1f}%)\n",
         "=" * 70 + "\n\n"]

for _, row in topic_info[topic_info['Topic'] != -1].iterrows():
    topic_id    = row['Topic']
    count       = row['Count']
    top_words   = topic_model.get_topic(topic_id)
    words_str   = ' | '.join([f"{w} ({round(s, 3)})" for w, s in top_words[:8]])
    line        = f"Topic {topic_id:02d} ({count:,} reviews): {words_str}"
    print(f"  {line}")
    lines.append(line + "\n")

with open(RESULTS_OUT, 'w') as f:
    f.writelines(lines)

print(f"\n  Results saved → {os.path.basename(RESULTS_OUT)}")
print("\nDone.")
