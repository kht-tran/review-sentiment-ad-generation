"""
BERTopic Hyperparameter Tuning
Grid search over UMAP/HDBSCAN parameters.
Reuses saved embeddings from bertopic_guitars.py (stage 2).

Target: 10-20 meaningful topics, outlier ratio < 20%
"""

import os
import pickle
import numpy as np
import pandas as pd
from itertools import product

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TEXTS_OUT   = os.path.join(SCRIPT_DIR, 'guitar_texts.pkl')
EMB_OUT     = os.path.join(SCRIPT_DIR, 'guitar_embeddings.npy')
RESULTS_OUT = os.path.join(SCRIPT_DIR, 'bertopic_tuning_results.txt')

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading embeddings and texts...")
with open(TEXTS_OUT, 'rb') as f:
    docs = pickle.load(f)
embeddings = np.load(EMB_OUT)
print(f"  {len(docs):,} reviews | embeddings: {embeddings.shape}\n")

# ── Parameter grid ────────────────────────────────────────────────────────────
n_neighbors_list      = [15, 30, 50]
min_cluster_size_list = [100, 200, 300]
min_samples_list      = [10, 30]

configs = list(product(n_neighbors_list, min_cluster_size_list, min_samples_list))
print(f"Testing {len(configs)} configurations...\n")

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))

results = []

for i, (n_neighbors, min_cluster_size, min_samples) in enumerate(configs):
    print(f"[{i+1}/{len(configs)}] n_neighbors={n_neighbors} | min_cluster_size={min_cluster_size} | min_samples={min_samples}")

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        prediction_data=True
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        language='english',
        calculate_probabilities=False,
        verbose=False
    )

    try:
        topics, _ = topic_model.fit_transform(docs, embeddings)
        topic_info = topic_model.get_topic_info()
        n_topics   = len(topic_info[topic_info['Topic'] != -1])
        n_outliers = int(topic_info[topic_info['Topic'] == -1]['Count'].values[0]) \
                     if -1 in topic_info['Topic'].values else 0
        outlier_pct = n_outliers / len(docs) * 100

        # Score: penalize if outside target range (10-20 topics, <20% outliers)
        topic_ok   = 10 <= n_topics <= 20
        outlier_ok = outlier_pct < 20
        status     = 'GOOD' if topic_ok and outlier_ok else \
                     'topics_ok' if topic_ok else \
                     'outlier_ok' if outlier_ok else 'BAD'

        results.append({
            'n_neighbors': n_neighbors,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'n_topics': n_topics,
            'outlier_pct': round(outlier_pct, 1),
            'status': status
        })
        print(f"  → {n_topics} topics | {outlier_pct:.1f}% outliers | {status}\n")

    except Exception as e:
        print(f"  → ERROR: {e}\n")
        results.append({
            'n_neighbors': n_neighbors,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'n_topics': -1,
            'outlier_pct': -1,
            'status': 'ERROR'
        })

# ── Summary ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df = df.sort_values(['status', 'outlier_pct'], ascending=[True, True])

print("\n── Full Results (sorted by status) ─────────────────────────────────")
print(df.to_string(index=False))

good = df[df['status'] == 'GOOD']
print(f"\n── GOOD configs ({len(good)} found) ────────────────────────────────")
print(good.to_string(index=False) if len(good) > 0 else "  None found — review results above")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(RESULTS_OUT, 'w') as f:
    f.write("BERTopic Hyperparameter Tuning\n")
    f.write("Target: 10-20 topics, <20% outliers\n")
    f.write("=" * 70 + "\n\n")
    f.write(df.to_string(index=False))
    f.write("\n\n── GOOD configs ──\n")
    f.write(good.to_string(index=False) if len(good) > 0 else "None found")

print(f"\nResults saved → {os.path.basename(RESULTS_OUT)}")
print("Done.")
