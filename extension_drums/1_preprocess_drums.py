"""
Extension — Preprocessing for Drums & Percussion Reviews
=========================================================
Mirrors lda_guitars_v2.py stages 1-2.
Run this FIRST. Saves tokens/dict/corpus for tune_params_drums.py and assign_topics_drums.py.

Output: drums_tokens.pkl, drums_dict.pkl, drums_corpus.pkl
"""

import os
import pickle
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NLP_DIR     = os.path.dirname(SCRIPT_DIR)

JOINED      = os.path.join(NLP_DIR, 'Musical_Instruments_joined.parquet')
TOKENS_OUT  = os.path.join(SCRIPT_DIR, 'drums_tokens.pkl')
DICT_OUT    = os.path.join(SCRIPT_DIR, 'drums_dict.pkl')
CORPUS_OUT  = os.path.join(SCRIPT_DIR, 'drums_corpus.pkl')

SUBCATEGORY = 'Drums & Percussion'


def stage_done(path):
    exists = os.path.exists(path)
    if exists:
        print(f"  [skip] Found: {os.path.basename(path)}")
    return exists


# ════════════════════════════════════════════════════════════════════════════
# Stage 1 — Filter & Preprocess
# ════════════════════════════════════════════════════════════════════════════
print("\n── Stage 1: Filter & Preprocess ────────────────────────────────────")

if stage_done(TOKENS_OUT):
    with open(TOKENS_OUT, 'rb') as f:
        tokens = pickle.load(f)
    print(f"  Loaded {len(tokens):,} tokenized reviews")
else:
    import re
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    print("  Loading joined parquet...")
    df = pd.read_parquet(JOINED, columns=['text', 'subcategory', 'rating'])
    subset = df[df['subcategory'] == SUBCATEGORY].copy()
    print(f"  {SUBCATEGORY} reviews: {len(subset):,}")

    # Mirrors guitars custom_stops exactly — only swapping instrument-specific terms.
    # 'drum'/'drums' appear in virtually every review → no topic signal, same logic as
    # 'guitar'/'guitars' in the guitars preprocessing.
    custom_stops = {
        'drum', 'drums', 'drumming', 'drummer', 'percussion',
        'product', 'buy', 'bought', 'purchase', 'ordered',
        'amazon', 'price', 'money', 'shipping', 'arrived', 'order', 'seller',
        'would', 'could', 'one', 'get', 'got', 'also', 'really', 'great',
        'good', 'nice', 'love', 'like', 'use', 'used', 'using', 'item',
        'came', 'come', 'says', 'said', 'well', 'much', 'very', 'even',
        'still', 'back', 'just', 'make', 'made', 'little', 'bit', 'lot',
        'way', 'time', 'star', 'stars', 'review', 'reviews'
    }
    stop_words = set(stopwords.words('english')) | custom_stops
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words
                 if w not in stop_words and len(w) > 3]
        return words

    print("  Preprocessing text...")
    subset = subset[subset['text'].notna()]
    tokens = subset['text'].apply(preprocess).tolist()

    with open(TOKENS_OUT, 'wb') as f:
        pickle.dump(tokens, f)
    print(f"  Saved {len(tokens):,} tokenized reviews → {os.path.basename(TOKENS_OUT)}")


# ════════════════════════════════════════════════════════════════════════════
# Stage 2 — Build Dictionary & Corpus
# ════════════════════════════════════════════════════════════════════════════
print("\n── Stage 2: Build Dictionary & Corpus ──────────────────────────────")

if stage_done(DICT_OUT) and stage_done(CORPUS_OUT):
    import gensim.corpora as corpora
    dictionary = corpora.Dictionary.load(DICT_OUT)
    with open(CORPUS_OUT, 'rb') as f:
        corpus = pickle.load(f)
    print(f"  Dictionary: {len(dictionary):,} terms")
else:
    import gensim.corpora as corpora

    dictionary = corpora.Dictionary(tokens)
    # Mirrors lda_guitars_v2.py: no_above=0.3 removes cross-topic generic words
    dictionary.filter_extremes(no_below=10, no_above=0.3)
    print(f"  Dictionary: {len(dictionary):,} terms after filtering")

    corpus = [dictionary.doc2bow(doc) for doc in tokens]

    dictionary.save(DICT_OUT)
    with open(CORPUS_OUT, 'wb') as f:
        pickle.dump(corpus, f)
    print(f"  Saved dictionary ({len(dictionary):,} terms) and corpus ({len(corpus):,} docs)")

print("\nDone.")
print("Next: sbatch run_tune_drums.sh")
