"""
LDA Topic Modeling on Guitar Reviews
Preprocesses guitar reviews, builds dictionary/corpus, trains LDA model.

Stages:
  1. Filter & preprocess  → guitar_tokens.pkl
  2. Build dictionary & corpus → guitar_dict.pkl, guitar_corpus.pkl
  3. Run LDA              → guitar_lda_model (folder)
  4. Save results         → lda_results.txt
"""

import os
import pickle
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NLP_DIR     = os.path.dirname(SCRIPT_DIR)

JOINED      = os.path.join(NLP_DIR, 'Musical_Instruments_joined.parquet')
TOKENS_OUT  = os.path.join(SCRIPT_DIR, 'guitar_tokens.pkl')
DICT_OUT    = os.path.join(SCRIPT_DIR, 'guitar_dict.pkl')
CORPUS_OUT  = os.path.join(SCRIPT_DIR, 'guitar_corpus.pkl')
MODEL_DIR   = os.path.join(SCRIPT_DIR, 'guitar_lda_model')
RESULTS_OUT = os.path.join(SCRIPT_DIR, 'lda_results.txt')

NUM_TOPICS  = 10
RANDOM_SEED = 42


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
    guitars = df[df['subcategory'] == 'Guitars'].copy()
    print(f"  Guitar reviews: {len(guitars):,}")

    custom_stops = {
        'guitar', 'guitars', 'product', 'buy', 'bought', 'purchase', 'ordered',
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
    guitars = guitars[guitars['text'].notna()]
    tokens = guitars['text'].apply(preprocess).tolist()

    with open(TOKENS_OUT, 'wb') as f:
        pickle.dump(tokens, f)
    print(f"  Saved {len(tokens):,} tokenized reviews")


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

    # no_above=0.3 removes words appearing in >30% of documents, reducing
    # cross-topic noise and producing more distinct topics
    dictionary.filter_extremes(no_below=10, no_above=0.3)
    print(f"  Dictionary: {len(dictionary):,} terms after filtering")

    corpus = [dictionary.doc2bow(doc) for doc in tokens]

    dictionary.save(DICT_OUT)
    with open(CORPUS_OUT, 'wb') as f:
        pickle.dump(corpus, f)
    print(f"  Saved dictionary and corpus")


# ════════════════════════════════════════════════════════════════════════════
# Stage 3 — Run LDA
# ════════════════════════════════════════════════════════════════════════════
print("\n── Stage 3: Run LDA ────────────────────────────────────────────────")

if stage_done(MODEL_DIR):
    from gensim.models import LdaMulticore
    lda = LdaMulticore.load(os.path.join(MODEL_DIR, 'model'))
    print(f"  Loaded LDA model ({NUM_TOPICS} topics)")
else:
    from gensim.models import LdaMulticore

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"  Training LDA with {NUM_TOPICS} topics...")

    lda = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        random_state=RANDOM_SEED,
        alpha='asymmetric',
        eta='auto',
        passes=20,
        workers=3
    )

    lda.save(os.path.join(MODEL_DIR, 'model'))
    print(f"  Saved LDA model")


# ════════════════════════════════════════════════════════════════════════════
# Stage 4 — Save Results
# ════════════════════════════════════════════════════════════════════════════
print("\n── Stage 4: Results ────────────────────────────────────────────────")

lines = [f"LDA Topic Model — Guitars ({NUM_TOPICS} topics, no_above=0.3)\n"]
lines.append("=" * 60 + "\n")

for idx, topic in lda.print_topics(num_topics=NUM_TOPICS, num_words=12):
    line = f"Topic {idx:02d}: {topic}"
    print(f"  {line}")
    lines.append(line + "\n")

from gensim.models import CoherenceModel
coherence = CoherenceModel(
    model=lda, texts=tokens, dictionary=dictionary, coherence='c_v'
).get_coherence()
lines.append(f"\nCoherence score (c_v): {coherence:.4f}\n")
print(f"\n  Coherence score (c_v): {coherence:.4f}")

with open(RESULTS_OUT, 'w') as f:
    f.writelines(lines)
print(f"  Results saved → {os.path.basename(RESULTS_OUT)}")

print("\nDone.")
