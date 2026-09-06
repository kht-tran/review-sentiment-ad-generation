"""
LDA Hyperparameter Tuning (no_above=0.3)
Sweeps alpha, eta, and passes combinations to find the best coherence score.
Reuses preprocessed data from lda_guitars.py (run stages 1-2 first).
"""

import os
import pickle
import gensim.corpora as corpora
from gensim.models import LdaMulticore, CoherenceModel

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TOKENS_OUT  = os.path.join(SCRIPT_DIR, 'guitar_tokens.pkl')
DICT_OUT    = os.path.join(SCRIPT_DIR, 'guitar_dict.pkl')
CORPUS_OUT  = os.path.join(SCRIPT_DIR, 'guitar_corpus.pkl')
RESULTS_OUT = os.path.join(SCRIPT_DIR, 'lda_param_tuning_results.txt')

RANDOM_SEED = 42

# ── Parameter grid ────────────────────────────────────────────────────────────
configs = [
    # (num_topics, alpha,        eta,         passes)
    (10,  'symmetric',  'symmetric',  10),
    (10,  'asymmetric', 'auto',       10),
    (10,  0.01,         0.01,         10),
    (10,  0.1,          0.1,          10),
    (10,  'asymmetric', 'auto',       20),
    (25,  'symmetric',  'symmetric',  10),
    (25,  'asymmetric', 'auto',       10),
    (25,  0.01,         0.01,         10),
]

# ── Load preprocessed data ────────────────────────────────────────────────────
print("Loading preprocessed data (no_above=0.3)...")
for path in [TOKENS_OUT, DICT_OUT, CORPUS_OUT]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {os.path.basename(path)}\n"
            "Run lda_guitars.py first to generate checkpoints."
        )

with open(TOKENS_OUT, 'rb') as f:
    tokens = pickle.load(f)
dictionary = corpora.Dictionary.load(DICT_OUT)
with open(CORPUS_OUT, 'rb') as f:
    corpus = pickle.load(f)
print(f"  {len(tokens):,} reviews | {len(dictionary):,} terms\n")

# ── Train & evaluate ──────────────────────────────────────────────────────────
results = []

for i, (n_topics, alpha, eta, passes) in enumerate(configs):
    print(f"[{i+1}/{len(configs)}] topics={n_topics} | alpha={alpha} | eta={eta} | passes={passes}")

    lda = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        alpha=alpha,
        eta=eta,
        passes=passes,
        random_state=RANDOM_SEED,
        workers=3
    )

    coherence = CoherenceModel(
        model=lda, texts=tokens, dictionary=dictionary, coherence='c_v'
    ).get_coherence()

    results.append((n_topics, alpha, eta, passes, coherence, lda))
    print(f"  Coherence: {coherence:.4f}\n")

# ── Summary table ─────────────────────────────────────────────────────────────
print("── Summary ─────────────────────────────────────────────────────────")
header = f"{'Topics':>8} | {'Alpha':>12} | {'Eta':>12} | {'Passes':>7} | {'Coherence':>10}"
divider = "-" * len(header)
print(header)
print(divider)
for n, alpha, eta, passes, score, _ in results:
    print(f"{n:>8} | {str(alpha):>12} | {str(eta):>12} | {passes:>7} | {score:>10.4f}")

best = max(results, key=lambda x: x[4])
print(f"\nBest: topics={best[0]} | alpha={best[1]} | eta={best[2]} | passes={best[3]} | coherence={best[4]:.4f}")

# ── Save results ──────────────────────────────────────────────────────────────
lines = ["LDA Hyperparameter Tuning — Guitars (no_above=0.3)\n", "=" * 70 + "\n\n"]
lines.append(header + "\n")
lines.append(divider + "\n")
for n, alpha, eta, passes, score, _ in results:
    lines.append(f"{n:>8} | {str(alpha):>12} | {str(eta):>12} | {passes:>7} | {score:>10.4f}\n")

lines.append(f"\nBest: topics={best[0]} | alpha={best[1]} | eta={best[2]} | passes={best[3]} | coherence={best[4]:.4f}\n")

lines.append("\n\n── Topics for best model ────────────────────────────────────────────\n")
for idx, topic in best[5].print_topics(num_topics=best[0], num_words=12):
    lines.append(f"Topic {idx:02d}: {topic}\n")

with open(RESULTS_OUT, 'w') as f:
    f.writelines(lines)

print(f"\nResults saved → {os.path.basename(RESULTS_OUT)}")
print("Done.")
