"""
Extension — LDA Hyperparameter Tuning for Drums & Percussion
=============================================================
Mirrors lda_tune_params_v2.py exactly — same 8-config grid, same evaluation.
Reuses drums_tokens.pkl/drums_dict.pkl/drums_corpus.pkl from preprocess_drums.py.

Output: tune_results_drums.txt
  → Read this file to find best (num_topics, alpha, eta, passes)
  → Copy those values into assign_topics_drums.py
"""

import os
import pickle
import gensim.corpora as corpora
from gensim.models import LdaMulticore, CoherenceModel

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TOKENS_FILE = os.path.join(SCRIPT_DIR, 'drums_tokens.pkl')
DICT_FILE   = os.path.join(SCRIPT_DIR, 'drums_dict.pkl')
CORPUS_FILE = os.path.join(SCRIPT_DIR, 'drums_corpus.pkl')
RESULTS_OUT = os.path.join(SCRIPT_DIR, 'tune_results_drums.txt')

RANDOM_SEED = 42

# ── Parameter grid (identical to lda_tune_params_v2.py) ──────────────────────
configs = [
    # (num_topics, alpha,        eta,          passes)
    (10,  'symmetric',  'symmetric',   10),
    (10,  'asymmetric', 'auto',        10),
    (10,  0.01,         0.01,          10),
    (10,  0.1,          0.1,           10),
    (10,  'asymmetric', 'auto',        20),
    (25,  'symmetric',  'symmetric',   10),
    (25,  'asymmetric', 'auto',        10),
    (25,  0.01,         0.01,          10),
]

# ── Load preprocessed data ───────────────────────────────────────────────────
print("Loading preprocessed drums data...")
for path in [TOKENS_FILE, DICT_FILE, CORPUS_FILE]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {os.path.basename(path)}\n"
            "Run preprocess_drums.py first."
        )

with open(TOKENS_FILE, 'rb') as f:
    tokens = pickle.load(f)
dictionary = corpora.Dictionary.load(DICT_FILE)
with open(CORPUS_FILE, 'rb') as f:
    corpus = pickle.load(f)
print(f"  {len(tokens):,} reviews | {len(dictionary):,} terms\n")

# ── Train & evaluate each config ─────────────────────────────────────────────
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
        workers=3,
    )

    coherence = CoherenceModel(
        model=lda, texts=tokens, dictionary=dictionary, coherence='c_v'
    ).get_coherence()

    results.append((n_topics, alpha, eta, passes, coherence, lda))
    print(f"  Coherence: {coherence:.4f}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print("── Summary ─────────────────────────────────────────────────────────")
header  = f"{'Topics':>8} | {'Alpha':>12} | {'Eta':>12} | {'Passes':>7} | {'Coherence':>10}"
divider = "-" * len(header)
print(header)
print(divider)
for n, alpha, eta, passes, score, _ in results:
    print(f"{n:>8} | {str(alpha):>12} | {str(eta):>12} | {passes:>7} | {score:>10.4f}")

best = max(results, key=lambda x: x[4])
print(f"\nBest: topics={best[0]} | alpha={best[1]} | eta={best[2]} | passes={best[3]} | coherence={best[4]:.4f}")

# ── Save results ──────────────────────────────────────────────────────────────
lines = ["LDA Hyperparameter Tuning — Drums & Percussion\n", "=" * 70 + "\n\n"]
lines.append("Copy the BEST row values into assign_topics_drums.py:\n")
lines.append("  NUM_TOPICS = <n>  |  ALPHA = '<alpha>'  |  ETA = '<eta>'  |  PASSES = <p>\n\n")
lines.append(header + "\n")
lines.append(divider + "\n")
for n, alpha, eta, passes, score, _ in results:
    lines.append(f"{n:>8} | {str(alpha):>12} | {str(eta):>12} | {passes:>7} | {score:>10.4f}\n")

lines.append(f"\nBest: topics={best[0]} | alpha={best[1]} | eta={best[2]} | passes={best[3]} | coherence={best[4]:.4f}\n")

lines.append("\n\n── Topics for best model (use these to choose FINAL_TOPICS) ──────────\n")
for idx, topic in best[5].print_topics(num_topics=best[0], num_words=15):
    lines.append(f"Topic {idx:02d}: {topic}\n")

with open(RESULTS_OUT, 'w') as f:
    f.writelines(lines)

print(f"\nResults saved → {os.path.basename(RESULTS_OUT)}")
print("\nNEXT STEP:")
print("  1. scp tune_results_drums.txt to your laptop")
print("  2. Read it — find best params, read topic words for best model")
print("  3. Fill in NUM_TOPICS/ALPHA/ETA/PASSES + FINAL_TOPICS in assign_topics_drums.py")
print("  4. sbatch run_assign_drums.sh")
print("Done.")
