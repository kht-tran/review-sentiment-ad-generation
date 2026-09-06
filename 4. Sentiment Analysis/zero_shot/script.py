"""
Zero-shot Sentiment Comparison (Guitars)
Inference-only, no training. Evaluated on the same test split as others
(test_size=0.2, random_state=42, stratify=labels).

  Model A: facebook/bart-large-mnli
  Model B: cross-encoder/nli-deberta-v3-large

Candidate labels: ["positive", "negative"]
Hypothesis template: "The sentiment of this review is {}."
Predict positive if score("positive") >= score("negative").

Checkpoints per-model predictions — safe to resubmit.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NLP_DIR     = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_FILE   = os.path.join(NLP_DIR, 'guitars_with_topics_v2.parquet')
RESULTS_OUT = os.path.join(SCRIPT_DIR, 'results.txt')
PREDICTIONS = os.path.join(SCRIPT_DIR, 'predictions.parquet')

MODELS = {
    'bart':    'facebook/bart-large-mnli',
    'deberta': 'cross-encoder/nli-deberta-v3-large',
}

CANDIDATE_LABELS    = ['positive', 'negative']
HYPOTHESIS_TEMPLATE = 'The sentiment of this review is {}.'
BATCH_SIZE          = 16  # zero-shot is heavier; smaller batches

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

# ── Load & filter ─────────────────────────────────────────────────────────────
print("\nLoading data...")
guitars = pd.read_parquet(DATA_FILE)
guitars = guitars[guitars['text'].notna()]
guitars = guitars[guitars['text'].str.strip() != '']
guitars = guitars[guitars['rating'] != 3].copy()
guitars['label'] = (guitars['rating'] >= 4).astype(int)
print(f"After filtering: {len(guitars):,} reviews")
print(f"Label distribution:\n{guitars['label'].value_counts()}\n")

texts  = guitars['text'].tolist()
labels = guitars['label'].values

# ── Same test split as steps 09/10 ───────────────────────────────────────────
indices = np.arange(len(guitars))
_, texts_test, _, y_test, _, idx_test = train_test_split(
    texts, labels, indices,
    test_size=0.2, random_state=42, stratify=labels
)
print(f"Test set: {len(texts_test):,} reviews\n")

# ── Per-model inference ───────────────────────────────────────────────────────
results = {}
for key, model_name in MODELS.items():
    ckpt_pred = os.path.join(SCRIPT_DIR, f'preds_{key}.npy')
    ckpt_prob = os.path.join(SCRIPT_DIR, f'probs_{key}.npy')

    if os.path.exists(ckpt_pred) and os.path.exists(ckpt_prob):
        print(f"[{key}] Loading cached predictions from checkpoint...")
        y_pred = np.load(ckpt_pred)
        y_prob = np.load(ckpt_prob)
    else:
        print(f"[{key}] Loading model: {model_name}")
        pipe = pipeline(
            'zero-shot-classification',
            model=model_name,
            device=device,
        )
        print(f"[{key}] Running inference on {len(texts_test):,} reviews "
              f"(batch_size={BATCH_SIZE})...")

        preds, probs = [], []
        for i in range(0, len(texts_test), BATCH_SIZE):
            batch = texts_test[i:i + BATCH_SIZE]
            out = pipe(
                batch,
                candidate_labels=CANDIDATE_LABELS,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
                truncation=True,
            )
            for r in out:
                label_scores = dict(zip(r['labels'], r['scores']))
                p_pos = label_scores.get('positive', 0)
                p_neg = label_scores.get('negative', 0)
                preds.append(1 if p_pos >= p_neg else 0)
                probs.append(p_pos)
            if (i // BATCH_SIZE + 1) % 100 == 0:
                done = min(i + BATCH_SIZE, len(texts_test))
                print(f"  {done:,} / {len(texts_test):,}")

        y_pred = np.array(preds)
        y_prob = np.array(probs)
        np.save(ckpt_pred, y_pred)
        np.save(ckpt_prob, y_prob)
        print(f"[{key}] Predictions saved to checkpoint.")
        del pipe

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    rep = classification_report(y_test, y_pred, target_names=['negative', 'positive'])
    results[key] = {
        'name': model_name, 'preds': y_pred, 'probs': y_prob,
        'acc': acc, 'f1': f1, 'report': rep,
    }
    print(f"[{key}] Accuracy: {acc:.4f}  |  Macro F1: {f1:.4f}\n")

# ── Per-topic breakdown ───────────────────────────────────────────────────────
guitar_reset = guitars.reset_index()
test_meta = guitar_reset.loc[idx_test, [
    'index', 'parent_asin', 'rating', 'label', 'topic_id', 'topic_label'
]].copy()
test_meta = test_meta.rename(columns={'index': 'orig_index'})
for key in MODELS:
    test_meta[f'pred_{key}'] = results[key]['preds']
    test_meta[f'prob_{key}'] = results[key]['probs']

# ── Build results file ────────────────────────────────────────────────────────
out_lines = []
out_lines.append("Step 12 — Zero-shot Sentiment Comparison (Guitars)\n")
out_lines.append(f"Candidate labels: {CANDIDATE_LABELS}\n")
out_lines.append(f"Hypothesis template: \"{HYPOTHESIS_TEMPLATE}\"\n")
out_lines.append("=" * 65 + "\n\n")

for key, r in results.items():
    print(f"\n{'='*60}")
    print(f"Model: {r['name']}")
    print(f"Accuracy: {r['acc']:.4f}  |  Macro F1: {r['f1']:.4f}")
    print(r['report'])

    out_lines.append(f"Model: {r['name']}\n")
    out_lines.append(f"  Accuracy:  {r['acc']:.4f}\n")
    out_lines.append(f"  Macro F1:  {r['f1']:.4f}\n\n")
    out_lines.append(f"Classification Report:\n{r['report']}\n")

    topic_rows  = test_meta[test_meta['topic_id'].notna()].copy()
    topic_lines = [f"\nPer-topic breakdown — {key}:\n"]
    topic_lines.append(f"{'Topic':<35} {'N':>6} {'Acc':>6} {'MacroF1':>8}\n")
    topic_lines.append("-" * 60 + "\n")
    print(f"\nPer-topic breakdown — {key}:")

    for (tid, tlabel), grp in topic_rows.groupby(['topic_id', 'topic_label']):
        if len(grp) < 10:
            continue
        t_acc = accuracy_score(grp['label'], grp[f'pred_{key}'])
        t_f1  = f1_score(grp['label'], grp[f'pred_{key}'], average='macro', zero_division=0)
        line  = f"Topic {int(tid):02d} — {tlabel:<28} {len(grp):>6} {t_acc:>6.3f} {t_f1:>8.3f}"
        topic_lines.append(line + "\n")
        print(line)

    out_lines.extend(topic_lines)
    out_lines.append("\n")

# ── Comparison summary ────────────────────────────────────────────────────────
summary = [
    "\n" + "=" * 65 + "\n",
    "COMPARISON SUMMARY\n",
    "=" * 65 + "\n",
    f"{'Model':<45} {'Accuracy':>9} {'Macro F1':>9}\n",
    "-" * 65 + "\n",
]
for key, r in results.items():
    summary.append(f"{r['name']:<45} {r['acc']:>9.4f} {r['f1']:>9.4f}\n")
out_lines.extend(summary)

print("\n" + "".join(summary))

with open(RESULTS_OUT, 'w') as f:
    f.writelines(out_lines)
print(f"Results saved to {RESULTS_OUT}")

# ── Save predictions ──────────────────────────────────────────────────────────
test_meta.to_parquet(PREDICTIONS, index=False)
print(f"Predictions saved to {PREDICTIONS}")
print("\nDone.")
