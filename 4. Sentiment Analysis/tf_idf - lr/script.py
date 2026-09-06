"""
TF-IDF + Logistic Regression (Guitars subset)
Document-level binary sentiment classification with grid search.
Loads guitars_with_topics.parquet from step08. Reports per-topic breakdown.
"""

import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NLP_DIR     = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_FILE   = os.path.join(NLP_DIR, 'guitars_with_topics_v2.parquet')
RESULTS_OUT = os.path.join(SCRIPT_DIR, 'results.txt')
PREDICTIONS = os.path.join(SCRIPT_DIR, 'predictions.parquet')

# ── Load & filter ─────────────────────────────────────────────────────────────
print("Loading guitars_with_topics.parquet...")
guitars = pd.read_parquet(DATA_FILE)
print(f"Loaded: {len(guitars):,} reviews")

guitars = guitars[guitars['text'].notna()]
guitars = guitars[guitars['text'].str.strip() != '']
guitars = guitars[guitars['rating'] != 3].copy()
guitars['label'] = (guitars['rating'] >= 4).astype(int)

print(f"After filtering (no rating=3): {len(guitars):,}")
print(f"Label distribution:\n{guitars['label'].value_counts()}\n")

# ── Train/test split ──────────────────────────────────────────────────────────
X = guitars['text']
y = guitars['label']

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, guitars.index,
    test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

# ── Grid search ───────────────────────────────────────────────────────────────
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english', min_df=5)),
    ('clf',   LogisticRegression(max_iter=1000, n_jobs=-1))
])

param_grid = {
    'tfidf__max_features': [30_000, 50_000, 100_000],
    'tfidf__ngram_range':  [(1, 1), (1, 2)],
    'tfidf__max_df':       [0.85, 0.95],
    'clf__C':              [0.1, 1.0, 10.0],
}

print(f"Grid search: {3*2*2*3} combinations × 5 folds = 180 fits")
print("Running grid search (scoring=macro F1)...")
t0 = time.time()

gs = GridSearchCV(
    pipe, param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1,
)
gs.fit(X_train, y_train)

print(f"Grid search done in {(time.time()-t0)/60:.1f} min")
print(f"Best params:      {gs.best_params_}")
print(f"Best CV macro F1: {gs.best_score_:.4f}\n")

# ── Evaluate best model on test set ──────────────────────────────────────────
best   = gs.best_estimator_
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
report   = classification_report(y_test, y_pred, target_names=['negative', 'positive'])

print(f"Test Accuracy:  {accuracy:.4f}")
print(f"Test Macro F1:  {macro_f1:.4f}")
print(f"\nClassification Report:\n{report}")

# ── Per-topic breakdown ───────────────────────────────────────────────────────
pred_df = guitars.loc[idx_test, ['parent_asin', 'rating', 'label', 'topic_id', 'topic_label']].copy()
pred_df['pred']      = y_pred
pred_df['pred_prob'] = y_prob
pred_df['correct']   = (pred_df['pred'] == pred_df['label']).astype(int)

topic_rows = pred_df[pred_df['topic_id'].notna()].copy()
topic_lines = ["\nPer-topic breakdown (test set, assigned reviews only):\n"]
topic_lines.append(f"{'Topic':<35} {'N':>6} {'Acc':>6} {'MacroF1':>8}\n")
topic_lines.append("-" * 60 + "\n")

for (tid, tlabel), grp in topic_rows.groupby(['topic_id', 'topic_label']):
    if len(grp) < 10:
        continue
    t_acc = accuracy_score(grp['label'], grp['pred'])
    t_f1  = f1_score(grp['label'], grp['pred'], average='macro', zero_division=0)
    line  = f"Topic {int(tid):02d} — {tlabel:<28} {len(grp):>6} {t_acc:>6.3f} {t_f1:>8.3f}"
    topic_lines.append(line + "\n")
    print(line)

# ── Save results ──────────────────────────────────────────────────────────────
with open(RESULTS_OUT, 'w') as f:
    f.write("TF-IDF + Logistic Regression (Guitars)\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Best params:      {gs.best_params_}\n")
    f.write(f"Best CV macro F1: {gs.best_score_:.4f}\n\n")
    f.write(f"Test Accuracy:    {accuracy:.4f}\n")
    f.write(f"Test Macro F1:    {macro_f1:.4f}\n\n")
    f.write(f"Classification Report:\n{report}\n")
    f.writelines(topic_lines)

print(f"\nResults saved to {RESULTS_OUT}")

# ── Save predictions ──────────────────────────────────────────────────────────
pred_df.to_parquet(PREDICTIONS, index=True)
print(f"Predictions saved to {PREDICTIONS}")
print("\nDone.")
