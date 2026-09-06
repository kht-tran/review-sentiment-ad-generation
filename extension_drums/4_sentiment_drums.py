"""
RoBERTa Sentiment Fine-tuning — Drums & Percussion Reviews
============================================================
Mirrors Fine-tuned bert/bert_finetune.py exactly.
Trains RoBERTa on rating-based sentiment labels, then runs per-topic inference.

Fixed project parameters: test_size=0.2, random_state=42
Primary metric: Macro F1
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NLP_DIR    = os.path.dirname(SCRIPT_DIR)
DATA_FILE  = os.path.join(NLP_DIR, 'drums_with_topics.parquet')

# ── Fixed project parameters (do not change) ──────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Argument parser ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for drums sentiment analysis")
parser.add_argument("--data",       type=str,   default=DATA_FILE)
parser.add_argument("--model",      type=str,   default="roberta-base")
parser.add_argument("--output_dir", type=str,   default=os.path.join(SCRIPT_DIR, "roberta_drums_model"))
parser.add_argument("--max_length", type=int,   default=256)
parser.add_argument("--batch_size", type=int,   default=16)
parser.add_argument("--epochs",     type=int,   default=3)
parser.add_argument("--lr",         type=float, default=2e-5)
parser.add_argument("--debug",      action="store_true",
                    help="Run on 500 samples only (for quick testing)")
args = parser.parse_args()

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("No GPU found — using CPU")

print(f"PyTorch version : {torch.__version__}")
print(f"Debug mode      : {args.debug}")

# ==============================================================================
# STEP 1 — Load Data & Create Labels
# ==============================================================================
print("\n[1/6] Loading data...")
df = pd.read_parquet(args.data)
print(f"  Loaded {len(df):,} reviews")

def assign_label(rating):
    if rating >= 4:
        return 1
    elif rating <= 2:
        return 0
    else:
        return None

df["label"] = df["rating"].apply(assign_label)
df_labelled = df.dropna(subset=["label"]).copy()
df_labelled["label"] = df_labelled["label"].astype(int)

df_all = df.copy()

print(f"  After dropping 3-star : {len(df_labelled):,}")
print(f"  Positive : {(df_labelled['label']==1).sum():,}  "
      f"({(df_labelled['label']==1).mean()*100:.1f}%)")
print(f"  Negative : {(df_labelled['label']==0).sum():,}  "
      f"({(df_labelled['label']==0).mean()*100:.1f}%)")

if args.debug:
    df_labelled = df_labelled.sample(n=500, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  [DEBUG] Using {len(df_labelled)} samples")

# ==============================================================================
# STEP 2 — Train / Validation / Test Split
# ==============================================================================
print("\n[2/6] Splitting data...")

train_df, test_df = train_test_split(
    df_labelled,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_labelled["label"]
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_df["label"]
)

print(f"  Train : {len(train_df):,}")
print(f"  Val   : {len(val_df):,}")
print(f"  Test  : {len(test_df):,}")

# ==============================================================================
# STEP 3 — Tokenisation
# ==============================================================================
print(f"\n[3/6] Loading tokenizer ({args.model})...")
tokenizer = RobertaTokenizer.from_pretrained(args.model)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.labels    = list(labels)
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = ReviewDataset(train_df["text"], train_df["label"], tokenizer, args.max_length)
val_dataset   = ReviewDataset(val_df["text"],   val_df["label"],   tokenizer, args.max_length)
test_dataset  = ReviewDataset(test_df["text"],  test_df["label"],  tokenizer, args.max_length)

print(f"  Train dataset : {len(train_dataset):,} samples")
print(f"  Val dataset   : {len(val_dataset):,} samples")
print(f"  Test dataset  : {len(test_dataset):,} samples")

# ==============================================================================
# STEP 4 — Load Model
# ==============================================================================
print(f"\n[4/6] Loading model ({args.model})...")
model = RobertaForSequenceClassification.from_pretrained(
    args.model,
    num_labels=2
)
model.to(DEVICE)
print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy" : accuracy_score(labels, predictions),
        "macro_f1" : f1_score(labels, predictions, average="macro"),
    }

# ==============================================================================
# STEP 5 — Training
# ==============================================================================
print("\n[5/6] Training...")

training_args = TrainingArguments(
    output_dir                  = args.output_dir,
    num_train_epochs            = args.epochs,
    per_device_train_batch_size = args.batch_size,
    per_device_eval_batch_size  = args.batch_size * 2,
    learning_rate               = args.lr,
    weight_decay                = 0.01,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "macro_f1",
    greater_is_better           = True,
    logging_steps               = 50,
    report_to                   = "none",
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"  Model saved to: {args.output_dir}")

# ==============================================================================
# STEP 6 — Evaluate on Test Set
# ==============================================================================
print("\n[6/6] Evaluating on test set...")

model.eval()
all_preds  = []
all_labels = []

dataloader = DataLoader(test_dataset, batch_size=32)
with torch.no_grad():
    for batch in dataloader:
        labels = batch.pop("labels")
        outputs = model(**{k: v.to(DEVICE) for k, v in batch.items()})
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

test_accuracy = accuracy_score(all_labels, all_preds)
test_macro_f1 = f1_score(all_labels, all_preds, average="macro")

print("\n" + "=" * 50)
print("TEST SET RESULTS")
print("=" * 50)
print(f"  Accuracy : {test_accuracy:.4f}")
print(f"  Macro F1 : {test_macro_f1:.4f}")
print("\nDetailed Report:")
print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

# ==============================================================================
# STEP 7 — Per-Topic Sentiment Inference
# ==============================================================================
print("Running per-topic inference on full dataset...")

inference_df = df_all.copy()
if args.debug:
    inference_df = inference_df.sample(n=300, random_state=RANDOM_STATE).reset_index(drop=True)

infer_dataset = ReviewDataset(
    inference_df["text"],
    [-1] * len(inference_df),
    tokenizer,
    args.max_length
)

model.eval()
all_infer_preds = []
all_infer_probs = []

infer_loader = DataLoader(infer_dataset, batch_size=64)
with torch.no_grad():
    for batch in infer_loader:
        batch.pop("labels")
        outputs = model(**{k: v.to(DEVICE) for k, v in batch.items()})
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_infer_preds.extend(preds.cpu().numpy())
        all_infer_probs.extend(probs[:, 1].cpu().numpy())

inference_df["pred_label"]     = all_infer_preds
inference_df["pred_pos_prob"]  = all_infer_probs
inference_df["pred_sentiment"] = inference_df["pred_label"].map({1: "positive", 0: "negative"})

topic_sentiment = (
    inference_df
    .groupby("topic_label")["pred_sentiment"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .rename(columns={"positive": "pos_rate", "negative": "neg_rate"})
    .sort_values("pos_rate", ascending=False)
)
topic_sentiment["avg_pos_prob"] = inference_df.groupby("topic_label")["pred_pos_prob"].mean()
topic_sentiment["count"]        = inference_df.groupby("topic_label").size()

print("\n=== Per-Topic Sentiment Distribution ===")
print(topic_sentiment.round(3).to_string())

inference_df.to_csv(os.path.join(SCRIPT_DIR, "drums_bert_predictions.csv"), index=False)
topic_sentiment.to_csv(os.path.join(SCRIPT_DIR, "drums_bert_topic_sentiment.csv"))
print("\nSaved: drums_bert_predictions.csv")
print("Saved: drums_bert_topic_sentiment.csv")
print("\nDone!")
