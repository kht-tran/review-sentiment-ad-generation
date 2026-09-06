"""
LLM Few-Shot Sentiment Analysis — HPC Full Script
==================================================
Uses Mistral-7B-Instruct-v0.3 via HuggingFace transformers pipeline.
Runs on the full test set (test_size=0.2, random_state=42).

Usage:
    python llm_fewshot_hpc.py --data /path/to/guitars_with_topics_v2.parquet
    python llm_fewshot_hpc.py --data /path/to/guitars_with_topics_v2.parquet --debug
"""

import argparse
import os
import re
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── Fixed project parameters — DO NOT CHANGE ──────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Few-shot examples (same as local debug notebook) ──────────────────────────
# These are loaded from the training set at runtime using fixed indices.
# Positive indices: [34991, 34219, 4328]
# Negative indices: [92221, 52135]
POS_INDICES = [34991, 34219, 4328]
NEG_INDICES = [92221, 52135]

# ── System prompt (same as local debug notebook) ───────────────────────────────
SYSTEM_PROMPT = (
    'You are a sentiment classifier for guitar product reviews. '
    'Your task is to classify each review as either "positive" or "negative". '
    'You MUST respond with exactly one word — either positive or negative. '
    'If the review is mostly positive, respond: positive. '
    'If the review is mostly negative, respond: negative. '
    'Never respond with anything else. No explanation. No punctuation. Just one word.'
)


def parse_args():
    parser = argparse.ArgumentParser(description='LLM few-shot sentiment analysis')
    parser.add_argument('--data',       type=str, required=True,
                        help='Path to guitars_with_topics_v2.parquet')
    parser.add_argument('--model',      type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.3',
                        help='HuggingFace model ID')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Inference batch size')
    parser.add_argument('--debug',      action='store_true',
                        help='Debug mode: run on 200 samples only')
    return parser.parse_args()


def load_and_prepare_data(data_path: str):
    """Load parquet, build sentiment labels, return cleaned DataFrame."""
    print(f'Loading data from: {data_path}')
    df_raw = pd.read_parquet(data_path)
    print(f'Raw dataset: {len(df_raw):,} rows')

    # Detect column names
    rating_col = 'overall' if 'overall' in df_raw.columns else 'rating'
    text_col   = next(c for c in ['reviewText', 'review_text', 'text'] if c in df_raw.columns)
    print(f'Using columns: rating="{rating_col}", text="{text_col}"')

    # Build labels: 4-5 → positive, 1-2 → negative, 3 → dropped
    df = df_raw[df_raw[rating_col] != 3].copy()
    df['sentiment'] = df[rating_col].apply(lambda r: 'positive' if r >= 4 else 'negative')
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    df['text'] = df[text_col].astype(str).str.strip()

    print(f'After cleaning: {len(df):,} reviews')
    print(df['sentiment'].value_counts().to_string())
    return df


def build_few_shot_examples(train_df: pd.DataFrame) -> list[dict]:
    """Load few-shot examples from training set using fixed indices."""
    examples = []
    for idx in POS_INDICES:
        examples.append({'text': train_df.loc[idx, 'text'], 'label': 'positive'})
    for idx in NEG_INDICES:
        examples.append({'text': train_df.loc[idx, 'text'], 'label': 'negative'})
    print(f'Loaded {len(examples)} few-shot examples from training set.')
    return examples


def build_prompt(review_text: str, few_shot_examples: list[dict]) -> str:
    """Build a few-shot prompt for a single review."""
    lines = [SYSTEM_PROMPT, '']
    for ex in few_shot_examples:
        ex_text = ex['text'][:300] + ('...' if len(ex['text']) > 300 else '')
        lines.append(f'Review: {ex_text}')
        lines.append(f'Sentiment: {ex["label"]}')
        lines.append('')
    truncated = review_text[:400] + ('...' if len(review_text) > 400 else '')
    lines.append(f'Review: {truncated}')
    lines.append('Sentiment:')
    return '\n'.join(lines)


def parse_label(raw_output: str) -> str:
    """Extract positive / negative from model output."""
    text = raw_output.strip().lower()
    first_word = re.split(r'[^a-z/]', text)[0]
    if first_word in ('positive', 'pos'):
        return 'positive'
    if first_word in ('negative', 'neg'):
        return 'negative'
    if 'positive' in text:
        return 'positive'
    if 'negative' in text:
        return 'negative'
    return 'N/A'


def load_model(model_id: str):
    """Load Mistral model and tokenizer onto GPU."""
    print(f'\nLoading model: {model_id}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}  |  GPU count: {torch.cuda.device_count()}')

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'   # for decoder-only models

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,    # half precision to fit in GPU memory
        device_map='auto',            # automatically spread across available GPUs
    )
    model.eval()
    print('Model loaded successfully.')
    return model, tokenizer, device


def run_inference_batch(texts: list[str], few_shot_examples: list[dict],
                        model, tokenizer, device: str, batch_size: int) -> list[str]:
    """
    Run inference on a list of review texts.
    Returns a list of parsed labels (positive / negative / N/A).
    """
    all_labels = []
    prompts    = [build_prompt(t, few_shot_examples) for t in texts]

    for i in tqdm(range(0, len(prompts), batch_size), desc='Inference'):
        batch_prompts = prompts[i: i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,      # we only need one word
                do_sample=False,       # greedy decoding for reproducibility
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (not the input prompt)
        input_lengths = inputs['input_ids'].shape[1]
        for output in outputs:
            new_tokens = output[input_lengths:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            all_labels.append(parse_label(raw))

    return all_labels


def compute_topic_summary(eval_df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute per-topic sentiment breakdown if a topic column exists."""
    topic_col = next(
        (c for c in ['topic', 'Topic', 'main_topic', 'assigned_topic'] if c in eval_df.columns),
        None
    )
    if topic_col is None:
        print('No topic column found — skipping topic summary.')
        return None

    valid_df = eval_df[eval_df['pred_sentiment'] != 'N/A'].copy()
    counts   = valid_df.groupby([topic_col, 'pred_sentiment']).size().unstack(fill_value=0)

    # Compute percentages
    totals   = counts.sum(axis=1)
    pct      = counts.div(totals, axis=0).rename(
        columns={'positive': 'pct_positive', 'negative': 'pct_negative'}
    )
    pct['total_reviews'] = totals
    pct = pct.reset_index().rename(columns={topic_col: 'topic'})

    if 'pct_negative' in pct.columns:
        pct = pct.sort_values('pct_negative', ascending=False)

    return pct


def main():
    args = parse_args()
    start_total = time.time()

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_and_prepare_data(args.data)

    train_df, test_df = train_test_split(
        df,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = df['sentiment']
    )
    print(f'\nTrain: {len(train_df):,}  |  Test: {len(test_df):,}')

    if args.debug:
        eval_df = test_df.sample(n=200, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f'DEBUG MODE: evaluating on {len(eval_df)} samples')
    else:
        eval_df = test_df.reset_index(drop=True)
        print(f'FULL MODE: evaluating on {len(eval_df):,} samples')

    # ── Few-shot examples ─────────────────────────────────────────────────────
    few_shot_examples = build_few_shot_examples(train_df)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer, device = load_model(args.model)

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f'\nRunning inference (batch_size={args.batch_size})...')
    start_inf = time.time()

    pred_labels = run_inference_batch(
        texts              = eval_df['text'].tolist(),
        few_shot_examples  = few_shot_examples,
        model              = model,
        tokenizer          = tokenizer,
        device             = device,
        batch_size         = args.batch_size,
    )

    elapsed_inf = time.time() - start_inf
    print(f'Inference done in {elapsed_inf/60:.1f} min  '
          f'({elapsed_inf/len(eval_df):.2f}s per sample)')

    eval_df['pred_sentiment'] = pred_labels

    # ── Evaluate ──────────────────────────────────────────────────────────────
    na_count  = (eval_df['pred_sentiment'] == 'N/A').sum()
    na_pct    = 100 * na_count / len(eval_df)
    valid_df  = eval_df[eval_df['pred_sentiment'] != 'N/A']

    print(f'\nN/A responses: {na_count}/{len(eval_df)} ({na_pct:.1f}%)')

    if len(valid_df) > 0:
        y_true    = valid_df['sentiment']
        y_pred    = valid_df['pred_sentiment']
        macro_f1  = f1_score(y_true, y_pred, average='macro',
                             labels=['positive', 'negative'])
        print(f'Macro F1 (valid predictions): {macro_f1:.4f}')
        print()
        print(classification_report(y_true, y_pred, labels=['positive', 'negative']))

    # ── Topic summary ─────────────────────────────────────────────────────────
    topic_summary = compute_topic_summary(eval_df)
    if topic_summary is not None:
        print('\nPer-topic sentiment:')
        print(topic_summary.to_string(index=False))

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))

    pred_path  = os.path.join(out_dir, 'llm_predictions.csv')
    eval_df.to_csv(pred_path, index=False)
    print(f'\nPredictions saved → {pred_path}')

    if topic_summary is not None:
        topic_path = os.path.join(out_dir, 'llm_topic_sentiment.csv')
        topic_summary.to_csv(topic_path, index=False)
        print(f'Topic summary saved → {topic_path}')

    total_elapsed = time.time() - start_total
    print(f'\nTotal time: {total_elapsed/60:.1f} min')


if __name__ == '__main__':
    main()
