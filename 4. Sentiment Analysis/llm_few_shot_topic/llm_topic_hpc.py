"""
LLM Few-Shot — Topic-Level Sentiment Analysis (HPC Full Script)
===============================================================
Supports checkpoint/resume: saves progress every 1000 samples.
If the job is killed and resubmitted, it resumes from the last checkpoint.

Usage:
    python llm_topic_hpc.py --data /path/to/guitars_with_topics_v2.parquet
    python llm_topic_hpc.py --data /path/to/guitars_with_topics_v2.parquet --debug
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── Few-shot examples ──────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        'topic':  'Shipping damage',
        'review': 'The guitar sounds absolutely incredible, but it arrived with a cracked headstock '
                  'and the box was completely crushed. Clearly it was not packaged properly.',
        'label':  'negative'
    },
    {
        'topic':  'Acoustic tone',
        'review': 'The guitar sounds absolutely incredible, but it arrived with a cracked headstock '
                  'and the box was completely crushed. Clearly it was not packaged properly.',
        'label':  'positive'
    },
    {
        'topic':  'Tuning stability',
        'review': 'I have been playing this guitar every day for three months and it holds its tuning '
                  'remarkably well. Even after aggressive strumming it stays in tune.',
        'label':  'positive'
    },
    {
        'topic':  'Setup / action',
        'review': 'The action is extremely high right out of the box. It is very hard to press the strings '
                  'down, especially on the higher frets. Needs a full professional setup before it is playable.',
        'label':  'negative'
    },
    {
        'topic':  'Visual appearance',
        'review': 'Very disappointed with this purchase. The frets are sharp and the action is too high. '
                  'However, I have to admit it looks stunning — the sunburst finish is beautiful.',
        'label':  'positive'
    },
]

SYSTEM_PROMPT = (
    'You are a sentiment classifier specializing in guitar product reviews. '
    'You will be given a review and a specific topic. '
    'Your task is to determine the sentiment expressed in the review SPECIFICALLY ABOUT that topic. '
    'Ignore the overall tone of the review — focus only on what the reviewer says about the given topic. '
    'You MUST respond with exactly one word: positive or negative. '
    'No explanation. No punctuation. Just one word.'
)

# Checkpoint save interval (number of samples)
CHECKPOINT_INTERVAL = 1000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       type=str, required=True)
    parser.add_argument('--model',      type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--debug',      action='store_true')
    return parser.parse_args()


def load_and_prepare_data(data_path: str, debug: bool) -> pd.DataFrame:
    print(f'Loading data from: {data_path}')
    df_raw = pd.read_parquet(data_path)
    print(f'Raw dataset: {len(df_raw):,} rows')

    df = df_raw.dropna(subset=['topic_label', 'text']).copy()
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 10].reset_index(drop=True)
    print(f'After filtering: {len(df):,} rows')
    print('\nTopic distribution:')
    print(df['topic_label'].value_counts().to_string())

    if debug:
        parts = [
            group.sample(n=min(10, len(group)), random_state=42)
            for _, group in df.groupby('topic_label')
        ]
        df = pd.concat(parts).reset_index(drop=True)
        print(f'\nDEBUG MODE: {len(df)} reviews')

    return df


def build_prompt(review_text: str, topic: str) -> str:
    lines = [SYSTEM_PROMPT, '']
    for ex in FEW_SHOT_EXAMPLES:
        ex_text = ex['review'][:300] + ('...' if len(ex['review']) > 300 else '')
        lines.append(f'Topic: {ex["topic"]}')
        lines.append(f'Review: {ex_text}')
        lines.append(f'Sentiment about "{ex["topic"]}": {ex["label"]}')
        lines.append('')
    truncated = review_text[:400] + ('...' if len(review_text) > 400 else '')
    lines.append(f'Topic: {topic}')
    lines.append(f'Review: {truncated}')
    lines.append(f'Sentiment about "{topic}":')
    return '\n'.join(lines)


def parse_label(raw: str) -> str:
    text = raw.strip().lower()
    first = re.split(r'[^a-z]', text)[0]
    if first in ('positive', 'pos'): return 'positive'
    if first in ('negative', 'neg'): return 'negative'
    if 'positive' in text: return 'positive'
    if 'negative' in text: return 'negative'
    return 'N/A'


def load_model(model_id: str):
    print(f'\nLoading model: {model_id}')
    print(f'Device: cuda  |  GPU count: {torch.cuda.device_count()}')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    model.eval()
    print('Model loaded successfully.')
    return model, tokenizer


def run_inference_with_checkpoint(df: pd.DataFrame, model, tokenizer,
                                  batch_size: int, checkpoint_path: str) -> list[str]:
    """
    Run inference with periodic checkpointing.
    If a checkpoint exists, resume from where it left off.
    """
    device = next(model.parameters()).device
    texts  = df['text'].tolist()
    topics = df['topic_label'].tolist()
    total  = len(texts)

    # ── Load checkpoint if it exists ──────────────────────────────────────────
    if os.path.exists(checkpoint_path):
        print(f'Checkpoint found: {checkpoint_path}')
        ckpt = pd.read_csv(checkpoint_path)
        all_labels  = ckpt['pred_sentiment'].tolist()
        start_idx   = len(all_labels)
        print(f'Resuming from sample {start_idx}/{total}')
    else:
        all_labels = []
        start_idx  = 0
        print('No checkpoint found, starting from scratch.')

    # ── Run inference from start_idx ──────────────────────────────────────────
    prompts = [build_prompt(t, top) for t, top in zip(texts[start_idx:], topics[start_idx:])]

    for i in tqdm(range(0, len(prompts), batch_size),
                  desc='Inference', initial=start_idx // batch_size,
                  total=total // batch_size):
        batch = prompts[i: i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        for output in outputs:
            raw = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            all_labels.append(parse_label(raw))

        # Save checkpoint every CHECKPOINT_INTERVAL samples
        global_idx = start_idx + i + len(batch)
        if global_idx % CHECKPOINT_INTERVAL < batch_size:
            ckpt_df = df.iloc[:len(all_labels)].copy()
            ckpt_df['pred_sentiment'] = all_labels
            ckpt_df[['topic_label', 'text', 'pred_sentiment']].to_csv(
                checkpoint_path, index=False
            )
            print(f'  Checkpoint saved at sample {len(all_labels)}/{total}')

    return all_labels


def compute_topic_summary(df: pd.DataFrame) -> pd.DataFrame:
    valid_df = df[df['pred_sentiment'] != 'N/A'].copy()
    rows = []
    for topic, group in valid_df.groupby('topic_label'):
        counts = group['pred_sentiment'].value_counts()
        pos    = counts.get('positive', 0)
        neg    = counts.get('negative', 0)
        total  = pos + neg
        rows.append({
            'topic':         topic,
            'total_reviews': total,
            'n_positive':    pos,
            'n_negative':    neg,
            'pct_positive':  round(100 * pos / total, 1) if total > 0 else None,
            'pct_negative':  round(100 * neg / total, 1) if total > 0 else None,
        })
    return pd.DataFrame(rows).sort_values('pct_negative', ascending=False)


def main():
    args = parse_args()
    start_total = time.time()

    df = load_and_prepare_data(args.data, args.debug)
    model, tokenizer = load_model(args.model)

    out_dir         = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(out_dir, 'llm_topic_checkpoint.csv')

    print(f'\nRunning inference on {len(df):,} reviews (batch_size={args.batch_size})...')
    print(f'Checkpoint interval: every {CHECKPOINT_INTERVAL} samples → {checkpoint_path}')
    start_inf = time.time()

    pred_labels = run_inference_with_checkpoint(
        df, model, tokenizer, args.batch_size, checkpoint_path
    )

    elapsed_inf = time.time() - start_inf
    print(f'\nInference done in {elapsed_inf/60:.1f} min '
          f'({elapsed_inf/len(df):.2f}s per sample)')

    df['pred_sentiment'] = pred_labels

    # N/A report
    na_count = (df['pred_sentiment'] == 'N/A').sum()
    print(f'N/A responses: {na_count}/{len(df)} ({100*na_count/len(df):.1f}%)')

    # Topic summary
    topic_summary = compute_topic_summary(df)
    print('\nPer-topic sentiment summary:')
    print(topic_summary.to_string(index=False))

    # Save final results
    pred_path = os.path.join(out_dir, 'llm_topic_predictions.csv')
    df[['topic_label', 'text', 'pred_sentiment']].to_csv(pred_path, index=False)
    print(f'\nPredictions saved → {pred_path}')

    summary_path = os.path.join(out_dir, 'llm_topic_sentiment.csv')
    topic_summary.to_csv(summary_path, index=False)
    print(f'Topic summary saved → {summary_path}')

    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print('Checkpoint file removed (job completed successfully).')

    print(f'\nTotal time: {(time.time() - start_total)/60:.1f} min')


if __name__ == '__main__':
    main()
