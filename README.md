# Review-Driven Ad Generation & Persuasion Strategy Evaluation

## Overview
This project explores whether customer reviews can be mined to automatically generate more persuasive ad copy. Using 134,000+ Amazon musical instrument reviews, it identifies the product features and pain points customers care about most, then tests whether ads built around those signals are judged more persuasive than generic alternatives — with an LLM acting as evaluator.

## Approach/Methods
- **Topic modeling**: LDA (Gensim) on guitar reviews to extract 16 instrument-feature topics; BERTopic tested as an alternative but discarded due to high outlier rates
- **Sentiment analysis**: benchmarked 8 modeling approaches per review — TF-IDF + logistic regression, sentence embeddings + logistic regression, off-the-shelf pretrained BERT models, zero-shot NLI classification, fine-tuned RoBERTa, and LLM few-shot classification (document-level and topic-aware)
- **Aspect-based sentiment analysis (ABSA)**: zero-shot classification applied per topic across all reviews to surface positive/negative sentiment patterns by product feature
- **Ad generation**: LLM-generated ad copy per topic, testing two strategies — amplifying positive sentiment vs. directly addressing common pain points
- **Evaluation**: LLM-as-judge scoring of generated ads, both pointwise and pairwise
- **Robustness check**: full pipeline (topic modeling → sentiment → ad generation) re-run on drums and keyboards review categories to test generalization beyond guitars

## Key Results
- Selected LDA over BERTopic after BERTopic produced ~48% outlier assignments on review text regardless of tuning
- Benchmarked 8 sentiment models spanning classical ML, embeddings, pretrained transformers, zero-shot, and fine-tuned approaches, to identify the most reliable sentiment signal per topic
- Findings held up on both drums and keyboards datasets, suggesting the pipeline generalizes beyond the original guitar category

## Data Note
Built on the Amazon Reviews 2023 dataset (Musical Instruments subset), a public research dataset. Raw review and metadata files are too large to include in this repository; download instructions are provided in the code. No proprietary or restricted data is used.

## Tools Used
Python — pandas, numpy, gensim (LDA), BERTopic, scikit-learn, sentence-transformers, HuggingFace Transformers (BERT, RoBERTa, zero-shot NLI models), PyTorch (RoBERTa fine-tuning), OpenAI/DeepSeek APIs (ad generation & LLM-as-judge)

## Repository Contents
- `01_join_datasets/`: merges reviews with product metadata
- `02_topic_modeling/`: LDA training and topic assignment (guitars)
- `03_sentiment_analysis/`: 8 benchmarked sentiment modeling approaches, plus model comparison and per-topic sentiment breakdown notebooks
- `04_ad_generation/`: ad copy generation and LLM-as-judge evaluation notebooks
- `05_extensions_drums_keyboards/`: robustness check pipeline (topic modeling, sentiment, ad generation) applied to drums and keyboards reviews
