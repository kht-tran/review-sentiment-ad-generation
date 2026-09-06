# Review-Driven Ad Generation & Persuasion Strategy Evaluation

## Overview
This project explores whether customer reviews can be mined to automatically generate more persuasive ad copy. 

Using 134,000+ Amazon musical instrument reviews, it identifies the product features and pain points customers care about most, then tests whether ads built around those signals are judged more persuasive than generic alternatives, with an LLM acting as evaluator.

*[Completed as part of a two-person team project. Both team members contributed collaboratively across all stages of the analysis.]*

## Approach/Methods
- **Topic modeling**: LDA (Gensim) on guitar reviews to extract 16 instrument-feature topics; BERTopic tested as an alternative but discarded due to high outlier rates
- **Sentiment analysis**: benchmarked 8 modeling approaches per review, i.e. TF-IDF + logistic regression, sentence embeddings + logistic regression, off-the-shelf pretrained BERT models, zero-shot NLI classification, fine-tuned RoBERTa, and LLM few-shot classification (document-level and topic-aware)
- **Aspect-based sentiment analysis (ABSA)**: zero-shot classification applied per topic across all reviews to surface positive/negative sentiment patterns by product feature
- **Ad generation**: LLM-generated ad copy per topic, testing two strategies — amplifying positive sentiment vs. directly addressing common pain points
- **Evaluation**: LLM-as-judge scoring of generated ads, both pointwise and pairwise
- **Robustness check**: full pipeline (topic modeling → sentiment → ad generation) re-run on drums and keyboards review categories to test generalization beyond guitars

## Key Results
- Selected LDA over BERTopic after BERTopic produced ~48% outlier assignments on review text regardless of tuning
- Benchmarked 8 sentiment models spanning classical ML, embeddings, pretrained transformers, zero-shot, and fine-tuned approaches, to identify the most reliable sentiment signal per topic
- Findings held up on both drums and keyboards datasets, suggesting the pipeline generalizes beyond the original guitar category

## Data Note
Built on the Amazon Reviews 2023 dataset (Musical Instruments subset), a public research dataset. Raw review and metadata files are too large to include in this repository. No proprietary or restricted data is used.

## Tools Used
- Python: pandas, numpy, gensim (LDA), BERTopic, scikit-learn, sentence-transformers, HuggingFace Transformers (BERT, RoBERTa, zero-shot NLI models)
- PyTorch (RoBERTa fine-tuning)
- OpenAI/DeepSeek APIs (ad generation & LLM-as-judge)

## Repository Contents
- `0_convert_files/`: file format conversion utilities
- `1_join_datasets/`: merges reviews with product metadata
- `2_eda/`: exploratory data analysis
- `3_topic_modelling/`: LDA training on guitar reviews (BERTopic tested as an alternative, not used in final pipeline)
- `3b_assign_topics/`: topic assignment output at different development stages
- `4_sentiment_analysis/`: 8 benchmarked sentiment modeling approaches, plus model comparison and per-topic sentiment breakdown notebooks
- `5_ad_generation/`: ad copy generation and LLM-as-judge evaluation notebooks
- `extension_drums/`, `extension_keyboards/`: robustness check: full pipeline (topic modeling, sentiment, ad generation) re-run on drums and keyboards reviews
- `environment.yml`: conda environment for reproducing the pipeline
  
Raw data files (review/metadata parquet and JSONL files) are not included.
