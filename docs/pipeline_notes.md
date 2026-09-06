# Pipeline Reference
Technical/internal reference documenting what each folder contains and what it produces.
 
All scripts were originally run via SLURM on an academic HPC cluster (`sbatch script.sh`), with checkpointing so jobs could be safely resubmitted if interrupted. The SLURM job wrapper (`.sh`) files themselves are cluster-specific and not included in this repo. Every script below can be run directly as a plain Python command, most of which take no command-line arguments (parameters hardcoded in the `.py` files); the two exceptions are noted in sections 4f and 4g below.

## Environment

Conda environment: `environment.yml`.
```bash
conda env create -f environment.yml
conda activate nlp_env
```

## HuggingFace Models

Compute nodes had no internet access, so models had to be downloaded on the login node before submitting jobs (SLURM scripts set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`):
- `all-mpnet-base-v2` — sentence embeddings
- `facebook/bart-large-mnli`: zero-shot classification
- `cross-encoder/nli-deberta-v3-large`: zero-shot classification
- `nlptown/bert-base-multilingual-uncased-sentiment`: pretrained sentiment
- `cardiffnlp/twitter-roberta-base-sentiment-latest`: pretrained sentiment
- `roberta-base`: base model for fine-tuning

---

## Data Files (root)

| File | Description |
|---|---|
| `Musical_Instruments.jsonl.gz` | Raw reviews (compressed) |
| `Musical_Instruments.parquet` | Raw reviews converted to parquet (output of step 0) |
| `Musical_Instruments_joined.parquet` | Reviews + metadata joined (main working file) |
| `full-00000-of-00002.parquet`,<br>`full-00001-of-00002.parquet` | Product metadata (two parts) |
| `meta_Musical_Instruments.jsonl` | Metadata in JSONL format |
| `guitars_with_topics_v2.parquet` | Guitar reviews with assigned LDA topics (copy of `3.2. Assign/`) |

Not included in the GitHub repo — see Data Note in `README.md`.

---

## 0. Convert files

**Folder:** `0. Convert files/`

Runs once, before anything else. Converts the raw `Musical_Instruments.jsonl.gz` into `Musical_Instruments.parquet` so every downstream step can load from parquet instead of re-parsing gzipped JSON each time.

```bash
python convert.py
```

Output: `Musical_Instruments.parquet` (saved one level up, at root)

---

## 1. Join 2 datasets

**Folder:** `1. Join 2 datasets/`

```bash
sbatch run.sh    # runs join.py
```

Joins the reviews parquet with both metadata parquet files on `parent_asin`, adds a `subcategory` column, and saves the result.

Output: `Musical_Instruments_joined.parquet` (saved one level up, at root)

`check_join.py` is a separate validation script - run it locally after the join to confirm match rates and row counts.

---

## 2. EDA

**Folder:** `2. EDA/`

Single notebook, `EDA_overview.ipynb`, run locally on `Musical_Instruments_joined.parquet`. 

Covers data quality, verified purchase behavior, rating and text-length distributions, and a per-subcategory scorecard (review count, review length, sentiment balance) used to select **guitars** as the subcategory for the topic-modeling and sentiment pipeline.

---

## 3. Topic Modelling (Guitars, LDA)

**Folder:** `3. Topic modelling/`

### Step 1: Tune hyperparameters

```bash
sbatch run_tune_params.sh    # runs lda_tune_params.py
```

Output: `lda_param_tuning_results.txt` - coherence scores for each config in the grid. Read this to pick the best `NUM_TOPICS`, `ALPHA`, `ETA`, `PASSES` and copy them into `lda_guitars.py`.

### Step 2: Train final LDA model

```bash
sbatch run_lda.sh            # runs lda_guitars.py
```

Outputs:
- `guitar_lda_model/`: saved Gensim LDA model
- `guitar_corpus.pkl`, `guitar_dict.pkl`, `guitar_tokens.pkl`: preprocessed corpus (used by downstream steps)

The script also prints the final topic word distributions. From the 25 raw topics, 16 instrument-feature-specific topics were manually selected and labelled for use in assignment and sentiment steps.

### BERTopic (alternative, not used in final pipeline)

**Subfolder:** `bertopic/`

```
bertopic_tune.py + run_tune_bertopic.sh   # tunes HDBSCAN params
bertopic_guitars.py + run_bertopic.sh     # runs final BERTopic model
```

Outputs: `bertopic_results.txt`, `guitar_embeddings.npy`, `guitar_texts.pkl`

BERTopic was tested as an alternative to LDA but produced ~48% outliers on review text regardless of parameter tuning. **Final pipeline uses LDA.**

---

## 3.2. Assign Topics
 
**Folder:** `3.2. Assign/`
 
`assign_compare.py` runs a small ablation to separate two design choices in topic assignment — **topic merging** vs. **assignment threshold** — rather than picking both at once:
 
| Version | Topics | Threshold | Purpose |
|---|---|---|---|
| v1 | 13 original topics | 0.15 | Baseline |
| v2 | 14 merged topics | 0.15 | Isolates effect of merging near-duplicate topics <br>(e.g. two separate "Accessories" and "Fret/neck setup" topics collapsed into one each)<br>**used in all downstream steps** |
| v3 | 14 merged topics | 0.30 | Isolates effect of a stricter assignment threshold |
 
Each config assigns every guitar review its top-scoring topic (if above threshold) and saves the result:
 
| File | Description |
|---|---|
| `guitars_with_topics_v1.parquet` | Baseline assignment |
| `guitars_with_topics_v2.parquet` | Merged topics - used in all downstream steps |
| `guitars_with_topics_v3.parquet` | Merged topics, stricter threshold |

---

## 4. Sentiment Analysis (Guitars)

**Folder:** `4. Sentiment Analysis/`

Six different modelling approaches, each in their own subfolder. All use the same binary task (positive/negative, rating 3 dropped) and the same 80/20 stratified train/test split.

After running the approaches, two notebooks at the folder root aggregate the results:
- `model_comparison.ipynb`: side-by-side comparison of all approaches
- `topic_feature_analysis.ipynb` → `topic_feature_analysis.csv`: per-topic sentiment breakdown

### 4a. TF-IDF + Logistic Regression
**Subfolder:** `tf_idf - lr/` · `python script.py` 

→ `results.txt`, `predictions.parquet`
 
### 4b. Sentence Transformer + Logistic Regression
**Subfolder:** `sentence_transformer - lr/` · `python script.py` 

→ `results.txt`, `predictions.parquet`, `embeddings.npy` (cached sentence embeddings)
 
### 4c. Pretrained BERT (off-the-shelf)
**Subfolder:** `pretrained_bert/` · `python script.py`: tests `nlptown` and `cardiffnlp` sentiment models with no fine-tuning (GPU required) 

→ `results.txt`, `predictions.parquet`, `preds_*.npy`, `probs_*.npy`
 
### 4d. Zero-shot Classification
**Subfolder:** `zero_shot/` · `python script.py`: `facebook/bart-large-mnli` and `cross-encoder/nli-deberta-v3-large` via NLI zero-shot, no training (GPU required) 

→ `results.txt`, `predictions.parquet`, `preds_*.npy`, `probs_*.npy`
 
### 4e. Fine-tuned RoBERTa
**Subfolder:** `fine_tuned_roberta/`
 
| File | Purpose |
|---|---|
| `bert_finetune.py` | Run - fine-tunes RoBERTa on the guitar train split |
| `bert_sentiment_finetune.ipynb` | Local/Colab version of the same fine-tuning |
| `bert_result.ipynb` | Open to view results - loads predictions, shows metrics |
| `bert_error_analysis.ipynb` | Open to view analysis - inspects misclassified examples |
 
Outputs: `bert_predictions.csv`, `bert_topic_sentiment.csv`

### 4f. LLM Few-shot (Document-level)
**Subfolder:** `llm_few_shot_doc/`

Runs `mistralai/Mistral-7B-Instruct-v0.3` locally (HuggingFace, GPU) in few-shot mode - not an external API call.
 
```bash
python llm_fewshot_hpc.py --data guitars_with_topics_v2.parquet --model mistralai/Mistral-7B-Instruct-v0.3 --batch_size 8
```

| File | Purpose |
|---|---|
| `llm_fewshot_hpc.py` + `submit_llm.sh` | Run on HPC - calls LLM API for each review |
| `llm_fewshot_local.ipynb` | Local version / development notebook |
| `llm_fewshot_result.ipynb` | Open to view results |

Output: `llm_predictions.csv`

### 4g. LLM Few-shot (Topic-aware)
**Subfolder:** `llm_few_shot_topic/` - same structure as 4f, but the LLM prompt includes the review's assigned topic label as context.

```bash
python llm_topic_hpc.py --data guitars_with_topics_v2.parquet --model mistralai/Mistral-7B-Instruct-v0.3 --batch_size 8
```

| File | Purpose |
|---|---|
| `llm_topic_hpc.py` + `submit_llm_topic.sh` | Run on HPC |
| `llm_topic_local.ipynb` | Local version |
| `llm_topic_result.ipynb` | Open to view results |

Outputs: `llm_topic_predictions.csv`, `llm_topic_sentiment.csv`

### 4h. Aspect-Based Sentiment (ABSA)
**Subfolder:** 
* `absa/` · `sbatch run.sh`: applies zero-shot BART per topic across all 134,068 guitar reviews to produce a positive/negative/not-mentioned distribution for each of the 16 topics. Checkpoints per topic into `ckpt/` (resubmit safely; completed topics are skipped). 
* `summarize.py` reads the checkpoints and prints the final summary - run locally once all checkpoints exist.

Outputs: `results.txt`, `guitars_absa.parquet`, `ckpt/t<id>_<label>.pkl` (one per topic)

---

## 5. Ad generation

**Folder:** `5. Ad generation/`

Three notebooks, run locally in order. Requires OpenAI/DeepSeek API keys in `.env`:

```
step1_data_prep.ipynb      # prepares per-topic pain point summaries → ad_generation_input.json
step2_ad_generation.ipynb  # generates two ad strategies per topic (A: amplify praise / B: address pain points) → ads_llm.json
step3_llm_judge.ipynb      # LLM-as-judge: pointwise + pairwise scoring → results/scores_pointwise.csv, scores_pairwise.csv, scores_*.json, results_summary.txt
```

---

## Extensions: Drums & Keyboards
 
**Folders:** `extension_drums/` and `extension_keyboards/`
 
Both extensions have identical structure. Run the numbered scripts in order (substitute `drums` or `keyboards` as appropriate):
 
```bash
python 1_preprocess_<category>.py         # tokenises reviews, builds LDA corpus
                                          # → <category>_tokens.pkl, _dict.pkl, _corpus.pkl
 
python 2_tune_params_<category>.py        # grid search over LDA hyperparameters
                                          # → results/tune_results_<category>.txt
 
python 3_assign_topics_<category>.py      # trains final LDA model, assigns topics to reviews
                                          # → <category>_lda_final/, results/lda_results_<category>.txt
 
python 4_sentiment_<category>.py          # fine-tunes RoBERTa, runs inference
                                          # → results/<category>_bert_predictions.csv, _bert_topic_sentiment.csv
```
 
Fine-tuned RoBERTa model saved to `roberta_<category>_model/` including intermediate training checkpoints.
 
Then three notebooks for ad generation and evaluation, run locally in order (same structure as main pipeline step 5):
 
```
5. adgen step1_data_prep.ipynb      # → ad_generation_input.json
5. adgen step2_ad_generation.ipynb  # → ads_llm.json
5. adgen step3_llm_judge.ipynb      # → results/scores_pointwise.csv, scores_pairwise.csv, scores_*.json, results_summary.txt
```
