# NLP Project — Amazon Musical Instruments Reviews
**Bocconi University** | Amazon Reviews 2023, Musical Instruments subset

All scripts are designed for the **Bocconi HPC cluster (SLURM)**. Run `.sh` files with `sbatch` from within the step's folder. Scripts use checkpointing — safe to resubmit if a job breaks.

## Environment

Conda environment: `environment.yml`. Create with:
```bash
conda env create -f environment.yml
conda activate nlp_env
```

## HuggingFace Models

HPC compute nodes have no internet access. The following models must be downloaded on the **login node** before submitting jobs (SLURM scripts set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`):

- `all-mpnet-base-v2` — sentence embeddings (step 4b, BERTopic)
- `facebook/bart-large-mnli` — zero-shot classification (step 4d, ABSA)
- `cross-encoder/nli-deberta-v3-large` — zero-shot classification (step 4d)
- `nlptown/bert-base-multilingual-uncased-sentiment` — pre-trained sentiment (step 4c)
- `cardiffnlp/twitter-roberta-base-sentiment-latest` — pre-trained sentiment (step 4c)
- `roberta-base` — base model for fine-tuning (step 4e, extensions)

---

## Data Files (root)

| File | Description |
|---|---|
| `Musical_Instruments.jsonl.gz` | Raw reviews (compressed) |
| `Musical_Instruments_joined.parquet` | Reviews + metadata joined — **main working file** |
| `full-00000-of-00002.parquet` | Product metadata part 1 |
| `full-00001-of-00002.parquet` | Product metadata part 2 |
| `meta_Musical_Instruments.jsonl` | Metadata in JSONL format |
| `guitars_with_topics_v2.parquet` | Guitar reviews with assigned LDA topics (copy of `3.2. Assign/`) |

---

## 1. Join 2 Datasets

**Folder:** `1. Join 2 datasets/`

```bash
sbatch run.sh    # runs join.py
```

Joins the reviews parquet with both metadata parquet files on `parent_asin`, adds a `subcategory` column, and saves the result.

Output: `Musical_Instruments_joined.parquet` (saved one level up, at root)

`check_join.py` is a separate validation script — run it locally after the join to confirm match rates and row counts.

---

## 3. Topic Modelling (Guitars, LDA)

**Folder:** `3. Topic modelling/`

### Step 1 — Tune hyperparameters

```bash
sbatch run_tune_params.sh    # runs lda_tune_params.py
```

Output: `lda_param_tuning_results.txt` — coherence scores for each config in the grid. Read this to pick the best `NUM_TOPICS`, `ALPHA`, `ETA`, `PASSES` and copy them into `lda_guitars.py`.

### Step 2 — Train final LDA model

```bash
sbatch run_lda.sh            # runs lda_guitars.py
```

Outputs:
- `guitar_lda_model/` — saved Gensim LDA model
- `guitar_corpus.pkl`, `guitar_dict.pkl`, `guitar_tokens.pkl` — preprocessed corpus (used by downstream steps)

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

Topic assignment is run as part of `lda_guitars.py` (Step 3, Step 2 above). This folder contains the output parquet files at different development versions:

| File | Description |
|---|---|
| `guitars_with_topics_v1.parquet` | Initial assignment |
| `guitars_with_topics_v2.parquet` | Revised labels — **used in all downstream steps** |
| `guitars_with_topics_v3.parquet` | Further refinement |

---

## 4. Sentiment Analysis (Guitars)

**Folder:** `4. Sentiment Analysis/`

Six different modelling approaches are each in their own subfolder. All use the same binary task (positive/negative, rating 3 dropped) and the same 80/20 stratified train/test split.

After running the approaches, two notebooks at the folder root aggregate the results:
- `model_comparison.ipynb` — side-by-side comparison of all approaches
- `topic_feature_analysis.ipynb` → `topic_feature_analysis.csv` — per-topic sentiment breakdown

---

### 4a. TF-IDF + Logistic Regression

**Subfolder:** `tf_idf - lr/`

```bash
sbatch run.sh    # runs script.py
```

Outputs: `results.txt`, `predictions.parquet`

---

### 4b. Sentence Transformer + Logistic Regression

**Subfolder:** `sentence_transformer - lr/`

```bash
sbatch run.sh    # runs script.py
```

Outputs: `results.txt`, `predictions.parquet`, `embeddings.npy` (cached sentence embeddings)

---

### 4c. Pre-trained BERT (off-the-shelf)

**Subfolder:** `pretrained_bert/`

```bash
sbatch run_step11.sh    # runs script.py
```

Tests two models (`nlptown/bert-base-multilingual-uncased-sentiment` and `cardiffnlp/twitter-roberta-base-sentiment-latest`) with no fine-tuning. GPU required.

Outputs: `results.txt`, `predictions.parquet`, `preds_nlptown.npy`, `probs_nlptown.npy`, `preds_cardiffnlp.npy`, `probs_cardiffnlp.npy`

Log files in `out/` and `err/`.

---

### 4d. Zero-shot Classification

**Subfolder:** `zero_shot/`

```bash
sbatch run.sh    # runs script.py
```

Uses `facebook/bart-large-mnli` and `cross-encoder/nli-deberta-v3-large` with NLI zero-shot (no training). GPU required.

Outputs: `results.txt`, `predictions.parquet`, `preds_bart.npy`, `probs_bart.npy`, `preds_deberta.npy`, `probs_deberta.npy`

---

### 4e. Fine-tuned RoBERTa

**Subfolder:** `fine_tuned_roberta/`

There are three files here — here is what each one is for:

| File | Purpose |
|---|---|
| `bert_finetune.py` + `submit_bert.sh` | **Run on HPC** — fine-tunes RoBERTa on the guitar train split |
| `bert_sentiment_finetune.ipynb` | Local/Colab version of the same fine-tuning |
| `bert_result.ipynb` | **Open to view results** — loads predictions, shows metrics |
| `bert_error_analysis.ipynb` | **Open to view analysis** — inspects misclassified examples |

```bash
sbatch submit_bert.sh    # runs bert_finetune.py (HPC)
```

Outputs: `bert_predictions.csv`, `bert_topic_sentiment.csv`

---

### 4f. LLM Few-shot (Document-level)

**Subfolder:** `llm_few_shot_doc/`

| File | Purpose |
|---|---|
| `llm_fewshot_hpc.py` + `submit_llm.sh` | **Run on HPC** — calls LLM API for each review |
| `llm_fewshot_local.ipynb` | Local version / development notebook |
| `llm_fewshot_result.ipynb` | **Open to view results** |

```bash
sbatch submit_llm.sh    # runs llm_fewshot_hpc.py (HPC)
```

Output: `llm_predictions.csv`

---

### 4g. LLM Few-shot (Topic-aware)

**Subfolder:** `llm_few_shot_topic/`

Same structure as 4f but the LLM prompt includes the review's assigned topic label as context.

| File | Purpose |
|---|---|
| `llm_topic_hpc.py` + `submit_llm_topic.sh` | **Run on HPC** |
| `llm_topic_local.ipynb` | Local version |
| `llm_topic_result.ipynb` | **Open to view results** |

```bash
sbatch submit_llm_topic.sh    # runs llm_topic_hpc.py (HPC)
```

Outputs: `llm_topic_predictions.csv`, `llm_topic_sentiment.csv`

---

### 4h. Aspect-Based Sentiment (ABSA)

**Subfolder:** `absa/`

Applies zero-shot BART per topic across all 134,068 guitar reviews to produce a positive/negative/not-mentioned distribution for each of the 16 topics.

```bash
sbatch run.sh    # runs script.py
```

The script checkpoints per topic into `ckpt/` (one `.pkl` per topic) — resubmit safely if the job times out; completed topics are skipped.

`summarize.py` reads the checkpoint files and prints the final summary — run this locally once all checkpoints exist.

Outputs: `results.txt`, `guitars_absa.parquet`, `ckpt/t<id>_<label>.pkl` (one per topic)

---

## Extensions — Drums & Keyboards

**Folders:** `extension_drums/` and `extension_keyboards/`

Both extensions have identical structure. Run the numbered scripts in order (substitute `drums` or `keyboards` as appropriate):

```bash
sbatch 1_run_preprocess_<category>.sh   # tokenises reviews, builds LDA corpus
                                        # → <category>_tokens.pkl, _dict.pkl, _corpus.pkl

sbatch 2_run_tune_<category>.sh         # grid search over LDA hyperparameters
                                        # → results/tune_results_<category>.txt
                                        #   (read this to pick final topic count & params)

sbatch 3_run_assign_<category>.sh       # trains final LDA model, assigns topics to reviews
                                        # → <category>_lda_final/ (saved model)
                                        #   results/lda_results_<category>.txt

sbatch 4_run_sentiment_<category>.sh    # fine-tunes RoBERTa, runs inference
                                        # → results/<category>_bert_predictions.csv
                                        #   results/<category>_bert_topic_sentiment.csv
```

Fine-tuned RoBERTa model saved to `roberta_<category>_model/` including intermediate training checkpoints.

Then three notebooks for ad generation and evaluation, run locally in order. Requires OpenAI/DeepSeek API keys in `.env`:

```
5. adgen step1_data_prep.ipynb      # prepares per-topic pain point summaries
                                    # → ad_generation_input.json

5. adgen step2_ad_generation.ipynb  # generates two ad strategies per topic
                                    # (A: amplify praise / B: address pain points)
                                    # → ads_llm.json

5. adgen step3_llm_judge.ipynb      # LLM-as-judge: pointwise + pairwise scoring
                                    # → results/scores_pointwise.csv
                                    #   results/scores_pairwise.csv
                                    #   results/scores_*.json
                                    #   results/results_summary.txt
```
