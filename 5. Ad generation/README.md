# 4. Ad Generation and Evaluation - Guitars (Report Section 5)

Primary subcategory. The pipeline runs in three notebooks executed in
order: data prep → ad generation → judge.

## Files

| File | Role |
|------|------|
| `step1_data_prep.ipynb`            | Builds the product × topic candidate set with ≥5 positive **and** ≥5 negative reviews; ranks products and selects the top 3 (post brand-dedup) |
| `step2_ad_generation.ipynb`        | Two-step LLM generation (insight extraction → A+B in a single call) using OpenAI GPT-5.4 |
| `step3_llm_judge.ipynb`            | Pairwise (primary) + pointwise (supplementary) evaluation by DeepSeek-V4-Flash |
| `ad_generation_input.json`         | Output of step 1, input to step 2 |
| `ads_llm.json`                     | Output of step 2, input to step 3 (18 ads = 3 products × 3 topics × 2 strategies) |
| `eligible_products.csv`            | Audit log of all 31 candidate products with ≥2 qualifying topics |
| `results/scores_pairwise.csv`      | Per-pair winner; 9 pairs × 2 orderings averaged |
| `results/scores_pairwise_raw.json` | Full judge reasoning for every pairwise call |
| `results/scores_pointwise.csv`     | Per-ad scores on 4 dimensions |
| `results/scores_pointwise_raw.json`| Full judge reasoning for every pointwise call |
| `results/results_summary.txt`      | Human-readable summary used in report Table 3 |

## Selected Products

- **B002RXXOX8** — Davison Full Size Electric Guitar (7 qualifying topics)
- **B006CYVD5E** — Directly Cheap Acoustic Pack (6 topics)
- **B002X49732** — Crescent MG38 Acoustic Starter Pack (6 topics)

`String quality` appears across all three products, enabling
within-topic cross-product comparisons.

## How to Run

1. Open `step1_data_prep.ipynb` and run all cells. This needs the
   joined parquet (see `../DATA_LINKS.txt`).
2. Open `step2_ad_generation.ipynb` — uses `OPENAI_API_KEY`. Total: 18
   API calls (9 analysis + 9 generation).
4. Open `step3_llm_judge.ipynb` — uses `DEEPSEEK_API_KEY`. Total: 9
   pairwise calls × 2 orderings + 18 pointwise calls = 36 calls.

## Result (Report Table 3)

Pairwise (primary): Strategy B wins 6/9 (66.7%); 2 ties; 1 A win.
Pointwise (supplementary): composite **3.17 vs 2.69** (Δ = 0.48).
Both methods agree → high confidence that Strategy B (problem-solving)
outperforms Strategy A (strength-amplifying) on this corpus.
