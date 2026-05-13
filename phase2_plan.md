# 🧠 Phase 2 Revised Plan: Knowledge Extraction & Graph (Zero Cost)

> **Same 6 steps as before. Every paid Gemini call replaced with a free open-source model.**

---

## Alignment: Old → New

| # | Step | ❌ Old (Paid) | ✅ New (Free) | Size | Where |
|---|------|-------------|-------------|------|-------|
| 1 | Entity Extraction | Gemini JSON mode | **SciBERT + SciERC** NER model | ~440 MB | Kaggle GPU |
| 2 | Entity Resolution | Fuzzy match + embeddings | **No change** | — | Local |
| 3 | Citation Graph | Semantic Scholar API → NetworkX | **No change** | — | Local |
| 4 | BERTopic Topics | Kaggle notebook (120K papers) | **Same, now 50K papers** | — | Kaggle GPU |
| 5 | Citation Classifier | Gemini LLM classification | **SciCite** by AllenAI | ~440 MB | Kaggle GPU |
| 6 | SOTA Tracker | Gemini extracts results → SQL | **Qwen2.5-3B** extracts results → SQL | ~6 GB | Kaggle GPU |

**Total cost: $0**

---

## Dataset Scope

- **CS-only**: cs.AI, cs.LG, cs.CL, cs.CV, cs.IR, cs.NE, cs.RO, cs.SE, cs.DB, cs.DC, cs.DS, cs.CR, cs.HC
- **50,000 papers** (2020+, drop oldest if over 50K)

---

## Step 1: Entity Extraction (SciBERT + SciERC)

**Replaces**: Gemini JSON mode on 10K papers

**Model**: `allenai/scibert_scivocab_uncased` fine-tuned on SciERC dataset

**What it extracts** (Token Classification / NER):
- `METHOD` — "LoRA", "Transformer", "DPO"
- `TASK` — "question answering", "text classification"
- `DATASET` — "SQuAD", "MMLU", "ImageNet"
- `METRIC` — "F1 score", "accuracy", "BLEU"

**How it works**:
```
Input:  "We fine-tune BERT using LoRA on the SQuAD dataset"
Output: [BERT → METHOD], [LoRA → METHOD], [SQuAD → DATASET]
```

**Processing**: All 50K abstracts on Kaggle T4 GPU in ~30 min (batch inference).

**Output**: `entities.parquet` → columns: `paper_id, entity_name, entity_type`

**Why SciBERT over generic BERT?** SciBERT was pre-trained on 1.14M scientific papers. Its vocabulary understands terms like "LSTM", "hyperparameter", and "ablation" natively, instead of splitting them into weird sub-word tokens.

---

## Step 2: Entity Resolution (No Change)

**Same as before**: Fuzzy string matching + embedding similarity to merge duplicates.

**How**:
1. Lowercase + strip hyphens/underscores: `"BERT-Large"` → `"bert large"`
2. Levenshtein distance > 0.85 → same entity
3. Embedding cosine similarity > 0.92 → same entity
4. Set `canonical_name` for each group

**Runs locally**, no model needed. Just string operations + numpy cosine.

**Output**: Updated entities with `canonical_name` column.

---

## Step 3: Citation Graph (No Change)

**Same as before**: Semantic Scholar API → NetworkX → PageRank.

**Data source**: Semantic Scholar Academic Graph API (completely free)
- Free API key: 1 request/sec
- Endpoint: `GET /paper/ArXiv:{id}?fields=references,citations`
- We fetch for top **15K most important papers** (not all 50K, to stay reasonable)
- Time: ~4 hours at 1 req/sec

**Graph algorithms** (NetworkX):
- **PageRank**: Identify most influential papers
- **Community Detection** (Louvain): Find paper clusters
- **Shortest Path**: Citation lineage between two papers

**Output**: `citations.parquet` → columns: `citing_paper_id, cited_paper_id`
Plus: PageRank scores stored back in `papers` table.

---

## Step 4: BERTopic Topics (Same, Now 50K Papers)

**Same as before**, just scaled down to 50K papers.

**How**:
1. Load 50K paper embeddings (from Phase 1 parquet)
2. UMAP dimensionality reduction (768d → 5d)
3. HDBSCAN clustering
4. c-TF-IDF for topic labeling

**Runs on Kaggle GPU** in ~15 min.

**Output**: `topics.parquet` → columns: `paper_id, topic_id, topic_label`

Expected: ~100-150 coherent topics like "Large Language Models", "Reinforcement Learning", "Computer Vision", etc.

---

## Step 5: Citation Classifier (SciCite)

**Replaces**: Gemini LLM classification

**Model**: `allenai/scicite` — SciBERT fine-tuned on 11K annotated citation sentences.

**Classification**:
| SciCite Output | Our Label | Meaning |
|---|---|---|
| `BACKGROUND` | **Mentioning** | Just references the work |
| `METHOD` | **Supporting** | Uses the cited method/approach |
| `RESULT_COMPARISON` | **Contrasting** | Compares or contradicts results |

**How it works**:
```
Input:  "Unlike Smith et al. [3], our method achieves higher accuracy."
Output: RESULT_COMPARISON → mapped to "Contrasting"
```

**Processing**: We extract citation sentences from abstracts (regex for patterns like "Author et al.", "[1]", "(2023)"), then classify each sentence. ~20 min on Kaggle GPU.

**Output**: `citation_intents.parquet` → columns: `citing_paper_id, cited_paper_id, sentence, classification`

---

## Step 6: SOTA Tracker (Qwen2.5-3B + SQL)

**Replaces**: Gemini extracting benchmark results

This step has two parts:

### Part A: Result Extraction (Qwen2.5-3B on Kaggle)

We need to extract structured tuples like:
```json
{"method": "LoRA", "dataset": "MMLU", "metric": "accuracy", "value": 86.4}
```

SciBERT can find the entity names, but it **cannot** link `86.4` to `LoRA + MMLU + accuracy`. For this we need a small LLM.

**Model**: `Qwen/Qwen2.5-3B-Instruct` — only 6GB, runs on Kaggle T4 GPU.

**Prompt**:
```
Extract benchmark results from this abstract. Return JSON only.
Abstract: "Our model achieves 86.4% accuracy on MMLU and 92.1 F1 on SQuAD 2.0"

Output: [
  {"method": "Our model", "dataset": "MMLU", "metric": "accuracy", "value": 86.4},
  {"method": "Our model", "dataset": "SQuAD 2.0", "metric": "F1", "value": 92.1}
]
```

**Processing**: We only run this on papers that SciBERT already tagged with METRIC + DATASET entities (~5K-10K papers). ~1-2 hours on Kaggle GPU.

**Output**: `results.parquet` → columns: `paper_id, method_name, dataset_name, metric_name, value`

### Part B: Leaderboard Queries (Pure SQL)

Once the `results` table is populated, SOTA tracking is just SQL:
```sql
SELECT method_name, value, paper_id
FROM results
WHERE dataset_name = 'MMLU' AND metric_name = 'accuracy'
ORDER BY value DESC;
```

No model needed for this part.

---

## Kaggle Notebooks to Create

| Notebook | Models Used | GPU Time |
|----------|-----------|----------|
| `kaggle_entity_extraction.py` | SciBERT + SciERC | ~30 min |
| `kaggle_bertopic_clustering.py` | BERTopic (UMAP + HDBSCAN) | ~15 min |
| `kaggle_citation_classification.py` | SciCite | ~20 min |
| `kaggle_sota_extraction.py` | Qwen2.5-3B-Instruct | ~1-2 hrs |

**Total Kaggle GPU time: ~2.5 hours** (well within free weekly quota of 30 hours)

---

## Backend Files to Create

```
backend/
├── app/knowledge/
│   ├── __init__.py
│   ├── entity_resolution.py     ← Step 2: Deduplicate entities
│   ├── citation_graph.py        ← Step 3: NetworkX + PageRank
│   ├── sota_tracker.py          ← Step 6B: SQL leaderboard queries
│   └── gap_analyzer.py          ← Research gap detection (Phase 3)
├── app/api/
│   ├── entities.py              ← Entity/KG query endpoints
│   ├── sota.py                  ← SOTA leaderboard endpoints
│   ├── topics.py                ← Topic browsing endpoints
│   └── citations.py             ← Citation context endpoints
├── ingestion/
│   ├── upload_entities.py       ← Push entities.parquet to Supabase
│   ├── upload_topics.py         ← Push topics to papers table
│   ├── upload_citations.py      ← Push citation pairs to Supabase
│   ├── upload_results.py        ← Push SOTA results to Supabase
│   └── fetch_citations_s2.py    ← Semantic Scholar API fetcher
└── scripts/
    └── setup_supabase_phase2.sql ← New tables (entities, relations, results, citation_contexts)
```

---

## Execution Order

| # | Task | Where | Time | Cost |
|---|------|-------|------|------|
| 1 | Write all Kaggle notebooks + backend code | Local | — | $0 |
| 2 | Run Phase 2 SQL in Supabase | Supabase | 1 min | $0 |
| 3 | Run entity extraction notebook | Kaggle GPU | ~30 min | $0 |
| 4 | Run BERTopic notebook | Kaggle GPU | ~15 min | $0 |
| 5 | Run SOTA extraction notebook | Kaggle GPU | ~1-2 hrs | $0 |
| 6 | Run citation classification notebook | Kaggle GPU | ~20 min | $0 |
| 7 | Run Semantic Scholar citation fetcher | Local | ~4 hrs | $0 |
| 8 | Run entity resolution | Local | ~5 min | $0 |
| 9 | Upload everything to Supabase | Local | ~10 min | $0 |

**Total cost: $0**
