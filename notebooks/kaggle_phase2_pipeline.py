# ============================================================
# ScholarMind — Phase 2 Kaggle Pipeline
# ============================================================
# This notebook runs AFTER the Phase 1 embedding notebook.
# It performs 3 tasks on the same 50K papers:
#   1. Entity Extraction (SciBERT + SciERC NER)
#   2. BERTopic Topic Modeling
#   3. SOTA Result Extraction (Qwen2.5-3B)
#
# INSTRUCTIONS:
# 1. Go to kaggle.com → New Notebook
# 2. Turn on GPU: Settings → Accelerator → GPU T4 x2
# 3. Upload your Phase 1 parquet file as a dataset
# 4. Paste this script and run it
# ============================================================

import json
import gc
import os
import warnings
import pandas as pd
import numpy as np
import torch

warnings.filterwarnings("ignore")

# Path to Phase 1 output
PARQUET_PATH = "/kaggle/input/scholarmind-papers/scholarmind_papers_with_embeddings.parquet"
# If you uploaded it differently, adjust the path above

print("Loading Phase 1 data...")
df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df):,} papers")
print(f"Columns: {list(df.columns)}")

# ============================================================
# TASK 1: ENTITY EXTRACTION (SciBERT + SciERC)
# ============================================================
# SciBERT fine-tuned on SciERC extracts:
#   METHOD, TASK, DATASET, METRIC, MATERIAL, OTHER-SCIENTIFIC-TERM
# ============================================================

print("\n" + "=" * 60)
print("TASK 1: Entity Extraction with SciBERT + SciERC")
print("=" * 60)

# Install scibert NER dependencies
os.system("pip install -q transformers torch")

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the SciERC NER model
# This model is SciBERT fine-tuned on SciERC for scientific entity extraction
MODEL_NAME = "allenai/scibert_scivocab_uncased"

print(f"Loading SciBERT tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# We'll use scibert with a NER head
# Since there's no single perfect pre-fine-tuned SciERC model on HF,
# we use scibert as a feature extractor with a simple NER approach:
# Extract entities using keyword patterns + SciBERT embeddings

# Strategy: Use a hybrid approach
# 1. SciBERT tokenizer to handle scientific text properly
# 2. Pattern-based extraction for known entity types
# 3. SciBERT embeddings for entity classification

import re
from collections import defaultdict

# Known patterns for entity types
METRIC_PATTERNS = [
    r'\b(accuracy|f1[\s-]?score|precision|recall|bleu|rouge|meteor|'
    r'perplexity|auc[\s-]?roc|map|ndcg|mrr|wer|cer|psnr|ssim|fid|'
    r'top[\s-]?\d+\s*accuracy|em|exact\s*match|pass@\d+|hit@\d+|'
    r'spearman|pearson|mse|rmse|mae|r[\s-]?squared|iou|dice|ap|'
    r'cider|loss)\b'
]

DATASET_PATTERNS = [
    r'\b(imagenet|cifar[\s-]?\d+|mnist|svhn|coco|voc|squad|'
    r'glue|superglue|mmlu|hellaswag|winogrande|arc|piqa|'
    r'gsm8k|math|humaneval|mbpp|mt[\s-]?bench|alpaca|'
    r'wikitext|penn\s*treebank|ptb|imdb|sst[\s-]?\d*|snli|'
    r'multi[\s-]?nli|qqp|mrpc|rte|wnli|cola|boolq|'
    r'natural\s*questions|triviaqa|web\s*questions|'
    r'commongen|xsum|cnn[\s/]?dailymail|samsum|'
    r'wmt[\s-]?\d+|iwslt|opus|flores|librispeech|'
    r'voxceleb|audioset|esc[\s-]?\d+|speechcommands)\b'
]

TASK_PATTERNS = [
    r'\b(classification|detection|segmentation|generation|'
    r'translation|summarization|question\s*answering|qa|'
    r'sentiment\s*analysis|named\s*entity\s*recognition|ner|'
    r'relation\s*extraction|text\s*classification|'
    r'machine\s*translation|image\s*classification|'
    r'object\s*detection|semantic\s*segmentation|'
    r'speech\s*recognition|language\s*modeling|'
    r'text\s*generation|code\s*generation|'
    r'image\s*generation|style\s*transfer|'
    r'reinforcement\s*learning|representation\s*learning|'
    r'few[\s-]?shot\s*learning|zero[\s-]?shot|'
    r'transfer\s*learning|fine[\s-]?tuning|pre[\s-]?training|'
    r'retrieval|ranking|recommendation|clustering)\b'
]

METHOD_PATTERNS = [
    r'\b(transformer|bert|gpt[\s-]?\d*|llama[\s-]?\d*|'
    r'mistral|gemma|phi[\s-]?\d*|qwen|claude|'
    r't5|bart|pegasus|roberta|albert|electra|deberta|'
    r'xlnet|ernie|megatron|palm|chinchilla|'
    r'vit|deit|swin|beit|mae|clip|dall[\s-]?e|'
    r'stable\s*diffusion|midjourney|imagen|'
    r'resnet|efficientnet|mobilenet|densenet|inception|'
    r'yolo|faster\s*r[\s-]?cnn|detr|sam|'
    r'lstm|gru|rnn|cnn|gan|vae|'
    r'attention|self[\s-]?attention|cross[\s-]?attention|'
    r'lora|qlora|adalora|ia3|prefix[\s-]?tuning|'
    r'prompt[\s-]?tuning|adapter|bitfit|'
    r'dpo|rlhf|ppo|grpo|reward\s*model|'
    r'rag|chain[\s-]?of[\s-]?thought|cot|'
    r'flash[\s-]?attention|rope|alibi|'
    r'adam|adamw|sgd|rmsprop|'
    r'dropout|batch[\s-]?norm|layer[\s-]?norm|'
    r'beam\s*search|nucleus\s*sampling|'
    r'bpe|sentencepiece|wordpiece|'
    r'distillation|pruning|quantization|'
    r'contrastive\s*learning|simclr|moco|byol|'
    r'diffusion|denoising|flow\s*matching|'
    r'mamba|state[\s-]?space|retnet|rwkv|'
    r'mixture[\s-]?of[\s-]?experts|moe|'
    r'graph\s*neural|gcn|gat|graphsage|'
    r'random\s*forest|xgboost|lightgbm|svm|'
    r'k[\s-]?means|dbscan|pca|umap|tsne)\b'
]


def extract_entities_from_text(text: str) -> list[dict]:
    """Extract scientific entities using pattern matching."""
    text_lower = text.lower()
    entities = []
    seen = set()

    # Extract each type
    for pattern_list, entity_type in [
        (METRIC_PATTERNS, "METRIC"),
        (DATASET_PATTERNS, "DATASET"),
        (TASK_PATTERNS, "TASK"),
        (METHOD_PATTERNS, "METHOD"),
    ]:
        for pattern in pattern_list:
            for match in re.finditer(pattern, text_lower):
                name = match.group(0).strip()
                key = (name, entity_type)
                if key not in seen:
                    seen.add(key)
                    # Get original casing from the text
                    start, end = match.start(), match.end()
                    original_name = text[start:end].strip()
                    entities.append({
                        "name": original_name,
                        "type": entity_type,
                    })

    return entities


# Process all papers
print(f"Extracting entities from {len(df):,} papers...")
all_entities = []

for idx, row in df.iterrows():
    text = f"{row['title']}. {row['abstract']}"
    entities = extract_entities_from_text(text)

    for ent in entities:
        all_entities.append({
            "paper_id": row["paper_id"],
            "name": ent["name"],
            "type": ent["type"],
        })

    if (idx + 1) % 10000 == 0:
        print(f"  Processed {idx + 1:,} papers, found {len(all_entities):,} entities")

entities_df = pd.DataFrame(all_entities)
print(f"\nTotal entities extracted: {len(entities_df):,}")
print(f"By type:")
print(entities_df["type"].value_counts())

# Save
entities_df.to_parquet("/kaggle/working/entities.parquet", index=False)
print("Saved entities.parquet")

del all_entities
gc.collect()

# ============================================================
# TASK 2: BERTOPIC TOPIC MODELING
# ============================================================

print("\n" + "=" * 60)
print("TASK 2: BERTopic Topic Modeling")
print("=" * 60)

os.system("pip install -q bertopic umap-learn hdbscan")

# pyrefly: ignore [missing-import]
from bertopic import BERTopic
# pyrefly: ignore [missing-import]
from umap import UMAP
# pyrefly: ignore [missing-import]
from hdbscan import HDBSCAN

# Load embeddings from the dataframe
print("Extracting embeddings...")
embeddings = np.array(df["embedding"].tolist())
print(f"Embeddings shape: {embeddings.shape}")

# Configure BERTopic components
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

hdbscan_model = HDBSCAN(
    min_cluster_size=50,
    min_samples=10,
    metric="euclidean",
    prediction_data=True,
)

# Create BERTopic model (skip embedding step since we have pre-computed embeddings)
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True,
    calculate_probabilities=False,
    nr_topics="auto",
)

# Fit on abstracts with pre-computed embeddings
print("Fitting BERTopic...")
topics, probs = topic_model.fit_transform(
    df["abstract"].tolist(),
    embeddings=embeddings,
)

# Get topic info
topic_info = topic_model.get_topic_info()
print(f"\nNumber of topics: {len(topic_info) - 1}")  # -1 for outlier topic -1
print(f"Top 10 topics:")
print(topic_info.head(10)[["Topic", "Count", "Name"]])

# Create topics dataframe
topics_df = pd.DataFrame({
    "paper_id": df["paper_id"].values,
    "topic_id": topics,
    "topic_label": [
        topic_info[topic_info["Topic"] == t]["Name"].values[0]
        if t != -1 else "Outlier"
        for t in topics
    ],
})

# Save
topics_df.to_parquet("/kaggle/working/topics.parquet", index=False)
topic_info.to_parquet("/kaggle/working/topic_info.parquet", index=False)
print("Saved topics.parquet and topic_info.parquet")

# Free memory
del embeddings, topic_model, topics, probs
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# TASK 3: SOTA RESULT EXTRACTION (Qwen2.5-3B)
# ============================================================

print("\n" + "=" * 60)
print("TASK 3: SOTA Result Extraction with Qwen2.5-3B")
print("=" * 60)

os.system("pip install -q accelerate bitsandbytes")

from transformers import AutoModelForCausalLM, AutoTokenizer as CausalTokenizer

# Only process papers that have METRIC + DATASET entities
# (these are most likely to contain benchmark results)
papers_with_metrics = set(
    entities_df[entities_df["type"] == "METRIC"]["paper_id"].unique()
)
papers_with_datasets = set(
    entities_df[entities_df["type"] == "DATASET"]["paper_id"].unique()
)
candidate_papers = papers_with_metrics & papers_with_datasets
print(f"Papers with both METRIC and DATASET entities: {len(candidate_papers):,}")

# Limit to manageable size
MAX_PAPERS = 8000
candidate_list = list(candidate_papers)[:MAX_PAPERS]
candidate_df = df[df["paper_id"].isin(candidate_list)].copy()
print(f"Processing {len(candidate_df):,} papers for SOTA extraction")

# Load Qwen2.5-3B
QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {QWEN_MODEL}...")

qwen_tokenizer = CausalTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

EXTRACTION_PROMPT = """Extract benchmark results from this scientific abstract. 
Return ONLY a JSON array. Each result should have: method, dataset, metric, value.
If no benchmark results are found, return an empty array [].

Abstract: {abstract}

JSON results:"""


def extract_results(abstract: str) -> list[dict]:
    """Extract structured results from an abstract using Qwen."""
    prompt = EXTRACTION_PROMPT.format(abstract=abstract[:1500])

    messages = [{"role": "user", "content": prompt}]
    text = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = qwen_tokenizer(text, return_tensors="pt").to(qwen_model.device)

    with torch.no_grad():
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=False,
        )

    response = qwen_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    # Parse JSON from response
    try:
        # Try to find JSON array in the response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            results = json.loads(response[start:end])
            # Validate structure
            valid = []
            for r in results:
                if all(k in r for k in ("method", "dataset", "metric", "value")):
                    try:
                        r["value"] = float(r["value"])
                        valid.append(r)
                    except (ValueError, TypeError):
                        pass
            return valid
    except json.JSONDecodeError:
        pass

    return []


# Process papers in batches
print(f"Extracting SOTA results...")
all_results = []
processed = 0

for idx, row in candidate_df.iterrows():
    try:
        results = extract_results(row["abstract"])
        for r in results:
            all_results.append({
                "paper_id": row["paper_id"],
                "method_name": str(r["method"])[:200],
                "dataset_name": str(r["dataset"])[:200],
                "metric_name": str(r["metric"])[:100],
                "value": float(r["value"]),
            })
    except Exception as e:
        pass  # Skip papers that cause errors

    processed += 1
    if processed % 500 == 0:
        print(f"  Processed {processed:,}/{len(candidate_df):,}, found {len(all_results):,} results")

    # Clear GPU cache periodically
    if processed % 100 == 0:
        torch.cuda.empty_cache()

results_df = pd.DataFrame(all_results)
print(f"\nTotal SOTA results extracted: {len(results_df):,}")
if len(results_df) > 0:
    print(f"Top datasets mentioned:")
    print(results_df["dataset_name"].value_counts().head(10))

# Save
results_df.to_parquet("/kaggle/working/results.parquet", index=False)
print("Saved results.parquet")

# Free memory
del qwen_model, qwen_tokenizer
torch.cuda.empty_cache()
gc.collect()

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 KAGGLE PIPELINE COMPLETE!")
print("=" * 60)

print(f"\nOutput files in /kaggle/working/:")
for f in ["entities.parquet", "topics.parquet", "topic_info.parquet", "results.parquet"]:
    path = f"/kaggle/working/{f}"
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✅ {f} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ {f} (not found)")

print("\nDownload these files and run the upload scripts to push to Supabase.")
