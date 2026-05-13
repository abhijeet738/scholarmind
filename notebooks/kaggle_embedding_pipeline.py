# ============================================================
# ScholarMind — Kaggle Embedding Pipeline
# ============================================================
# INSTRUCTIONS:
# 1. Go to kaggle.com → New Notebook
# 2. Turn on GPU: Settings → Accelerator → GPU T4 x2
# 3. Add the arXiv dataset: + Add Data → search "arxiv" →
#    select "arXiv Dataset" by Cornell University
# 4. Paste this ENTIRE script into a single cell and run it
# 5. Download the output parquet file from /kaggle/working/
# ============================================================

import json
import pandas as pd
import numpy as np
from datetime import datetime
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
import torch
import gc
import os

# ============================================================
# STEP 1: Configuration
# ============================================================

# Path to the arXiv dataset on Kaggle
ARXIV_DATA_PATH = "/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json"

# Categories we want (CS, Math, Physics, Quantitative Biology)
TARGET_CATEGORIES = {
    # Computer Science
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.IR", "cs.NE", "cs.RO",
    "cs.MA", "cs.HC", "cs.CY", "cs.SE", "cs.DB", "cs.DC", "cs.DS",
    # Mathematics
    "math.ST", "math.OC", "math.PR", "math.NA", "math.CO",
    # Statistics / Machine Learning
    "stat.ML", "stat.TH", "stat.ME", "stat.AP",
    # Physics (relevant subfields)
    "physics.data-an", "physics.comp-ph",
    # Quantitative Biology
    "q-bio.QM", "q-bio.NC", "q-bio.GN",
    # Electrical Engineering (signal processing, systems)
    "eess.SP", "eess.SY", "eess.IV", "eess.AS",
}

# Minimum publication year
MIN_YEAR = 2020

# Target number of papers
TARGET_COUNT = 120_000

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768  # bge-base outputs 768 dimensions

# Batch size for embedding (adjust if you run out of GPU memory)
BATCH_SIZE = 256

print(f"Configuration:")
print(f"  Target categories: {len(TARGET_CATEGORIES)} sub-categories")
print(f"  Min year: {MIN_YEAR}")
print(f"  Target papers: {TARGET_COUNT:,}")
print(f"  Embedding model: {EMBEDDING_MODEL}")
print(f"  GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")

# ============================================================
# STEP 2: Parse and Filter the arXiv Dataset
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Parsing and filtering arXiv metadata...")
print("=" * 60)


def parse_date(date_str):
    """Parse arXiv update_date string to year."""
    try:
        # Format is typically "YYYY-MM-DD" or similar
        return int(date_str.strip()[:4])
    except (ValueError, IndexError):
        return None


def has_target_category(categories_str):
    """Check if paper has at least one target category."""
    paper_cats = set(categories_str.strip().split())
    return bool(paper_cats & TARGET_CATEGORIES)


def get_matching_categories(categories_str):
    """Return the matching target categories for a paper."""
    paper_cats = set(categories_str.strip().split())
    return list(paper_cats & TARGET_CATEGORIES)


# Stream through the JSON file (it's too large to load entirely)
papers = []
total_scanned = 0

with open(ARXIV_DATA_PATH, "r") as f:
    for line in f:
        total_scanned += 1

        if total_scanned % 500_000 == 0:
            print(
                f"  Scanned {total_scanned:,} papers, "
                f"collected {len(papers):,}/{TARGET_COUNT:,}..."
            )

        # Stop when we have enough
        if len(papers) >= TARGET_COUNT:
            break

        try:
            paper = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Filter by category
        if not has_target_category(paper.get("categories", "")):
            continue

        # Filter by year
        year = parse_date(paper.get("update_date", ""))
        if year is None or year < MIN_YEAR:
            continue

        # Filter out papers with missing/tiny abstracts
        abstract = paper.get("abstract", "").strip()
        if len(abstract) < 50:
            continue

        title = paper.get("title", "").strip().replace("\n", " ")
        abstract = abstract.replace("\n", " ")

        # Parse authors (stored as a string in the dataset)
        authors_raw = paper.get("authors", "")
        # Simple split — arXiv uses comma-separated author names
        authors = [a.strip() for a in authors_raw.split(",") if a.strip()][:20]

        papers.append(
            {
                "paper_id": paper["id"],
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "categories": get_matching_categories(paper["categories"]),
                "all_categories": paper["categories"].strip().split(),
                "published_at": paper.get("update_date", "").strip(),
                "year": year,
            }
        )

print(f"\nTotal scanned: {total_scanned:,}")
print(f"Papers collected: {len(papers):,}")

# Convert to DataFrame
df = pd.DataFrame(papers)
del papers
gc.collect()

print(f"\nDataset shape: {df.shape}")
print(f"\nYear distribution:")
print(df["year"].value_counts().sort_index())
print(f"\nTop 15 categories (papers can have multiple):")
from collections import Counter

cat_counts = Counter()
for cats in df["categories"]:
    cat_counts.update(cats)
for cat, count in cat_counts.most_common(15):
    print(f"  {cat}: {count:,}")

# ============================================================
# STEP 3: Generate Embeddings with GPU
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Generating embeddings with GPU...")
print("=" * 60)

# Load the embedding model on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {EMBEDDING_MODEL} on {device}...")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)

# Prepare text for embedding: "Title: ... Abstract: ..."
# BGE models work best with the instruction prefix for retrieval
texts = [
    f"Represent this scientific paper for retrieval: {row['title']}. {row['abstract']}"
    for _, row in df.iterrows()
]

print(f"Embedding {len(texts):,} papers in batches of {BATCH_SIZE}...")
print(f"Estimated time: ~{len(texts) / BATCH_SIZE * 0.8:.0f} seconds on T4 GPU")

# Generate embeddings in batches
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,  # L2 normalize for cosine similarity
    device=device,
)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings dtype: {embeddings.dtype}")

# Free GPU memory
del model, texts
torch.cuda.empty_cache()
gc.collect()

# ============================================================
# STEP 4: Combine and Save
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Saving to Parquet...")
print("=" * 60)

# Convert embeddings to a list of lists for parquet storage
df["embedding"] = [emb.tolist() for emb in embeddings]
del embeddings
gc.collect()

# Convert list columns to strings for clean parquet storage
df["authors"] = df["authors"].apply(json.dumps)
df["categories"] = df["categories"].apply(json.dumps)
df["all_categories"] = df["all_categories"].apply(json.dumps)

# Save as parquet (much smaller and faster than CSV)
OUTPUT_PATH = "/kaggle/working/scholarmind_papers_with_embeddings.parquet"
df.to_parquet(OUTPUT_PATH, index=False)

file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"\nSaved to: {OUTPUT_PATH}")
print(f"File size: {file_size_mb:.1f} MB")
print(f"Total papers: {len(df):,}")
print(f"Embedding dimension: {EMBEDDING_DIM}")

# ============================================================
# STEP 5: Quick Sanity Check
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Sanity Check")
print("=" * 60)

# Reload and verify
df_check = pd.read_parquet(OUTPUT_PATH)
print(f"Reloaded shape: {df_check.shape}")
print(f"Columns: {list(df_check.columns)}")
print(f"\nSample paper:")
sample = df_check.iloc[0]
print(f"  ID: {sample['paper_id']}")
print(f"  Title: {sample['title'][:80]}...")
print(f"  Abstract: {sample['abstract'][:100]}...")
print(f"  Authors: {sample['authors'][:80]}...")
print(f"  Categories: {sample['categories']}")
print(f"  Year: {sample['year']}")
print(f"  Embedding length: {len(json.loads(sample['embedding']) if isinstance(sample['embedding'], str) else sample['embedding'])}")

print("\n" + "=" * 60)
print("DONE! Download 'scholarmind_papers_with_embeddings.parquet'")
print("from the Output tab on the right side of this notebook.")
print("=" * 60)
