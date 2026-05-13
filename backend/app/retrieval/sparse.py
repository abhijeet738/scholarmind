"""
ScholarMind — Sparse Retrieval via BM25

Loads a pre-built BM25 index from disk and performs keyword-based
retrieval. The index is built during the ingestion phase from
paper titles and abstracts.
"""

import pickle
import re
from pathlib import Path
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
from app.config import get_settings

# Module-level index (loaded once)
_bm25_index: BM25Okapi | None = None
_paper_ids: list[str] | None = None
_paper_metadata: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer with basic cleanup."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    tokens = text.split()
    # Remove very short tokens
    return [t for t in tokens if len(t) > 1]


def _load_bm25_index():
    """Load the pre-built BM25 index from disk."""
    global _bm25_index, _paper_ids, _paper_metadata

    settings = get_settings()
    index_path = Path(settings.bm25_index_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {index_path}. "
            "Run the ingestion pipeline first to build it."
        )

    with open(index_path, "rb") as f:
        data = pickle.load(f)

    _bm25_index = data["bm25"]
    _paper_ids = data["paper_ids"]
    _paper_metadata = data["paper_metadata"]


def sparse_search(query: str, top_k: int = 50) -> list[dict]:
    """
    Perform BM25 keyword search over the paper corpus.

    Args:
        query: The user's search query string.
        top_k: Number of results to return.

    Returns:
        List of paper dicts with BM25 scores, sorted by relevance.
    """
    global _bm25_index, _paper_ids, _paper_metadata

    # Lazy-load index
    if _bm25_index is None:
        _load_bm25_index()

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Get BM25 scores for all documents
    scores = _bm25_index.get_scores(query_tokens)

    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # only include docs with non-zero relevance
            paper = _paper_metadata[idx].copy()
            paper["bm25_score"] = float(scores[idx])
            results.append(paper)

    return results
