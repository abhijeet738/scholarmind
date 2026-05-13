"""
ScholarMind — Cross-Encoder Reranker (BAAI/bge-reranker-base)

Takes the top-20 candidates from RRF fusion and re-scores them
using a cross-encoder that processes (query, document) pairs together
through full transformer cross-attention.

This is the "retrieve-then-rerank" paradigm used by Google and Bing.
Cross-encoders are ~10-30% more accurate than bi-encoders but too
slow to run on the full corpus — hence we only use them on top candidates.
"""

# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder
from app.config import get_settings

# Module-level model (loaded once, ~1.1 GB)
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        settings = get_settings()
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """
    Re-rank candidates using the cross-encoder.

    The cross-encoder processes each (query, document) pair through
    full transformer attention, producing a much more accurate
    relevance score than bi-encoder cosine similarity.

    Args:
        query: The user's original search query.
        candidates: List of paper dicts from RRF fusion.
        top_k: Number of final results to return.

    Returns:
        List of paper dicts with reranker scores, sorted by final relevance.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    # Build (query, document) pairs for the cross-encoder
    # We use "Title. Abstract" as the document text
    pairs = [
        (query, f"{paper['title']}. {paper['abstract'][:500]}")
        for paper in candidates
    ]

    # Score all pairs in a single batch
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Attach scores and sort
    scored_papers = []
    for paper, score in zip(candidates, scores):
        paper_copy = paper.copy()
        paper_copy["reranker_score"] = float(score)
        scored_papers.append(paper_copy)

    scored_papers.sort(key=lambda p: p["reranker_score"], reverse=True)

    return scored_papers[:top_k]
