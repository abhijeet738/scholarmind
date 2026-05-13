"""
ScholarMind — Reciprocal Rank Fusion (RRF)

Merges ranked lists from Dense (vector) and Sparse (BM25) retrieval
into a single, unified ranking. RRF is used by Elasticsearch 8+,
Pinecone, and Weaviate — it's the industry standard for hybrid search.

Formula: RRF_score(doc) = Σ 1 / (k + rank_i(doc))
Where k=60 (standard constant) and rank_i is the rank from system i.
"""

from app.config import get_settings


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int = 20,
    k: int = 60,
) -> list[dict]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    Args:
        dense_results: Papers from dense (vector) search, ranked by similarity.
        sparse_results: Papers from sparse (BM25) search, ranked by BM25 score.
        top_k: Number of fused results to return.
        k: RRF constant (default 60, the standard value).

    Returns:
        List of paper dicts with RRF scores, sorted by fused relevance.
    """
    rrf_scores: dict[str, float] = {}
    paper_data: dict[str, dict] = {}

    # Score from dense retrieval
    for rank, paper in enumerate(dense_results):
        pid = paper["paper_id"]
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        paper_data[pid] = paper

    # Score from sparse retrieval
    for rank, paper in enumerate(sparse_results):
        pid = paper["paper_id"]
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        # Keep the richer metadata (dense results have similarity scores)
        if pid not in paper_data:
            paper_data[pid] = paper

    # Sort by fused RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)

    results = []
    for pid in sorted_ids[:top_k]:
        paper = paper_data[pid].copy()
        paper["rrf_score"] = rrf_scores[pid]
        results.append(paper)

    return results
