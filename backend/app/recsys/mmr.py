"""
ScholarMind — MMR (Maximal Marginal Relevance) Diversity Re-ranking

Ensures the final recommendation list is diverse — not all papers
about the same narrow topic. Works Day 1 with no training.

Formula:
    MMR(dᵢ) = λ × Relevance(dᵢ) - (1-λ) × max[Sim(dᵢ, dⱼ)]
                                             for dⱼ in already_selected

λ = 0.7 means 70% relevance, 30% diversity

Reference: Carbonell & Goldstein, "The Use of MMR, Diversity-Based
           Reranking for Reordering Documents", SIGIR 1998
"""

import numpy as np


def mmr_rerank(
    candidates: list[dict],
    top_k: int = 10,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Re-rank candidates using Maximal Marginal Relevance.

    Args:
        candidates: list of dicts with 'final_score' and 'embedding' keys
        top_k: number of results to return
        lambda_param: balance between relevance (1.0) and diversity (0.0)

    Returns:
        Re-ranked list of candidates
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        return candidates

    # Extract embeddings (if available)
    has_embeddings = all(c.get("embedding") is not None for c in candidates)

    if not has_embeddings:
        # Without embeddings, use topic diversity as proxy
        return _topic_diverse_rerank(candidates, top_k)

    embeddings = np.array([c["embedding"] for c in candidates])

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_norm = embeddings / norms

    # Similarity matrix
    sim_matrix = embeddings_norm @ embeddings_norm.T

    # Normalize relevance scores to [0, 1]
    scores = np.array([c.get("final_score", 0) for c in candidates])
    if scores.max() > scores.min():
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores_norm = np.ones_like(scores)

    # Greedy MMR selection
    selected_indices = []
    remaining = set(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        if not remaining:
            break

        best_idx = None
        best_mmr = float("-inf")

        for idx in remaining:
            relevance = scores_norm[idx]

            if selected_indices:
                max_sim = max(sim_matrix[idx][j] for j in selected_indices)
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


def _topic_diverse_rerank(
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Diversity re-ranking using topic IDs when embeddings aren't available.

    Strategy: greedy selection ensuring no more than 3 papers per topic.
    """
    from collections import defaultdict

    topic_counts = defaultdict(int)
    selected = []
    remaining = list(candidates)

    # Sort by score descending
    remaining.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    max_per_topic = 3

    for candidate in remaining:
        if len(selected) >= top_k:
            break

        topic = candidate.get("topic_id", -1)
        if topic_counts[topic] < max_per_topic:
            selected.append(candidate)
            topic_counts[topic] += 1

    # If we didn't get enough, add remaining regardless of topic
    if len(selected) < top_k:
        for candidate in remaining:
            if candidate not in selected and len(selected) < top_k:
                selected.append(candidate)

    return selected
