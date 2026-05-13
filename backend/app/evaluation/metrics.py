"""
ScholarMind — Recommendation Metrics Library

Pure math functions that compute standard recommendation quality metrics.
No database calls — these operate on Python lists so they can be used
in both offline evaluation scripts and online monitoring.

Metrics:
  - NDCG@K    : ranking quality (position-aware)
  - Hit Rate@K: did the relevant item appear at all?
  - Precision@K: fraction of recommended items that are relevant
  - Recall@K  : fraction of relevant items that were recommended
  - MRR       : how far down is the first relevant item?
  - Coverage  : what fraction of the catalogue gets recommended?
  - Diversity : how varied are the topics in a recommendation list?
"""

import math
from collections import Counter

import numpy as np


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    Measures ranking quality: rewards placing relevant items at the TOP.
    A relevant item at position 1 scores more than one at position 10.

    Args:
        recommended: ordered list of recommended paper_ids
        relevant: set of paper_ids the user actually interacted with
        k: cutoff position

    Returns:
        NDCG score between 0.0 and 1.0
    """
    recommended = recommended[:k]

    # DCG: sum of (1 / log2(position + 1)) for each relevant item
    dcg = 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)  # +2 because positions are 1-indexed

    # Ideal DCG: if all relevant items were at the top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int = 10) -> float:
    """
    Hit Rate @ K.

    Binary: 1.0 if ANY relevant item appears in top K, else 0.0.
    Simple but useful — "did we get at least one right?"
    """
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def precision_at_k(recommended: list[str], relevant: set[str], k: int = 10) -> float:
    """
    Precision @ K.

    Of the K items we recommended, how many were relevant?
    precision = |recommended ∩ relevant| / K
    """
    recommended = recommended[:k]
    if not recommended:
        return 0.0
    hits = sum(1 for item in recommended if item in relevant)
    return hits / len(recommended)


def recall_at_k(recommended: list[str], relevant: set[str], k: int = 10) -> float:
    """
    Recall @ K.

    Of all relevant items, how many did we recommend?
    recall = |recommended ∩ relevant| / |relevant|
    """
    if not relevant:
        return 0.0
    recommended = recommended[:k]
    hits = sum(1 for item in recommended if item in relevant)
    return hits / len(relevant)


def mrr(recommended: list[str], relevant: set[str]) -> float:
    """
    Mean Reciprocal Rank.

    1 / position_of_first_relevant_item.
    If the first relevant item is at position 3, MRR = 1/3 = 0.33.
    """
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def catalogue_coverage(
    all_recommendations: list[list[str]],
    total_items: int,
) -> float:
    """
    Catalogue Coverage.

    What fraction of ALL papers in the database ever appear in any
    user's recommendation list? Low coverage = popularity bias.

    Args:
        all_recommendations: list of recommendation lists (one per user)
        total_items: total number of papers in the catalogue

    Returns:
        Coverage between 0.0 and 1.0
    """
    if total_items == 0:
        return 0.0
    unique_recommended = set()
    for rec_list in all_recommendations:
        unique_recommended.update(rec_list)
    return len(unique_recommended) / total_items


def intra_list_diversity(
    recommended: list[str],
    paper_topics: dict[str, int],
) -> float:
    """
    Intra-List Diversity (ILD).

    Measures how many different topics appear in a single recommendation list.
    All same topic = 0.0. All different topics = 1.0.

    Args:
        recommended: list of paper_ids
        paper_topics: mapping from paper_id → topic_id

    Returns:
        Diversity score between 0.0 and 1.0
    """
    if len(recommended) <= 1:
        return 0.0

    topics = [paper_topics.get(pid, -1) for pid in recommended]
    unique_topics = len(set(topics))
    return (unique_topics - 1) / (len(topics) - 1)


def compute_all_metrics(
    recommended: list[str],
    relevant: set[str],
    k: int = 10,
    paper_topics: dict[str, int] | None = None,
) -> dict[str, float]:
    """
    Compute all metrics for a single user's recommendation list.

    Returns a dict with all metric values.
    """
    results = {
        f"ndcg@{k}": ndcg_at_k(recommended, relevant, k),
        f"hit_rate@{k}": hit_rate_at_k(recommended, relevant, k),
        f"precision@{k}": precision_at_k(recommended, relevant, k),
        f"recall@{k}": recall_at_k(recommended, relevant, k),
        "mrr": mrr(recommended, relevant),
    }

    if paper_topics:
        results[f"diversity@{k}"] = intra_list_diversity(recommended[:k], paper_topics)

    return results


def aggregate_metrics(user_metrics: list[dict[str, float]]) -> dict[str, float]:
    """
    Aggregate per-user metrics into overall averages.

    Args:
        user_metrics: list of metric dicts (one per user)

    Returns:
        Dict of averaged metrics
    """
    if not user_metrics:
        return {}

    keys = user_metrics[0].keys()
    aggregated = {}

    for key in keys:
        values = [m[key] for m in user_metrics if key in m]
        aggregated[key] = round(np.mean(values), 4)
        aggregated[f"{key}_std"] = round(np.std(values), 4)

    return aggregated
