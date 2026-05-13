"""
ScholarMind — Evaluator Orchestrator

Runs the full evaluation pipeline:
1. Loads user interaction data from Supabase
2. Splits into train (80%) and test (20%) by time
3. Runs each recommendation algorithm on test users
4. Computes all metrics per algorithm
5. Saves results to metric_snapshots table
"""

from datetime import datetime, timezone

import numpy as np

from app.db.database import get_supabase_client
from app.evaluation.metrics import (
    compute_all_metrics,
    aggregate_metrics,
    catalogue_coverage,
)


def run_offline_evaluation(k: int = 10, save_to_db: bool = True) -> dict:
    """
    Run offline evaluation on existing interaction data.

    Strategy (leave-one-out):
    - For each user, hold out their LAST interaction as ground truth
    - Use everything else to generate recommendations
    - Measure if the held-out item appears in the top K

    Returns:
        Dict with per-algorithm aggregated metrics
    """
    supabase = get_supabase_client()

    print("Loading user sessions...")

    # Get all completed sessions with sequences
    sessions = (
        supabase.table("user_sessions")
        .select("user_id, paper_sequence")
        .eq("is_active", False)
        .execute()
    )

    if not sessions.data:
        return {"error": "No completed sessions found"}

    # Filter sessions with at least 3 papers (need train + test)
    valid_sessions = [
        s for s in sessions.data
        if s.get("paper_sequence") and len(s["paper_sequence"]) >= 3
    ]

    if not valid_sessions:
        return {"error": "No sessions with enough papers (need 3+)"}

    print(f"Found {len(valid_sessions)} valid sessions.")

    # Get paper topics for diversity metric
    papers = (
        supabase.table("papers")
        .select("paper_id, topic_id")
        .limit(10000)
        .execute()
    )
    paper_topics = {p["paper_id"]: p.get("topic_id", -1) for p in (papers.data or [])}
    total_papers = len(paper_topics)

    # ── Evaluate each algorithm ──
    algorithms = {
        "sasrec_fallback": _eval_sasrec_fallback,
        "lightgcn_fallback": _eval_jaccard_fallback,
        "popularity": _eval_popularity,
        "random": _eval_random,
    }

    all_results = {}

    for algo_name, eval_fn in algorithms.items():
        print(f"Evaluating {algo_name}...")

        user_metrics = []
        all_recs = []

        for session in valid_sessions:
            sequence = session["paper_sequence"]
            user_id = session["user_id"]

            # Leave-one-out: train on all but last, test on last
            train_seq = sequence[:-1]
            test_item = sequence[-1]
            ground_truth = {test_item}

            # Get recommendations from this algorithm
            try:
                recommended = eval_fn(user_id, train_seq, k=k)
            except Exception:
                continue

            if not recommended:
                continue

            # Compute metrics
            metrics = compute_all_metrics(
                recommended, ground_truth, k=k, paper_topics=paper_topics
            )
            user_metrics.append(metrics)
            all_recs.append(recommended)

        if not user_metrics:
            all_results[algo_name] = {"error": "No valid evaluations"}
            continue

        # Aggregate
        agg = aggregate_metrics(user_metrics)
        agg["coverage"] = round(catalogue_coverage(all_recs, total_papers), 4)
        agg["num_users_evaluated"] = len(user_metrics)
        all_results[algo_name] = agg

        print(f"  {algo_name}: NDCG@{k}={agg.get(f'ndcg@{k}', 0):.4f}, "
              f"HitRate@{k}={agg.get(f'hit_rate@{k}', 0):.4f}")

    # Save to database
    if save_to_db:
        _save_metrics_to_db(all_results)

    return all_results


# ============================================================
# Algorithm evaluation functions
# ============================================================

def _eval_sasrec_fallback(user_id: str, train_seq: list[str], k: int) -> list[str]:
    """Evaluate SASRec fallback (weighted embedding average)."""
    from app.recsys.sasrec import _fallback_predict

    results = _fallback_predict(train_seq, top_k=k)
    return [r["paper_id"] for r in results]


def _eval_jaccard_fallback(user_id: str, train_seq: list[str], k: int) -> list[str]:
    """Evaluate LightGCN fallback (Jaccard CF)."""
    from app.recsys.lightgcn import _fallback_jaccard

    results = _fallback_jaccard(user_id, top_k=k, exclude_paper_ids=train_seq)
    return [r["paper_id"] for r in results]


def _eval_popularity(user_id: str, train_seq: list[str], k: int) -> list[str]:
    """Baseline: recommend most popular papers (by citation count)."""
    supabase = get_supabase_client()

    exclude = set(train_seq)
    response = (
        supabase.table("papers")
        .select("paper_id")
        .order("citation_count", desc=True)
        .limit(k + len(exclude))
        .execute()
    )

    results = [
        r["paper_id"] for r in (response.data or [])
        if r["paper_id"] not in exclude
    ]
    return results[:k]


def _eval_random(user_id: str, train_seq: list[str], k: int) -> list[str]:
    """Baseline: recommend random papers."""
    import random

    supabase = get_supabase_client()

    exclude = set(train_seq)
    response = (
        supabase.table("papers")
        .select("paper_id")
        .limit(500)
        .execute()
    )

    pool = [r["paper_id"] for r in (response.data or []) if r["paper_id"] not in exclude]
    return random.sample(pool, min(k, len(pool)))


# ============================================================
# Save results
# ============================================================

def _save_metrics_to_db(all_results: dict):
    """Save metric snapshots to database for historical tracking."""
    supabase = get_supabase_client()

    for algo_name, metrics in all_results.items():
        if "error" in metrics:
            continue

        for metric_name, value in metrics.items():
            if metric_name.endswith("_std") or metric_name == "num_users_evaluated":
                continue

            snapshot = {
                "metric_name": metric_name,
                "metric_value": value,
                "algorithm": algo_name,
                "metadata": {
                    "num_users": metrics.get("num_users_evaluated", 0),
                },
            }

            try:
                supabase.table("metric_snapshots").insert(snapshot).execute()
            except Exception:
                pass

    print("Metrics saved to metric_snapshots table.")
