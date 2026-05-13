"""
ScholarMind — Offline Evaluation Script

Run this after generating synthetic data to measure how good
each recommendation algorithm is.

Usage:
    python scripts/evaluate_offline.py

Output:
    A table comparing all algorithms on NDCG@10, HitRate@10,
    Precision@10, Recall@10, MRR, Coverage, and Diversity.
"""

import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.evaluation.evaluator import run_offline_evaluation


def main():
    print("=" * 60)
    print("ScholarMind — Offline Recommendation Evaluation")
    print("=" * 60)
    print()

    results = run_offline_evaluation(k=10, save_to_db=True)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    # Print comparison table
    print()
    print("=" * 80)
    print(f"{'Algorithm':<22} {'NDCG@10':>9} {'HitRate':>9} {'Prec@10':>9} "
          f"{'Recall':>9} {'MRR':>9} {'Diversity':>9} {'Coverage':>9}")
    print("-" * 80)

    for algo_name, metrics in results.items():
        if "error" in metrics:
            print(f"{algo_name:<22} {'ERROR':>9}")
            continue

        print(
            f"{algo_name:<22} "
            f"{metrics.get('ndcg@10', 0):>9.4f} "
            f"{metrics.get('hit_rate@10', 0):>9.4f} "
            f"{metrics.get('precision@10', 0):>9.4f} "
            f"{metrics.get('recall@10', 0):>9.4f} "
            f"{metrics.get('mrr', 0):>9.4f} "
            f"{metrics.get('diversity@10', 0):>9.4f} "
            f"{metrics.get('coverage', 0):>9.4f}"
        )

    print("=" * 80)
    print()
    print("Results saved to metric_snapshots table in Supabase.")
    print("You can track these over time as you retrain models.")


if __name__ == "__main__":
    main()
