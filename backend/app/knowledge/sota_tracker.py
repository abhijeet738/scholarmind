"""
ScholarMind — SOTA Benchmark Tracker

Queries the results table to build leaderboards for any
dataset+metric combination. Pure SQL/Python — no ML model needed.
"""

from app.db.database import get_supabase_client


def get_leaderboard(dataset_name: str, metric_name: str, limit: int = 20) -> list[dict]:
    """
    Get SOTA leaderboard for a dataset+metric pair.

    Returns ranked list of methods with their scores and paper IDs.
    """
    supabase = get_supabase_client()

    response = (
        supabase.table("results")
        .select("method_name, value, paper_id")
        .ilike("dataset_name", f"%{dataset_name}%")
        .ilike("metric_name", f"%{metric_name}%")
        .order("value", desc=True)
        .limit(limit)
        .execute()
    )

    return response.data


def get_method_benchmarks(method_name: str) -> list[dict]:
    """Get all benchmark results for a specific method."""
    supabase = get_supabase_client()

    response = (
        supabase.table("results")
        .select("dataset_name, metric_name, value, paper_id")
        .ilike("method_name", f"%{method_name}%")
        .order("dataset_name")
        .execute()
    )

    return response.data


def get_all_datasets() -> list[str]:
    """Get list of all unique datasets in the results table."""
    supabase = get_supabase_client()

    response = (
        supabase.table("results")
        .select("dataset_name")
        .execute()
    )

    datasets = sorted(set(r["dataset_name"] for r in response.data))
    return datasets


def get_dataset_timeline(dataset_name: str, metric_name: str) -> list[dict]:
    """
    Get the SOTA progress over time for a dataset.

    Returns papers ordered by publication date showing how the
    best score on this benchmark has improved.
    """
    supabase = get_supabase_client()

    response = (
        supabase.table("results")
        .select("method_name, value, paper_id")
        .ilike("dataset_name", f"%{dataset_name}%")
        .ilike("metric_name", f"%{metric_name}%")
        .order("value", desc=True)
        .execute()
    )

    return response.data
