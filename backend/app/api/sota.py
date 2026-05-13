"""
ScholarMind — SOTA Leaderboard API

Endpoints for querying benchmark leaderboards and method performance.
"""

from fastapi import APIRouter, Query
from app.knowledge.sota_tracker import (
    get_leaderboard,
    get_method_benchmarks,
    get_all_datasets,
    get_dataset_timeline,
)

router = APIRouter()


@router.get("/sota/leaderboard")
async def leaderboard(
    dataset: str = Query(..., description="Dataset name (e.g., MMLU, SQuAD)"),
    metric: str = Query("accuracy", description="Metric name (e.g., accuracy, f1)"),
    limit: int = Query(20, ge=1, le=50),
):
    """Get the SOTA leaderboard for a dataset+metric pair."""
    results = get_leaderboard(dataset, metric, limit)
    return {
        "dataset": dataset,
        "metric": metric,
        "leaderboard": results,
    }


@router.get("/sota/methods/{method_name}")
async def method_results(method_name: str):
    """Get all benchmark results for a specific method."""
    results = get_method_benchmarks(method_name)
    return {
        "method": method_name,
        "benchmarks": results,
    }


@router.get("/sota/datasets")
async def list_datasets():
    """List all datasets with benchmark results."""
    datasets = get_all_datasets()
    return {"datasets": datasets, "total": len(datasets)}


@router.get("/sota/timeline")
async def sota_timeline(
    dataset: str = Query(...),
    metric: str = Query("accuracy"),
):
    """Get SOTA progress over time for a benchmark."""
    results = get_dataset_timeline(dataset, metric)
    return {
        "dataset": dataset,
        "metric": metric,
        "timeline": results,
    }
