"""
ScholarMind — Monitoring & Metrics API

GET /api/v1/metrics/overview          → system health + engagement numbers
GET /api/v1/metrics/algorithms        → per-algorithm comparison
GET /api/v1/metrics/experiments       → list all A/B experiments
GET /api/v1/metrics/experiments/{name} → specific experiment results
POST /api/v1/metrics/evaluate         → trigger offline evaluation
"""

from fastapi import APIRouter, Query

from app.db.database import get_supabase_client
from app.evaluation.ab_testing import (
    create_experiment,
    start_experiment,
    end_experiment,
    get_experiment_results,
)

router = APIRouter()


@router.get("/metrics/overview")
async def metrics_overview():
    """
    System-wide health and engagement metrics.

    Returns total users, events, sessions, and engagement rates.
    """
    supabase = get_supabase_client()

    # Count users
    users = supabase.table("user_profiles").select("user_id", count="exact").execute()
    total_users = users.count or 0

    # Count events
    events = supabase.table("user_events").select("id", count="exact").execute()
    total_events = events.count or 0

    # Count sessions
    sessions = supabase.table("user_sessions").select("session_id", count="exact").execute()
    total_sessions = sessions.count or 0

    # Count saves
    saves = supabase.table("user_saves").select("user_id", count="exact").execute()
    total_saves = saves.count or 0

    # Avg interactions per user
    avg_interactions = total_events / total_users if total_users > 0 else 0

    # Save rate (saves / clicks)
    clicks = (
        supabase.table("user_events")
        .select("id", count="exact")
        .eq("event_type", "CLICK")
        .execute()
    )
    total_clicks = clicks.count or 0
    save_rate = total_saves / total_clicks if total_clicks > 0 else 0

    return {
        "total_users": total_users,
        "total_events": total_events,
        "total_sessions": total_sessions,
        "total_saves": total_saves,
        "avg_interactions_per_user": round(avg_interactions, 1),
        "save_rate": round(save_rate, 4),
    }


@router.get("/metrics/algorithms")
async def algorithm_comparison():
    """
    Compare algorithms using the most recent metric snapshots.

    Returns latest NDCG, HitRate, Precision, etc. for each algorithm.
    """
    supabase = get_supabase_client()

    # Get the most recent snapshot date
    latest = (
        supabase.table("metric_snapshots")
        .select("snapshot_date")
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )

    if not latest.data:
        return {"message": "No evaluation data yet. Run evaluate_offline.py first."}

    latest_date = latest.data[0]["snapshot_date"]

    # Get all metrics from that date
    snapshots = (
        supabase.table("metric_snapshots")
        .select("algorithm, metric_name, metric_value")
        .eq("snapshot_date", latest_date)
        .execute()
    )

    # Pivot: {algorithm: {metric: value}}
    results = {}
    for s in (snapshots.data or []):
        algo = s["algorithm"]
        if algo not in results:
            results[algo] = {}
        results[algo][s["metric_name"]] = s["metric_value"]

    return {
        "snapshot_date": latest_date,
        "algorithms": results,
    }


@router.get("/metrics/history")
async def metric_history(
    metric: str = Query("ndcg@10", description="Metric name"),
    algorithm: str = Query("sasrec_fallback", description="Algorithm name"),
    days: int = Query(30, ge=1, le=365),
):
    """
    Get historical trend of a metric for an algorithm.

    Useful for tracking improvement over time as you retrain models.
    """
    supabase = get_supabase_client()

    snapshots = (
        supabase.table("metric_snapshots")
        .select("snapshot_date, metric_value")
        .eq("metric_name", metric)
        .eq("algorithm", algorithm)
        .order("snapshot_date", desc=True)
        .limit(days)
        .execute()
    )

    return {
        "metric": metric,
        "algorithm": algorithm,
        "datapoints": snapshots.data or [],
    }


# ── A/B Testing endpoints ──

@router.get("/metrics/experiments")
async def list_experiments():
    """List all A/B experiments."""
    supabase = get_supabase_client()

    experiments = (
        supabase.table("experiments")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return {"experiments": experiments.data or []}


@router.get("/metrics/experiments/{experiment_name}")
async def experiment_results(experiment_name: str):
    """Get results of a specific A/B experiment."""
    return get_experiment_results(experiment_name)


@router.post("/metrics/experiments")
async def create_new_experiment(
    name: str = Query(...),
    description: str = Query(""),
    variant_a: str = Query("sasrec"),
    variant_b: str = Query("sasrec_fallback"),
    traffic_split: float = Query(0.5, ge=0.0, le=1.0),
):
    """Create and start a new A/B experiment."""
    experiment = create_experiment(
        name=name,
        description=description,
        variants={"A": variant_a, "B": variant_b},
        traffic_split=traffic_split,
    )
    start_experiment(name)
    return {"status": "created_and_started", "experiment": experiment}


@router.post("/metrics/evaluate")
async def trigger_evaluation(k: int = Query(10, ge=1, le=50)):
    """
    Trigger offline evaluation.

    Runs leave-one-out evaluation across all algorithms
    and saves results to metric_snapshots.
    """
    from app.evaluation.evaluator import run_offline_evaluation

    results = run_offline_evaluation(k=k, save_to_db=True)
    return {"status": "completed", "results": results}
