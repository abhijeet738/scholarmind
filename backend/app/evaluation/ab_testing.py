"""
ScholarMind — A/B Testing Infrastructure

Simple, database-driven A/B testing:
1. Create an experiment (e.g. "sasrec_vs_fallback")
2. Users are assigned to variants using a deterministic hash
3. The recommendation pipeline checks the user's variant and serves accordingly
4. After enough data, compute which variant wins
"""

import hashlib
from datetime import datetime, timezone

from app.db.database import get_supabase_client


def create_experiment(
    name: str,
    description: str,
    variants: dict[str, str],
    traffic_split: float = 0.5,
) -> dict:
    """
    Create a new A/B experiment.

    Args:
        name: unique experiment name (e.g. "sasrec_vs_fallback")
        description: what we're testing
        variants: {"A": "sasrec", "B": "sasrec_fallback"}
        traffic_split: fraction of users in variant A (0.5 = 50/50)

    Returns:
        The created experiment record.
    """
    supabase = get_supabase_client()

    experiment = {
        "name": name,
        "description": description,
        "variants": variants,
        "traffic_split": traffic_split,
        "status": "draft",
    }

    response = supabase.table("experiments").insert(experiment).execute()
    return response.data[0] if response.data else experiment


def start_experiment(experiment_name: str) -> dict:
    """Activate an experiment (start assigning users)."""
    supabase = get_supabase_client()

    response = (
        supabase.table("experiments")
        .update({
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("name", experiment_name)
        .execute()
    )

    return response.data[0] if response.data else {}


def end_experiment(experiment_name: str) -> dict:
    """End an experiment."""
    supabase = get_supabase_client()

    response = (
        supabase.table("experiments")
        .update({
            "status": "completed",
            "ended_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("name", experiment_name)
        .execute()
    )

    return response.data[0] if response.data else {}


def get_user_variant(user_id: str, experiment_name: str) -> str | None:
    """
    Get which variant a user is assigned to for an experiment.

    Uses deterministic hashing: hash(user_id + experiment_name) → A or B.
    This ensures:
    - Same user always gets the same variant (consistent experience)
    - No database lookup needed after first assignment
    - Statistically random distribution

    Returns:
        "A" or "B", or None if experiment doesn't exist / isn't running.
    """
    supabase = get_supabase_client()

    # Check if experiment is running
    experiment = (
        supabase.table("experiments")
        .select("experiment_id, traffic_split, status")
        .eq("name", experiment_name)
        .single()
        .execute()
    )

    if not experiment.data or experiment.data["status"] != "running":
        return None

    exp_id = experiment.data["experiment_id"]
    traffic_split = experiment.data["traffic_split"]

    # Check if already assigned
    existing = (
        supabase.table("experiment_assignments")
        .select("variant")
        .eq("experiment_id", exp_id)
        .eq("user_id", user_id)
        .execute()
    )

    if existing.data:
        return existing.data[0]["variant"]

    # Deterministic hash assignment
    hash_input = f"{user_id}:{experiment_name}"
    hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
    hash_fraction = (hash_value % 10000) / 10000.0  # 0.0 to 1.0

    variant = "A" if hash_fraction < traffic_split else "B"

    # Save assignment
    supabase.table("experiment_assignments").insert({
        "experiment_id": exp_id,
        "user_id": user_id,
        "variant": variant,
    }).execute()

    return variant


def get_experiment_results(experiment_name: str) -> dict:
    """
    Compute A/B test results: CTR per variant.

    CTR = (clicked recommendations) / (total recommendations shown)
    """
    supabase = get_supabase_client()

    # Get experiment
    experiment = (
        supabase.table("experiments")
        .select("experiment_id, variants")
        .eq("name", experiment_name)
        .single()
        .execute()
    )

    if not experiment.data:
        return {"error": "Experiment not found"}

    exp_id = experiment.data["experiment_id"]
    variants = experiment.data["variants"]

    # Get all user assignments for this experiment
    assignments = (
        supabase.table("experiment_assignments")
        .select("user_id, variant")
        .eq("experiment_id", exp_id)
        .execute()
    )

    if not assignments.data:
        return {"error": "No users assigned yet"}

    # Group users by variant
    variant_users = {"A": [], "B": []}
    for a in assignments.data:
        variant_users[a["variant"]].append(a["user_id"])

    # Compute CTR for each variant from recommendation_log
    results = {}
    for variant_key, user_ids in variant_users.items():
        if not user_ids:
            results[variant_key] = {
                "algorithm": variants.get(variant_key, "unknown"),
                "users": 0,
                "total_shown": 0,
                "total_clicked": 0,
                "ctr": 0.0,
            }
            continue

        # Count recommendations shown and clicked
        recs = (
            supabase.table("recommendation_log")
            .select("was_clicked")
            .in_("user_id", user_ids[:200])
            .execute()
        )

        total_shown = len(recs.data) if recs.data else 0
        total_clicked = sum(1 for r in (recs.data or []) if r.get("was_clicked"))

        ctr = total_clicked / total_shown if total_shown > 0 else 0.0

        results[variant_key] = {
            "algorithm": variants.get(variant_key, "unknown"),
            "users": len(user_ids),
            "total_shown": total_shown,
            "total_clicked": total_clicked,
            "ctr": round(ctr, 4),
        }

    # Determine winner
    ctr_a = results.get("A", {}).get("ctr", 0)
    ctr_b = results.get("B", {}).get("ctr", 0)

    if ctr_a > ctr_b:
        winner = "A"
        lift = ((ctr_a - ctr_b) / ctr_b * 100) if ctr_b > 0 else 0
    elif ctr_b > ctr_a:
        winner = "B"
        lift = ((ctr_b - ctr_a) / ctr_a * 100) if ctr_a > 0 else 0
    else:
        winner = "tie"
        lift = 0

    return {
        "experiment": experiment_name,
        "variants": results,
        "winner": winner,
        "lift_percent": round(lift, 2),
    }
