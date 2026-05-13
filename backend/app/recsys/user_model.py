"""
ScholarMind — User Model

Builds and maintains user taste vectors — a running weighted average
of paper embeddings based on interactions. This is the foundation
for content-based recommendations and the DeepFM feature vector.
"""

import numpy as np
from app.db.database import get_supabase_client


def get_or_create_profile(user_id: str, username: str | None = None) -> dict:
    """Get user profile, creating one if it doesn't exist."""
    supabase = get_supabase_client()

    response = (
        supabase.table("user_profiles")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )
    
    if response.data:
        return response.data[0]

    # Create new profile
    profile = {
        "user_id": user_id,
        "username": username,
        "total_interactions": 0,
        "preferred_topics": [],
        "exploration_alpha": [],
        "exploration_beta": [],
    }

    supabase.table("user_profiles").insert(profile).execute()
    return profile


def update_taste_vector(user_id: str):
    """
    Recompute user taste vector from recent interactions.

    The taste vector is a weighted average of paper embeddings,
    where weights come from event_weight (SAVE=4.0 matters more
    than CLICK=1.0) and recency (recent papers matter more).
    """
    supabase = get_supabase_client()

    # Get recent interactions with weights
    events = (
        supabase.table("user_events")
        .select("paper_id, event_weight, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(200)
        .execute()
    )

    if not events.data:
        return

    # Get embeddings for interacted papers
    paper_ids = list(set(e["paper_id"] for e in events.data))
    papers = (
        supabase.table("papers")
        .select("paper_id, embedding, topic_id")
        .in_("paper_id", paper_ids[:100])
        .execute()
    )

    if not papers.data:
        return

    # Build paper_id → embedding lookup
    paper_embeddings = {}
    paper_topics = {}
    for p in papers.data:
        if p.get("embedding"):
            paper_embeddings[p["paper_id"]] = np.array(p["embedding"])
            paper_topics[p["paper_id"]] = p.get("topic_id")

    if not paper_embeddings:
        return

    # Compute weighted average embedding
    # Weight = event_weight × recency_decay
    total_weight = 0.0
    weighted_sum = np.zeros(len(next(iter(paper_embeddings.values()))))

    for i, event in enumerate(events.data):
        pid = event["paper_id"]
        if pid not in paper_embeddings:
            continue

        recency_decay = 1.0 / (1.0 + i * 0.05)  # newer = higher weight
        weight = event["event_weight"] * recency_decay
        weighted_sum += paper_embeddings[pid] * weight
        total_weight += weight

    if total_weight > 0:
        taste_vector = (weighted_sum / total_weight).tolist()
    else:
        return

    # Find preferred topics (top 5 by interaction count)
    from collections import Counter
    topic_counts = Counter()
    for event in events.data:
        topic = paper_topics.get(event["paper_id"])
        if topic is not None and topic != -1:
            topic_counts[topic] += 1
    preferred_topics = [tid for tid, _ in topic_counts.most_common(5)]

    # Update profile
    supabase.table("user_profiles").update({
        "taste_vector": taste_vector,
        "preferred_topics": preferred_topics,
    }).eq("user_id", user_id).execute()


def get_taste_vector(user_id: str) -> np.ndarray | None:
    """Get the user's taste vector as a numpy array."""
    supabase = get_supabase_client()

    response = (
        supabase.table("user_profiles")
        .select("taste_vector")
        .eq("user_id", user_id)
        .single()
        .execute()
    )

    if response.data and response.data.get("taste_vector"):
        return np.array(response.data["taste_vector"])
    return None


def content_based_score(user_id: str, paper_embedding: np.ndarray) -> float:
    """
    Score a candidate paper by cosine similarity to user taste vector.

    This is the simplest personalization signal — works from
    the very first interaction.
    """
    taste = get_taste_vector(user_id)
    if taste is None:
        return 0.0

    # Cosine similarity
    dot = np.dot(taste, paper_embedding)
    norm = np.linalg.norm(taste) * np.linalg.norm(paper_embedding)
    if norm == 0:
        return 0.0
    return float(dot / norm)
