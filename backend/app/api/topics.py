"""
ScholarMind — Topics API

Endpoints for browsing BERTopic clusters and topic-based paper discovery.
"""

from fastapi import APIRouter, Query
from app.db.database import get_supabase_client

router = APIRouter()


@router.get("/topics")
async def list_topics(limit: int = Query(50, ge=1, le=200)):
    """List all topic clusters with paper counts."""
    supabase = get_supabase_client()

    # Get distinct topics with counts
    response = (
        supabase.table("papers")
        .select("topic_id, topic_label")
        .not_.is_("topic_id", "null")
        .execute()
    )

    # Aggregate counts
    from collections import Counter
    topic_counts = Counter()
    topic_labels = {}
    for r in response.data:
        tid = r["topic_id"]
        topic_counts[tid] += 1
        topic_labels[tid] = r.get("topic_label", f"Topic {tid}")

    topics = [
        {"topic_id": tid, "label": topic_labels[tid], "paper_count": count}
        for tid, count in topic_counts.most_common(limit)
        if tid != -1  # exclude outlier topic
    ]

    return {"topics": topics, "total": len(topics)}


@router.get("/topics/{topic_id}/papers")
async def get_topic_papers(
    topic_id: int,
    limit: int = Query(20, ge=1, le=100),
):
    """Get papers belonging to a specific topic cluster."""
    supabase = get_supabase_client()

    response = (
        supabase.table("papers")
        .select("paper_id, title, abstract, authors, categories, year, topic_label")
        .eq("topic_id", topic_id)
        .order("year", desc=True)
        .limit(limit)
        .execute()
    )

    return {
        "topic_id": topic_id,
        "total": len(response.data),
        "papers": response.data,
    }
