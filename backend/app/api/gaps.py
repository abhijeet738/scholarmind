"""
ScholarMind — Research Gaps API

GET /api/v1/gaps — Find under-explored research topic intersections.
"""

from fastapi import APIRouter, Query
from app.knowledge.gap_analyzer import compute_research_gaps

router = APIRouter()


@router.get("/gaps")
async def get_research_gaps(limit: int = Query(20, ge=1, le=50)):
    """
    Find research gaps: popular topic pairs that are rarely combined.

    High gap_score = popular topics × low co-occurrence = opportunity.
    No LLM needed — pure computation over BERTopic clusters.
    """
    gaps = compute_research_gaps(limit=limit)
    return {
        "total": len(gaps),
        "gaps": gaps,
    }
