"""
ScholarMind — Recommendations API

GET /api/v1/recommendations — Get personalized paper recommendations.
"""

from fastapi import APIRouter, Query

from app.recsys.mixer import get_recommendations

router = APIRouter()


@router.get("/recommendations")
async def recommend(
    user_id: str = Query(..., description="User UUID"),
    session_id: str | None = Query(None, description="Current session UUID"),
    query: str | None = Query(None, description="Optional search query"),
    limit: int = Query(10, ge=1, le=50),
    diversity: float = Query(0.7, ge=0.0, le=1.0, description="MMR diversity (0=pure relevance, 1=max diversity)"),
):
    """
    Get personalized recommendations using the 3-stage pipeline:

    1. **Candidate Generation**: SASRec + LightGCN + KGAT + Search
    2. **Scoring**: DeepFM context-aware scoring
    3. **Re-ranking**: MMR diversity + DQN exploration + PGPR explanations

    Each recommendation includes:
    - Paper metadata (title, abstract, authors)
    - Final score
    - Source algorithm (which model suggested it)
    - Explanation (why this paper was recommended)
    """
    results = get_recommendations(
        user_id=user_id,
        session_id=session_id,
        query=query,
        top_k=limit,
        lambda_diversity=diversity,
    )

    # Clean up response (remove embeddings)
    clean_results = []
    for r in results:
        clean = {
            "paper_id": r.get("paper_id"),
            "title": r.get("title", ""),
            "abstract": r.get("abstract", "")[:300],
            "authors": r.get("authors", []),
            "year": r.get("year"),
            "categories": r.get("categories", []),
            "topic_label": r.get("topic_label"),
            "score": round(r.get("final_score", 0), 4),
            "source": r.get("source", "unknown"),
            "sources": r.get("sources", []),
            "explanation": r.get("explanation", ""),
            "was_explored": r.get("was_explored", False),
        }
        clean_results.append(clean)

    return {
        "user_id": user_id,
        "total": len(clean_results),
        "recommendations": clean_results,
    }
