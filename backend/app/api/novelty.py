"""
ScholarMind — Novelty Assessment API

POST /api/v1/novelty — Assess how novel a research idea is.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class NoveltyRequest(BaseModel):
    idea: str


class NoveltyResponse(BaseModel):
    idea: str
    assessment: str
    novelty_scores: dict
    prior_art_count: int
    hallucination_score: float


@router.post("/novelty", response_model=NoveltyResponse)
async def assess_novelty(request: NoveltyRequest):
    """
    Assess the novelty of a research idea.

    The agent will:
    1. Parse the idea into method/domain/application
    2. Search for closest prior art
    3. Score novelty on 3 dimensions (0-10 each)
    4. Suggest differentiation strategies
    """
    from app.agent.graph import get_agent
    from app.agent.state import AgentState

    agent = get_agent()
    state: AgentState = {
        "query": request.idea,
        "intent": "novelty",  # force novelty path
        "documents": [],
        "doc_grades": [],
        "rewritten_queries": [],
        "iteration_count": 0,
        "generation": "",
        "hallucination_check": False,
        "hallucination_score": 0.0,
        "error": None,
    }

    result = agent.invoke(state)

    return NoveltyResponse(
        idea=request.idea,
        assessment=result.get("generation", "Assessment failed."),
        novelty_scores=result.get("novelty_scores", {}),
        prior_art_count=len(result.get("prior_art", [])),
        hallucination_score=result.get("hallucination_score", 0.0),
    )
