"""
ScholarMind — Literature Review API

POST /api/v1/review — Generate a structured literature review on a topic.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.agent.graph import run_agent

router = APIRouter()


class ReviewRequest(BaseModel):
    topic: str
    depth: str = "comprehensive"  # "brief" or "comprehensive"


class ReviewResponse(BaseModel):
    topic: str
    review: str
    sub_topics: list[str]
    papers_used: int
    hallucination_score: float


@router.post("/review", response_model=ReviewResponse)
async def generate_review(request: ReviewRequest):
    """
    Generate a comprehensive literature review on a topic.

    The agent will:
    1. Decompose the topic into sub-topics
    2. Search for papers per sub-topic
    3. Synthesize each section
    4. Assemble a full Markdown review with gaps + references
    """
    # Override intent to lit_review
    from app.agent.graph import get_agent
    from app.agent.state import AgentState

    agent = get_agent()
    state: AgentState = {
        "query": request.topic,
        "intent": "lit_review",  # force lit review path
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

    return ReviewResponse(
        topic=request.topic,
        review=result.get("generation", "Review generation failed."),
        sub_topics=result.get("sub_queries", []),
        papers_used=len(result.get("documents", [])),
        hallucination_score=result.get("hallucination_score", 0.0),
    )
