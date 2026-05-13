"""
ScholarMind — Scientific Consensus API

POST /api/v1/consensus — Measure scientific consensus on a question.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ConsensusRequest(BaseModel):
    question: str


class ConsensusResponse(BaseModel):
    question: str
    verdict: str
    agreement_percentage: int
    evidence_strength: str
    analysis: str
    evidence_count: int
    hallucination_score: float


@router.post("/consensus", response_model=ConsensusResponse)
async def measure_consensus(request: ConsensusRequest):
    """
    Measure scientific consensus on a research question.

    The agent will:
    1. Retrieve 20+ papers on the topic
    2. Classify each paper's stance (supports/contradicts/inconclusive)
    3. Aggregate into a consensus verdict with confidence
    """
    from app.agent.graph import get_agent
    from app.agent.state import AgentState

    agent = get_agent()
    state: AgentState = {
        "query": request.question,
        "intent": "consensus",  # force consensus path
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
    consensus = result.get("consensus_result", {})

    return ConsensusResponse(
        question=request.question,
        verdict=consensus.get("verdict", "UNKNOWN"),
        agreement_percentage=consensus.get("agreement_percentage", 0),
        evidence_strength=consensus.get("evidence_strength", "Unknown"),
        analysis=result.get("generation", "Analysis failed."),
        evidence_count=len(result.get("evidence_cards", [])),
        hallucination_score=result.get("hallucination_score", 0.0),
    )
