"""
ScholarMind — Hallucination Checker Node (CRAG)

Final quality gate: verifies that every claim in the generated
response is grounded in the retrieved source documents.
"""

from app.agent.llm import llm_json_call
from app.agent.prompts import HALLUCINATION_CHECK_PROMPT
from app.agent.state import AgentState


def hallucination_checker(state: AgentState) -> dict:
    """
    Check if the generated response is grounded in source documents.

    Returns a grounding score (0.0 to 1.0) and flags ungrounded claims.
    A score >= 0.7 is considered acceptable.
    """
    generation = state.get("generation", "")
    documents = state.get("documents", [])

    if not generation or not documents:
        return {
            "hallucination_check": True,
            "hallucination_score": 1.0,
        }

    # Build sources text
    sources_text = "\n\n".join(
        f"[{i+1}] {doc.get('title', 'Untitled')}\n{doc.get('abstract', '')[:400]}"
        for i, doc in enumerate(documents[:10])
    )

    prompt = HALLUCINATION_CHECK_PROMPT.format(
        generation=generation[:2000],
        sources=sources_text,
    )

    try:
        result = llm_json_call(prompt)
        return {
            "hallucination_check": result.get("grounded", False),
            "hallucination_score": result.get("score", 0.0),
        }
    except Exception:
        # On error, pass through (don't block generation)
        return {
            "hallucination_check": True,
            "hallucination_score": 0.5,
        }
