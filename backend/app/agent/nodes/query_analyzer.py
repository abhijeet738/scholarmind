"""
ScholarMind — Query Analyzer Node

First node in the LangGraph. Classifies the user's intent
and extracts key search terms using Gemini.
"""

from app.agent.llm import llm_json_call
from app.agent.prompts import QUERY_ANALYZER_PROMPT
from app.agent.state import AgentState


def query_analyzer(state: AgentState) -> dict:
    """
    Analyze user query to determine intent and extract key terms.

    Intent types:
    - search: find specific papers
    - lit_review: generate a literature review
    - novelty: assess research idea novelty
    - consensus: measure scientific consensus
    - gap_analysis: find research gaps
    """
    query = state["query"]
    prompt = QUERY_ANALYZER_PROMPT.format(query=query)

    try:
        result = llm_json_call(prompt)
        return {
            "intent": result.get("intent", "search"),
            "query": query,
        }
    except Exception as e:
        # Default to search on failure
        return {
            "intent": "search",
            "query": query,
            "error": f"Query analysis failed: {str(e)}",
        }
