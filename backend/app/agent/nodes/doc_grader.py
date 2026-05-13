"""
ScholarMind — Document Grader Node (CRAG)

Grades each retrieved document for relevance to the query.
Part of the Corrective RAG loop.
"""

from app.agent.llm import llm_json_call
from app.agent.prompts import DOC_GRADER_PROMPT
from app.agent.state import AgentState


def doc_grader(state: AgentState) -> dict:
    """
    Grade each retrieved document for relevance.

    If <60% of documents are relevant, the CRAG loop
    will trigger a query rewrite.
    """
    query = state["query"]
    documents = state.get("documents", [])
    grades = []

    for doc in documents:
        prompt = DOC_GRADER_PROMPT.format(
            query=query,
            title=doc.get("title", ""),
            abstract=doc.get("abstract", "")[:500],
        )

        try:
            result = llm_json_call(prompt)
            grades.append(result.get("relevant", False))
        except Exception:
            # On error, assume relevant to avoid dropping good docs
            grades.append(True)

    return {"doc_grades": grades}
