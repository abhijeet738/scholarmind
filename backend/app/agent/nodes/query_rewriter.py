"""
ScholarMind — Query Rewriter Node (CRAG)

When retrieved documents are mostly irrelevant, this node
rewrites the query to be more targeted and retries retrieval.
"""

from app.agent.llm import llm_json_call
from app.agent.prompts import QUERY_REWRITER_PROMPT
from app.agent.state import AgentState


def query_rewriter(state: AgentState) -> dict:
    """
    Rewrite the query for better retrieval results.

    Called when <60% of documents pass the relevance grader.
    Extracts failed topics to avoid repeating the same query.
    """
    query = state["query"]
    documents = state.get("documents", [])
    doc_grades = state.get("doc_grades", [])

    # Summarize what the bad results were about
    failed_topics = []
    for doc, grade in zip(documents, doc_grades):
        if not grade:
            failed_topics.append(doc.get("title", "unknown")[:100])

    failed_summary = "; ".join(failed_topics[:5]) if failed_topics else "unrelated topics"

    prompt = QUERY_REWRITER_PROMPT.format(
        query=query,
        failed_topics=failed_summary,
    )

    try:
        result = llm_json_call(prompt)
        rewritten = result.get("rewritten_query", query)

        # Track rewrites
        existing_rewrites = state.get("rewritten_queries", [])
        existing_rewrites.append(rewritten)

        return {
            "query": rewritten,
            "rewritten_queries": existing_rewrites,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    except Exception:
        return {
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
