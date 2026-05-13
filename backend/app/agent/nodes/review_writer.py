"""
ScholarMind — Literature Review Writer Node

Multi-step pipeline:
1. Decompose topic into sub-topics
2. Targeted retrieval per sub-topic (uses Phase 1 search)
3. Synthesize each section
4. Assemble final review with gaps + references
"""

from app.agent.llm import llm_json_call, llm_call
from app.agent.prompts import (
    LIT_REVIEW_DECOMPOSE_PROMPT,
    LIT_REVIEW_SECTION_PROMPT,
    LIT_REVIEW_ASSEMBLE_PROMPT,
)
from app.agent.state import AgentState
from app.retrieval.dense import dense_search
from app.retrieval.sparse import sparse_search
from app.retrieval.fusion import reciprocal_rank_fusion


def _search_papers(query: str, top_k: int = 5) -> list[dict]:
    """Run hybrid search using Phase 1 retrieval engine."""
    try:
        dense_results = dense_search(query, top_k=top_k * 2)
        sparse_results = sparse_search(query, top_k=top_k * 2)
        fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)
        return fused
    except Exception:
        return []


def review_writer(state: AgentState) -> dict:
    """
    Generate a comprehensive literature review.

    Steps:
    1. Decompose the topic into 4-6 sub-queries
    2. Search for papers per sub-query
    3. Synthesize each section with LLM
    4. Assemble the final Markdown review
    """
    topic = state["query"]

    # ── Step 1: Decompose topic into sub-topics ──
    decompose_prompt = LIT_REVIEW_DECOMPOSE_PROMPT.format(topic=topic)
    try:
        decomposition = llm_json_call(decompose_prompt)
        sub_topics = decomposition.get("sub_topics", [])
    except Exception:
        # Fallback: use the topic directly
        sub_topics = [{"title": topic, "search_query": topic}]

    if not sub_topics:
        sub_topics = [{"title": topic, "search_query": topic}]

    sub_queries = [st["search_query"] for st in sub_topics]

    # ── Step 2: Targeted retrieval per sub-topic ──
    all_documents = []
    section_papers = {}

    for st in sub_topics:
        papers = _search_papers(st["search_query"], top_k=5)
        section_papers[st["title"]] = papers
        all_documents.extend(papers)

    # ── Step 3: Synthesize each section ──
    section_drafts = []

    for st in sub_topics:
        papers = section_papers.get(st["title"], [])
        if not papers:
            continue

        papers_text = "\n\n".join(
            f"Paper: {p.get('title', 'Untitled')}\n"
            f"Authors: {', '.join(p.get('authors', [])[:3])}\n"
            f"Year: {p.get('year', 'N/A')}\n"
            f"Abstract: {p.get('abstract', '')[:400]}"
            for p in papers
        )

        section_prompt = LIT_REVIEW_SECTION_PROMPT.format(
            section_title=st["title"],
            papers_text=papers_text,
        )

        try:
            section_content = llm_call(section_prompt, temperature=0.4)
        except Exception:
            section_content = f"Section on {st['title']} — papers found but synthesis failed."

        section_drafts.append({
            "title": st["title"],
            "content": section_content,
            "papers": [p.get("title", "") for p in papers],
        })

    # ── Step 4: Assemble final review ──
    sections_text = "\n\n---\n\n".join(
        f"## {sd['title']}\n\n{sd['content']}"
        for sd in section_drafts
    )

    assemble_prompt = LIT_REVIEW_ASSEMBLE_PROMPT.format(
        topic=topic,
        sections_text=sections_text,
    )

    try:
        final_review = llm_call(assemble_prompt, temperature=0.3)
    except Exception:
        # Fallback: just concatenate sections
        final_review = f"# Literature Review: {topic}\n\n{sections_text}"

    return {
        "generation": final_review,
        "sub_queries": sub_queries,
        "section_drafts": section_drafts,
        "documents": all_documents,
    }
