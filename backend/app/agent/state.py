"""
ScholarMind — Agent State Definition

Shared state that flows through every node in the LangGraph.
Each node reads from and writes to this state object.
"""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state for the ScholarMind LangGraph agent.

    Every node in the graph reads from and updates this state.
    Fields marked with total=False are optional.
    """

    # ── Input ──
    query: str                          # original user query
    intent: str                         # classified intent: search, lit_review, novelty, consensus, gap_analysis

    # ── Retrieval ──
    documents: list[dict]               # retrieved papers
    doc_grades: list[bool]              # per-document relevance grades (True=relevant)
    rewritten_queries: list[str]        # CRAG query rewrites
    iteration_count: int                # CRAG loop counter (max 3)

    # ── Generation ──
    generation: str                     # LLM-generated response
    hallucination_check: bool           # did the response pass grounding check?
    hallucination_score: float          # 0.0 to 1.0 grounding confidence

    # ── Literature Review ──
    sub_queries: list[str]              # decomposed sub-topics for lit review
    section_drafts: list[dict]          # drafted sections {title, content, papers}

    # ── Novelty Assessment ──
    parsed_idea: dict                   # {method, domain, application}
    prior_art: list[dict]               # closest existing papers
    novelty_scores: dict                # {method: 0-10, domain: 0-10, combination: 0-10}

    # ── Consensus Meter ──
    evidence_cards: list[dict]          # {paper_id, stance, quote, confidence}
    consensus_result: dict              # {verdict, agreement_pct, strength}

    # ── Research Gaps ──
    gap_results: list[dict]             # {topic_a, topic_b, gap_score}

    # ── Error Handling ──
    error: str | None                   # error message if something fails
