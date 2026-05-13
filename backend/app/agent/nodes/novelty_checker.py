"""
ScholarMind — Novelty Assessor Node

Evaluates how novel a research idea is by:
1. Parsing the idea into method/domain/application
2. Searching for prior art across all three dimensions
3. Scoring novelty on 3 axes (0-10 each)
4. Suggesting differentiation strategies
"""

from app.agent.llm import llm_json_call
from app.agent.prompts import NOVELTY_PARSE_PROMPT, NOVELTY_SCORE_PROMPT
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


def novelty_checker(state: AgentState) -> dict:
    """
    Assess the novelty of a research idea.

    Steps:
    1. Parse idea into components (method, domain, application)
    2. Search for prior art across each dimension
    3. Score novelty using LLM analysis
    """
    idea = state["query"]

    # ── Step 1: Parse the idea ──
    parse_prompt = NOVELTY_PARSE_PROMPT.format(idea=idea)
    try:
        parsed = llm_json_call(parse_prompt)
    except Exception:
        parsed = {
            "method": idea,
            "domain": "general",
            "application": "research",
            "search_queries": [idea],
        }

    method = parsed.get("method", "")
    domain = parsed.get("domain", "")
    application = parsed.get("application", "")
    search_queries = parsed.get("search_queries", [idea])

    # ── Step 2: Search for prior art ──
    all_prior_art = []
    for sq in search_queries[:3]:
        papers = _search_papers(sq, top_k=5)
        all_prior_art.extend(papers)

    # Deduplicate by paper_id
    seen = set()
    unique_prior_art = []
    for p in all_prior_art:
        pid = p.get("paper_id", "")
        if pid not in seen:
            seen.add(pid)
            unique_prior_art.append(p)

    # ── Step 3: Score novelty ──
    prior_art_text = "\n\n".join(
        f"- {p.get('title', 'Untitled')} ({p.get('year', 'N/A')})\n"
        f"  Abstract: {p.get('abstract', '')[:300]}"
        for p in unique_prior_art[:10]
    )

    score_prompt = NOVELTY_SCORE_PROMPT.format(
        idea=idea,
        method=method,
        domain=domain,
        application=application,
        prior_art_text=prior_art_text if prior_art_text else "No closely related prior art found.",
    )

    try:
        scores = llm_json_call(score_prompt)
    except Exception:
        scores = {
            "method_novelty": {"score": 5, "explanation": "Could not assess"},
            "domain_novelty": {"score": 5, "explanation": "Could not assess"},
            "combination_novelty": {"score": 5, "explanation": "Could not assess"},
            "overall_novelty": 5,
            "differentiation_suggestions": [],
        }

    # ── Build response ──
    overall = scores.get("overall_novelty", 5)
    generation = (
        f"# Novelty Assessment\n\n"
        f"**Research Idea:** {idea}\n\n"
        f"## Scores\n"
        f"- **Method Novelty:** {scores.get('method_novelty', {}).get('score', '?')}/10 — "
        f"{scores.get('method_novelty', {}).get('explanation', '')}\n"
        f"- **Domain Novelty:** {scores.get('domain_novelty', {}).get('score', '?')}/10 — "
        f"{scores.get('domain_novelty', {}).get('explanation', '')}\n"
        f"- **Combination Novelty:** {scores.get('combination_novelty', {}).get('score', '?')}/10 — "
        f"{scores.get('combination_novelty', {}).get('explanation', '')}\n"
        f"- **Overall:** {overall}/10\n\n"
        f"## Closest Prior Art\n"
    )

    for p in unique_prior_art[:5]:
        generation += f"- **{p.get('title', 'Untitled')}** ({p.get('year', '')})\n"

    suggestions = scores.get("differentiation_suggestions", [])
    if suggestions:
        generation += "\n## Differentiation Suggestions\n"
        for s in suggestions:
            generation += f"- {s}\n"

    return {
        "generation": generation,
        "parsed_idea": parsed,
        "prior_art": unique_prior_art[:10],
        "novelty_scores": scores,
        "documents": unique_prior_art,
    }
