"""
ScholarMind — Consensus Builder Node

Measures scientific consensus on a factual research question:
1. Retrieves 20+ papers related to the question
2. Classifies each paper's stance (supports/contradicts/inconclusive)
3. Aggregates into a consensus verdict with confidence
"""

from app.agent.llm import llm_json_call
from app.agent.prompts import CONSENSUS_STANCE_PROMPT, CONSENSUS_VERDICT_PROMPT
from app.agent.state import AgentState
from app.retrieval.dense import dense_search
from app.retrieval.sparse import sparse_search
from app.retrieval.fusion import reciprocal_rank_fusion


def _search_papers(query: str, top_k: int = 20) -> list[dict]:
    """Run hybrid search for consensus evidence."""
    try:
        dense_results = dense_search(query, top_k=top_k * 2)
        sparse_results = sparse_search(query, top_k=top_k * 2)
        fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)
        return fused
    except Exception:
        return []


def consensus_builder(state: AgentState) -> dict:
    """
    Measure scientific consensus on a question.

    Steps:
    1. Retrieve relevant papers
    2. Classify stance of each paper
    3. Aggregate into verdict
    """
    question = state["query"]

    # ── Step 1: Retrieve evidence ──
    papers = _search_papers(question, top_k=20)

    if not papers:
        return {
            "generation": "No relevant papers found to measure consensus.",
            "evidence_cards": [],
            "consensus_result": {
                "verdict": "INSUFFICIENT",
                "agreement_percentage": 0,
                "evidence_strength": "Insufficient",
            },
        }

    # ── Step 2: Classify stance for each paper ──
    papers_text = "\n\n".join(
        f"Paper {i+1}:\n"
        f"  paper_id: {p.get('paper_id', '')}\n"
        f"  title: {p.get('title', 'Untitled')}\n"
        f"  abstract: {p.get('abstract', '')[:400]}"
        for i, p in enumerate(papers[:20])
    )

    stance_prompt = CONSENSUS_STANCE_PROMPT.format(
        question=question,
        papers_text=papers_text,
    )

    try:
        stance_result = llm_json_call(stance_prompt)
        evidence_cards = stance_result.get("evidence", [])
    except Exception:
        evidence_cards = []

    # ── Step 3: Aggregate into verdict ──
    supports = sum(1 for e in evidence_cards if e.get("stance") == "SUPPORTS")
    contradicts = sum(1 for e in evidence_cards if e.get("stance") == "CONTRADICTS")
    inconclusive = sum(1 for e in evidence_cards if e.get("stance") == "INCONCLUSIVE")

    verdict_prompt = CONSENSUS_VERDICT_PROMPT.format(
        question=question,
        supports_count=supports,
        contradicts_count=contradicts,
        inconclusive_count=inconclusive,
    )

    try:
        verdict = llm_json_call(verdict_prompt)
    except Exception:
        total = supports + contradicts + inconclusive
        agreement = (supports / max(total, 1)) * 100
        verdict = {
            "verdict": "YES" if agreement > 60 else ("NO" if contradicts > supports else "MIXED"),
            "agreement_percentage": round(agreement),
            "evidence_strength": "Moderate",
            "summary": f"{supports} papers support, {contradicts} contradict, {inconclusive} inconclusive.",
        }

    # ── Build response ──
    consensus_result = {
        "verdict": verdict.get("verdict", "MIXED"),
        "agreement_percentage": verdict.get("agreement_percentage", 0),
        "evidence_strength": verdict.get("evidence_strength", "Unknown"),
    }

    generation = (
        f"# Scientific Consensus\n\n"
        f"**Question:** {question}\n\n"
        f"## Verdict: {consensus_result['verdict']} "
        f"({consensus_result['agreement_percentage']}% agreement, "
        f"{consensus_result['evidence_strength']} evidence)\n\n"
        f"{verdict.get('summary', '')}\n\n"
        f"## Evidence Breakdown\n"
        f"- ✅ **Supports:** {supports} papers\n"
        f"- ❌ **Contradicts:** {contradicts} papers\n"
        f"- ➖ **Inconclusive:** {inconclusive} papers\n\n"
        f"## Evidence Cards\n"
    )

    for card in evidence_cards[:10]:
        emoji = {"SUPPORTS": "✅", "CONTRADICTS": "❌", "INCONCLUSIVE": "➖"}.get(card.get("stance"), "❓")
        generation += (
            f"\n### {emoji} {card.get('title', 'Untitled')}\n"
            f"> {card.get('key_quote', 'No quote available')}\n\n"
        )

    return {
        "generation": generation,
        "evidence_cards": evidence_cards,
        "consensus_result": consensus_result,
        "documents": papers,
    }
