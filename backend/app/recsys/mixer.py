"""
ScholarMind — Recommendation Mixer (3-Stage Pipeline)

Orchestrates the full recommendation pipeline:
  Stage 1: Candidate Generation (SASRec + LightGCN + KGAT + Search)
  Stage 2: Scoring (Cross-Encoder + DeepFM)
  Stage 3: Re-ranking (MMR + DQN exploration + PGPR explanations)

This is the single entry point that combines all 7 algorithms.
"""

import numpy as np

from app.recsys.sasrec import predict_next_papers
from app.recsys.lightgcn import get_collaborative_picks
from app.recsys.kgat import get_kg_aware_picks
from app.recsys.deepfm import score_candidates
from app.recsys.dqn_agent import get_explore_exploit_decision
from app.recsys.pgpr import generate_explanation
from app.recsys.mmr import mmr_rerank
from app.recsys.user_model import content_based_score, get_taste_vector
from app.recsys.event_tracker import get_session_papers
from app.db.database import get_supabase_client


def get_recommendations(
    user_id: str,
    session_id: str | None = None,
    query: str | None = None,
    top_k: int = 10,
    lambda_diversity: float = 0.7,
) -> list[dict]:
    """
    Full 3-stage recommendation pipeline.

    Args:
        user_id: UUID of the user
        session_id: Current session ID (for sequential recs)
        query: Search query (if triggered by search)
        top_k: Number of final recommendations
        lambda_diversity: MMR diversity parameter (0.7 = 70% relevance)

    Returns:
        List of recommended papers with scores and explanations.
    """

    # ── Stage 1: Candidate Generation ──
    candidates = _stage1_candidate_generation(
        user_id, session_id, query, top_k * 10
    )

    if not candidates:
        return []

    # ── Stage 2: Scoring ──
    scored = _stage2_scoring(candidates, user_id, session_id)

    # ── Stage 3: Re-ranking ──
    final = _stage3_reranking(
        scored, user_id, session_id, top_k, lambda_diversity
    )

    return final


def _stage1_candidate_generation(
    user_id: str,
    session_id: str | None,
    query: str | None,
    pool_size: int,
) -> list[dict]:
    """
    Stage 1: Generate candidate pool from multiple sources.

    Sources:
      A. SASRec (session-based sequential prediction)
      B. LightGCN (collaborative filtering)
      C. KGAT (knowledge-graph-aware)
      D. Search results (if query provided)
    """
    all_candidates = {}

    # Get papers the user already read (to exclude)
    exclude = _get_user_read_papers(user_id)

    # Source A: SASRec — session sequential predictions
    if session_id:
        session_papers = get_session_papers(session_id)
        if session_papers:
            sasrec_picks = predict_next_papers(session_papers, top_k=pool_size // 4)
            for p in sasrec_picks:
                pid = p["paper_id"]
                if pid not in all_candidates:
                    all_candidates[pid] = {**p, "sources": [p["source"]]}
                else:
                    all_candidates[pid]["sources"].append(p["source"])

    # Source B: LightGCN — collaborative filtering
    collab_picks = get_collaborative_picks(user_id, top_k=pool_size // 4, exclude_paper_ids=exclude)
    for p in collab_picks:
        pid = p["paper_id"]
        if pid not in all_candidates:
            all_candidates[pid] = {**p, "sources": [p["source"]]}
        else:
            all_candidates[pid]["sources"].append(p["source"])

    # Source C: KGAT — knowledge-graph-aware
    kg_picks = get_kg_aware_picks(user_id, top_k=pool_size // 4, exclude_paper_ids=exclude)
    for p in kg_picks:
        pid = p["paper_id"]
        if pid not in all_candidates:
            all_candidates[pid] = {**p, "sources": [p["source"]]}
        else:
            all_candidates[pid]["sources"].append(p["source"])

    # Source D: Search results (if query)
    if query:
        from app.retrieval.dense import dense_search
        from app.retrieval.sparse import sparse_search
        from app.retrieval.fusion import reciprocal_rank_fusion

        try:
            dense = dense_search(query, top_k=pool_size // 4)
            sparse = sparse_search(query, top_k=pool_size // 4)
            search_results = reciprocal_rank_fusion(dense, sparse, top_k=pool_size // 4)
            for p in search_results:
                pid = p.get("paper_id", "")
                if pid and pid not in all_candidates:
                    all_candidates[pid] = {
                        **p,
                        "score": p.get("rrf_score", 0),
                        "source": "search",
                        "sources": ["search"],
                    }
                elif pid:
                    all_candidates[pid]["sources"].append("search")
        except Exception:
            pass

    # Convert to list and enrich with paper metadata
    candidates = list(all_candidates.values())
    candidates = _enrich_candidates(candidates)

    return candidates[:pool_size]


def _stage2_scoring(
    candidates: list[dict],
    user_id: str,
    session_id: str | None,
) -> list[dict]:
    """
    Stage 2: Score each candidate using multiple signals.

    Computes:
      - content_score: from Phase 1 retrieval
      - personal_score: cosine sim to user taste vector
      - session_score: SASRec prediction score
      - collab_score: LightGCN score
      - kg_score: KGAT score

    Then feeds ALL features to DeepFM for final scoring.
    """
    taste = get_taste_vector(user_id)

    for c in candidates:
        # Content score (already from retrieval or source)
        c["content_score"] = c.get("score", 0)

        # Personal score (cosine sim to taste vector)
        if taste is not None and c.get("embedding"):
            emb = np.array(c["embedding"])
            dot = np.dot(taste, emb)
            norm = np.linalg.norm(taste) * np.linalg.norm(emb)
            c["personal_score"] = float(dot / norm) if norm > 0 else 0.0
        else:
            c["personal_score"] = 0.0

        # Session/collab/kg scores come from the source
        sources = c.get("sources", [])
        c["session_score"] = c["score"] if any("sasrec" in s for s in sources) else 0.0
        c["collab_score"] = c["score"] if any("lightgcn" in s for s in sources) else 0.0
        c["kg_score"] = c["score"] if any("kgat" in s for s in sources) else 0.0

        # Multi-source boost: papers from multiple sources are more relevant
        c["source_count"] = len(set(sources))
        c["content_score"] += c["source_count"] * 0.1

    # DeepFM scoring (or linear fallback)
    scored = score_candidates(candidates, user_id)
    return scored


def _stage3_reranking(
    scored: list[dict],
    user_id: str,
    session_id: str | None,
    top_k: int,
    lambda_diversity: float,
) -> list[dict]:
    """
    Stage 3: Final re-ranking with diversity + exploration + explanations.

    1. MMR diversity re-ranking
    2. DQN exploration injection (add 2-3 surprises)
    3. PGPR explanation generation
    """
    # MMR diversity re-ranking (take more than needed for DQN injection)
    diverse = mmr_rerank(scored, top_k=top_k + 3, lambda_param=lambda_diversity)

    # DQN explore/exploit decision
    decision = get_explore_exploit_decision(user_id)

    if decision["action"] == "explore" and len(diverse) > top_k:
        # Replace last 2-3 safe picks with exploratory ones
        explore_topics = decision.get("explore_topics", [])
        if explore_topics:
            # Find candidates from exploratory topics that aren't in final list
            final_pids = {c["paper_id"] for c in diverse[:top_k - 2]}
            explore_candidates = [
                c for c in scored
                if c.get("topic_id") in explore_topics and c["paper_id"] not in final_pids
            ]

            # Inject 2-3 exploration picks
            num_explore = min(3, len(explore_candidates))
            final = diverse[:top_k - num_explore] + explore_candidates[:num_explore]
        else:
            final = diverse[:top_k]
    else:
        final = diverse[:top_k]

    # Generate explanations for each recommendation
    for rec in final:
        rec["explanation"] = generate_explanation(
            user_id,
            rec["paper_id"],
            source=rec.get("source", "unknown"),
        )
        rec["was_explored"] = decision["action"] == "explore" and rec.get("topic_id") in decision.get("explore_topics", [])

    return final


# ============================================================
# Helpers
# ============================================================

def _get_user_read_papers(user_id: str) -> list[str]:
    """Get list of paper IDs the user has already interacted with."""
    supabase = get_supabase_client()

    events = (
        supabase.table("user_events")
        .select("paper_id")
        .eq("user_id", user_id)
        .limit(500)
        .execute()
    )

    return list(set(e["paper_id"] for e in (events.data or [])))


def _enrich_candidates(candidates: list[dict]) -> list[dict]:
    """Enrich candidates with paper metadata from the database."""
    supabase = get_supabase_client()

    paper_ids = [c["paper_id"] for c in candidates if c.get("paper_id")]
    if not paper_ids:
        return candidates

    # Fetch in batches
    papers = {}
    for i in range(0, len(paper_ids), 100):
        batch = paper_ids[i:i + 100]
        response = (
            supabase.table("papers")
            .select("paper_id, title, abstract, authors, categories, year, "
                    "topic_id, topic_label, citation_count, pagerank_score, embedding")
            .in_("paper_id", batch)
            .execute()
        )
        for p in (response.data or []):
            papers[p["paper_id"]] = p

    # Merge metadata into candidates
    for c in candidates:
        pid = c.get("paper_id", "")
        if pid in papers:
            p = papers[pid]
            c.update({
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "authors": p.get("authors", []),
                "categories": p.get("categories", []),
                "year": p.get("year"),
                "topic_id": p.get("topic_id"),
                "topic_label": p.get("topic_label"),
                "citation_count": p.get("citation_count", 0),
                "pagerank_score": p.get("pagerank_score", 0),
                "embedding": p.get("embedding"),
                "paper_age_days": 365,  # placeholder
            })

    return candidates
