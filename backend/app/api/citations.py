"""
ScholarMind — Citations API

Endpoints for citation context classification and citation graph queries.
"""

from fastapi import APIRouter, Query
from app.db.database import get_supabase_client

router = APIRouter()


@router.get("/papers/{paper_id}/citations")
async def get_paper_citations(paper_id: str):
    """
    Get citation breakdown for a paper:
    - How many papers cite it (and with what intent)
    - How many papers it references
    - Supporting vs Contrasting vs Mentioning counts
    """
    supabase = get_supabase_client()

    # Papers that cite this paper (with classification)
    citing = (
        supabase.table("citation_contexts")
        .select("citing_paper_id, classification, confidence")
        .eq("cited_paper_id", paper_id)
        .execute()
    )

    # Papers this paper references
    referenced = (
        supabase.table("citation_edges")
        .select("cited_paper_id")
        .eq("citing_paper_id", paper_id)
        .execute()
    )

    # Count by classification
    from collections import Counter
    class_counts = Counter(r["classification"] for r in citing.data)

    return {
        "paper_id": paper_id,
        "cited_by_count": len(citing.data),
        "references_count": len(referenced.data),
        "classification_breakdown": {
            "supporting": class_counts.get("SUPPORTING", 0),
            "contrasting": class_counts.get("CONTRASTING", 0),
            "mentioning": class_counts.get("MENTIONING", 0),
        },
        "reliability_score": (
            class_counts.get("SUPPORTING", 0)
            / max(class_counts.get("SUPPORTING", 0) + class_counts.get("CONTRASTING", 0), 1)
        ),
        "citations": citing.data[:50],
    }


@router.get("/papers/{paper_id}/graph")
async def get_paper_graph(paper_id: str, depth: int = Query(1, ge=1, le=3)):
    """
    Get the local citation graph around a paper.
    Returns nodes and edges for visualization.
    """
    supabase = get_supabase_client()

    nodes = {paper_id}
    edges = []

    current_papers = [paper_id]

    for _ in range(depth):
        next_papers = []

        for pid in current_papers:
            # Outgoing (this paper cites)
            refs = (
                supabase.table("citation_edges")
                .select("cited_paper_id")
                .eq("citing_paper_id", pid)
                .limit(10)
                .execute()
            )
            for r in refs.data:
                target = r["cited_paper_id"]
                edges.append({"source": pid, "target": target, "type": "cites"})
                if target not in nodes:
                    nodes.add(target)
                    next_papers.append(target)

            # Incoming (papers that cite this)
            cites = (
                supabase.table("citation_edges")
                .select("citing_paper_id")
                .eq("cited_paper_id", pid)
                .limit(10)
                .execute()
            )
            for r in cites.data:
                source = r["citing_paper_id"]
                edges.append({"source": source, "target": pid, "type": "cites"})
                if source not in nodes:
                    nodes.add(source)
                    next_papers.append(source)

        current_papers = next_papers

    # Fetch paper titles for all nodes
    node_list = list(nodes)
    papers = (
        supabase.table("papers")
        .select("paper_id, title, year")
        .in_("paper_id", node_list[:100])
        .execute()
    )

    paper_map = {p["paper_id"]: p for p in papers.data}
    graph_nodes = [
        {"id": pid, "title": paper_map.get(pid, {}).get("title", "Unknown"), "year": paper_map.get(pid, {}).get("year")}
        for pid in node_list
    ]

    return {
        "center": paper_id,
        "nodes": graph_nodes,
        "edges": edges,
    }
