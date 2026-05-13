"""
ScholarMind — Research Gap Analyzer

Finds under-explored intersections between popular research topics.
Uses BERTopic clusters + entity co-occurrence. No LLM needed.

gap_score(i, j) = popularity(i) × popularity(j) / (co_occurrence(i,j) + 1)
High gap_score = popular topics rarely combined = research opportunity.
"""

from collections import Counter, defaultdict

from app.db.database import get_supabase_client


def compute_research_gaps(limit: int = 20) -> list[dict]:
    """
    Compute research gaps from topic co-occurrence.

    Returns list of (topic_a, topic_b, gap_score) sorted by gap_score desc.
    """
    supabase = get_supabase_client()

    # Get all papers with topic assignments
    response = (
        supabase.table("papers")
        .select("paper_id, topic_id, topic_label")
        .not_.is_("topic_id", "null")
        .execute()
    )

    papers = response.data
    if not papers:
        return []

    # Count topic popularity
    topic_counts = Counter()
    topic_labels = {}
    paper_topics = defaultdict(set)

    for p in papers:
        tid = p["topic_id"]
        if tid == -1:  # skip outlier topic
            continue
        topic_counts[tid] += 1
        topic_labels[tid] = p.get("topic_label", f"Topic {tid}")
        paper_topics[p["paper_id"]].add(tid)

    # Get entity co-occurrence: papers sharing entities from different topics
    # A simpler proxy: count papers that have multiple topics via shared entities
    entity_response = (
        supabase.table("entities")
        .select("paper_id, canonical_name")
        .execute()
    )

    # Build entity → papers mapping
    entity_papers = defaultdict(set)
    for e in entity_response.data:
        entity_papers[e["canonical_name"]].add(e["paper_id"])

    # Count co-occurrence: how many papers share entities between two topics
    co_occurrence = Counter()
    topic_ids = list(topic_counts.keys())

    # For each entity, find which topics it appears in
    entity_topics = defaultdict(set)
    for entity, pids in entity_papers.items():
        for pid in pids:
            for tid in paper_topics.get(pid, set()):
                entity_topics[entity].add(tid)

    # Count pairs of topics that share entities
    for entity, tids in entity_topics.items():
        tids_list = sorted(tids)
        for i in range(len(tids_list)):
            for j in range(i + 1, len(tids_list)):
                co_occurrence[(tids_list[i], tids_list[j])] += 1

    # Compute gap scores
    gaps = []
    for i in range(len(topic_ids)):
        for j in range(i + 1, len(topic_ids)):
            tid_a, tid_b = topic_ids[i], topic_ids[j]
            pop_a = topic_counts[tid_a]
            pop_b = topic_counts[tid_b]

            # Only consider topics with decent popularity (>20 papers)
            if pop_a < 20 or pop_b < 20:
                continue

            co_occ = co_occurrence.get((tid_a, tid_b), 0) + co_occurrence.get((tid_b, tid_a), 0)
            gap_score = (pop_a * pop_b) / (co_occ + 1)

            gaps.append({
                "topic_a": topic_labels.get(tid_a, f"Topic {tid_a}"),
                "topic_a_id": tid_a,
                "topic_a_papers": pop_a,
                "topic_b": topic_labels.get(tid_b, f"Topic {tid_b}"),
                "topic_b_id": tid_b,
                "topic_b_papers": pop_b,
                "co_occurrence": co_occ,
                "gap_score": round(gap_score, 1),
            })

    # Sort by gap score descending
    gaps.sort(key=lambda x: x["gap_score"], reverse=True)
    return gaps[:limit]
