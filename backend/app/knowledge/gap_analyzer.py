"""
ScholarMind — Research Gap Analyzer (Knowledge Graph Version)

Finds under-explored intersections between popular research topics by analyzing
the semantic edges in the pre-computed SciBERT Knowledge Graph.

gap_score(i, j) = popularity(i) × popularity(j) / (co_occurrence(i,j) + 1)
High gap_score = popular topics that are rarely connected by semantic edges = research opportunity.
"""

import os
import pickle
from collections import Counter, defaultdict

_graph_cache = None

def get_graph_data():
    """Lazy-load the knowledge graph from disk."""
    global _graph_cache
    if _graph_cache is None:
        file_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge_graph.pkl")
        if os.path.exists(file_path):
            print("Loading SciBERT Knowledge Graph...")
            with open(file_path, "rb") as f:
                _graph_cache = pickle.load(f)
        else:
            print(f"Warning: Graph file not found at {file_path}")
            _graph_cache = {}
    return _graph_cache


def compute_research_gaps(limit: int = 20) -> list[dict]:
    """
    Compute research gaps from topic co-occurrence along Graph Edges.
    """
    data = get_graph_data()
    if not data or "topics" not in data or "graph" not in data:
        return []

    G = data["graph"]
    topics_dict = data["topics"]
    topic_info_list = topics_dict.get("topic_info", [])
    paper_topics = topics_dict.get("paper_topics", {})

    # Extract topic popularity and names
    topic_counts = Counter()
    topic_labels = {}
    
    for t_info in topic_info_list:
        tid = t_info["Topic"]
        if tid == -1:  # skip outlier topic
            continue
        topic_counts[tid] = t_info["Count"]
        # Make the label look pretty (remove the leading ID)
        raw_name = t_info.get("Name", f"Topic {tid}")
        clean_name = raw_name.split("_", 1)[1].replace("_", " ").title() if "_" in raw_name else raw_name
        topic_labels[tid] = clean_name

    # Count co-occurrence: how many semantic edges connect two different topics
    co_occurrence = Counter()
    
    for u, v in G.edges():
        tid_u = paper_topics.get(u, -1)
        tid_v = paper_topics.get(v, -1)
        
        # We only care about connections between DIFFERENT topics (ignoring outliers)
        if tid_u != -1 and tid_v != -1 and tid_u != tid_v:
            # Sort to ensure (A, B) is identical to (B, A)
            pair = tuple(sorted([tid_u, tid_v]))
            co_occurrence[pair] += 1

    # Compute gap scores
    gaps = []
    topic_ids = list(topic_counts.keys())
    
    for i in range(len(topic_ids)):
        for j in range(i + 1, len(topic_ids)):
            tid_a, tid_b = topic_ids[i], topic_ids[j]
            pop_a = topic_counts[tid_a]
            pop_b = topic_counts[tid_b]

            # Only consider topics with decent popularity (>50 papers)
            # to avoid false 'gaps' from ultra-niche micro topics
            if pop_a < 50 or pop_b < 50:
                continue

            # The pair is always sorted
            pair = tuple(sorted([tid_a, tid_b]))
            co_occ = co_occurrence.get(pair, 0)
            
            # gap score formula
            gap_score = (pop_a * pop_b) / (co_occ + 1)

            gaps.append({
                "topic_a": topic_labels.get(tid_a, f"Topic {tid_a}"),
                "topic_a_id": tid_a,
                "topic_a_papers": pop_a,
                "topic_b": topic_labels.get(tid_b, f"Topic {tid_b}"),
                "topic_b_id": tid_b,
                "topic_b_papers": pop_b,
                "co_occurrence": co_occ, # semantic bridges between them
                "gap_score": round(gap_score, 1),
            })

    # Sort by gap score descending
    gaps.sort(key=lambda x: x["gap_score"], reverse=True)
    return gaps[:limit]
