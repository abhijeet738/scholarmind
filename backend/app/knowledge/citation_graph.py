"""
ScholarMind — Citation Graph (NetworkX + PageRank)

Builds a directed citation graph from citation edges and runs
graph algorithms: PageRank, community detection, shortest paths.
"""

import pickle
from pathlib import Path

import networkx as nx
import pandas as pd


def build_citation_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed citation graph from edge data.

    Args:
        edges_df: DataFrame with columns [citing_paper_id, cited_paper_id]

    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()

    for _, row in edges_df.iterrows():
        G.add_edge(row["citing_paper_id"], row["cited_paper_id"])

    print(f"Citation graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def compute_pagerank(G: nx.DiGraph, alpha: float = 0.85) -> dict[str, float]:
    """
    Compute PageRank scores for all papers in the graph.

    Higher PageRank = more influential paper (cited by other influential papers).
    """
    pr = nx.pagerank(G, alpha=alpha)
    print(f"PageRank computed for {len(pr):,} papers")

    # Show top 10
    top10 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 papers by PageRank:")
    for pid, score in top10:
        print(f"  {pid}: {score:.6f}")

    return pr


def detect_communities(G: nx.DiGraph) -> dict[str, int]:
    """
    Detect communities using Louvain method on the undirected version.

    Returns dict mapping paper_id → community_id.
    """
    G_undirected = G.to_undirected()
    communities = nx.community.louvain_communities(G_undirected, seed=42)

    paper_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            paper_to_community[node] = i

    print(f"Detected {len(communities)} communities")
    return paper_to_community


def find_citation_path(G: nx.DiGraph, source: str, target: str) -> list[str] | None:
    """Find the shortest citation path between two papers."""
    try:
        path = nx.shortest_path(G, source, target)
        return path
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        # Try undirected if directed path doesn't exist
        try:
            path = nx.shortest_path(G.to_undirected(), source, target)
            return path
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None


def get_paper_influence(G: nx.DiGraph, paper_id: str) -> dict:
    """Get influence metrics for a specific paper."""
    if paper_id not in G:
        return {"in_degree": 0, "out_degree": 0, "pagerank": 0.0}

    return {
        "in_degree": G.in_degree(paper_id),    # how many papers cite this
        "out_degree": G.out_degree(paper_id),   # how many papers this cites
        "predecessors": list(G.predecessors(paper_id))[:20],  # papers that cite this
        "successors": list(G.successors(paper_id))[:20],      # papers this cites
    }


def save_graph(G: nx.DiGraph, path: str):
    """Save the graph to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")


def load_graph(path: str) -> nx.DiGraph:
    """Load the graph from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--citations", default="data/citations.parquet")
    parser.add_argument("--output-graph", default="data/citation_graph.pkl")
    parser.add_argument("--output-pagerank", default="data/pagerank_scores.parquet")
    args = parser.parse_args()

    # Load citation edges
    edges_df = pd.read_parquet(args.citations)
    print(f"Loaded {len(edges_df):,} citation edges")

    # Build graph
    G = build_citation_graph(edges_df)

    # Compute PageRank
    pr_scores = compute_pagerank(G)

    # Save graph
    save_graph(G, args.output_graph)

    # Save PageRank scores
    pr_df = pd.DataFrame([
        {"paper_id": pid, "pagerank_score": score}
        for pid, score in pr_scores.items()
    ])
    pr_df.to_parquet(args.output_pagerank, index=False)
    print(f"PageRank scores saved to {args.output_pagerank}")


if __name__ == "__main__":
    main()
