"""
ScholarMind — Fetch Citation Data from Semantic Scholar API

Downloads citation/reference data for papers in our corpus.
Uses the free Semantic Scholar Academic Graph API (1 req/sec with free API key).

Usage:
    python -m ingestion.fetch_citations_s2 \
        --parquet data/scholarmind_papers_with_embeddings.parquet \
        --output data/citations.parquet \
        --max-papers 15000
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
RATE_LIMIT_DELAY = 1.1  # seconds between requests (free tier = 1 req/sec)


def fetch_paper_citations(arxiv_id: str, api_key: str | None = None) -> dict:
    """
    Fetch references and citations for a paper from Semantic Scholar.

    Args:
        arxiv_id: The arXiv paper ID (e.g., "2106.09685")
        api_key: Optional S2 API key for higher rate limits

    Returns:
        Dict with 'references' and 'citations' lists of paper IDs
    """
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_API_BASE}/paper/ArXiv:{arxiv_id}"
    params = {"fields": "references.externalIds,citations.externalIds"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)

        if resp.status_code == 429:
            # Rate limited — wait and retry
            time.sleep(5)
            resp = requests.get(url, params=params, headers=headers, timeout=10)

        if resp.status_code != 200:
            return {"references": [], "citations": []}

        data = resp.json()

        # Extract arXiv IDs from references
        references = []
        for ref in data.get("references", []) or []:
            ext_ids = ref.get("externalIds", {}) or {}
            if ext_ids.get("ArXiv"):
                references.append(ext_ids["ArXiv"])

        # Extract arXiv IDs from citations
        citations = []
        for cit in data.get("citations", []) or []:
            ext_ids = cit.get("externalIds", {}) or {}
            if ext_ids.get("ArXiv"):
                citations.append(ext_ids["ArXiv"])

        return {"references": references, "citations": citations}

    except (requests.RequestException, json.JSONDecodeError):
        return {"references": [], "citations": []}


def main():
    parser = argparse.ArgumentParser(description="Fetch citation data from Semantic Scholar")
    parser.add_argument("--parquet", type=str, required=True, help="Path to Phase 1 parquet")
    parser.add_argument("--output", type=str, default="data/citations.parquet", help="Output path")
    parser.add_argument("--max-papers", type=int, default=15000, help="Max papers to fetch")
    parser.add_argument("--api-key", type=str, default=None, help="Semantic Scholar API key (optional)")
    args = parser.parse_args()

    # Load papers
    print(f"Loading papers from {args.parquet}...")
    df = pd.read_parquet(args.parquet, columns=["paper_id", "year"])

    # Sort by year descending (prioritize recent papers)
    df = df.sort_values("year", ascending=False)

    # Take top N
    paper_ids = df["paper_id"].tolist()[:args.max_papers]
    print(f"Will fetch citations for {len(paper_ids):,} papers")

    # Our corpus paper IDs (for filtering — only keep citations within our corpus)
    corpus_ids = set(df["paper_id"].tolist())

    # Fetch citations
    all_edges = []
    fetched = 0
    errors = 0

    print(f"Fetching from Semantic Scholar API ({RATE_LIMIT_DELAY}s delay)...")
    print(f"Estimated time: ~{len(paper_ids) * RATE_LIMIT_DELAY / 3600:.1f} hours")

    for i, pid in enumerate(paper_ids):
        result = fetch_paper_citations(pid, api_key=args.api_key)

        # Add reference edges (this paper cites those papers)
        for ref_id in result["references"]:
            all_edges.append({
                "citing_paper_id": pid,
                "cited_paper_id": ref_id,
            })

        # Add citation edges (those papers cite this paper)
        for cit_id in result["citations"]:
            if cit_id in corpus_ids:  # only keep in-corpus citations
                all_edges.append({
                    "citing_paper_id": cit_id,
                    "cited_paper_id": pid,
                })

        fetched += 1
        if fetched % 500 == 0:
            print(f"  [{fetched:,}/{len(paper_ids):,}] Fetched, {len(all_edges):,} edges found, {errors} errors")

        # Rate limit
        time.sleep(RATE_LIMIT_DELAY)

    # Deduplicate edges
    edges_df = pd.DataFrame(all_edges).drop_duplicates()
    print(f"\nTotal citation edges: {len(edges_df):,}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")

    # Stats
    print(f"\nStats:")
    print(f"  Papers fetched: {fetched:,}")
    print(f"  Total edges: {len(edges_df):,}")
    print(f"  Unique citing papers: {edges_df['citing_paper_id'].nunique():,}")
    print(f"  Unique cited papers: {edges_df['cited_paper_id'].nunique():,}")


if __name__ == "__main__":
    main()
