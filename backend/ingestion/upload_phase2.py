"""
ScholarMind — Upload Phase 2 Data to Supabase

Uploads entities, topics, SOTA results, and citation edges
from parquet files to Supabase tables.

Usage:
    python -m ingestion.upload_phase2 \
        --supabase-url https://your-project.supabase.co \
        --supabase-key your-service-role-key \
        --data-dir data/
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from supabase import create_client

BATCH_SIZE = 500


def upload_entities(client, entities_path: str):
    """Upload resolved entities to the entities table."""
    df = pd.read_parquet(entities_path)
    print(f"\nUploading {len(df):,} entities...")

    # Aggregate mention counts per (canonical_name, type, paper_id)
    uploaded = 0
    for start in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[start:start + BATCH_SIZE]
        batch = [
            {
                "name": str(row["name"]),
                "canonical_name": str(row.get("canonical_name", row["name"])),
                "type": str(row["type"]),
                "paper_id": str(row["paper_id"]),
            }
            for _, row in batch_df.iterrows()
        ]

        try:
            client.table("entities").insert(batch).execute()
            uploaded += len(batch)
        except Exception as e:
            print(f"  ❌ Batch error: {e}")

        pct = min(100, (start + len(batch)) / len(df) * 100)
        print(f"  [{pct:5.1f}%] Entities: {uploaded:,}/{len(df):,}", end="\r")

    print(f"\n  ✅ Entities uploaded: {uploaded:,}")


def upload_topics(client, topics_path: str):
    """Update papers table with topic assignments."""
    df = pd.read_parquet(topics_path)
    print(f"\nUpdating {len(df):,} papers with topics...")

    updated = 0
    for _, row in df.iterrows():
        try:
            client.table("papers").update({
                "topic_id": int(row["topic_id"]),
                "topic_label": str(row["topic_label"]),
            }).eq("paper_id", str(row["paper_id"])).execute()
            updated += 1
        except Exception:
            pass

        if updated % 1000 == 0:
            print(f"  Topics updated: {updated:,}/{len(df):,}", end="\r")

    print(f"\n  ✅ Topics updated: {updated:,}")


def upload_results(client, results_path: str):
    """Upload SOTA results to the results table."""
    df = pd.read_parquet(results_path)
    print(f"\nUploading {len(df):,} SOTA results...")

    uploaded = 0
    for start in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[start:start + BATCH_SIZE]
        batch = [
            {
                "method_name": str(row["method_name"]),
                "dataset_name": str(row["dataset_name"]),
                "metric_name": str(row["metric_name"]),
                "value": float(row["value"]),
                "paper_id": str(row["paper_id"]),
            }
            for _, row in batch_df.iterrows()
        ]

        try:
            client.table("results").insert(batch).execute()
            uploaded += len(batch)
        except Exception as e:
            print(f"  ❌ Batch error: {e}")

    print(f"  ✅ Results uploaded: {uploaded:,}")


def upload_citation_edges(client, citations_path: str):
    """Upload citation edges to the citation_edges table."""
    df = pd.read_parquet(citations_path)
    print(f"\nUploading {len(df):,} citation edges...")

    uploaded = 0
    for start in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[start:start + BATCH_SIZE]
        batch = [
            {
                "citing_paper_id": str(row["citing_paper_id"]),
                "cited_paper_id": str(row["cited_paper_id"]),
            }
            for _, row in batch_df.iterrows()
        ]

        try:
            client.table("citation_edges").upsert(
                batch, on_conflict="citing_paper_id,cited_paper_id"
            ).execute()
            uploaded += len(batch)
        except Exception as e:
            print(f"  ❌ Batch error: {e}")

    print(f"  ✅ Citation edges uploaded: {uploaded:,}")


def upload_pagerank(client, pagerank_path: str):
    """Update papers table with PageRank scores."""
    df = pd.read_parquet(pagerank_path)
    print(f"\nUpdating {len(df):,} papers with PageRank scores...")

    updated = 0
    for _, row in df.iterrows():
        try:
            client.table("papers").update({
                "pagerank_score": float(row["pagerank_score"]),
            }).eq("paper_id", str(row["paper_id"])).execute()
            updated += 1
        except Exception:
            pass

        if updated % 1000 == 0:
            print(f"  PageRank updated: {updated:,}/{len(df):,}", end="\r")

    print(f"\n  ✅ PageRank updated: {updated:,}")


def main():
    parser = argparse.ArgumentParser(description="Upload Phase 2 data to Supabase")
    parser.add_argument("--supabase-url", required=True)
    parser.add_argument("--supabase-key", required=True)
    parser.add_argument("--data-dir", default="data/", help="Directory with parquet files")
    parser.add_argument("--skip", nargs="*", default=[], help="Steps to skip: entities topics results citations pagerank")
    args = parser.parse_args()

    client = create_client(args.supabase_url, args.supabase_key)
    data_dir = Path(args.data_dir)

    if "entities" not in args.skip and (data_dir / "entities_resolved.parquet").exists():
        upload_entities(client, str(data_dir / "entities_resolved.parquet"))

    if "topics" not in args.skip and (data_dir / "topics.parquet").exists():
        upload_topics(client, str(data_dir / "topics.parquet"))

    if "results" not in args.skip and (data_dir / "results.parquet").exists():
        upload_results(client, str(data_dir / "results.parquet"))

    if "citations" not in args.skip and (data_dir / "citations.parquet").exists():
        upload_citation_edges(client, str(data_dir / "citations.parquet"))

    if "pagerank" not in args.skip and (data_dir / "pagerank_scores.parquet").exists():
        upload_pagerank(client, str(data_dir / "pagerank_scores.parquet"))

    print("\n🎉 Phase 2 upload complete!")


if __name__ == "__main__":
    main()
