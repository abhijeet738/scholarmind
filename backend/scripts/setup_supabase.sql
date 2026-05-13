-- ============================================================
-- ScholarMind — Supabase Database Setup
-- ============================================================
-- INSTRUCTIONS:
-- 1. Go to your Supabase Dashboard → SQL Editor
-- 2. Paste this entire script and click "Run"
-- ============================================================

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- PAPERS TABLE (Core data store)
-- ============================================================
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT[] NOT NULL DEFAULT '{}',
    categories TEXT[] NOT NULL DEFAULT '{}',
    all_categories TEXT[] NOT NULL DEFAULT '{}',
    published_at TEXT,
    year INT,
    embedding VECTOR(768),  -- BAAI/bge-base-en-v1.5 outputs 768 dims
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast cosine similarity search
-- This makes vector search go from O(n) brute force to O(log n)
CREATE INDEX IF NOT EXISTS idx_papers_embedding 
ON papers USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Regular indexes for filtering
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers USING GIN(categories);

-- ============================================================
-- VECTOR SEARCH FUNCTION (called from Python)
-- ============================================================
-- This function takes a query embedding and returns the most
-- similar papers using cosine distance via the HNSW index.
CREATE OR REPLACE FUNCTION match_papers(
    query_embedding VECTOR(768),
    match_count INT DEFAULT 50,
    filter_year INT DEFAULT NULL
)
RETURNS TABLE (
    paper_id TEXT,
    title TEXT,
    abstract TEXT,
    authors TEXT[],
    categories TEXT[],
    year INT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.paper_id,
        p.title,
        p.abstract,
        p.authors,
        p.categories,
        p.year,
        1 - (p.embedding <=> query_embedding) AS similarity
    FROM papers p
    WHERE 
        p.embedding IS NOT NULL
        AND (filter_year IS NULL OR p.year >= filter_year)
    ORDER BY p.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================
-- Verify setup
-- ============================================================
-- Run this to check everything was created:
SELECT 
    'pgvector' AS component,
    EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector') AS ready
UNION ALL
SELECT 
    'papers_table',
    EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'papers')
UNION ALL
SELECT 
    'match_papers_fn',
    EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'match_papers');
