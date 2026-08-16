-- ============================================================
-- ScholarMind — 1-Week MVP Schema
-- ============================================================
-- Run this entire script in your Supabase SQL Editor to prep
-- your database for the OpenAI embeddings.
-- ============================================================

-- 1. Enable the pgvector extension (Required for vector math)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Drop the old papers table if it exists (since we changed the vector size)
DROP TABLE IF EXISTS papers CASCADE;

-- 3. Create the simplified MVP papers table
CREATE TABLE papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    arxiv_id TEXT UNIQUE NOT NULL,      -- original kaggle 'id'
    title TEXT NOT NULL,
    authors TEXT,
    published_date DATE,
    core_category TEXT,
    summary TEXT NOT NULL,              -- The abstract text
    embedding vector(768),              -- Gemini embedding size
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Create an HNSW index for blazingly fast vector search
-- (We use cosine distance `<=>` which is standard for OpenAI embeddings)
CREATE INDEX ON papers USING hnsw (embedding vector_cosine_ops);

-- 5. Drop any old recommendation tables you might still have lying around
DROP TABLE IF EXISTS user_events CASCADE;
DROP TABLE IF EXISTS user_profiles CASCADE;
DROP TABLE IF EXISTS user_saves CASCADE;
DROP TABLE IF EXISTS user_sessions CASCADE;
DROP TABLE IF EXISTS recommendation_log CASCADE;
DROP TABLE IF EXISTS experiments CASCADE;
DROP TABLE IF EXISTS experiment_assignments CASCADE;
DROP TABLE IF EXISTS metric_snapshots CASCADE;
DROP TABLE IF EXISTS results CASCADE;

-- Success! Your database is now ready for the data ingestion script.
