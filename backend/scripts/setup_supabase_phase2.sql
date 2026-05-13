-- ============================================================
-- ScholarMind — Phase 2: Knowledge Graph Tables
-- ============================================================
-- Run this in your Supabase SQL Editor AFTER setup_supabase.sql
-- ============================================================

-- ============================================================
-- ENTITIES TABLE (extracted by SciBERT NER)
-- ============================================================
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    canonical_name TEXT,                -- set by entity resolution
    type TEXT NOT NULL,                 -- METHOD, DATASET, METRIC, TASK
    paper_id TEXT REFERENCES papers(paper_id),
    mention_count INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_entities_paper ON entities(paper_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);

-- ============================================================
-- RELATIONS TABLE (KG edges)
-- ============================================================
CREATE TABLE IF NOT EXISTS relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity TEXT NOT NULL,        -- entity name
    target_entity TEXT NOT NULL,        -- entity name
    relation_type TEXT NOT NULL,        -- uses, evaluated_on, outperforms, extends
    paper_id TEXT REFERENCES papers(paper_id),
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

-- ============================================================
-- RESULTS TABLE (SOTA benchmark data)
-- ============================================================
CREATE TABLE IF NOT EXISTS results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    method_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value FLOAT NOT NULL,
    paper_id TEXT REFERENCES papers(paper_id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_results_dataset ON results(dataset_name);
CREATE INDEX IF NOT EXISTS idx_results_method ON results(method_name);

-- ============================================================
-- CITATION CONTEXTS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS citation_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    citing_paper_id TEXT NOT NULL,
    cited_paper_id TEXT NOT NULL,
    sentence TEXT,
    classification TEXT NOT NULL,       -- SUPPORTING, CONTRASTING, MENTIONING
    confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_citctx_citing ON citation_contexts(citing_paper_id);
CREATE INDEX IF NOT EXISTS idx_citctx_cited ON citation_contexts(cited_paper_id);
CREATE INDEX IF NOT EXISTS idx_citctx_class ON citation_contexts(classification);

-- ============================================================
-- CITATION EDGES TABLE (for graph building)
-- ============================================================
CREATE TABLE IF NOT EXISTS citation_edges (
    citing_paper_id TEXT NOT NULL,
    cited_paper_id TEXT NOT NULL,
    PRIMARY KEY (citing_paper_id, cited_paper_id)
);

CREATE INDEX IF NOT EXISTS idx_citedge_citing ON citation_edges(citing_paper_id);
CREATE INDEX IF NOT EXISTS idx_citedge_cited ON citation_edges(cited_paper_id);

-- ============================================================
-- ADD COLUMNS TO PAPERS TABLE
-- ============================================================
ALTER TABLE papers ADD COLUMN IF NOT EXISTS topic_id INT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS topic_label TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS pagerank_score FLOAT DEFAULT 0.0;

CREATE INDEX IF NOT EXISTS idx_papers_topic ON papers(topic_id);

-- ============================================================
-- Verify
-- ============================================================
SELECT 'entities' AS tbl, COUNT(*) AS rows FROM entities
UNION ALL SELECT 'relations', COUNT(*) FROM relations
UNION ALL SELECT 'results', COUNT(*) FROM results
UNION ALL SELECT 'citation_contexts', COUNT(*) FROM citation_contexts
UNION ALL SELECT 'citation_edges', COUNT(*) FROM citation_edges;
