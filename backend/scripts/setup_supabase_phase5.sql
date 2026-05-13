-- ============================================================
-- ScholarMind — Phase 5: Evaluation & A/B Testing Tables
-- ============================================================
-- Run this in your Supabase SQL Editor AFTER Phase 4 script
-- ============================================================

-- ============================================================
-- EXPERIMENTS (A/B test definitions)
-- ============================================================
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,             -- e.g. "sasrec_vs_fallback"
    description TEXT,
    variants JSONB NOT NULL,               -- {"A": "sasrec", "B": "fallback"}
    traffic_split FLOAT DEFAULT 0.5,       -- % of users in variant A
    status TEXT DEFAULT 'draft',           -- draft, running, completed
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- EXPERIMENT ASSIGNMENTS (which user → which variant)
-- ============================================================
CREATE TABLE IF NOT EXISTS experiment_assignments (
    experiment_id UUID REFERENCES experiments(experiment_id),
    user_id UUID REFERENCES user_profiles(user_id),
    variant TEXT NOT NULL,                  -- "A" or "B"
    assigned_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (experiment_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_assignments_experiment ON experiment_assignments(experiment_id);
CREATE INDEX IF NOT EXISTS idx_assignments_user ON experiment_assignments(user_id);

-- ============================================================
-- METRIC SNAPSHOTS (daily performance logs)
-- ============================================================
CREATE TABLE IF NOT EXISTS metric_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name TEXT NOT NULL,             -- ndcg_10, hit_rate_10, ctr, etc.
    metric_value FLOAT NOT NULL,
    algorithm TEXT,                         -- sasrec, lightgcn, deepfm, overall
    experiment_id UUID,                    -- NULL if not part of an experiment
    snapshot_date DATE DEFAULT CURRENT_DATE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_metric ON metric_snapshots(metric_name);
CREATE INDEX IF NOT EXISTS idx_snapshots_date ON metric_snapshots(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_algorithm ON metric_snapshots(algorithm);
