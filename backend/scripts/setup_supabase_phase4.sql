-- ============================================================
-- ScholarMind — Phase 4: Personalization Engine Tables
-- ============================================================
-- Run this in your Supabase SQL Editor AFTER Phase 1 & 2 scripts
-- ============================================================

-- ============================================================
-- USER PROFILES (taste vectors + preferences)
-- ============================================================
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT UNIQUE,
    taste_vector VECTOR(768),              -- running avg of read paper embeddings
    total_interactions INT DEFAULT 0,
    preferred_topics INT[],                -- top BERTopic cluster IDs
    exploration_alpha FLOAT[] DEFAULT '{}', -- Thompson Sampling α per topic
    exploration_beta FLOAT[] DEFAULT '{}',  -- Thompson Sampling β per topic
    lightgcn_embedding VECTOR(64),         -- learned by LightGCN (NULL until trained)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- USER EVENTS (every interaction — training data for all models)
-- ============================================================
CREATE TABLE IF NOT EXISTS user_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES user_profiles(user_id),
    session_id UUID NOT NULL,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    event_type TEXT NOT NULL,              -- CLICK, EXPAND, DWELL_30S, DWELL_60S, SAVE, UPVOTE, DOWNVOTE
    event_weight FLOAT NOT NULL,           -- 1.0 to 5.0 (or -3.0 for downvote)
    query_text TEXT,                        -- what they searched for (if applicable)
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_user ON user_events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON user_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_paper ON user_events(paper_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON user_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_created ON user_events(created_at DESC);

-- ============================================================
-- USER SESSIONS (reading sequences for SASRec)
-- ============================================================
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES user_profiles(user_id),
    paper_sequence TEXT[] DEFAULT '{}',    -- ordered list of paper_ids read
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON user_sessions(is_active) WHERE is_active = TRUE;

-- ============================================================
-- USER SAVES (bookmarks)
-- ============================================================
CREATE TABLE IF NOT EXISTS user_saves (
    user_id UUID REFERENCES user_profiles(user_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_saves_user ON user_saves(user_id);

-- ============================================================
-- RECOMMENDATION LOG (for evaluation + DQN replay buffer)
-- ============================================================
CREATE TABLE IF NOT EXISTS recommendation_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES user_profiles(user_id),
    session_id UUID,
    paper_id TEXT NOT NULL,
    score FLOAT NOT NULL,                  -- final recommendation score
    source TEXT,                            -- which algorithm generated this: sasrec, lightgcn, kgat, search
    explanation TEXT,                       -- PGPR explanation
    was_clicked BOOLEAN DEFAULT FALSE,     -- did the user click it?
    was_explored BOOLEAN DEFAULT FALSE,    -- was this a DQN exploration pick?
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reclog_user ON recommendation_log(user_id);
CREATE INDEX IF NOT EXISTS idx_reclog_session ON recommendation_log(session_id);
