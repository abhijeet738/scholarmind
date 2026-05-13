# 🧠 Layer 3: Personalization Engine — Deep Implementation Plan

> How ScholarMind learns from every user interaction and delivers increasingly smarter recommendations.

---

## The Big Picture

```
User lands on platform (anonymous or logged in)
        │
        ▼
┌──────────────────────────────────┐
│     EVENT TRACKING LAYER         │
│  Every click, save, search,     │
│  dwell time → logged to Supabase│
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│     USER PROFILE BUILDER         │
│  Compute taste vector from       │
│  interaction-weighted embeddings │
└──────────────┬───────────────────┘
               │
     ┌─────────┴──────────┐
     ▼                    ▼
┌──────────┐     ┌───────────────┐
│ Session  │     │  Long-Term    │
│ Model    │     │  Profile      │
│ (anon)   │     │  (logged in)  │
└────┬─────┘     └──────┬────────┘
     │                  │
     └────────┬─────────┘
              ▼
┌──────────────────────────────────┐
│     RECOMMENDATION MIXER         │
│  Blend: content + collaborative  │
│  + session + bandit exploration  │
└──────────────┬───────────────────┘
              ▼
┌──────────────────────────────────┐
│     EXPLAINABILITY LAYER         │
│  KG path reasoning: "Why this?"  │
└──────────────────────────────────┘
```

---

## 1. Event Tracking System

### What Events to Track

| Event | Trigger | Weight | Why |
|-------|---------|--------|-----|
| `SEARCH` | User submits a query | — | Captures research intent |
| `CLICK` | User clicks a paper card | 1.0 | Basic interest signal |
| `EXPAND` | User expands abstract | 1.5 | Stronger interest |
| `DWELL_30S` | User stays on paper >30s | 2.0 | Reading = real engagement |
| `DWELL_60S` | User stays on paper >60s | 3.0 | Deep reading |
| `SAVE` | User bookmarks paper | 4.0 | Explicit positive signal |
| `UPVOTE` | User upvotes recommendation | 5.0 | Strongest positive signal |
| `DOWNVOTE` | User downvotes recommendation | -3.0 | Negative signal |
| `CITE_VIEW` | User views citation details | 1.0 | Interest in paper's impact |
| `RELATED_CLICK` | User clicks "related papers" | 1.5 | Wants to go deeper |

### Supabase Schema

```sql
-- Users table (managed by Supabase Auth)
-- Supabase Auth handles: id, email, created_at, etc.

-- User profiles (our custom extension)
CREATE TABLE user_profiles (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id),
    taste_vector VECTOR(384),           -- running embedding average
    taste_vector_updated_at TIMESTAMPTZ,
    total_interactions INT DEFAULT 0,
    preferred_topics INT[],             -- top BERTopic cluster IDs
    exploration_alpha FLOAT[] DEFAULT ARRAY[]::FLOAT[],  -- bandit params
    exploration_beta FLOAT[] DEFAULT ARRAY[]::FLOAT[],   -- bandit params
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Event log (the core tracking table)
CREATE TABLE user_events (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    session_id UUID NOT NULL,           -- groups events in one visit
    paper_id TEXT NOT NULL,             -- arXiv paper ID
    event_type TEXT NOT NULL,           -- CLICK, SAVE, DWELL_30S, etc.
    event_weight FLOAT NOT NULL,        -- weight from table above
    query_text TEXT,                    -- the search query (if from search)
    metadata JSONB DEFAULT '{}',       -- extra context
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX idx_events_user ON user_events(user_id, created_at DESC);
CREATE INDEX idx_events_session ON user_events(session_id, created_at);
CREATE INDEX idx_events_paper ON user_events(paper_id);

-- User's saved papers (bookmarks)
CREATE TABLE user_saves (
    user_id UUID REFERENCES auth.users(id),
    paper_id TEXT NOT NULL,
    saved_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, paper_id)
);

-- User's search history
CREATE TABLE search_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    session_id UUID NOT NULL,
    query_text TEXT NOT NULL,
    results_count INT,
    clicked_paper_ids TEXT[],           -- which results they clicked
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Session tracking
CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),  -- NULL for anonymous
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_active_at TIMESTAMPTZ DEFAULT NOW(),
    paper_sequence TEXT[] DEFAULT ARRAY[]::TEXT[],  -- ordered papers viewed
    is_active BOOLEAN DEFAULT TRUE
);
```

### Row-Level Security (Supabase RLS)

```sql
-- Users can only read their own events
ALTER TABLE user_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users read own events" ON user_events
    FOR SELECT USING (auth.uid() = user_id);

-- Backend service role can insert for any user
CREATE POLICY "Service inserts events" ON user_events
    FOR INSERT WITH CHECK (TRUE);
    -- (use service_role key from backend only)
```

---

## 2. User Profile Builder

### How the Taste Vector Works

The taste vector is a **384-dimensional embedding** that represents the user's research interests. It's computed as a weighted rolling average of the embeddings of papers they've interacted with.

```python
import numpy as np
from datetime import datetime, timedelta

class UserProfileBuilder:
    def __init__(self, embedding_dim=384):
        self.dim = embedding_dim
    
    def compute_taste_vector(self, events: list[dict], paper_embeddings: dict) -> np.ndarray:
        """
        Compute user taste vector from their interaction history.
        
        Args:
            events: List of {paper_id, event_weight, created_at}
            paper_embeddings: Dict of paper_id -> embedding (384-d)
        """
        if not events:
            return np.zeros(self.dim)
        
        weighted_sum = np.zeros(self.dim)
        total_weight = 0.0
        now = datetime.utcnow()
        
        for event in events:
            paper_emb = paper_embeddings.get(event['paper_id'])
            if paper_emb is None:
                continue
            
            # Time decay: recent interactions matter more
            days_ago = (now - event['created_at']).days
            time_decay = np.exp(-0.05 * days_ago)  # half-life ~14 days
            
            # Final weight = event_weight × time_decay
            weight = event['event_weight'] * time_decay
            
            weighted_sum += paper_emb * weight
            total_weight += weight
        
        if total_weight == 0:
            return np.zeros(self.dim)
        
        # Normalize to unit vector
        taste_vector = weighted_sum / total_weight
        taste_vector = taste_vector / (np.linalg.norm(taste_vector) + 1e-8)
        
        return taste_vector
    
    def extract_preferred_topics(self, events: list[dict], paper_topics: dict, top_k=5) -> list[int]:
        """Find user's top-k preferred BERTopic clusters."""
        topic_weights = {}
        for event in events:
            topic_id = paper_topics.get(event['paper_id'])
            if topic_id is not None:
                topic_weights[topic_id] = topic_weights.get(topic_id, 0) + event['event_weight']
        
        sorted_topics = sorted(topic_weights.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_topics[:top_k]]
```

### When to Recompute

| Trigger | Action |
|---------|--------|
| Every 10 new events | Recompute taste vector |
| User logs in after >24h | Recompute on first request |
| Explicit save/upvote | Immediate incremental update |
| Background job (daily) | Batch recompute all active users |

### Incremental Update (Fast Path)

```python
def incremental_update(current_vector: np.ndarray, new_paper_emb: np.ndarray, 
                        event_weight: float, total_interactions: int) -> np.ndarray:
    """Update taste vector without recomputing from scratch."""
    # Exponential moving average
    alpha = event_weight / (total_interactions + event_weight)
    updated = (1 - alpha) * current_vector + alpha * new_paper_emb
    return updated / (np.linalg.norm(updated) + 1e-8)
```

---

## 3. Session Model (Cold-Start / Anonymous Users)

Even without login, we can give good recommendations using the current session.

### Implementation

```python
class SessionRecommender:
    def __init__(self, decay_rate=0.7):
        self.decay = decay_rate
    
    def get_session_embedding(self, session_paper_ids: list[str], 
                               paper_embeddings: dict) -> np.ndarray:
        """
        Compute session intent vector with recency weighting.
        More recent papers in the session get higher weight.
        """
        if not session_paper_ids:
            return None
        
        embeddings = []
        weights = []
        for i, pid in enumerate(session_paper_ids):
            emb = paper_embeddings.get(pid)
            if emb is not None:
                embeddings.append(emb)
                # Exponential decay: last paper = weight 1.0, previous = 0.7, etc.
                weights.append(self.decay ** (len(session_paper_ids) - 1 - i))
        
        if not embeddings:
            return None
        
        weighted = np.average(embeddings, axis=0, weights=weights)
        return weighted / (np.linalg.norm(weighted) + 1e-8)
    
    def recommend_next(self, session_paper_ids: list[str], 
                        paper_embeddings: dict, 
                        all_paper_ids: list[str], k=5) -> list[str]:
        """Recommend next papers based on session history."""
        session_emb = self.get_session_embedding(session_paper_ids, paper_embeddings)
        if session_emb is None:
            return []  # fall back to popular/trending
        
        # Find nearest papers (excluding already viewed)
        viewed_set = set(session_paper_ids)
        candidates = []
        for pid in all_paper_ids:
            if pid not in viewed_set:
                emb = paper_embeddings[pid]
                sim = np.dot(session_emb, emb)
                candidates.append((pid, sim))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in candidates[:k]]
```

### Session Lifecycle on Supabase

```python
# FastAPI middleware to manage sessions
async def session_middleware(request, call_next):
    session_id = request.cookies.get("scholar_session")
    
    if not session_id:
        session_id = str(uuid4())
        # Create session in Supabase
        await supabase.table("user_sessions").insert({
            "session_id": session_id,
            "user_id": request.state.user_id,  # None if anonymous
            "paper_sequence": []
        }).execute()
    
    request.state.session_id = session_id
    response = await call_next(request)
    response.set_cookie("scholar_session", session_id, max_age=3600)
    return response
```

---

## 4. Recommendation Mixer (The Blending Layer)

This is where all signals combine into a final ranked list.

### The Pipeline

```
User makes request (search or "recommend me papers")
        │
        ├── Content-Based Score ──────────┐
        │   (hybrid retrieval + rerank)   │
        │                                 │
        ├── Personalization Score ─────────┤
        │   (cosine(paper, taste_vector)) │
        │                                 ├──→ WEIGHTED BLEND ──→ MMR ──→ Final Top-10
        ├── Session Score ────────────────┤
        │   (cosine(paper, session_emb))  │
        │                                 │
        ├── Collaborative Score ──────────┤
        │   (similar users read this)     │
        │                                 │
        └── Bandit Exploration ───────────┘
            (inject 2-3 exploration picks)
```

### Blending Implementation

```python
class RecommendationMixer:
    def __init__(self):
        # Weights — tune these based on RAGAS/NDCG results
        self.w_content = 0.40      # retrieval + reranking score
        self.w_personal = 0.25     # long-term taste match
        self.w_session = 0.20      # current session relevance
        self.w_collab = 0.15       # collaborative filtering score
    
    def blend(self, paper_id: str, scores: dict) -> float:
        """Compute final blended score for a paper."""
        return (
            self.w_content * scores.get('content', 0) +
            self.w_personal * scores.get('personal', 0) +
            self.w_session * scores.get('session', 0) +
            self.w_collab * scores.get('collaborative', 0)
        )
    
    def build_recommendations(self, user, session, query=None, k=10):
        """Full recommendation pipeline."""
        
        # 1. Get candidates
        if query:
            # Search mode: hybrid retrieval → rerank → get top-30
            candidates = self.retrieval_engine.search(query, top_k=30)
        else:
            # Feed mode: get popular + topic-relevant papers
            candidates = self.get_feed_candidates(user, top_k=50)
        
        # 2. Score each candidate across all signals
        scored = []
        for paper in candidates:
            scores = {
                'content': paper.rerank_score if query else 0,
                'personal': cosine_sim(paper.embedding, user.taste_vector) 
                            if user.taste_vector is not None else 0,
                'session': cosine_sim(paper.embedding, session.embedding) 
                           if session.embedding is not None else 0,
                'collaborative': self.collab_score(user, paper),
            }
            final = self.blend(paper.id, scores)
            scored.append((paper, final, scores))
        
        # 3. Sort by blended score
        scored.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored[:20]
        
        # 4. MMR diversity re-ranking (Capability 12)
        diverse_results = mmr_rerank(
            [p for p, _, _ in top_candidates],
            query_embedding=session.embedding or user.taste_vector,
            lambda_=0.6, k=k-2  # reserve 2 slots for exploration
        )
        
        # 5. Bandit exploration (Capability 15)
        explore_papers = self.bandit.get_exploration_papers(
            user=user, 
            exclude=[p.id for p in diverse_results],
            n=2
        )
        
        # 6. Combine: main results + exploration
        final_results = diverse_results + explore_papers
        
        # 7. Generate explanations (Capability 11)
        for paper in final_results:
            paper.explanation = self.explainer.explain(user, paper)
        
        return final_results
    
    def collab_score(self, user, paper) -> float:
        """Simple collaborative signal: how many similar users read this paper?"""
        if not user or user.total_interactions < 3:
            return 0.0
        
        # Query Supabase: users with similar taste vectors who interacted with this paper
        similar_user_count = self.get_similar_users_who_read(user.id, paper.id)
        return min(similar_user_count / 10.0, 1.0)  # normalize to 0-1
```

---

## 5. Simple Collaborative Filtering (Supabase-Native)

No need for LightGCN. A SQL-based approach works great:

```sql
-- Find papers read by users similar to the current user
-- "Similar users" = users who have saved 3+ of the same papers

WITH similar_users AS (
    SELECT us2.user_id, COUNT(*) as overlap
    FROM user_saves us1
    JOIN user_saves us2 ON us1.paper_id = us2.paper_id
    WHERE us1.user_id = $current_user_id
      AND us2.user_id != $current_user_id
    GROUP BY us2.user_id
    HAVING COUNT(*) >= 3
    ORDER BY overlap DESC
    LIMIT 20
),
collab_papers AS (
    SELECT us.paper_id, COUNT(*) as popularity, AVG(su.overlap) as avg_overlap
    FROM user_saves us
    JOIN similar_users su ON us.user_id = su.user_id
    WHERE us.paper_id NOT IN (
        SELECT paper_id FROM user_saves WHERE user_id = $current_user_id
    )
    GROUP BY us.paper_id
    ORDER BY popularity * avg_overlap DESC
    LIMIT 20
)
SELECT * FROM collab_papers;
```

This is **pure SQL collaborative filtering** — no model training needed, runs entirely on Supabase, and is easy to explain in interviews.

---

## 6. Bandit Exploration (Thompson Sampling on Supabase)

### Storage

The bandit parameters live in `user_profiles.exploration_alpha` and `exploration_beta` (arrays, one entry per topic).

### Implementation

```python
class TopicBandit:
    def __init__(self, n_topics: int):
        self.n_topics = n_topics
    
    async def get_exploration_papers(self, user, exclude: list[str], n=2) -> list:
        """Select papers from under-explored topics."""
        
        # Load bandit params from Supabase
        alpha = user.exploration_alpha or np.ones(self.n_topics)
        beta = user.exploration_beta or np.ones(self.n_topics)
        
        # Thompson Sampling: sample from Beta distribution
        samples = [np.random.beta(alpha[i], beta[i]) for i in range(self.n_topics)]
        
        # Exclude topics user is already heavy in (those are "exploit")
        user_topics = set(user.preferred_topics or [])
        explore_candidates = [
            (i, s) for i, s in enumerate(samples) if i not in user_topics
        ]
        explore_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Pick top-n exploration topics
        selected_topics = [t[0] for t in explore_candidates[:n]]
        
        # Get best paper from each topic (not in exclude list)
        papers = []
        for topic_id in selected_topics:
            paper = await self.get_top_paper_in_topic(topic_id, exclude)
            if paper:
                paper.is_exploration = True
                paper.explanation = f"🔍 Exploring: {self.topic_labels[topic_id]}"
                papers.append(paper)
        
        return papers
    
    async def update(self, user_id: str, topic_id: int, engaged: bool):
        """Update bandit after user feedback."""
        if engaged:
            # Increment alpha (success)
            await supabase.rpc('increment_bandit_alpha', {
                'uid': user_id, 'topic_idx': topic_id
            }).execute()
        else:
            # Increment beta (failure)
            await supabase.rpc('increment_bandit_beta', {
                'uid': user_id, 'topic_idx': topic_id
            }).execute()
```

### Supabase RPC Functions

```sql
-- Postgres function to increment bandit alpha
CREATE OR REPLACE FUNCTION increment_bandit_alpha(uid UUID, topic_idx INT)
RETURNS VOID AS $$
BEGIN
    UPDATE user_profiles
    SET exploration_alpha[topic_idx + 1] = 
        COALESCE(exploration_alpha[topic_idx + 1], 1.0) + 1.0
    WHERE user_id = uid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

---

## 7. Data Flow — Complete User Journey

```
ANONYMOUS USER                          LOGGED-IN USER
─────────────                           ──────────────
1. Lands on platform                    1. Logs in via Supabase Auth
2. Gets session_id (cookie)             2. Gets session_id + user_id
3. Searches "transformer attention"     3. Searches "transformer attention"
4. Event logged: SEARCH                 4. Event logged: SEARCH
5. Clicks paper #3                      5. Clicks paper #3
6. Event logged: CLICK                  6. Event logged: CLICK
7. Session embedding updated            7. Session + taste vector updated
8. "Continue Reading" sidebar           8. "Continue Reading" + "For You"
   shows session-based recs                shows personalized recs
9. No long-term memory                  9. Returns next day → 
                                           taste vector persists
                                        10. Collaborative: "Users like you
                                            also read..."
                                        11. Bandit: "Discover: Federated
                                            Learning" (new topic)
```

---

## 8. Supabase Integration Summary

| Supabase Feature | Usage |
|-----------------|-------|
| **Auth** | User signup/login, JWT tokens, session management |
| **PostgreSQL** | Papers, embeddings (pgvector), events, user profiles |
| **RLS** | Users can only read their own events/saves |
| **RPC Functions** | Bandit updates, collaborative filtering queries |
| **Realtime** (optional) | Live-update "Continue Reading" sidebar as user browses |
| **Edge Functions** (optional) | Run taste vector recomputation on event triggers |

---

## 9. API Endpoints for Layer 3

```
POST /events/track              → Log a user event (click, save, etc.)
GET  /recommendations/feed      → Personalized paper feed (no query)
GET  /recommendations/search    → Search with personalized reranking
GET  /recommendations/session   → Session-based "Continue Reading"
GET  /recommendations/explain/{paper_id}  → Why was this recommended?
POST /feedback/upvote           → Upvote a recommendation
POST /feedback/downvote         → Downvote a recommendation
GET  /profile/interests         → User's extracted interests/topics
GET  /profile/history           → Reading history
```
