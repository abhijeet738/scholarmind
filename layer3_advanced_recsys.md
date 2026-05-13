# 🔬 Layer 3 Advanced RecSys — Algorithm Upgrades

> Replacing primitive algorithms with research-grade methods from [arXiv:2407.13699v1](https://arxiv.org/html/2407.13699v1)

---

## What Changed: Primitive → Advanced

| Component | ❌ Old (Primitive) | ✅ New (Research-Grade) | Paper Section |
|-----------|-------------------|------------------------|---------------|
| Sequential Rec | Weighted embedding avg | **SASRec** (Self-Attentive Sequential Rec) | §7.2 |
| Collaborative Filtering | Jaccard similarity SQL | **LightGCN** (Graph Convolution) | §7.1 |
| Knowledge-Aware Rec | Simple KG path lookup | **KGAT** (KG Attention Network) | §7.3 |
| Explore/Exploit | Thompson Sampling bandit | **DQN-based RL Agent** | §7.4 |
| Context-Aware Scoring | Linear weighted blend | **DeepFM** (Deep Factorization Machine) | §8.1 |
| Explainability | Template-fill from paths | **PGPR** (Policy-Guided Path Reasoning) | §7.3/§7.4 |

---

## 1. SASRec — Self-Attentive Sequential Recommendation

**Paper**: *Self-Attentive Sequential Recommendation* (Kang & McAuley, 2018)

**Why**: Our old approach (weighted embedding average) treats all session items equally with a simple decay. SASRec uses **self-attention** to learn WHICH past papers matter most for predicting the next one.

**Architecture**:
```
Session: [paper₁, paper₂, paper₃, paper₄] → predict paper₅

paper₁ ──→ ┌──────────────────────┐
paper₂ ──→ │  Transformer Encoder │ ──→ next_paper_embedding
paper₃ ──→ │  (2 layers, 2 heads) │
paper₄ ──→ └──────────────────────┘
```

**Implementation**:
```python
class SASRec(nn.Module):
    def __init__(self, num_items, embed_dim=384, num_heads=2, num_layers=2, max_len=50):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, item_seq):  # item_seq: (batch, seq_len)
        positions = torch.arange(item_seq.size(1)).unsqueeze(0)
        x = self.item_embedding(item_seq) + self.pos_embedding(positions)
        
        # Causal mask: each position can only attend to previous positions
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool()
        output = self.transformer(x, mask=mask)
        
        return output[:, -1, :]  # last position = predicted next item embedding
```

**Training**: BPR (Bayesian Personalized Ranking) loss on historical sessions.  
**Why not BERT4Rec?** SASRec is unidirectional (causal) — better for real-time "predict next" vs BERT4Rec's bidirectional (better for fill-in-the-blank).

---

## 2. LightGCN — Simplified Graph Collaborative Filtering

**Paper**: *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* (He et al., SIGIR 2020)

**Why**: Our old SQL Jaccard approach can't capture **multi-hop collaborative signals** (User A → Paper X → User B → Paper Y). LightGCN propagates embeddings across the user-paper bipartite graph.

**Key Insight**: LightGCN removes all the heavy operations (feature transformation, nonlinear activation) from standard GCNs — keeping ONLY neighborhood aggregation. This makes it both simpler and better-performing.

**Architecture**:
```
User-Paper Bipartite Graph:
    User₁ ←→ Paper_A, Paper_B
    User₂ ←→ Paper_B, Paper_C
    User₃ ←→ Paper_A, Paper_C, Paper_D

Layer 0: Initial embeddings (learnable)
Layer 1: Aggregate 1-hop neighbors
Layer 2: Aggregate 2-hop neighbors  
Layer 3: Aggregate 3-hop neighbors
Final:   Mean of all layers → prediction
```

**Implementation**:
```python
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=3):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.num_layers = num_layers
    
    def forward(self, adj_matrix):
        # adj_matrix: normalized adjacency of user-item bipartite graph
        all_embs = torch.cat([self.user_emb.weight, self.item_emb.weight])
        layer_embs = [all_embs]
        
        for _ in range(self.num_layers):
            all_embs = torch.sparse.mm(adj_matrix, all_embs)  # message passing
            layer_embs.append(all_embs)
        
        # Mean pooling across all layers (key LightGCN insight)
        final_embs = torch.stack(layer_embs).mean(dim=0)
        user_embs, item_embs = torch.split(final_embs, [num_users, num_items])
        return user_embs, item_embs
    
    def predict(self, user_id, item_id):
        return torch.dot(self.user_embs[user_id], self.item_embs[item_id])
```

**Training**: BPR loss + negative sampling (MixGCF-style hard negatives).  
**Supabase integration**: Store learned embeddings in `user_profiles` and `papers` tables. Retrain daily as a batch job.

---

## 3. KGAT — Knowledge Graph Attention Network

**Paper**: *KGAT: Knowledge Graph Attention Network for Recommendation* (Wang et al., KDD 2019)

**Why**: Our KG has entities (methods, datasets, tasks) and relations. KGAT uses **graph attention** to learn which KG relations matter most for each user, making recommendations both accurate AND explainable.

**Architecture**:
```
Knowledge Graph:
    Paper_A --uses_method--> Transformer
    Paper_A --evaluated_on--> GLUE
    Paper_B --uses_method--> Transformer  
    Paper_B --extends--> LoRA
    User₁ --clicked--> Paper_A

KGAT propagates through KG:
    User₁ → Paper_A → Transformer → Paper_B (recommended!)
    Attention weights tell us WHY: "because both use Transformers"
```

**Key Operations**:
1. **Embedding layer**: Initialize entity + relation embeddings (TransR-style)
2. **Attentive embedding propagation**: For each entity, aggregate neighbor info weighted by learned attention
3. **Prediction**: `score(user, paper) = user_emb · paper_emb` after K layers of propagation

```python
# Attention coefficient for edge (head, relation, tail)
def attention(self, h_emb, r_emb, t_emb):
    # TransR-style: project into relation space
    score = (h_emb + r_emb - t_emb).norm(p=2)
    return torch.exp(-score)  # higher score = more relevant neighbor
```

**Why KGAT over RippleNet?** KGAT jointly models user-item AND knowledge graph in a single unified graph. RippleNet only propagates from user side.

---

## 4. DQN-Based RL Recommendation Agent

**Paper**: Based on DRN (Zheng et al., WWW 2018) and KERL (Wang et al., SIGIR 2020)

**Why**: Thompson Sampling is a simple bandit — it treats each topic independently. A DQN agent models the **sequential nature** of recommendation: "if I show paper X now, what's the best paper to show NEXT?"

**Architecture**:
```
State = [user_taste_vector, session_embedding, last_3_actions, context_features]
Action = select a paper from candidate pool
Reward = click(+1), save(+3), dwell_60s(+5), ignore(-0.5)

┌───────────┐    ┌─────────────┐    ┌──────────┐
│   State   │───→│  DQN (MLP)  │───→│ Q-values │
│  (384+d)  │    │  3 layers   │    │ per paper│
└───────────┘    └─────────────┘    └──────────┘
```

**Implementation**:
```python
class RecommenderDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)  # Q-values for each candidate paper

# Double DQN training with experience replay
class DQNAgent:
    def __init__(self):
        self.policy_net = RecommenderDQN(state_dim=512, action_dim=100)
        self.target_net = RecommenderDQN(state_dim=512, action_dim=100)
        self.replay_buffer = deque(maxlen=10000)
        self.epsilon = 0.1  # exploration rate
    
    def select_action(self, state, candidates):
        if random.random() < self.epsilon:
            return random.choice(candidates)  # explore
        q_values = self.policy_net(state)
        return candidates[q_values[candidates].argmax()]  # exploit
```

**Practical approach**: Start with Thompson Sampling (fast to deploy), then upgrade to DQN once you have enough logged interaction data (~10K+ sessions).

---

## 5. DeepFM — Deep Factorization Machine for Scoring

**Paper**: *DeepFM: A Factorization-Machine based Neural Network* (Guo et al., 2017)

**Why**: Our old linear blend (`0.4*content + 0.25*personal + ...`) can't capture **feature interactions**. DeepFM learns both low-order (FM component) AND high-order (DNN component) feature interactions automatically.

**Architecture**:
```
Input Features:
    [content_score, personal_score, session_score, collab_score,
     topic_id, author_count, paper_age, user_tenure, ...]

            ┌──────────┐
Features ──→│    FM    │──→ Low-order interactions  ──┐
            └──────────┘                              ├──→ σ(sum) → P(click)
            ┌──────────┐                              │
Features ──→│   DNN    │──→ High-order interactions ──┘
            └──────────┘
```

**Implementation**:
```python
class DeepFM(nn.Module):
    def __init__(self, feature_dims, embed_dim=16, hidden_dims=[256, 128]):
        super().__init__()
        # FM component
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in feature_dims
        ])
        self.linear = nn.Linear(sum(feature_dims), 1)
        
        # DNN component
        dnn_input = len(feature_dims) * embed_dim
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(dnn_input, h), nn.ReLU(), nn.Dropout(0.2)]
            dnn_input = h
        layers.append(nn.Linear(dnn_input, 1))
        self.dnn = nn.Sequential(*layers)
    
    def forward(self, x):
        # FM: sum of pairwise embedding interactions
        embeds = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        fm_input = torch.stack(embeds, dim=1)
        square_of_sum = fm_input.sum(dim=1).pow(2)
        sum_of_square = fm_input.pow(2).sum(dim=1)
        fm_out = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)
        
        # DNN: high-order interactions
        dnn_input = torch.cat(embeds, dim=1)
        dnn_out = self.dnn(dnn_input)
        
        return torch.sigmoid(fm_out + dnn_out + self.linear(x.float()))
```

**Replaces**: The static `0.4*content + 0.25*personal` weights → learned automatically from click data.

---

## 6. PGPR — Policy-Guided Path Reasoning for Explainability

**Paper**: *Reinforcement Knowledge Graph Reasoning for Explainable Recommendation* (Xian et al., SIGIR 2019)

**Why**: Instead of finding ALL paths and scoring them, PGPR trains an **RL agent to walk the KG** from user to item, finding the most meaningful explanation path.

**Architecture**:
```
Knowledge Graph Walk:
    Agent starts at: User₁
    Action space: follow any edge (clicked, uses_method, evaluated_on, etc.)
    Goal: reach recommended Paper_B in ≤4 steps
    Reward: +1 if path reaches target, bonus for diverse edge types

    User₁ --clicked→ Paper_A --uses_method→ Transformer --used_by→ Paper_B ✓
    
    Explanation: "Recommended because you read Paper_A, which uses 
    Transformers, and Paper_B also uses Transformers."
```

---

## Multi-Stage Pipeline (Production Architecture)

```
User Request
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 1: CANDIDATE GENERATION     │  ← Fast, broad retrieval
│  • Hybrid search (BM25 + Dense)    │     ~1000 candidates
│  • LightGCN collaborative picks    │
│  • SASRec session predictions      │
│  • KGAT knowledge-aware picks      │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Stage 2: SCORING / RANKING        │  ← Accurate, feature-rich
│  • Cross-encoder reranking         │     ~100 candidates
│  • DeepFM context-aware scoring    │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Stage 3: RE-RANKING               │  ← Quality, diversity
│  • MMR diversity re-ranking        │     ~10 final results
│  • DQN exploration injection (2-3) │
│  • PGPR explanation generation     │
└─────────────────────────────────────┘
```

---

## Practical Implementation Strategy

| Algorithm | Complexity | When to Build | Fallback |
|-----------|-----------|--------------|----------|
| **SASRec** | Medium | Phase 3 | Weighted avg (already have) |
| **LightGCN** | Medium | Phase 4 (needs user data) | SQL Jaccard CF |
| **KGAT** | High | Phase 4 (needs KG built) | Simple KG path scoring |
| **DeepFM** | Medium | Phase 4 (needs click logs) | Linear weight blend |
| **DQN Agent** | High | Phase 5 (needs 10K+ sessions) | Thompson Sampling |
| **PGPR** | High | Phase 5 | Template path explanation |

> [!TIP]
> **Build order**: Start with the fallbacks (they work immediately), then upgrade to the advanced versions once you have enough data. This is exactly how Netflix and Spotify do it — ship simple, iterate to complex.

---

## Interview Talking Points

> "My recommendation pipeline uses a **3-stage architecture** similar to YouTube/Netflix: candidate generation (LightGCN + SASRec + hybrid retrieval), scoring (DeepFM for learned feature interactions), and re-ranking (MMR + RL exploration). I started with simpler baselines and progressively upgraded to these research-grade algorithms as user data accumulated."

> "For sequential recommendation, I implemented **SASRec** with self-attention over reading sessions — it outperformed weighted averages by learning which past interactions are most predictive of future interest."

> "For explainability, I use **PGPR** — an RL agent that walks the knowledge graph to find the most meaningful path between a user and a recommended paper, generating natural language explanations like 'Recommended because you read LoRA, which uses low-rank adaptation, and QLoRA extends this method.'"
