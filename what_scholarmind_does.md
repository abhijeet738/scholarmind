# ScholarMind — What It Actually Does (Interview Opening Script)

## The Problem It Solves

A researcher today has to:
1. **Search** across thousands of papers on Google Scholar — results are keyword-only, no semantic understanding
2. **Read** dozens of papers manually to write a literature review
3. **Guess** whether their research idea is novel — no way to systematically check
4. **Wonder** what the scientific consensus is on a question — no tool aggregates evidence
5. **Miss** research gaps — no tool tells you which popular topics are rarely combined

**ScholarMind solves all 5 problems in one platform.**

---

## The 5 Things a User Can Do

### 1. 🔍 Smart Paper Search (`POST /api/v1/search`)
**User says:** *"Find papers on transformer efficiency"*

**What happens behind the scenes:**
```
User query
  → bge-base-en-v1.5 embeds query into 768-dim vector
  → pgvector HNSW index finds top 50 semantically similar papers
  → BM25 finds top 50 keyword-matched papers
  → Reciprocal Rank Fusion merges both lists into top 20
  → Cross-encoder (bge-reranker-base) re-scores top 20
  → Returns top 10 most relevant papers with timing metrics
```

**What the user gets:** The 10 most relevant papers, ranked by both meaning AND keywords, with millisecond-level timing breakdown.

**Why it's better than Google Scholar:** Google Scholar is keyword-only. ScholarMind understands that "visual categorization" and "image classification" mean the same thing.

---

### 2. 📝 Automated Literature Review (`POST /api/v1/review`)
**User says:** *"Write a literature review on federated learning in healthcare"*

**What happens behind the scenes:**
```
User topic
  → LLM decomposes into 4-6 sub-topics:
      "privacy-preserving FL", "FL in medical imaging", 
      "communication-efficient FL", "FL with non-IID data"
  → For EACH sub-topic:
      → Runs hybrid search → finds 5 real papers
      → LLM synthesizes a paragraph citing [Author, Year]
  → LLM assembles all sections into a full review:
      Introduction → Thematic Sections → Comparison Table → Gaps → References
  → Hallucination checker verifies every claim is grounded
```

**What the user gets:** A complete, structured literature review in Markdown — with real citations, a comparison table, and identified gaps. Plus a hallucination score (0.0-1.0) showing how trustworthy it is.

**This is "citation-backed"** because every paper cited is a REAL paper from the database, not a hallucinated reference.

---

### 3. 💡 Novelty Assessment (`POST /api/v1/novelty`)
**User says:** *"How novel is applying graph neural networks to drug discovery using protein structures?"*

**What happens behind the scenes:**
```
User idea
  → LLM parses into 3 dimensions:
      Method: "graph neural networks"
      Domain: "drug discovery"  
      Application: "protein structures"
  → Generates 3 targeted search queries
  → Hybrid search finds closest prior art for each dimension
  → LLM scores novelty on 3 axes (0-10 each):
      - Method novelty: Has GNN been used before?
      - Domain novelty: Has this been done in drug discovery?
      - Combination novelty: Has this EXACT combo been tried?
  → Suggests ways to differentiate
```

**What the user gets:** A novelty report with scores, closest prior art listed, and concrete suggestions to make the idea more original.

---

### 4. ⚖️ Scientific Consensus Meter (`POST /api/v1/consensus`)
**User says:** *"Does batch normalization improve training stability?"*

**What happens behind the scenes:**
```
User question
  → Hybrid search retrieves 20 relevant papers
  → LLM classifies each paper's stance:
      ✅ SUPPORTS | ❌ CONTRADICTS | ➖ INCONCLUSIVE
  → Extracts a key quote from each paper
  → Aggregates into a verdict:
      "YES — 75% agreement, Strong evidence"
```

**What the user gets:** A clear verdict (YES/NO/MIXED), agreement percentage, evidence strength, and individual evidence cards with quotes from each paper.

---

### 5. 🔬 Research Gap Finder (`GET /api/v1/gaps`)
**User says:** *"Show me unexplored research opportunities"*

**What happens behind the scenes:**
```
BERTopic has already clustered all 50K papers into ~150 topics
  → For every pair of popular topics (>20 papers each):
      gap_score = popularity(A) × popularity(B) / (co_occurrence + 1)
  → High gap_score = two popular topics that are RARELY combined
  → Returns top 20 research gaps, sorted by opportunity score
```

**What the user gets:** A ranked list like:
- "Reinforcement Learning × Medical Imaging" — gap score: 4500 (both popular, rarely combined)
- "Federated Learning × Graph Neural Networks" — gap score: 3200

**No LLM needed** — this is pure algorithmic computation over BERTopic clusters.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        USER REQUEST                          │
│         (search / review / novelty / consensus / gaps)        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (13 routers)                │
│                   Deployed on HF Spaces via CI/CD             │
└──────────────────────┬───────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌──────────────┐ ┌──────────┐ ┌────────────────┐
│ SEARCH ENGINE│ │ LANGGRAPH│ │ KNOWLEDGE LAYER│
│              │ │  AGENT   │ │                │
│ • BM25       │ │          │ │ • Citation     │
│ • pgvector   │ │ • Query  │ │   Graph        │
│   (dense)    │ │   Analyzer│ │   (NetworkX)  │
│ • RRF Fusion │ │ • CRAG   │ │ • Louvain      │
│ • Cross-     │ │   Loop   │ │   Clustering   │
│   Encoder    │ │ • Review │ │ • PageRank     │
│   Reranker   │ │   Writer │ │ • Gap Analyzer │
│              │ │ • Novelty│ │ • Entity       │
│              │ │ • Consen-│ │   Resolution   │
│              │ │   sus    │ │                │
│              │ │ • Hallu- │ │                │
│              │ │   cination│ │               │
│              │ │   Check  │ │                │
└──────┬───────┘ └────┬─────┘ └───────┬────────┘
       │              │               │
       └──────────────┼───────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────┐
│              Supabase (PostgreSQL + pgvector)                 │
│                                                              │
│  papers (50K+) │ entities │ citations │ topics │ results     │
│  + HNSW vector index (768-dim embeddings)                    │
└──────────────────────────────────────────────────────────────┘
                      ▲
                      │ (offline batch ingestion)
┌──────────────────────────────────────────────────────────────┐
│                   DATA PIPELINE (Kaggle GPU)                  │
│                                                              │
│  arXiv Fetcher → Embedding Gen → Entity Extraction →          │
│  BERTopic Clustering → Citation Fetching → Supabase Upload    │
└──────────────────────────────────────────────────────────────┘
```

---

## Your 2-Minute Interview Pitch

> *"ScholarMind is a Research Intelligence System I built to solve the problem of information overload in academic research.*
>
> *At its core, it's a hybrid search engine over 50,000+ arXiv papers. It combines BM25 keyword search with dense vector retrieval using pgvector, fuses results with Reciprocal Rank Fusion, and reranks them with a cross-encoder — so users get both keyword precision and semantic understanding.*
>
> *On top of the search engine, I built a LangGraph-based agentic layer with 5 capabilities. The most interesting one is citation-backed literature reviews — the agent decomposes a topic into sub-queries, retrieves real papers for each, synthesizes sections with proper citations, and then a hallucination checker verifies that every claim is grounded in the source papers. It also has a novelty assessor, a consensus meter, and a research gap finder.*
>
> *The knowledge layer uses a citation graph built with NetworkX — I run PageRank for paper influence ranking, Louvain community detection to find research clusters, and BERTopic topic modeling to discover latent research themes. The gap analyzer computes co-occurrence scores between topics to surface unexplored research directions.*
>
> *The whole system runs on a FastAPI backend with PostgreSQL/pgvector via Supabase, containerized with Docker, and deployed to Hugging Face Spaces through a GitHub Actions CI/CD pipeline.*
>
> *I think this is directly relevant to OpenText because you solve the same core problem — helping organizations search, understand, and extract intelligence from massive document collections — just at an enterprise scale."*

Then the interviewer will pick a keyword and ask you to deep-dive. You're prepared for all of them.
