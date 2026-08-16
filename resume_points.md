# ScholarMind — Resume Bullet Points

> **Project Title**: ScholarMind — AI-Powered Academic Research Intelligence Engine  
> **Tech Stack**: Python, FastAPI, Supabase (PostgreSQL + pgvector), LangGraph, Gemini 2.0 Flash, PyTorch, Sentence-Transformers, NetworkX, Docker  
> **Codebase**: ~7,800 lines of Python + SQL across 84 modules | 18 REST API endpoints | 7 recommendation algorithms  
> **GitHub**: github.com/abhijeet738/scholarmind  

---

## Project Description (2-3 lines for resume header)

> Built an end-to-end AI research assistant that combines hybrid semantic search, knowledge graph analysis, agentic LLM workflows, and a 7-algorithm personalized recommendation engine to help researchers discover, evaluate, and track 50K+ academic papers. Deployed on Hugging Face Spaces via CI/CD.

---

## Bullet Points by Category

### Information Retrieval & Search (Phase 1)

- Designed a **hybrid search engine** combining dense retrieval (BAAI/bge-base-en-v1.5 embeddings, 768-dim, pgvector HNSW indexing) with sparse retrieval (BM25) fused via **Reciprocal Rank Fusion**, achieving sub-200ms latency on 50K+ papers
- Implemented a **cross-encoder re-ranking pipeline** (BAAI/bge-reranker-base) that refines the top-20 fused candidates down to 10 final results, improving nDCG@10 over fusion-only retrieval
- Built a **vector ingestion pipeline** for arXiv papers with automated embedding generation, BM25 index serialization, and batch upload to Supabase with pgvector-backed HNSW indexes (m=16, ef_construction=64)

### Knowledge Graph & Analytics (Phase 2)

- Constructed a **research knowledge graph** over 50K papers with SciBERT-extracted entities (METHOD, DATASET, METRIC, TASK), entity resolution via canonical name normalization, and typed relation edges (uses, evaluated_on, outperforms, extends)
- Built a **citation graph engine** using NetworkX with PageRank computation, Louvain community detection, shortest-path analysis, and paper influence scoring (in-degree, out-degree, predecessor/successor chains)
- Developed a **SOTA benchmark tracker** that monitors method-dataset-metric triples across papers and automatically detects state-of-the-art improvements over time
- Implemented **research gap detection** by computing topic co-occurrence matrices from BERTopic clusters and identifying high-potential, under-explored cross-topic research directions

### Agentic Intelligence with LLM (Phase 3)

- Architected a **multi-capability LangGraph agent** with intent-based routing (search, literature review, novelty assessment, consensus analysis, gap analysis) powered by Gemini 2.0 Flash, with a Corrective RAG loop that automatically rewrites queries when <60% of retrieved documents pass relevance grading
- Implemented **automated literature review generation** that synthesizes multi-paper narratives from retrieved documents, with structured sections, citation-backed claims, and LLM-graded hallucination detection
- Built a **novelty assessment pipeline** that analyzes a user's research idea against the existing literature and knowledge graph, identifying overlapping methods, underexplored datasets, and potential novel contributions
- Designed a **research consensus meter** that classifies and aggregates findings across papers on a topic to quantify the level of scientific agreement or disagreement

### Personalized Recommendation Engine (Phase 4)

- Built a **3-stage recommendation pipeline** (Candidate Generation → Scoring → Re-ranking) orchestrating 7 algorithms: SASRec (session-based transformers), LightGCN (GNN collaborative filtering), KGAT (knowledge-graph attention), DeepFM (hybrid deep factorization), DQN (exploration/exploitation), MMR (diversity re-ranking), and PGPR (explainable path-based reasoning)
- Implemented **SASRec** with multi-head self-attention and causal masking for sequential paper prediction, including position embeddings, layer normalization, and heuristic fallback for cold-start sessions
- Designed a **LightGCN-based collaborative filtering module** with multi-hop graph convolution over user-paper bipartite interaction graphs, with a co-interaction popularity fallback for cold-start scenarios
- Developed a **KGAT module** combining knowledge graph entity relations with attention-weighted neighbor aggregation for knowledge-aware recommendations, including path-scoring heuristics for fallback
- Built a **DeepFM scoring model** combining factorization-machine-style feature interactions with a deep MLP, taking 5 input signals (content, personal, session, collaborative, KG scores) plus source count features
- Implemented a **DQN-based explore/exploit agent** with Thompson Sampling bandit fallback, dynamically deciding whether to serve safe high-confidence recommendations or inject serendipitous under-explored topic papers
- Developed **MMR (Maximal Marginal Relevance) re-ranking** using cosine similarity over embedding space to enforce topic diversity in the final recommendation list
- Implemented **PGPR-style explainability** that generates natural-language reasoning paths ("Recommended because you read X, which shares method Y with this paper") for each recommendation

### Evaluation & A/B Testing (Phase 5)

- Built an **offline evaluation suite** implementing 7 ranking metrics (NDCG@K, Hit Rate@K, Precision@K, Recall@K, MRR, Catalogue Coverage, Intra-List Diversity) to benchmark algorithm quality using leave-one-out methodology
- Designed a **database-driven A/B testing framework** with deterministic hash-based user-to-variant assignment, experiment lifecycle management, and automated click-through-rate comparison between algorithm variants
- Created **monitoring API endpoints** (6 routes) exposing real-time system overview, per-algorithm metric history with time-series snapshots, and A/B experiment result dashboards

### Data Engineering & Infrastructure

- Engineered a **multi-script data pipeline** (arXiv fetcher → SciBERT embedding generator → NER entity extractor → BERTopic clusterer → Semantic Scholar citation fetcher → batch Supabase uploader) for end-to-end corpus ingestion
- Designed a **normalized PostgreSQL schema** (4 SQL migration scripts, 12+ tables) with pgvector extensions, GIN indexes for array columns, HNSW indexes for vector search, and UUID primary keys
- Built a **synthetic user data generator** creating research-persona-based interaction sequences (ML Engineer, NLP Researcher, CV Researcher, etc.) with realistic temporal browsing patterns for model training
- Containerized the full backend with a **multi-stage Dockerfile** (CPU-only PyTorch to reduce image from 2GB to ~500MB) and implemented a **GitHub Actions CI/CD pipeline** for automated deployment to Hugging Face Spaces on every push to main

### API Design

- Designed a **RESTful API** with 18 versioned endpoints (under `/api/v1`) organized across 5 functional domains (Search, Knowledge Graph, Agentic Intelligence, Recommendations, Evaluation), with Swagger/OpenAPI documentation auto-generated by FastAPI

---

## Skills to Highlight

| Category | Technologies |
|----------|-------------|
| **Languages** | Python, SQL |
| **ML/DL Frameworks** | PyTorch, Sentence-Transformers, Cross-Encoders |
| **NLP** | BAAI/bge embeddings, BM25 (rank_bm25), SciBERT NER, BERTopic |
| **LLM & Agents** | LangGraph, Gemini 2.0 Flash, Corrective RAG, Prompt Engineering |
| **RecSys** | SASRec, LightGCN, KGAT, DeepFM, DQN, MMR, PGPR |
| **Databases** | PostgreSQL, Supabase, pgvector, Redis |
| **Graph Analysis** | NetworkX, PageRank, Louvain Communities, Citation Graphs |
| **Backend** | FastAPI, Pydantic, REST API Design, CORS |
| **DevOps** | Docker, GitHub Actions CI/CD, Hugging Face Spaces |
| **Evaluation** | NDCG, MRR, Precision/Recall, A/B Testing, Leave-One-Out |

---

## Quantifiable Metrics to Mention

- **50,000+** arXiv papers indexed with 768-dimensional vector embeddings
- **7** recommendation algorithms integrated into a single pipeline
- **18** REST API endpoints across 5 functional domains
- **~7,800** lines of production Python + SQL across 84 modules
- **Sub-200ms** hybrid search latency with HNSW indexing
- **$0** infrastructure cost (Supabase free tier + HF Spaces free tier)
- **3-stage** pipeline: Candidate Generation (4 sources) → Multi-signal Scoring → Diversity Re-ranking
- **Corrective RAG** loop with up to 3 automatic query rewrites on low relevance
