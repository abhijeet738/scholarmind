# 🚀 Phase 1 Implementation Plan: Data & Retrieval Foundation

> **Goal**: Set up the core backend infrastructure, ingest a highly relevant AI research dataset, and build the hybrid retrieval engine (BM25 + Dense + RRF + Cross-Encoder) that forms the baseline for everything else.

---

## 1. Dataset Selection & Curation

Instead of hitting the live arXiv API 35,000 times (which will get us rate-limited and take days), we will use the official arXiv dataset snapshot.

*   **Source Dataset**: The [arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) (maintained by Cornell University). It contains a single `arxiv-metadata-oai-snapshot.json` file with metadata for all 2.4+ million papers.
*   **Filtering Strategy**: We don't want physics or math papers for this project. We want a highly concentrated AI/ML corpus.
    *   **Categories**: `cs.AI` (Artificial Intelligence), `cs.LG` (Machine Learning), `cs.CL` (Computation and Language / NLP).
    *   **Timeframe**: Papers published from **2020 to Present**.
    *   **Target Size**: We will stream the JSON file and extract the first **35,000 papers** that match these criteria.
*   **What we extract**: `id`, `title`, `abstract`, `authors`, `categories`, `update_date`.
*   *(Note: Since downloading from Kaggle requires an account, I will write a script that can also pull directly from Hugging Face datasets like `gfissore/arxiv-abstracts-2021` or use the arXiv OAI-PMH bulk API as a fallback if you don't want to download the 3GB Kaggle file manually).*

---

## 2. Technical Scaffolding

We will set up the foundational project structure.

1.  **Environment**: Python 3.11+ with a virtual environment.
2.  **Dependencies**:
    *   `fastapi`, `uvicorn`: API framework.
    *   `supabase`: Database client.
    *   `sentence-transformers`: For local embedding generation (`BAAI/bge-base-en-v1.5`) and Cross-Encoder reranking.
    *   `rank_bm25`: For sparse retrieval.
    *   `redis`: For caching.
3.  **Docker Compose**: A simple `docker-compose.yml` to spin up local Redis. (We will use a hosted Supabase project for the database to make vector setup and RLS easy, though we can use local Supabase if preferred).

---

## 3. Database Setup (Supabase)

We will execute SQL commands in your Supabase SQL Editor to set up the Layer 1 tables.

1.  **Enable pgvector**: `CREATE EXTENSION IF NOT EXISTS vector;`
2.  **Create Papers Table**:
    ```sql
    CREATE TABLE papers (
        paper_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        abstract TEXT NOT NULL,
        authors TEXT[] NOT NULL,
        categories TEXT[] NOT NULL,
        published_at TIMESTAMPTZ,
        embedding VECTOR(384), -- Dimension for bge-base-en-v1.5
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    ```
3.  **Vector Index**: Create an HNSW index for lightning-fast cosine similarity search.
    ```sql
    CREATE INDEX ON papers USING hnsw (embedding vector_cosine_ops);
    ```
4.  **Search RPC**: A Postgres function `match_papers` to execute the dense vector search directly in the database.

---

## 4. Ingestion & Embedding Pipeline

We will create a command-line script (`ingest.py`) to process the dataset.

1.  **Parser**: Reads the raw JSON metadata and filters for AI/ML papers.
2.  **Embedder**: Loads `BAAI/bge-base-en-v1.5` on your CPU/Mac (it's small and fast). It takes the string `"Title: {title}\nAbstract: {abstract}"` and converts it into a 384-dimensional vector.
3.  **Batch Uploader**: Pushes the metadata and vectors to Supabase in chunks of 500 to optimize network traffic.
4.  **BM25 Index Builder**: While processing, it tokenizes the text and builds a `rank_bm25` index. Because 35K abstracts is small (~30MB in RAM), we will save this index to disk as a `.pkl` file so FastAPI can load it instantly on startup.

---

## 5. The Hybrid Retrieval Engine

We will build the core `SearchEngine` class inside FastAPI.

1.  **Dense Retrieval**: Takes user query → embeds it → calls Supabase `match_papers` RPC → returns top 50 matches.
2.  **Sparse Retrieval**: Takes user query → tokenizes it → queries the in-memory BM25 index → returns top 50 matches.
3.  **RRF Fusion**: Combines the two lists using Reciprocal Rank Fusion: `score = 1 / (60 + rank)`. This gives us the top 20 candidate papers that are both semantically and keyword-relevant.
4.  **Cross-Encoder Reranker**: Passes the top 20 candidates through `cross-encoder/ms-marco-MiniLM-L-6-v2`. This model scores the exact relationship between the Query and the Paper text.
5.  Returns the absolute best Top 10 papers.

---

## 6. FastAPI Endpoints

1.  **`/health`**: Basic check to ensure API, Supabase, and Redis are connected.
2.  **`/search`**:
    *   **Method**: POST
    *   **Input**: `{"query": "parameter efficient fine tuning", "top_k": 10}`
    *   **Output**: Ranked list of JSON paper objects with their final reranker scores.

---

## Step-by-Step Execution Plan for Us:

1.  **Step 1**: I will write the `requirements.txt` and initial directory structure.
2.  **Step 2**: I will provide you with the SQL to run in your Supabase dashboard to set up the DB.
3.  **Step 3**: I will write the dataset fetcher and ingestion pipeline (`ingestion/arxiv_fetcher.py`).
4.  **Step 4**: We will run the ingestion script to populate your database with 35K papers.
5.  **Step 5**: I will write the retrieval engine (`retrieval/dense.py`, `sparse.py`, `fusion.py`, `reranker.py`).
6.  **Step 6**: I will write the FastAPI `main.py` and test the `/search` endpoint.

Are you ready to begin Step 1? Make sure you have an empty Supabase project ready, and let me know if you prefer to use the Kaggle JSON download method or if I should use a Hugging Face dataset library to pull the data automatically.
