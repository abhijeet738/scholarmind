"""
ScholarMind — Dense Retrieval via Supabase pgvector

Embeds the user query with bge-base-en-v1.5, then calls the
match_papers RPC function in Supabase to find the most
semantically similar papers using HNSW cosine search.
"""

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from app.config import get_settings
from app.db.database import get_supabase_client

# Module-level model (loaded once on first call)
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _embed_model
    if _embed_model is None:
        settings = get_settings()
        _embed_model = SentenceTransformer(settings.embedding_model)
    return _embed_model


def embed_query(query: str) -> list[float]:
    """
    Embed a user query into a 768-dim vector.
    
    BGE models expect the instruction prefix for queries
    (not for documents — those were embedded without it on Kaggle).
    """
    model = _get_embed_model()
    # BGE instruction prefix for retrieval queries
    instruction = "Represent this sentence for searching relevant passages: "
    embedding = model.encode(instruction + query, normalize_embeddings=True)
    return embedding.tolist()


def dense_search(query: str, top_k: int = 50, filter_year: int | None = None) -> list[dict]:
    """
    Perform dense vector search against Supabase pgvector.

    Args:
        query: The user's search query string.
        top_k: Number of results to return.
        filter_year: Optional minimum publication year filter.

    Returns:
        List of paper dicts with similarity scores, sorted by relevance.
    """
    query_embedding = embed_query(query)
    supabase = get_supabase_client()

    # Call the match_papers RPC function we created in Supabase
    response = supabase.rpc(
        "match_papers",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "filter_year": filter_year,
        },
    ).execute()

    return response.data
