"""
ScholarMind — Application Configuration

Loads environment variables from .env file using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "ScholarMind"
    debug: bool = False

    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""  # anon/public key
    supabase_service_key: str = ""  # service role key (for admin ops)

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Gemini LLM (Phase 3 — Agentic Intelligence)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Models
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    reranker_model: str = "BAAI/bge-reranker-base"

    # Retrieval settings
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_top_k: int = 20
    rerank_top_k: int = 10
    rrf_k: int = 60  # RRF constant

    # Paths
    bm25_index_path: str = "data/bm25_index.pkl"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — loaded once, reused everywhere."""
    return Settings()
