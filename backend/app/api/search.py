"""
ScholarMind — Search API Endpoint

The core /search endpoint that runs the full hybrid retrieval pipeline:
Dense (pgvector) → Sparse (BM25) → RRF Fusion → Cross-Encoder Rerank
"""

import time
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.retrieval.dense import dense_search
from app.retrieval.sparse import sparse_search
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank

router = APIRouter()


# ---- Request / Response Models ----

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    filter_year: int | None = Field(default=None, description="Minimum publication year")


class PaperResult(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: list[str] | None = None
    categories: list[str] | None = None
    year: int | None = None
    reranker_score: float | None = None
    rrf_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[PaperResult]
    timings: dict[str, float]  # ms for each pipeline stage


# ---- Endpoint ----

@router.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """
    Full hybrid search pipeline:
    1. Dense retrieval (pgvector cosine search) → top 50
    2. Sparse retrieval (BM25 keyword search) → top 50
    3. Reciprocal Rank Fusion → top 20
    4. Cross-Encoder reranking → top K
    """
    settings = get_settings()
    timings = {}

    # Stage 1: Dense retrieval
    t0 = time.time()
    dense_results = dense_search(
        query=request.query,
        top_k=settings.dense_top_k,
        filter_year=request.filter_year,
    )
    timings["dense_ms"] = round((time.time() - t0) * 1000, 1)

    # Stage 2: Sparse retrieval
    t0 = time.time()
    sparse_results = sparse_search(
        query=request.query,
        top_k=settings.sparse_top_k,
    )
    timings["sparse_ms"] = round((time.time() - t0) * 1000, 1)

    # Stage 3: RRF Fusion
    t0 = time.time()
    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=settings.rrf_top_k,
        k=settings.rrf_k,
    )
    timings["fusion_ms"] = round((time.time() - t0) * 1000, 1)

    # Stage 4: Cross-Encoder Reranking
    t0 = time.time()
    final_results = rerank(
        query=request.query,
        candidates=fused_results,
        top_k=request.top_k,
    )
    timings["rerank_ms"] = round((time.time() - t0) * 1000, 1)

    timings["total_ms"] = round(sum(timings.values()), 1)

    return SearchResponse(
        query=request.query,
        total_results=len(final_results),
        results=[PaperResult(**paper) for paper in final_results],
        timings=timings,
    )
