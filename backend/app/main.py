"""
ScholarMind — FastAPI Application Entry Point

Starts the API server with the search endpoint and health check.
Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.search import router as search_router
from app.api.entities import router as entities_router
from app.api.sota import router as sota_router
from app.api.topics import router as topics_router
from app.api.citations import router as citations_router
from app.api.review import router as review_router
from app.api.novelty import router as novelty_router
from app.api.consensus import router as consensus_router
from app.api.gaps import router as gaps_router
from app.api.events import router as events_router
from app.api.recommendations import router as recommendations_router
from app.api.users import router as users_router
from app.api.metrics import router as metrics_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="AI Research Intelligence Agent — Hybrid Retrieval Engine",
    version="0.5.0",
)

# CORS — allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes — Phase 1
app.include_router(search_router, prefix="/api/v1", tags=["Search"])

# Register routes — Phase 2
app.include_router(entities_router, prefix="/api/v1", tags=["Knowledge Graph"])
app.include_router(sota_router, prefix="/api/v1", tags=["SOTA Tracker"])
app.include_router(topics_router, prefix="/api/v1", tags=["Topics"])
app.include_router(citations_router, prefix="/api/v1", tags=["Citations"])

# Register routes — Phase 3 (Agentic Intelligence)
app.include_router(review_router, prefix="/api/v1", tags=["Literature Review"])
app.include_router(novelty_router, prefix="/api/v1", tags=["Novelty Assessment"])
app.include_router(consensus_router, prefix="/api/v1", tags=["Consensus Meter"])
app.include_router(gaps_router, prefix="/api/v1", tags=["Research Gaps"])

# Register routes — Phase 4 (Personalization Engine)
app.include_router(events_router, prefix="/api/v1", tags=["Event Tracking"])
app.include_router(recommendations_router, prefix="/api/v1", tags=["Recommendations"])
app.include_router(users_router, prefix="/api/v1", tags=["Users"])

# Register routes — Phase 5 (Evaluation & Monitoring)
app.include_router(metrics_router, prefix="/api/v1", tags=["Metrics & Evaluation"])


@app.get("/health")
async def health_check():
    """Basic health check — verifies the API is running."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": "0.5.0",
        "gemini_configured": bool(settings.gemini_api_key),
        "supabase_configured": bool(settings.supabase_url),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": f"Welcome to {settings.app_name} API",
        "docs": "/docs",
        "health": "/health",
        "search": "POST /api/v1/search",
    }
