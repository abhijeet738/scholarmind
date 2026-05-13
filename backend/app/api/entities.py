"""
ScholarMind — Entities API

Endpoints for querying the Knowledge Graph: entities, relations,
and paper-entity connections.
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.db.database import get_supabase_client

router = APIRouter()


class EntityResponse(BaseModel):
    name: str
    canonical_name: str | None = None
    type: str
    mention_count: int = 1


class EntityWithPapers(BaseModel):
    canonical_name: str
    type: str
    total_papers: int
    paper_ids: list[str]


@router.get("/entities", response_model=list[EntityResponse])
async def list_entities(
    type: str | None = Query(None, description="Filter by type: METHOD, DATASET, METRIC, TASK"),
    limit: int = Query(50, ge=1, le=200),
    search: str | None = Query(None, description="Search entity names"),
):
    """List top entities, optionally filtered by type."""
    supabase = get_supabase_client()
    query = supabase.table("entities").select("name, canonical_name, type, mention_count")

    if type:
        query = query.eq("type", type.upper())
    if search:
        query = query.ilike("canonical_name", f"%{search}%")

    response = query.order("mention_count", desc=True).limit(limit).execute()
    return response.data


@router.get("/entities/{name}/papers")
async def get_entity_papers(name: str, limit: int = Query(20, ge=1, le=100)):
    """Get all papers that mention a specific entity."""
    supabase = get_supabase_client()

    # Find papers with this entity
    response = (
        supabase.table("entities")
        .select("paper_id")
        .ilike("canonical_name", f"%{name}%")
        .limit(limit)
        .execute()
    )

    paper_ids = list(set(r["paper_id"] for r in response.data))

    if not paper_ids:
        return {"entity": name, "papers": []}

    # Fetch paper details
    papers = (
        supabase.table("papers")
        .select("paper_id, title, abstract, authors, categories, year")
        .in_("paper_id", paper_ids)
        .execute()
    )

    return {"entity": name, "total": len(papers.data), "papers": papers.data}


@router.get("/kg/relations")
async def get_relations(
    paper_id: str | None = Query(None),
    entity: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """Get Knowledge Graph relations for a paper or entity."""
    supabase = get_supabase_client()
    query = supabase.table("relations").select("*")

    if paper_id:
        query = query.eq("paper_id", paper_id)
    elif entity:
        query = query.or_(
            f"source_entity.ilike.%{entity}%,target_entity.ilike.%{entity}%"
        )

    response = query.limit(limit).execute()
    return {"relations": response.data}
