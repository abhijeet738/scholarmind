"""
ScholarMind — Events API

POST /api/v1/events — Log user interactions (clicks, saves, dwells).
POST /api/v1/sessions — Create a new session.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.recsys.event_tracker import log_event, create_session, end_session
from app.recsys.user_model import update_taste_vector

router = APIRouter()


class EventRequest(BaseModel):
    user_id: str
    session_id: str
    paper_id: str
    event_type: str  # CLICK, EXPAND, DWELL_30S, DWELL_60S, SAVE, UPVOTE, DOWNVOTE
    query_text: str | None = None
    metadata: dict | None = None


class SessionRequest(BaseModel):
    user_id: str | None = None


@router.post("/events")
async def record_event(request: EventRequest):
    """
    Log a user interaction event.

    This feeds ALL recommendation algorithms:
    - Updates user taste vector
    - Appends to session sequence (for SASRec)
    - Stores in user_events (for LightGCN, DeepFM training)
    """
    event = log_event(
        user_id=request.user_id,
        session_id=request.session_id,
        paper_id=request.paper_id,
        event_type=request.event_type,
        query_text=request.query_text,
        metadata=request.metadata,
    )

    # Update taste vector asynchronously (on every 5th interaction)
    try:
        from app.db.database import get_supabase_client
        supabase = get_supabase_client()
        profile = (
            supabase.table("user_profiles")
            .select("total_interactions")
            .eq("user_id", request.user_id)
            .single()
            .execute()
        )
        if profile.data and profile.data.get("total_interactions", 0) % 5 == 0:
            update_taste_vector(request.user_id)
    except Exception:
        pass

    return {"status": "recorded", "event": event}


@router.post("/sessions")
async def start_session(request: SessionRequest):
    """Create a new reading session."""
    session_id = create_session(request.user_id)
    return {"session_id": session_id}


@router.post("/sessions/{session_id}/end")
async def stop_session(session_id: str):
    """End an active session."""
    end_session(session_id)
    return {"status": "ended", "session_id": session_id}
