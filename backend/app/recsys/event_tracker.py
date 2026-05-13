"""
ScholarMind — Event Tracker

Captures every user interaction (click, expand, dwell, save, vote)
and stores it in the user_events table. This data feeds ALL
recommendation algorithms in Phase 4.

Event weights:
    CLICK    = 1.0    (clicked paper card)
    EXPAND   = 1.5    (expanded abstract)
    DWELL_30S = 2.0   (stayed 30+ seconds)
    DWELL_60S = 3.0   (stayed 60+ seconds)
    SAVE     = 4.0    (bookmarked)
    UPVOTE   = 5.0    (upvoted recommendation)
    DOWNVOTE = -3.0   (downvoted recommendation)
"""

import uuid
from datetime import datetime, timezone

from app.db.database import get_supabase_client


# Event type → weight mapping
EVENT_WEIGHTS = {
    "CLICK": 1.0,
    "EXPAND": 1.5,
    "DWELL_30S": 2.0,
    "DWELL_60S": 3.0,
    "SAVE": 4.0,
    "UPVOTE": 5.0,
    "DOWNVOTE": -3.0,
}


def log_event(
    user_id: str,
    session_id: str,
    paper_id: str,
    event_type: str,
    query_text: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Log a user interaction event.

    Args:
        user_id: UUID of the user
        session_id: UUID of the current session
        paper_id: ID of the paper interacted with
        event_type: One of CLICK, EXPAND, DWELL_30S, DWELL_60S, SAVE, UPVOTE, DOWNVOTE
        query_text: The search query that led to this interaction (optional)
        metadata: Any additional data (optional)

    Returns:
        The created event record.
    """
    event_type = event_type.upper()
    weight = EVENT_WEIGHTS.get(event_type, 1.0)

    supabase = get_supabase_client()

    event = {
        "user_id": user_id,
        "session_id": session_id,
        "paper_id": paper_id,
        "event_type": event_type,
        "event_weight": weight,
        "query_text": query_text,
        "metadata": metadata or {},
    }

    response = supabase.table("user_events").insert(event).execute()

    # Also update the session's paper sequence
    _update_session_sequence(session_id, paper_id, event_type)

    # Update user profile interaction count
    _increment_user_interactions(user_id)

    return response.data[0] if response.data else event


def _update_session_sequence(session_id: str, paper_id: str, event_type: str):
    """Append paper to session sequence on CLICK events."""
    if event_type != "CLICK":
        return

    supabase = get_supabase_client()

    # Get current sequence
    session = (
        supabase.table("user_sessions")
        .select("paper_sequence")
        .eq("session_id", session_id)
        .single()
        .execute()
    )

    if session.data:
        current = session.data.get("paper_sequence", [])
        if paper_id not in current:
            current.append(paper_id)
            supabase.table("user_sessions").update(
                {"paper_sequence": current}
            ).eq("session_id", session_id).execute()


def _increment_user_interactions(user_id: str):
    """Increment the user's total interaction count."""
    supabase = get_supabase_client()
    try:
        profile = (
            supabase.table("user_profiles")
            .select("total_interactions")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        if profile.data:
            count = profile.data.get("total_interactions", 0) + 1
            supabase.table("user_profiles").update(
                {"total_interactions": count}
            ).eq("user_id", user_id).execute()
    except Exception:
        pass


def create_session(user_id: str | None = None) -> str:
    """Create a new user session, return session_id."""
    supabase = get_supabase_client()
    session_id = str(uuid.uuid4())

    session = {
        "session_id": session_id,
        "user_id": user_id,
        "paper_sequence": [],
        "is_active": True,
    }

    supabase.table("user_sessions").insert(session).execute()
    return session_id


def end_session(session_id: str):
    """Mark a session as ended."""
    supabase = get_supabase_client()
    supabase.table("user_sessions").update({
        "is_active": False,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }).eq("session_id", session_id).execute()


def get_session_papers(session_id: str) -> list[str]:
    """Get the ordered list of paper IDs read in a session."""
    supabase = get_supabase_client()

    session = (
        supabase.table("user_sessions")
        .select("paper_sequence")
        .eq("session_id", session_id)
        .single()
        .execute()
    )

    if session.data:
        return session.data.get("paper_sequence", [])
    return []


def get_user_history(user_id: str, limit: int = 100) -> list[dict]:
    """Get recent interaction history for a user."""
    supabase = get_supabase_client()

    response = (
        supabase.table("user_events")
        .select("paper_id, event_type, event_weight, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    return response.data
