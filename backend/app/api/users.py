"""
ScholarMind — Users API

GET /api/v1/users/{user_id}/profile — Get user taste profile.
POST /api/v1/users — Create a new user profile.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.recsys.user_model import get_or_create_profile, update_taste_vector
from app.recsys.event_tracker import get_user_history

router = APIRouter()


class CreateUserRequest(BaseModel):
    username: str | None = None


@router.post("/users")
async def create_user(request: CreateUserRequest):
    """Create a new user profile."""
    import uuid
    user_id = str(uuid.uuid4())
    profile = get_or_create_profile(user_id, request.username)
    return {"user_id": user_id, "profile": profile}


@router.get("/users/{user_id}/profile")
async def get_profile(user_id: str):
    """
    Get a user's taste profile.

    Returns:
    - preferred_topics: top research topics
    - total_interactions: engagement count
    - recent_history: last 20 interactions
    """
    profile = get_or_create_profile(user_id)
    history = get_user_history(user_id, limit=20)

    return {
        "user_id": user_id,
        "username": profile.get("username"),
        "total_interactions": profile.get("total_interactions", 0),
        "preferred_topics": profile.get("preferred_topics", []),
        "has_taste_vector": profile.get("taste_vector") is not None,
        "recent_history": history,
    }


@router.post("/users/{user_id}/refresh-taste")
async def refresh_taste(user_id: str):
    """Force recompute the user's taste vector from interaction history."""
    update_taste_vector(user_id)
    return {"status": "updated", "user_id": user_id}
