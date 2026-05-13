"""
ScholarMind — Supabase Database Client

Provides a singleton Supabase client used across the application.
"""

from supabase import create_client, Client
from app.config import get_settings

# Module-level client (initialized on first import)
_client: Client | None = None


def get_supabase_client() -> Client:
    """Get or create the Supabase client singleton."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = create_client(settings.supabase_url, settings.supabase_service_key)
    return _client
