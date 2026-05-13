"""
ScholarMind — PGPR (Policy-Guided Path Reasoning)

Generates natural language explanations for recommendations
by finding meaningful paths through the Knowledge Graph from
the user's reading history to the recommended paper.

Example output:
  "Recommended because you read 'LoRA: Low-Rank Adaptation'
   → which uses the method 'Low-Rank Decomposition'
   → and 'QLoRA' extends this method with quantization."

Training: RL agent on KG paths (Kaggle GPU)
Fallback: Template-fill from NetworkX shortest paths (Day 1)

Reference: Xian et al., "Reinforcement Knowledge Graph Reasoning
           for Explainable Recommendation", SIGIR 2019
"""

from app.db.database import get_supabase_client


def generate_explanation(
    user_id: str,
    paper_id: str,
    source: str = "unknown",
) -> str:
    """
    Generate a natural language explanation for why a paper
    is recommended to this user.

    Uses PGPR trained model if available, otherwise template fallback.
    """
    # For now, always use fallback (PGPR model training comes later)
    return _fallback_explanation(user_id, paper_id, source)


def _fallback_explanation(
    user_id: str,
    paper_id: str,
    source: str,
) -> str:
    """
    Day 1 fallback: template-based explanation using KG paths.

    Strategy:
    1. Find entities in the recommended paper
    2. Find entities in the user's recently read papers
    3. Find shared entities = the connection
    4. Fill template
    """
    supabase = get_supabase_client()

    # Get the recommended paper's title and entities
    paper = (
        supabase.table("papers")
        .select("title, topic_label")
        .eq("paper_id", paper_id)
        .single()
        .execute()
    )
    paper_title = paper.data.get("title", "this paper") if paper.data else "this paper"

    paper_entities = (
        supabase.table("entities")
        .select("canonical_name, type")
        .eq("paper_id", paper_id)
        .limit(10)
        .execute()
    )

    # Get user's recent papers and their entities
    user_events = (
        supabase.table("user_events")
        .select("paper_id")
        .eq("user_id", user_id)
        .in_("event_type", ["CLICK", "SAVE"])
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )

    if not user_events.data:
        return _source_explanation(source, paper_title)

    user_paper_ids = [e["paper_id"] for e in user_events.data]

    user_entities = (
        supabase.table("entities")
        .select("canonical_name, type, paper_id")
        .in_("paper_id", user_paper_ids[:5])
        .limit(50)
        .execute()
    )

    # Find shared entities (the KG connection)
    paper_entity_names = {e["canonical_name"] for e in (paper_entities.data or []) if e.get("canonical_name")}
    user_entity_names = {}
    for e in (user_entities.data or []):
        if e.get("canonical_name"):
            user_entity_names[e["canonical_name"]] = e

    shared = paper_entity_names & set(user_entity_names.keys())

    if shared:
        # Find which user paper contained the shared entity
        shared_entity = list(shared)[0]
        user_entity = user_entity_names[shared_entity]
        user_paper_id = user_entity.get("paper_id", "")

        # Get user paper title
        user_paper = (
            supabase.table("papers")
            .select("title")
            .eq("paper_id", user_paper_id)
            .single()
            .execute()
        )
        user_paper_title = user_paper.data.get("title", "a paper you read") if user_paper.data else "a paper you read"

        entity_type = user_entity.get("type", "concept").lower()

        return (
            f"Recommended because you read \"{user_paper_title}\" "
            f"→ which involves the {entity_type} \"{shared_entity}\" "
            f"→ and \"{paper_title}\" also uses this {entity_type}."
        )

    # No shared entities found — use topic/source based explanation
    return _source_explanation(source, paper_title)


def _source_explanation(source: str, paper_title: str) -> str:
    """Generate explanation based on recommendation source."""
    explanations = {
        "sasrec": f"Based on your reading pattern, \"{paper_title}\" is the likely next paper in your research trajectory.",
        "sasrec_fallback": f"\"{paper_title}\" is similar to papers you've recently read.",
        "lightgcn": f"Researchers with similar interests to yours also found \"{paper_title}\" valuable.",
        "lightgcn_fallback": f"Users who read similar papers also read \"{paper_title}\".",
        "kgat": f"\"{paper_title}\" connects to your research interests through the Knowledge Graph.",
        "kgat_fallback": f"\"{paper_title}\" shares key methods and datasets with your recent reading.",
        "search": f"\"{paper_title}\" is highly relevant to your search query.",
    }
    return explanations.get(source, f"\"{paper_title}\" matches your research interests.")
