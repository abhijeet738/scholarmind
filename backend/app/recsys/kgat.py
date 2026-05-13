"""
ScholarMind — KGAT (Knowledge Graph Attention Network)

Recommends papers using attention-weighted propagation through
the Knowledge Graph. Combines user-item interactions with KG
relations (uses_method, evaluated_on, extends) in one unified graph.

Key insight: attention = exp(-||h + r - t||₂) (TransR-style)
learns WHICH KG relations matter for each user.

Training: BPR + TransR loss on Kaggle GPU
Fallback: NetworkX path scoring through the KG

Reference: Wang et al., "KGAT", KDD 2019
"""

import pickle
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.db.database import get_supabase_client


# ============================================================
# KGAT Model (PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    class KGATModel(nn.Module):
        """Knowledge Graph Attention Network."""

        def __init__(
            self,
            num_entities: int,    # users + papers + KG entities
            num_relations: int,   # interaction types + KG relation types
            embed_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_layers = num_layers

            self.entity_embedding = nn.Embedding(num_entities, embed_dim)
            self.relation_embedding = nn.Embedding(num_relations, embed_dim)

            # Attention layers (one per GNN layer)
            self.attention_layers = nn.ModuleList([
                nn.Linear(embed_dim * 3, 1) for _ in range(num_layers)
            ])

            self.dropout = nn.Dropout(dropout)

            nn.init.xavier_uniform_(self.entity_embedding.weight)
            nn.init.xavier_uniform_(self.relation_embedding.weight)

        def compute_attention(
            self,
            head_embs: "torch.Tensor",
            rel_embs: "torch.Tensor",
            tail_embs: "torch.Tensor",
            layer_idx: int,
        ) -> "torch.Tensor":
            """Compute attention weights for edges."""
            # TransR-style: score = exp(-||h + r - t||)
            transr_score = -torch.norm(head_embs + rel_embs - tail_embs, dim=-1)

            # Learned attention
            concat = torch.cat([head_embs, rel_embs, tail_embs], dim=-1)
            attn_score = self.attention_layers[layer_idx](concat).squeeze(-1)

            # Combine
            alpha = F.softmax(transr_score + attn_score, dim=0)
            return alpha

        def forward(
            self,
            edge_index: "torch.Tensor",
            edge_type: "torch.Tensor",
        ) -> "torch.Tensor":
            """
            Forward pass: propagate through KG with attention.

            Args:
                edge_index: (2, num_edges) — head, tail indices
                edge_type: (num_edges,) — relation type indices

            Returns:
                Updated entity embeddings
            """
            x = self.entity_embedding.weight

            for layer_idx in range(self.num_layers):
                heads = edge_index[0]
                tails = edge_index[1]

                head_embs = x[heads]
                tail_embs = x[tails]
                rel_embs = self.relation_embedding(edge_type)

                # Compute attention
                alpha = self.compute_attention(head_embs, rel_embs, tail_embs, layer_idx)

                # Weighted aggregation
                weighted_msgs = alpha.unsqueeze(-1) * tail_embs

                # Scatter add to update embeddings
                out = torch.zeros_like(x)
                out.scatter_add_(0, heads.unsqueeze(-1).expand_as(weighted_msgs), weighted_msgs)

                x = self.dropout(F.relu(x + out))

            return x


# ============================================================
# KGAT Wrapper
# ============================================================

_kgat_embeddings = None
_entity_id_to_idx = None
_idx_to_entity_id = None


def _load_model():
    """Load precomputed KGAT embeddings."""
    global _kgat_embeddings, _entity_id_to_idx, _idx_to_entity_id

    embeddings_path = Path("data/models/kgat_embeddings.pkl")

    if not embeddings_path.exists():
        return False

    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)
        _kgat_embeddings = data["embeddings"]
        _entity_id_to_idx = data["entity_id_to_idx"]
        _idx_to_entity_id = data["idx_to_entity_id"]

    return True


def get_kg_aware_picks(
    user_id: str,
    top_k: int = 50,
    exclude_paper_ids: list[str] | None = None,
) -> list[dict]:
    """
    Get knowledge-graph-aware recommendations.

    Uses trained KGAT if available, otherwise falls back
    to NetworkX path scoring.
    """
    if _kgat_embeddings is None:
        if not _load_model():
            return _fallback_kg_paths(user_id, top_k, exclude_paper_ids)

    return _model_predict(user_id, top_k, exclude_paper_ids)


def _model_predict(
    user_id: str,
    top_k: int,
    exclude_paper_ids: list[str] | None,
) -> list[dict]:
    """Predict using trained KGAT embeddings."""
    user_idx = _entity_id_to_idx.get(f"user_{user_id}")
    if user_idx is None:
        return _fallback_kg_paths(user_id, top_k, exclude_paper_ids)

    user_emb = _kgat_embeddings[user_idx]

    # Score all paper entities
    exclude_set = set(exclude_paper_ids or [])
    results = []

    for entity_id, idx in _entity_id_to_idx.items():
        if not entity_id.startswith("paper_"):
            continue
        pid = entity_id.replace("paper_", "")
        if pid in exclude_set:
            continue

        item_emb = _kgat_embeddings[idx]
        score = float(np.dot(user_emb, item_emb))
        results.append({"paper_id": pid, "score": score, "source": "kgat"})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def _fallback_kg_paths(
    user_id: str,
    top_k: int,
    exclude_paper_ids: list[str] | None,
) -> list[dict]:
    """
    Day 1 fallback: score papers by shortest KG path from user interests.

    1. Get entities from user's recently read papers
    2. Find other papers that share entities
    3. Score by: shared_entity_count / path_length
    """
    supabase = get_supabase_client()

    # Get user's recent papers
    events = (
        supabase.table("user_events")
        .select("paper_id")
        .eq("user_id", user_id)
        .in_("event_type", ["CLICK", "SAVE"])
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )

    if not events.data:
        return []

    my_papers = set(r["paper_id"] for r in events.data)
    exclude_set = set(exclude_paper_ids or []) | my_papers

    # Get entities from user's papers
    my_entities = (
        supabase.table("entities")
        .select("canonical_name, type")
        .in_("paper_id", list(my_papers)[:20])
        .execute()
    )

    if not my_entities.data:
        return []

    entity_names = list(set(e["canonical_name"] for e in my_entities.data if e.get("canonical_name")))

    if not entity_names:
        return []

    # Find papers that share these entities
    related = (
        supabase.table("entities")
        .select("paper_id, canonical_name")
        .in_("canonical_name", entity_names[:30])
        .limit(500)
        .execute()
    )

    # Score by shared entity count
    from collections import Counter
    paper_scores = Counter()
    for r in related.data:
        if r["paper_id"] not in exclude_set:
            paper_scores[r["paper_id"]] += 1

    results = [
        {"paper_id": pid, "score": float(count), "source": "kgat_fallback"}
        for pid, count in paper_scores.most_common(top_k)
    ]

    return results
