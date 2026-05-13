"""
ScholarMind — LightGCN (Light Graph Convolutional Network)

Graph-based collaborative filtering: propagates embeddings
through the user-paper bipartite graph to capture multi-hop
collaborative signals.

Key insight: removes ALL nonlinearities from GCN — just
neighborhood aggregation + mean pooling across layers.
This actually IMPROVES CF performance (proven in the paper).

Architecture: 3-layer GCN with mean pooling
Training: BPR loss + MixGCF hard negatives on Kaggle GPU
Fallback: Jaccard similarity CF (works Day 1)

Reference: He et al., "LightGCN", SIGIR 2020
"""

import pickle
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.db.database import get_supabase_client


# ============================================================
# LightGCN Model (PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    class LightGCNModel(nn.Module):
        """Light Graph Convolutional Network for collaborative filtering."""

        def __init__(
            self,
            num_users: int,
            num_items: int,
            embed_dim: int = 64,
            num_layers: int = 3,
        ):
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.num_layers = num_layers

            self.user_embedding = nn.Embedding(num_users, embed_dim)
            self.item_embedding = nn.Embedding(num_items, embed_dim)

            # Xavier initialization
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)

        def forward(self, adj: "torch.Tensor") -> tuple:
            """
            Forward pass: propagate through normalized adjacency matrix.

            Args:
                adj: Normalized adjacency matrix (sparse tensor)

            Returns:
                (user_embeddings, item_embeddings) after GCN propagation
            """
            # Concatenate all embeddings
            all_embeddings = torch.cat([
                self.user_embedding.weight,
                self.item_embedding.weight,
            ])

            # Multi-layer propagation (no nonlinearity!)
            layer_embeddings = [all_embeddings]
            for _ in range(self.num_layers):
                all_embeddings = torch.sparse.mm(adj, all_embeddings)
                layer_embeddings.append(all_embeddings)

            # Mean pooling across all layers
            final_embeddings = torch.stack(layer_embeddings).mean(dim=0)

            user_embs = final_embeddings[:self.num_users]
            item_embs = final_embeddings[self.num_users:]
            return user_embs, item_embs

        def predict(self, user_emb: "torch.Tensor", item_embs: "torch.Tensor") -> "torch.Tensor":
            """Score items for a specific user."""
            return (user_emb @ item_embs.T).squeeze()


# ============================================================
# LightGCN Wrapper
# ============================================================

_lightgcn_model = None
_user_id_to_idx = None
_paper_id_to_idx = None
_idx_to_paper_id = None
_adj_matrix = None
_user_embeddings = None
_item_embeddings = None


def _load_model():
    """Load trained LightGCN model and mappings."""
    global _lightgcn_model, _user_id_to_idx, _paper_id_to_idx
    global _idx_to_paper_id, _user_embeddings, _item_embeddings

    model_path = Path("data/models/lightgcn.pt")
    mapping_path = Path("data/models/lightgcn_mapping.pkl")
    embeddings_path = Path("data/models/lightgcn_embeddings.pkl")

    if not model_path.exists() or not TORCH_AVAILABLE:
        return False

    with open(mapping_path, "rb") as f:
        mappings = pickle.load(f)
        _user_id_to_idx = mappings["user_id_to_idx"]
        _paper_id_to_idx = mappings["paper_id_to_idx"]
        _idx_to_paper_id = mappings["idx_to_paper_id"]

    # Load precomputed embeddings (faster than running forward pass)
    with open(embeddings_path, "rb") as f:
        embs = pickle.load(f)
        _user_embeddings = embs["user_embeddings"]  # numpy array
        _item_embeddings = embs["item_embeddings"]  # numpy array

    return True


def get_collaborative_picks(
    user_id: str,
    top_k: int = 50,
    exclude_paper_ids: list[str] | None = None,
) -> list[dict]:
    """
    Get collaborative filtering recommendations for a user.

    Uses trained LightGCN if available, otherwise falls back
    to Jaccard similarity.
    """
    if _user_embeddings is None:
        if not _load_model():
            return _fallback_jaccard(user_id, top_k, exclude_paper_ids)

    return _model_predict(user_id, top_k, exclude_paper_ids)


def _model_predict(
    user_id: str,
    top_k: int,
    exclude_paper_ids: list[str] | None,
) -> list[dict]:
    """Predict using trained LightGCN embeddings."""
    user_idx = _user_id_to_idx.get(user_id)
    if user_idx is None:
        return _fallback_jaccard(user_id, top_k, exclude_paper_ids)

    user_emb = _user_embeddings[user_idx]  # (embed_dim,)
    scores = _item_embeddings @ user_emb    # (num_items,)

    # Exclude already-read papers
    exclude_set = set(exclude_paper_ids or [])
    exclude_indices = {_paper_id_to_idx[pid] for pid in exclude_set if pid in _paper_id_to_idx}

    # Get top-K
    sorted_indices = np.argsort(scores)[::-1]
    results = []
    for idx in sorted_indices:
        if idx in exclude_indices:
            continue
        pid = _idx_to_paper_id.get(int(idx))
        if pid:
            results.append({
                "paper_id": pid,
                "score": float(scores[idx]),
                "source": "lightgcn",
            })
        if len(results) >= top_k:
            break

    return results


def _fallback_jaccard(
    user_id: str,
    top_k: int,
    exclude_paper_ids: list[str] | None,
) -> list[dict]:
    """
    Day 1 fallback: Jaccard similarity collaborative filtering.

    Find users who saved/clicked similar papers, then recommend
    papers THEY liked that this user hasn't seen.
    """
    supabase = get_supabase_client()

    # Get this user's saved/clicked papers
    user_papers = (
        supabase.table("user_events")
        .select("paper_id")
        .eq("user_id", user_id)
        .in_("event_type", ["CLICK", "SAVE", "UPVOTE"])
        .execute()
    )

    if not user_papers.data:
        return []

    my_papers = set(r["paper_id"] for r in user_papers.data)
    exclude_set = set(exclude_paper_ids or []) | my_papers

    # Find other users who interacted with the same papers
    similar_users_papers = (
        supabase.table("user_events")
        .select("user_id, paper_id")
        .in_("paper_id", list(my_papers)[:20])
        .neq("user_id", user_id)
        .in_("event_type", ["CLICK", "SAVE", "UPVOTE"])
        .limit(500)
        .execute()
    )

    if not similar_users_papers.data:
        return []

    # Count paper frequency among similar users (excluding already seen)
    from collections import Counter
    paper_counts = Counter()
    for r in similar_users_papers.data:
        if r["paper_id"] not in exclude_set:
            paper_counts[r["paper_id"]] += 1

    results = [
        {
            "paper_id": pid,
            "score": float(count),
            "source": "lightgcn_fallback",
        }
        for pid, count in paper_counts.most_common(top_k)
    ]

    return results
