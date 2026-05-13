"""
ScholarMind — DeepFM (Deep Factorization Machine)

Hybrid scoring model: FM layer captures low-order feature
interactions (pairs), DNN layer captures high-order interactions.
Replaces hardcoded weight blending with a LEARNED scoring function.

Input features per candidate paper:
  - content_score, personal_score, session_score, collab_score, kg_score
  - topic_id (categorical), author_overlap (binary)
  - paper_age_days, citation_count, pagerank_score, user_tenure_days

Training: Binary cross-entropy on click/no-click data (Kaggle GPU)
Fallback: Linear weighted blend (works Day 1)

Reference: Guo et al., "DeepFM", IJCAI 2017
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


# ============================================================
# DeepFM Model (PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    class DeepFMModel(nn.Module):
        """Deep Factorization Machine for context-aware scoring."""

        def __init__(
            self,
            num_continuous: int = 10,      # continuous feature count
            num_topics: int = 200,          # categorical: topic ID
            embed_dim: int = 8,            # FM embedding dimension
            hidden_dims: list[int] | None = None,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.num_continuous = num_continuous

            # FM: embedding for each continuous feature
            self.fm_continuous_embeddings = nn.ModuleList([
                nn.Linear(1, embed_dim) for _ in range(num_continuous)
            ])

            # FM: embedding for categorical feature (topic_id)
            self.fm_topic_embedding = nn.Embedding(num_topics + 1, embed_dim, padding_idx=0)

            # FM first-order weights
            self.fm_first_order = nn.Linear(num_continuous + 1, 1)

            # DNN
            if hidden_dims is None:
                hidden_dims = [128, 64, 32]

            total_input = (num_continuous + 1) * embed_dim  # all FM embeddings concatenated
            dnn_layers = []
            prev_dim = total_input
            for dim in hidden_dims:
                dnn_layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = dim
            dnn_layers.append(nn.Linear(prev_dim, 1))
            self.dnn = nn.Sequential(*dnn_layers)

        def forward(
            self,
            continuous_features: "torch.Tensor",   # (batch, num_continuous)
            topic_ids: "torch.Tensor",              # (batch,) int
        ) -> "torch.Tensor":
            """
            Forward pass.

            Returns: (batch,) click probability scores
            """
            # FM embeddings
            fm_embeddings = []
            for i in range(self.num_continuous):
                feat = continuous_features[:, i:i+1]  # (batch, 1)
                emb = self.fm_continuous_embeddings[i](feat)  # (batch, embed_dim)
                fm_embeddings.append(emb)

            topic_emb = self.fm_topic_embedding(topic_ids)  # (batch, embed_dim)
            fm_embeddings.append(topic_emb)

            fm_stack = torch.stack(fm_embeddings, dim=1)  # (batch, n_fields, embed_dim)

            # FM second-order: sum of pairwise interactions
            sum_of_embs = fm_stack.sum(dim=1)  # (batch, embed_dim)
            sum_of_squares = (fm_stack ** 2).sum(dim=1)
            fm_second = 0.5 * (sum_of_embs ** 2 - sum_of_squares).sum(dim=1)  # (batch,)

            # FM first-order
            first_order_input = torch.cat([
                continuous_features,
                topic_ids.unsqueeze(1).float(),
            ], dim=1)
            fm_first = self.fm_first_order(first_order_input).squeeze(-1)  # (batch,)

            # DNN
            dnn_input = fm_stack.view(fm_stack.size(0), -1)  # (batch, n_fields * embed_dim)
            dnn_out = self.dnn(dnn_input).squeeze(-1)  # (batch,)

            # Combine
            output = torch.sigmoid(fm_first + fm_second + dnn_out)
            return output


# ============================================================
# DeepFM Wrapper
# ============================================================

_deepfm_model = None


def _load_model():
    """Load trained DeepFM model."""
    global _deepfm_model

    model_path = Path("data/models/deepfm.pt")
    if not model_path.exists() or not TORCH_AVAILABLE:
        return False

    _deepfm_model = DeepFMModel()
    _deepfm_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    _deepfm_model.eval()
    return True


def score_candidates(
    candidates: list[dict],
    user_id: str,
) -> list[dict]:
    """
    Score candidates using DeepFM or linear fallback.

    Each candidate dict must have:
        paper_id, content_score, personal_score, session_score,
        collab_score, kg_score, topic_id, citation_count,
        pagerank_score, paper_age_days
    """
    if _deepfm_model is None:
        if not _load_model():
            return _fallback_linear_blend(candidates)

    return _model_score(candidates)


def _model_score(candidates: list[dict]) -> list[dict]:
    """Score using trained DeepFM."""
    if not candidates:
        return []

    # Build feature tensors
    continuous = []
    topics = []

    for c in candidates:
        continuous.append([
            c.get("content_score", 0),
            c.get("personal_score", 0),
            c.get("session_score", 0),
            c.get("collab_score", 0),
            c.get("kg_score", 0),
            c.get("author_overlap", 0),
            c.get("paper_age_days", 365),
            c.get("citation_count", 0),
            c.get("pagerank_score", 0),
            c.get("user_tenure_days", 0),
        ])
        topics.append(c.get("topic_id", 0) or 0)

    continuous_tensor = torch.tensor(continuous, dtype=torch.float32)
    topic_tensor = torch.tensor(topics, dtype=torch.long)

    with torch.no_grad():
        scores = _deepfm_model(continuous_tensor, topic_tensor).numpy()

    for i, c in enumerate(candidates):
        c["deepfm_score"] = float(scores[i])
        c["final_score"] = float(scores[i])

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates


def _fallback_linear_blend(candidates: list[dict]) -> list[dict]:
    """
    Day 1 fallback: hardcoded linear weight blend.

    Weights tuned for reasonable defaults:
      content=0.35, personal=0.25, session=0.20, collab=0.10, kg=0.10
    """
    WEIGHTS = {
        "content_score": 0.35,
        "personal_score": 0.25,
        "session_score": 0.20,
        "collab_score": 0.10,
        "kg_score": 0.10,
    }

    for c in candidates:
        score = sum(
            c.get(key, 0) * weight
            for key, weight in WEIGHTS.items()
        )
        # Small boost for highly cited papers
        citation_boost = min(c.get("citation_count", 0) / 1000, 0.1)
        c["deepfm_score"] = score + citation_boost
        c["final_score"] = score + citation_boost

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates
