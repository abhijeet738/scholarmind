"""
ScholarMind — SASRec (Self-Attentive Sequential Recommendation)

Transformer-based model that predicts the next paper a user will
want based on their reading sequence. Uses causal self-attention
so each position can only attend to previous papers.

Architecture: 2-layer Transformer Encoder with causal mask
Training: BPR loss on Kaggle GPU
Fallback: Weighted embedding average (works Day 1)

Reference: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018
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
from app.config import get_settings


# ============================================================
# SASRec Model (PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    class SASRecModel(nn.Module):
        """Self-Attentive Sequential Recommendation model."""

        def __init__(
            self,
            num_items: int,
            embed_dim: int = 384,
            num_heads: int = 2,
            num_layers: int = 2,
            max_seq_len: int = 50,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.num_items = num_items
            self.embed_dim = embed_dim
            self.max_seq_len = max_seq_len

            # Embeddings
            self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
            self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(embed_dim)

            # Transformer encoder with causal mask
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward(self, seq: "torch.Tensor") -> "torch.Tensor":
            """
            Forward pass.

            Args:
                seq: (batch_size, seq_len) — paper ID indices

            Returns:
                (batch_size, embed_dim) — next paper prediction embedding
            """
            batch_size, seq_len = seq.size()

            # Create embeddings
            positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
            x = self.item_embedding(seq) + self.position_embedding(positions)
            x = self.layer_norm(self.dropout(x))

            # Causal mask: each position can only see itself and previous
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=seq.device),
                diagonal=1,
            ).bool()

            # Padding mask: ignore padded positions (id=0)
            padding_mask = (seq == 0)

            # Transformer forward
            output = self.transformer(
                x,
                mask=causal_mask,
                src_key_padding_mask=padding_mask,
            )

            # Return last position's output = next item prediction
            return output[:, -1, :]

        def predict_next(self, seq: "torch.Tensor") -> "torch.Tensor":
            """Get scores for all items given a sequence."""
            output = self.forward(seq)  # (batch, embed_dim)
            scores = output @ self.item_embedding.weight.T  # (batch, num_items+1)
            return scores


# ============================================================
# SASRec Wrapper (handles model loading + fallback)
# ============================================================

_sasrec_model = None
_paper_id_to_idx = None
_idx_to_paper_id = None


def _load_model():
    """Load trained SASRec model and ID mappings from disk."""
    global _sasrec_model, _paper_id_to_idx, _idx_to_paper_id

    model_path = Path("data/models/sasrec.pt")
    mapping_path = Path("data/models/sasrec_mapping.pkl")

    if not model_path.exists() or not TORCH_AVAILABLE:
        return False

    with open(mapping_path, "rb") as f:
        mappings = pickle.load(f)
        _paper_id_to_idx = mappings["paper_id_to_idx"]
        _idx_to_paper_id = mappings["idx_to_paper_id"]

    num_items = len(_paper_id_to_idx)
    _sasrec_model = SASRecModel(num_items=num_items)
    _sasrec_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    _sasrec_model.eval()
    return True


def predict_next_papers(
    session_paper_ids: list[str],
    top_k: int = 50,
) -> list[dict]:
    """
    Predict next papers for a session.

    Uses the trained SASRec model if available,
    otherwise falls back to weighted embedding average.
    """
    global _sasrec_model

    if _sasrec_model is None:
        if not _load_model():
            return _fallback_predict(session_paper_ids, top_k)

    return _model_predict(session_paper_ids, top_k)


def _model_predict(session_paper_ids: list[str], top_k: int) -> list[dict]:
    """Predict using the trained SASRec Transformer."""
    # Convert paper IDs to indices
    seq_indices = []
    for pid in session_paper_ids[-50:]:  # max sequence length
        idx = _paper_id_to_idx.get(pid, 0)
        seq_indices.append(idx)

    if not seq_indices:
        return []

    # Pad to length 50
    while len(seq_indices) < 50:
        seq_indices.insert(0, 0)  # pad left

    seq_tensor = torch.tensor([seq_indices], dtype=torch.long)

    with torch.no_grad():
        scores = _sasrec_model.predict_next(seq_tensor)[0]  # (num_items+1,)

    # Get top-K (exclude padding index 0 and already-read papers)
    already_read = set(_paper_id_to_idx.get(pid, -1) for pid in session_paper_ids)
    scores[0] = float("-inf")  # padding
    for idx in already_read:
        if 0 <= idx < len(scores):
            scores[idx] = float("-inf")

    top_indices = torch.topk(scores, min(top_k, len(scores))).indices.tolist()

    results = []
    for idx in top_indices:
        pid = _idx_to_paper_id.get(idx, None)
        if pid:
            results.append({
                "paper_id": pid,
                "score": float(scores[idx]),
                "source": "sasrec",
            })

    return results


def _fallback_predict(session_paper_ids: list[str], top_k: int) -> list[dict]:
    """
    Day 1 fallback: weighted average of session paper embeddings.

    Recent papers get higher weight. Find papers closest to
    this weighted average in embedding space.
    """
    if not session_paper_ids:
        return []

    supabase = get_supabase_client()

    # Get embeddings for session papers
    papers = (
        supabase.table("papers")
        .select("paper_id, embedding")
        .in_("paper_id", session_paper_ids[-10:])
        .execute()
    )

    if not papers.data:
        return []

    # Weighted average (recent = higher weight)
    embeddings = []
    weights = []
    for i, p in enumerate(papers.data):
        if p.get("embedding"):
            embeddings.append(np.array(p["embedding"]))
            weights.append(1.0 + i * 0.5)  # more recent = higher

    if not embeddings:
        return []

    weights = np.array(weights) / sum(weights)
    avg_embedding = np.average(embeddings, axis=0, weights=weights).tolist()

    # Use pgvector similarity search with this average
    settings = get_settings()
    results = supabase.rpc(
        "match_papers",
        {
            "query_embedding": avg_embedding,
            "match_count": top_k + len(session_paper_ids),
        },
    ).execute()

    # Filter out already-read papers
    already_read = set(session_paper_ids)
    candidates = [
        {
            "paper_id": r["paper_id"],
            "score": float(r.get("similarity", 0)),
            "source": "sasrec_fallback",
        }
        for r in (results.data or [])
        if r["paper_id"] not in already_read
    ]

    return candidates[:top_k]
