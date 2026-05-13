"""
ScholarMind — DQN Explore/Exploit Agent

Decides when to show safe recommendations vs. surprising
discoveries using Deep Q-Learning. Optimizes long-term
user satisfaction, not just immediate clicks.

Training: Double DQN with experience replay on Kaggle GPU
Fallback: Thompson Sampling bandit (works Day 1, no training)

Reference: Zheng et al., "DRN: A Deep Reinforcement Learning Framework
           for News Recommendation", WWW 2018
"""

import pickle
import random
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
# DQN Model (PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    class DQNModel(nn.Module):
        """Double DQN for explore/exploit decision."""

        def __init__(self, state_dim: int = 20, num_actions: int = 2):
            """
            Args:
                state_dim: Size of state vector (user features)
                num_actions: 0=exploit (safe recs), 1=explore (diverse recs)
            """
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_actions),
            )

        def forward(self, state: "torch.Tensor") -> "torch.Tensor":
            return self.network(state)


# ============================================================
# Thompson Sampling Fallback (Day 1)
# ============================================================

def thompson_sampling_decision(
    user_id: str,
    num_topics: int = 50,
) -> dict:
    """
    Thompson Sampling bandit for explore/exploit.

    Each topic has a Beta(α, β) distribution:
    - α = number of times a paper from this topic was clicked
    - β = number of times it was ignored

    Sample from each, recommend topics with highest samples.
    Naturally balances explore (uncertain topics) vs exploit (known good topics).
    """
    supabase = get_supabase_client()

    # Get user's alpha/beta parameters
    profile = (
        supabase.table("user_profiles")
        .select("exploration_alpha, exploration_beta, preferred_topics")
        .eq("user_id", user_id)
        .single()
        .execute()
    )

    if not profile.data:
        # New user: pure exploration
        return {
            "action": "explore",
            "explore_topics": [],
            "confidence": 0.0,
        }

    alpha = profile.data.get("exploration_alpha", [])
    beta = profile.data.get("exploration_beta", [])
    preferred = profile.data.get("preferred_topics", [])

    if not alpha or not beta:
        # Initialize with uniform priors
        alpha = [1.0] * num_topics
        beta = [1.0] * num_topics

    # Sample from Beta distributions
    samples = [
        np.random.beta(a + 1, b + 1) for a, b in zip(alpha, beta)
    ]

    # Top topics by sample value
    top_topics = sorted(range(len(samples)), key=lambda i: samples[i], reverse=True)

    # Decide: explore if top sample is from a non-preferred topic
    explore_topics = [t for t in top_topics[:5] if t not in preferred]

    if len(explore_topics) >= 2:
        action = "explore"
    else:
        action = "exploit"

    return {
        "action": action,
        "explore_topics": explore_topics[:3],
        "exploit_topics": preferred[:3],
        "confidence": max(samples) if samples else 0.0,
    }


def update_thompson_params(user_id: str, topic_id: int, was_clicked: bool):
    """Update Thompson Sampling parameters after a recommendation."""
    supabase = get_supabase_client()

    profile = (
        supabase.table("user_profiles")
        .select("exploration_alpha, exploration_beta")
        .eq("user_id", user_id)
        .single()
        .execute()
    )

    if not profile.data:
        return

    alpha = list(profile.data.get("exploration_alpha", []))
    beta = list(profile.data.get("exploration_beta", []))

    # Extend if needed
    while len(alpha) <= topic_id:
        alpha.append(1.0)
        beta.append(1.0)

    if was_clicked:
        alpha[topic_id] += 1.0
    else:
        beta[topic_id] += 1.0

    supabase.table("user_profiles").update({
        "exploration_alpha": alpha,
        "exploration_beta": beta,
    }).eq("user_id", user_id).execute()


# ============================================================
# DQN Wrapper
# ============================================================

_dqn_model = None


def _load_model():
    """Load trained DQN model."""
    global _dqn_model

    model_path = Path("data/models/dqn.pt")
    if not model_path.exists() or not TORCH_AVAILABLE:
        return False

    _dqn_model = DQNModel()
    _dqn_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    _dqn_model.eval()
    return True


def get_explore_exploit_decision(
    user_id: str,
    session_diversity: float = 0.5,
    satisfaction_trend: float = 0.5,
) -> dict:
    """
    Decide whether to explore or exploit.

    Uses trained DQN if available, otherwise Thompson Sampling.
    """
    if _dqn_model is None:
        if not _load_model():
            return thompson_sampling_decision(user_id)

    # Build state vector for DQN
    supabase = get_supabase_client()
    profile = (
        supabase.table("user_profiles")
        .select("total_interactions, preferred_topics, taste_vector")
        .eq("user_id", user_id)
        .single()
        .execute()
    )

    if not profile.data:
        return thompson_sampling_decision(user_id)

    # Construct state
    taste = profile.data.get("taste_vector", [0] * 20)[:15]
    state_features = taste + [
        profile.data.get("total_interactions", 0) / 100,
        session_diversity,
        satisfaction_trend,
        len(profile.data.get("preferred_topics", [])) / 10,
        random.random(),  # epsilon-like noise
    ]

    state_tensor = torch.tensor([state_features[:20]], dtype=torch.float32)

    with torch.no_grad():
        q_values = _dqn_model(state_tensor)[0]

    action = "explore" if q_values[1] > q_values[0] else "exploit"

    return {
        "action": action,
        "exploit_q": float(q_values[0]),
        "explore_q": float(q_values[1]),
        "confidence": float(abs(q_values[1] - q_values[0])),
    }
