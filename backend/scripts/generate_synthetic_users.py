"""
ScholarMind — Synthetic User Data Generator

This script generates synthetic users, reading sessions, and interactions (clicks, saves, etc.)
to populate the database for training and testing the Phase 4 recommendation models
(SASRec, LightGCN, DeepFM, etc.).

Run with: python scripts/generate_synthetic_users.py
"""

import sys
import os
import uuid
import random
from datetime import datetime, timedelta, timezone

# Add the backend directory to Python path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.database import get_supabase_client
from app.recsys.user_model import update_taste_vector
import numpy as np

NUM_USERS = 200
SESSIONS_PER_USER = (1, 5)  # min, max
PAPERS_PER_SESSION = (2, 10) # min, max
EXPLORATION_RATE = 0.2  # 20% chance to click a random topic instead of preferred

def generate_synthetic_data():
    supabase = get_supabase_client()

    print("Fetching papers and topics from Supabase...")
    # Fetch papers to interact with (limit to a reasonable number for simulation)
    papers_res = supabase.table("papers").select("paper_id, topic_id").limit(5000).execute()
    papers = papers_res.data

    if not papers:
        print("No papers found in the database. Run Phase 1 & 2 ingestion first.")
        return

    # Group papers by topic for persona-based simulation
    topic_to_papers = {}
    all_paper_ids = []
    for p in papers:
        tid = p.get("topic_id", -1)
        pid = p["paper_id"]
        all_paper_ids.append(pid)
        if tid not in topic_to_papers:
            topic_to_papers[tid] = []
        topic_to_papers[tid].append(pid)

    all_topics = list(topic_to_papers.keys())
    if -1 in all_topics:
        all_topics.remove(-1) # Remove outlier topic if present

    if not all_topics:
        # Fallback if no topics exist yet
        all_topics = [0]
        topic_to_papers[0] = all_paper_ids

    print(f"Loaded {len(papers)} papers across {len(all_topics)} topics.")
    print(f"Generating {NUM_USERS} synthetic users...")

    total_events_created = 0

    for i in range(NUM_USERS):
        user_id = str(uuid.uuid4())
        username = f"synthetic_researcher_{i}"

        # 1. Assign Persona (1-3 preferred topics)
        num_preferred = random.randint(1, min(3, len(all_topics)))
        preferred_topics = random.sample(all_topics, num_preferred)

        # Create user profile
        profile = {
            "user_id": user_id,
            "username": username,
            "total_interactions": 0,
            "preferred_topics": preferred_topics,
            "exploration_alpha": [1.0] * 50, # Dummy values for Thompson sampling
            "exploration_beta": [1.0] * 50,
            "created_at": (datetime.now(timezone.utc) - timedelta(days=random.randint(1, 30))).isoformat()
        }
        supabase.table("user_profiles").insert(profile).execute()

        num_sessions = random.randint(*SESSIONS_PER_USER)
        user_interaction_count = 0
        user_saves = []

        # 2. Generate Sessions
        for s in range(num_sessions):
            session_id = str(uuid.uuid4())
            session_start = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 14), hours=random.randint(0, 23))
            
            num_papers = random.randint(*PAPERS_PER_SESSION)
            session_sequence = []
            
            for p_idx in range(num_papers):
                # Decide topic: exploit (preferred) or explore (random)
                if random.random() > EXPLORATION_RATE:
                    chosen_topic = random.choice(preferred_topics)
                else:
                    chosen_topic = random.choice(all_topics)
                
                # Pick a random paper from that topic
                if not topic_to_papers.get(chosen_topic):
                    continue # Should be rare
                    
                paper_id = random.choice(topic_to_papers[chosen_topic])
                
                # Avoid duplicates in same session
                if paper_id in session_sequence:
                    continue

                session_sequence.append(paper_id)
                event_time = session_start + timedelta(minutes=p_idx * random.randint(1, 10))

                # Create CLICK event (base)
                event = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "paper_id": paper_id,
                    "event_type": "CLICK",
                    "event_weight": 1.0,
                    "created_at": event_time.isoformat()
                }
                supabase.table("user_events").insert(event).execute()
                user_interaction_count += 1
                total_events_created += 1

                # Randomly add high-value events (DWELL, SAVE, UPVOTE)
                rand_val = random.random()
                extra_event_type = None
                extra_weight = 1.0

                if rand_val > 0.95:
                    extra_event_type = "UPVOTE"
                    extra_weight = 5.0
                elif rand_val > 0.85:
                    extra_event_type = "SAVE"
                    extra_weight = 4.0
                    user_saves.append(paper_id)
                elif rand_val > 0.70:
                    extra_event_type = "DWELL_60S"
                    extra_weight = 3.0
                elif rand_val > 0.50:
                    extra_event_type = "DWELL_30S"
                    extra_weight = 2.0
                
                if extra_event_type:
                    extra_time = event_time + timedelta(seconds=random.randint(5, 60))
                    extra_event = {
                        "user_id": user_id,
                        "session_id": session_id,
                        "paper_id": paper_id,
                        "event_type": extra_event_type,
                        "event_weight": extra_weight,
                        "created_at": extra_time.isoformat()
                    }
                    supabase.table("user_events").insert(extra_event).execute()
                    user_interaction_count += 1
                    total_events_created += 1

            # Save session record
            if session_sequence:
                session_end = session_start + timedelta(minutes=num_papers * 10)
                session_record = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "paper_sequence": session_sequence,
                    "started_at": session_start.isoformat(),
                    "ended_at": session_end.isoformat(),
                    "is_active": False
                }
                supabase.table("user_sessions").insert(session_record).execute()

        # Update user total interactions
        supabase.table("user_profiles").update(
            {"total_interactions": user_interaction_count}
        ).eq("user_id", user_id).execute()

        # Add saves to user_saves table
        for pid in set(user_saves):
            try:
                supabase.table("user_saves").insert({
                    "user_id": user_id,
                    "paper_id": pid
                }).execute()
            except Exception:
                pass # Ignore duplicate primary key errors if they randomly saved it twice

        # Finally, compute the taste vector based on all these new interactions
        update_taste_vector(user_id)

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{NUM_USERS} users... ({total_events_created} events total)")

    print(f"\\n✅ Synthetic data generation complete!")
    print(f"Created {NUM_USERS} users and {total_events_created} interaction events.")
    print("You can now use this data to train SASRec, LightGCN, and DeepFM!")

if __name__ == "__main__":
    generate_synthetic_data()
