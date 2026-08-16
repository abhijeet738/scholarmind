import os
import pickle
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm
import datetime

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_to_supabase(file_path):
    print(f"Loading {file_path}...")
    with open(file_path, "rb") as f:
        papers = pickle.load(f)
        
    print(f"Loaded {len(papers)} papers with embeddings.")
    
    CHUNK_SIZE = 100
    
    for i in tqdm(range(0, len(papers), CHUNK_SIZE)):
        batch_papers = papers[i:i + CHUNK_SIZE]
        supabase_data = []
        
        for paper in batch_papers:
            pub_date = paper.get("published_date")
            if isinstance(pub_date, int):
                pub_date = datetime.datetime.fromtimestamp(pub_date/1000.0).strftime('%Y-%m-%d')
                
            supabase_data.append({
                "arxiv_id": str(paper.get("id")),
                "title": str(paper.get("title")),
                "authors": str(paper.get("authors")),
                "published_date": pub_date,
                "core_category": str(paper.get("core_category")),
                "summary": str(paper.get("summary")),
                "embedding": paper.get("embedding")
            })
            
        max_retries = 3
        for attempt in range(max_retries):
            try:
                supabase.table("papers").upsert(supabase_data, on_conflict="arxiv_id").execute()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Timeout/Error on chunk {i}. Retrying in 2s...")
                    time.sleep(2)
                else:
                    print(f"Failed to upload chunk {i}: {e}")

if __name__ == "__main__":
    FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "scholarmind_with_embeddings.pkl")
    upload_to_supabase(FILE_PATH)
