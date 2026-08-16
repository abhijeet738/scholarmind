import os
import json
import asyncio
import httpx
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm
import time

# Load environment variables
load_dotenv()

# Initialize Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing Gemini API key in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

BATCH_SIZE = 100  # Number of papers to process concurrently
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "scholarmind_papers_mvp.json")

async def process_batch(batch: list[dict]):
    # Use HTTP REST API to use the batchEmbedContents feature to avoid rate limits
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={GEMINI_API_KEY}"
    
    requests = []
    for paper in batch:
        text = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('summary', '')}"
        requests.append({
            "model": "models/gemini-embedding-2",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": 768
        })
        
    payload = {"requests": requests}
    embeddings = None
    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                resp = await http_client.post(url, json=payload)
                if resp.status_code == 429:
                    error_details = resp.json() if resp.text else "No details"
                    print(f"\nRate limit hit (429). Error details: {error_details}")
                    print(f"Sleeping for 60 seconds before retrying (Attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(60)
                    continue
                resp.raise_for_status()
                data = resp.json()
                embeddings = [emb["values"] for emb in data.get("embeddings", [])]
                break # Success, break out of retry loop
        except Exception as e:
            print(f"Failed to generate batch embeddings: {e}")
            if attempt == max_retries - 1:
                return # Give up after max retries
            await asyncio.sleep(5) # Small sleep for other random errors
            
    if embeddings is None:
        print("Failed to generate embeddings after maximum retries.")
        return
        
    # Prepare data for Supabase
    supabase_data = []
    for i, paper in enumerate(batch):
        if embeddings[i] is None:
            continue
            
        # Convert date to standard ISO format string if needed
        pub_date = paper.get("published_date")
        if isinstance(pub_date, int):
            import datetime
            pub_date = datetime.datetime.fromtimestamp(pub_date/1000.0).strftime('%Y-%m-%d')
            
        supabase_data.append({
            "arxiv_id": str(paper.get("id")),
            "title": str(paper.get("title")),
            "authors": str(paper.get("authors")),
            "published_date": pub_date, 
            "core_category": str(paper.get("core_category")),
            "summary": str(paper.get("summary")),
            "embedding": embeddings[i]
        })
    
    # Insert into Supabase
    if supabase_data:
        try:
            supabase.table("papers").upsert(supabase_data, on_conflict="arxiv_id").execute()
        except Exception as e:
            print(f"Error inserting batch into Supabase: {e}")

async def main():
    print("Loading JSON data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}")
        
    with open(DATA_PATH, "r") as f:
        papers = json.load(f)
        
    print(f"Loaded {len(papers)} papers from JSON.")
    
    # ---- RESUME LOGIC ----
    print("Checking Supabase for already processed papers (so we don't start over)...")
    existing_ids = set()
    start = 0
    step = 1000
    
    while True:
        res = supabase.table("papers").select("arxiv_id").range(start, start + step - 1).execute()
        if not res.data:
            break
        for row in res.data:
            existing_ids.add(row["arxiv_id"])
        start += step
        
    print(f"Found {len(existing_ids)} papers already in the database.")
    
    # Filter out papers that are already in the database
    papers_to_process = [p for p in papers if str(p.get("id")) not in existing_ids]
    
    print(f"Remaining papers to process: {len(papers_to_process)}")
    
    if not papers_to_process:
        print("All papers are already ingested!")
        return
        
    print("Starting ingestion...")
    
    # Process in batches
    for i in tqdm(range(0, len(papers_to_process), BATCH_SIZE)):
        batch = papers_to_process[i:i + BATCH_SIZE]
        await process_batch(batch)
        
    print("Ingestion complete!")

if __name__ == "__main__":
    asyncio.run(main())
