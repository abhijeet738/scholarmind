import os
import json
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import AsyncOpenAI
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

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key in .env")

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

BATCH_SIZE = 100  # Number of papers to process concurrently
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "scholarmind_papers_mvp.json")

async def get_embedding(text: str) -> list[float]:
    try:
        response = await aclient.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Failed to generate embedding: {e}")
        return None

async def process_batch(batch: list[dict]):
    # Generate embeddings concurrently for the batch
    tasks = []
    for paper in batch:
        # We embed a combination of title + summary for best semantic search
        text_to_embed = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('summary', '')}"
        tasks.append(get_embedding(text_to_embed))
    
    embeddings = await asyncio.gather(*tasks)
    
    # Prepare data for Supabase
    supabase_data = []
    for paper, embedding in zip(batch, embeddings):
        if embedding is None:
            continue # Skip failed embeddings
            
        # Convert date to standard ISO format string if needed
        pub_date = paper.get("published_date")
        if isinstance(pub_date, int):
            # If Kaggle exported it as Unix timestamp ms
            import datetime
            pub_date = datetime.datetime.fromtimestamp(pub_date/1000.0).strftime('%Y-%m-%d')
            
        supabase_data.append({
            "arxiv_id": str(paper.get("id")),
            "title": str(paper.get("title")),
            "authors": str(paper.get("authors")),
            "published_date": pub_date, 
            "core_category": str(paper.get("core_category")),
            "summary": str(paper.get("summary")),
            "embedding": embedding
        })
    
    # Insert into Supabase
    # Upsert allows us to run the script multiple times safely if it crashes
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
        
    print(f"Loaded {len(papers)} papers. Starting ingestion...")
    
    # Process in batches
    for i in tqdm(range(0, len(papers), BATCH_SIZE)):
        batch = papers[i:i + BATCH_SIZE]
        await process_batch(batch)
        # Small sleep to avoid hitting OpenAI API rate limits on lower tiers
        time.sleep(0.5) 
        
    print("Ingestion complete!")

if __name__ == "__main__":
    asyncio.run(main())
