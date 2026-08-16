import os
from dotenv import load_dotenv
from openai import OpenAI

def test_openai_key():
    # Load the environment variables from .env
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY is not set in your .env file.")
        return
        
    print(f"✅ Found API Key starting with: {api_key[:8]}...")
    
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Make a tiny request to the embedding model
        print("⏳ Testing connection to OpenAI text-embedding-3-small...")
        response = client.embeddings.create(
            input="This is a test string to check if the API is working.",
            model="text-embedding-3-small"
        )
        
        # Check if we got an embedding back
        embedding = response.data[0].embedding
        print(f"✅ Success! OpenAI API is working perfectly.")
        print(f"📊 Received embedding with {len(embedding)} dimensions.")
        print("You are ready to run the full ingestion script!")
        
    except Exception as e:
        print("\n❌ Error connecting to OpenAI API!")
        print("This usually means one of two things:")
        print("1. Your API key is incorrect.")
        print("2. You have not added a credit card / billing info to your OpenAI account (you might see a 'RateLimitError' or 'Insufficient Quota' message).")
        print(f"\nDetailed Error message:\n{str(e)}")

if __name__ == "__main__":
    test_openai_key()
