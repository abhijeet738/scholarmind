import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

def test_gemini_key():
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY is not set in your .env file.")
        return
        
    print(f"✅ Found API Key starting with: {api_key[:8]}...")
    
    try:
        # Initialize the new Google GenAI client
        client = genai.Client(api_key=api_key)
        
        print("⏳ Testing connection to Gemini gemini-embedding-2...")
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents="This is a test string to check if the API is working.",
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        
        embedding = response.embeddings[0].values
        print(f"✅ Success! Gemini API is working perfectly.")
        print(f"📊 Received embedding with {len(embedding)} dimensions.")
        print("You are ready to run the Gemini ingestion script!")
        
    except Exception as e:
        print("\n❌ Error connecting to Gemini API!")
        print(f"\nDetailed Error message:\n{str(e)}")

if __name__ == "__main__":
    test_gemini_key()
