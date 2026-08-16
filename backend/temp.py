import os, httpx, asyncio
from dotenv import load_dotenv

async def test():
    load_dotenv()
    key=os.getenv('GEMINI_API_KEY')
    url=f'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={key}'
    payload={'requests': [
        {'model': 'models/gemini-embedding-2', 'content': {'parts': [{'text': 't1'}]}, 'outputDimensionality': 768},
        {'model': 'models/gemini-embedding-2', 'content': {'parts': [{'text': 't2'}]}, 'outputDimensionality': 768}
    ]}
    async with httpx.AsyncClient() as client:
        r=await client.post(url, json=payload)
        print(r.status_code)
        print(len(r.json().get('embeddings', [])))

if __name__ == '__main__':
    asyncio.run(test())
