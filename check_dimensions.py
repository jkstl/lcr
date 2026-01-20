"""Check actual embedding dimensions from nomic-embed-text."""
import httpx
import asyncio

async def check_dimensions():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": "test"}
        )
        data = response.json()
        embedding = data.get("embedding", [])
        print(f"nomic-embed-text produces {len(embedding)} dimensions")
        
        # Also check what the model info says
        tags_response = await client.get("http://localhost:11434/api/tags")
        models = tags_response.json().get("models", [])
        for model in models:
            if "nomic" in model.get("name", ""):
                print(f"Model: {model.get('name')}")
                print(f"Size: {model.get('size')} bytes")

if __name__ == "__main__":
    asyncio.run(check_dimensions())
