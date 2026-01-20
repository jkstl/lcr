"""Test Ollama embeddings endpoint directly."""
import httpx
import asyncio

async def test_embeddings():
    async with httpx.AsyncClient() as client:
        # Test 1: API tags (should work)
        print("Testing /api/tags...")
        try:
            response = await client.get("http://localhost:11434/api/tags")
            print(f"✓ Status: {response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Test 2: Embeddings endpoint
        print("\nTesting /api/embeddings...")
        try:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": "test"}
            )
            print(f"✓ Status: {response.status_code}")
            data = response.json()
            print(f"✓ Response keys: {list(data.keys())}")
        except httpx.HTTPStatusError as e:
            print(f"✗ HTTP Error: {e.response.status_code} {e.response.text}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Test 3: Alternative endpoint /api/embed
        print("\nTesting /api/embed...")
        try:
            response = await client.post(
                "http://localhost:11434/api/embed",
                json={"model": "nomic-embed-text", "input": "test"}
            )
            print(f"✓ Status: {response.status_code}")
            data = response.json()
            print(f"✓ Response keys: {list(data.keys())}")
        except httpx.HTTPStatusError as e:
            print(f"✗ HTTP Error: {e.response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_embeddings())
