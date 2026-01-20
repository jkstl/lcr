"""Test the exact same code path as the app uses."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.llm import OllamaClient

async def test_app_embedding():
    client = OllamaClient()
    
    print("Testing embedding through OllamaClient...")
    try:
        result = await client.embed("nomic-embed-text", "test message")
        print(f"✓ Success! Embedding length: {len(result)}")
        print(f"  First 5 values: {result[:5]}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_app_embedding())
