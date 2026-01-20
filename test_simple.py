"""Simple test: embed a message and verify dimension."""
import asyncio
from src.models.embedder import Embedder
from src.memory.vector_store import EMBED_DIM

async def test():
    print(f"Expected EMBED_DIM: {EMBED_DIM}")
    
    embedder = Embedder()
    result = await embedder.embed("test message")
    
    print(f"Actual embedding size: {len(result)}")
    
    if len(result) == EMBED_DIM:
        print("✅ SUCCESS: Dimensions match!")
        return True
    else:
        print(f"❌ FAIL: Expected {EMBED_DIM}, got {len(result)}")
        return False

if __name__ == "__main__":
    asyncio.run(test())
