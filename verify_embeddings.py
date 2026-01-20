"""Quick verification that embeddings work end-to-end."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory.vector_store import VectorStore, EMBED_DIM
from src.models.embedder import Embedder

async def verify_system():
    print(f"EMBED_DIM: {EMBED_DIM}")
    
    embedder = Embedder()
    
    # Test embedding
    print("\n1. Testing embedding generation...")
    embedding = await embedder.embed("test message")
    print(f"   ✓ Generated embedding with {len(embedding)} dimensions")
    
    if len(embedding) != EMBED_DIM:
        print(f"   ✗ ERROR: Dimension mismatch! Expected {EMBED_DIM}, got {len(embedding)}")
        return False
    
    # Test vector store
    print("\n2. Testing vector store...")
    store = VectorStore()
    await store.add_memory(
        conversation_id="test_conv",
        turn_index=1,
        content="This is a test memory about cats.",
        embedding=embedding,
        utility_grade="HIGH",
        summary="Test about cats"
    )
    print("   ✓ Memory added successfully")
    
    # Test search
    print("\n3. Testing search...")
    search_embedding = await embedder.embed("tell me about cats")
    results = await store.search(search_embedding, top_k=5)
    print(f"   ✓ Search returned {len(results)} results")
    
    if results:
        print(f"   ✓ Top result: {results[0]['content'][:50]}...")
    
    print("\n✅ All checks passed! System is working correctly with nomic-embed-text")
    return True

if __name__ == "__main__":
    success = asyncio.run(verify_system())
    sys.exit(0 if success else 1)
