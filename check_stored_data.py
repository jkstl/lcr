"""Quick check of stored data without full dependencies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

print("=" * 80)
print("STORED DATA INSPECTION")
print("=" * 80)

# Check LanceDB
try:
    from src.memory.vector_store import init_vector_store
    from src.config import settings

    print("\n[LanceDB Vector Store]")
    table = init_vector_store()
    count = table.count_rows()
    print(f"Total memory chunks: {count}")

    if count > 0:
        # Get all rows
        rows = table.to_arrow().to_pylist()
        print(f"\nLatest {min(count, 5)} memories:")
        for i, row in enumerate(sorted(rows, key=lambda x: x.get("created_at", ""), reverse=True)[:5]):
            print(f"\n  [{i+1}] Turn {row.get('turn_index', '?')} - Utility: {row.get('utility_score', 0):.2f}")
            print(f"      Summary: {row.get('summary', 'N/A')[:80]}")
            print(f"      Queries: {row.get('retrieval_queries', [])}")

except Exception as e:
    print(f"[LanceDB] Error: {e}")

# Check FalkorDB
try:
    from src.memory.graph_store import create_graph_store

    print("\n" + "=" * 80)
    print("[FalkorDB Graph Store]")

    graph_store = create_graph_store()
    store_type = type(graph_store).__name__
    print(f"Store type: {store_type}")

    if hasattr(graph_store, 'entities'):
        # InMemoryGraphStore
        print(f"\nEntities: {len(graph_store.entities)}")
        for name, data in list(graph_store.entities.items())[:10]:
            attrs = data.get('attributes', {})
            attr_str = f" {attrs}" if attrs else ""
            print(f"  - {name} ({data.get('type')}){attr_str}")

        print(f"\nRelationships: {len(graph_store.relationships)}")
        for rel in graph_store.relationships[:15]:
            valid = "✓" if rel.metadata.get("still_valid", True) else "✗"
            print(f"  {valid} [{rel.predicate}] {rel.subject} → {rel.object}")
    else:
        # FalkorGraphStore - try to query
        import asyncio

        async def check_graph():
            entities = ["User", "Giana", "Mom", "Justine", "Philadelphia", "West Boylston", "Oregon Diner"]
            rels = await graph_store.search_relationships(entities, limit=50)

            print(f"\nRelationships found: {len(rels)}")
            for rel in rels[:15]:
                print(f"  [{rel.predicate}] {rel.subject} → {rel.object}")

        asyncio.run(check_graph())

except Exception as e:
    print(f"[FalkorDB] Error: {e}")

print("\n" + "=" * 80)
