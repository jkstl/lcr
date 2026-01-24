#!/usr/bin/env python3
"""
Clean hallucinated data from the graph store.

Removes specific bad relationships that were stored from assistant hallucinations.
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.graph_store import create_graph_store
from src.config import settings


async def main():
    """Remove hallucinated data from graph store."""
    print("Connecting to graph store...")
    graph_store = create_graph_store()

    # List of hallucinated relationships to remove
    # Format: (subject, predicate, object)
    hallucinated_facts = [
        ("User", "SHOPPING", "tomatoes, chickpeas, olive oil"),
    ]

    print(f"\nSearching for {len(hallucinated_facts)} hallucinated fact(s)...\n")

    for subject, predicate, obj in hallucinated_facts:
        print(f"Looking for: {subject} {predicate} {obj}")

        # Query for this specific relationship
        relationships = await graph_store.query(subject, predicate)

        # Find exact match
        matches = [r for r in relationships if r.object == obj]

        if matches:
            for rel in matches:
                print(f"  ✓ Found relationship ID {rel.id}")
                print(f"    Created: {rel.created_at}")
                print(f"    Confidence: {getattr(rel, 'confidence', 'N/A')}")

                # Delete the relationship
                if hasattr(graph_store, 'graph') and graph_store.graph is not None:
                    # FalkorDB backend
                    try:
                        query = f"MATCH ()-[r]->() WHERE ID(r) = {rel.id} DELETE r"
                        graph_store.graph.query(query)
                        print(f"    ✗ Deleted from FalkorDB")
                    except Exception as e:
                        print(f"    ✗ Error deleting: {e}")
                elif hasattr(graph_store, 'relationships'):
                    # In-memory backend
                    if rel.id in graph_store.relationships:
                        del graph_store.relationships[rel.id]
                        print(f"    ✗ Deleted from in-memory store")
                else:
                    print(f"    ⚠ Unknown graph store type, cannot delete")
        else:
            print(f"  ⚠ Not found (may have been already deleted)")

    print("\n✓ Cleanup complete!")
    print("\nRun `python scripts/inspect_memory.py` to verify.")


if __name__ == "__main__":
    asyncio.run(main())
