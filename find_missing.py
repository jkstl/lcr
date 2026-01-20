"""Find the missing memory for turn 5 (the big project description)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory.vector_store import init_vector_store

def main():
    vector_table = init_vector_store()
    all_rows = vector_table.to_arrow().to_pylist()
    
    conv1 = "afe867da-4e75-4ae4-9be5-76cf8dd27a2f"
    conv1_rows = [r for r in all_rows if r.get("source_conversation_id") == conv1]
    
    print(f"Total memories from conversation 1: {len(conv1_rows)}")
    print(f"\nAll turn indices found: {sorted([r.get('turn_index') for r in conv1_rows])}")
    print("\nAll turns:")
    
    for row in sorted(conv1_rows, key=lambda x: x.get("turn_index", 0)):
        print(f"\nTurn {row.get('turn_index')}: Utility={row.get('utility_score')}, Fact Type={row.get('fact_type')}")
        print(f"Content preview: {row.get('content', '')[:150]}...")
        
if __name__ == "__main__":
    main()
