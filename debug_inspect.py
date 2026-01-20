"""Debug script to inspect LCR memories for the review_chat conversation."""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory.vector_store import init_vector_store

def main():
    # Initialize vector store
    print("=== INSPECTING VECTOR STORE ===\n")
    vector_table = init_vector_store()
    count = vector_table.count_rows()
    print(f"Total memory chunks: {count}\n")
    
    # Write to file
    output_file = "memory_analysis.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== INSPECTING VECTOR STORE ===\n\n")
        f.write(f"Total memory chunks: {count}\n\n")
        
        if count > 0:
            # Get all rows
            all_rows = vector_table.to_arrow().to_pylist()
            
            # Filter for the two conversation IDs from the review_chat.txt
            conv1 = "afe867da-4e75-4ae4-9be5-76cf8dd27a2f"
            conv2 = "460035c4-0340-4db7-bb4d-31403b610f31"
            
            f.write(f"\n=== CONVERSATION 1: {conv1} ===\n\n")
            conv1_rows = [r for r in all_rows if r.get("source_conversation_id") == conv1]
            f.write(f"Found {len(conv1_rows)} memories\n\n")
            
            for idx, row in enumerate(sorted(conv1_rows, key=lambda x: x.get("turn_index", 0)), 1):
                f.write(f"--- Memory {idx} ---\n")
                f.write(f"Turn: {row.get('turn_index')}\n")
                f.write(f"Utility: {row.get('utility_score'):.2f}\n")
                f.write(f"Fact Type: {row.get('fact_type', 'N/A')}\n")
                f.write(f"Summary: {row.get('summary', 'N/A')}\n")
                f.write(f"Content: {row.get('content', '')}\n")
                f.write(f"Retrieval Queries: {row.get('retrieval_queries', [])}\n")
                f.write("\n")
            
            f.write(f"\n=== CONVERSATION 2: {conv2} ===\n\n")
            conv2_rows = [r for r in all_rows if r.get("source_conversation_id") == conv2]
            f.write(f"Found {len(conv2_rows)} memories\n\n")
            
            for idx, row in enumerate(sorted(conv2_rows, key=lambda x: x.get("turn_index", 0)), 1):
                f.write(f"--- Memory {idx} ---\n")
                f.write(f"Turn: {row.get('turn_index')}\n")
                f.write(f"Utility: {row.get('utility_score'):.2f}\n")
                f.write(f"Fact Type: {row.get('fact_type', 'N/A')}\n")
                f.write(f"Summary: {row.get('summary', 'N/A')}\n")
                f.write(f"Content: {row.get('content', '')}\n")
                f.write(f"Retrieval Queries: {row.get('retrieval_queries', [])}\n")
                f.write("\n")
    
    print(f"Analysis written to {output_file}")

if __name__ == "__main__":
    main()
