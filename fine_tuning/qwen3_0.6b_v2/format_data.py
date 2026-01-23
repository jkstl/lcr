#!/usr/bin/env python3
"""
Format training data for Qwen3 instruction fine-tuning.

Combines utility grading and entity extraction into proper format.
"""

import json
import random
from pathlib import Path


def format_utility_example(example: dict) -> dict:
    """Format utility grading example for instruction tuning."""
    conversation = f"USER: {example['user_message']}\nASSISTANT: {example['assistant_message']}"
    
    instruction = """Rate the memory-worthiness of this conversation using a 3-level system:
- DISCARD: Pure acknowledgments with zero information
- STORE: Factual information, preferences, opinions worth remembering  
- IMPORTANT: Critical life facts (job, relationships, location, possessions, major events)

Respond with exactly one word: DISCARD, STORE, or IMPORTANT"""
    
    return {
        "instruction": instruction,
        "input": conversation,
        "output": example['label']
    }


def format_extraction_example(example: dict) -> dict:
    """Format entity extraction example for instruction tuning."""
    conversation = f"USER: {example['user_message']}\nASSISTANT: {example['assistant_message']}"
    
    instruction = """Extract entities and relationships from this conversation.
Output valid JSON with:
- entities: list of {name, type, attributes}
- relationships: list of {subject, predicate, object, temporal}

Use "User" for the user, actual names for others. Include temporal states (ongoing/completed/future)."""
    
    output = {
        "entities": example['entities'],
        "relationships": example['relationships']
    }
    
    return {
        "instruction": instruction,
        "input": conversation,
        "output": json.dumps(output, ensure_ascii=False)
    }


def main():
    """Combine and format training data."""
    
    input_dir = Path("fine_tuning/qwen3_0.6b_v2")
    output_dir = Path("fine_tuning/qwen3_0.6b_v2")
    
    print("=" * 70)
    print("Formatting Training Data for Fine-tuning")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    with open(input_dir / "utility_grading.jsonl") as f:
        utility_data = [json.loads(line) for line in f]
    
    with open(input_dir / "entity_extraction.jsonl") as f:
        extraction_data = [json.loads(line) for line in f]
    
    print(f"  Utility examples: {len(utility_data)}")
    print(f"  Extraction examples: {len(extraction_data)}")
    
    # Format examples
    print("\nFormatting examples...")
    all_examples = []
    
    for example in utility_data:
        all_examples.append(format_utility_example(example))
    
    for example in extraction_data:
        all_examples.append(format_extraction_example(example))
    
    print(f"  Total formatted: {len(all_examples)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_examples)
    
    # Train/val split (80/20)
    split_idx = int(len(all_examples) * 0.8)
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val: {len(val_data)} examples")
    
    # Save
    print("\nSaving formatted data...")
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    
    with open(train_file, 'w') as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    with open(val_file, 'w') as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"  ✓ Train: {train_file}")
    print(f"  ✓ Val: {val_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Formatting Complete!")
    print("=" * 70)
    print(f"\nTotal examples: {len(all_examples)}")
    print(f"  Training: {len(train_data)} (80%)")
    print(f"  Validation: {len(val_data)} (20%)")
    print()
    print("Ready for fine-tuning!")
    print()


if __name__ == "__main__":
    main()
