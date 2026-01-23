#!/usr/bin/env python3
"""
Format training data for LiquidAI LFM2.5-1.2B-Instruct using proper ChatML format.

Key fixes from failed Qwen3 attempt:
1. Use tokenizer.apply_chat_template() instead of manual formatting
2. Proper message structure with role/content
3. Test formatting before fine-tuning
"""

import json
import random
from pathlib import Path
from transformers import AutoTokenizer


MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"


def format_utility_example(tokenizer, example: dict) -> dict:
    """Format utility grading example using proper ChatML."""
    
    # Input format: {user_message, assistant_message, label}
    conversation = f"USER: {example['user_message']}\nASSISTANT: {example['assistant_message']}"
    
    # CRITICAL: Add system message to specify the task
    messages = [
        {
            "role": "system",
            "content": "You are a utility grading assistant. Rate conversations as DISCARD (no information), STORE (preferences/opinions), or IMPORTANT (identity/relationships/life facts). Respond with exactly one word."
        },
        {
            "role": "user",
            "content": conversation
        },
        {
            "role": "assistant",
            "content": example['label']  # DISCARD, STORE, or IMPORTANT
        }
    ]
    
    # Use tokenizer to format correctly (ChatML)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False  # We have the assistant response
    )
    
    return {"text": text}


def format_extraction_example(tokenizer, example: dict) -> dict:
    """Format entity extraction example using proper ChatML."""
    
    # Input format: {user_message, assistant_message, entities, relationships}
    conversation = f"USER: {example['user_message']}\nASSISTANT: {example['assistant_message']}"
    
    # Build JSON output
    output = {
        "entities": example['entities'],
        "relationships": example['relationships']
    }
    
    # CRITICAL: Add system message to specify the task
    messages = [
        {
            "role": "system",
            "content": "You are an entity extraction assistant. Extract entities and relationships from conversations in JSON format with entities (name, type, attributes) and relationships (subject, predicate, object, temporal)."
        },
        {
            "role": "user", 
            "content": conversation
        },
        {
            "role": "assistant",
            "content": json.dumps(output, ensure_ascii=False)
        }
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def main():
    """Format and combine training data."""
    
    print("=" * 70)
    print("Formatting Training Data for LFM2.5-1.2B-Instruct")
    print("=" * 70)
    print()
    
    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✓ Tokenizer loaded")
    print()
    
    # Source data from previous attempt (good data, wrong formatting)
    source_dir = Path("fine_tuning/qwen3_0.6b_v2")
    output_dir = Path("fine_tuning/lfm_1.2b_v1")
    
    # Load existing data
    print("Loading source data...")
    with open(source_dir / "utility_grading.jsonl") as f:
        utility_data = [json.loads(line) for line in f]
    
    with open(source_dir / "entity_extraction.jsonl") as f:
        extraction_data = [json.loads(line) for line in f]
    
    print(f"  Utility: {len(utility_data)} examples")
    print(f"  Extraction: {len(extraction_data)} examples")
    print()
    
    # Format examples using proper method
    print("Formatting with apply_chat_template...")
    formatted_data = []
    
    for ex in utility_data:
        formatted_data.append(format_utility_example(tokenizer, ex))
    
    for ex in extraction_data:
        formatted_data.append(format_extraction_example(tokenizer, ex))
    
    print(f"  Total formatted: {len(formatted_data)}")
    print()
    
    # Show example
    print("Example formatted text (first 500 chars):")
    print("-" * 70)
    print(formatted_data[0]['text'][:500])
    print("-" * 70)
    print()
    
    # Shuffle
    random.seed(42)
    random.shuffle(formatted_data)
    
    # Split 80/20
    split_idx = int(len(formatted_data) * 0.8)
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]
    
    print(f"Split:")
    print(f"  Train: {len(train_data)} examples (80%)")
    print(f"  Val: {len(val_data)} examples (20%)")
    print()
    
    # Save
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    
    print("Saving...")
    with open(train_file, 'w') as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    with open(val_file, 'w') as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"  ✓ {train_file}")
    print(f"  ✓ {val_file}")
    print()
    
    print("=" * 70)
    print("Formatting Complete!")
    print("=" * 70)
    print()
    print("Next: Test base model before fine-tuning")
    print()


if __name__ == "__main__":
    main()
