#!/usr/bin/env python3
"""
Training data generator using GPT-4o for knowledge distillation.

Generates labeled conversation examples for fine-tuning qwen3:0.6b on utility grading.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import random
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.training.conversation_templates import get_all_templates, generate_from_template


class GPT4oClient:
    """Client for GPT-4o API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    async def generate(self, prompt: str, system: str = None) -> str:
        """Generate completion from GPT-4o."""
        import aiohttp
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 0.3,  # Low temperature for consistency
            "max_tokens": 500,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"GPT-4o API error: {error_text}")
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]


# Utility grading prompt for GPT-4o labeler
LABELING_PROMPT = """Rate this conversation turn using the 3-level system.

CONVERSATION:
USER: {user_message}
ASSISTANT: {assistant_message}

SYSTEM (3 levels):
- DISCARD: Pure acknowledgments with zero content (e.g., "thanks", "ok")
- STORE: Any factual information worth remembering (preferences, opinions, general facts)
- IMPORTANT: Critical life facts - user's identity, relationships, possessions, major events

INSTRUCTIONS:
1. Decide: DISCARD, STORE, or IMPORTANT
2. Provide brief reasoning (1 sentence)
3. Give confidence score (0.0-1.0)

Respond in JSON format:
{{
    "label": "DISCARD|STORE|IMPORTANT",
    "reasoning": "one sentence explanation",
    "confidence": 0.95
}}

Respond with ONLY the JSON, no other text."""


async def label_conversation(client: GPT4oClient, conversation: Dict[str, str], max_retries: int = 5) -> Dict[str, Any]:
    """Use GPT-4o to label a conversation with retry logic."""
    prompt = LABELING_PROMPT.format(
        user_message=conversation['user_message'],
        assistant_message=conversation['assistant_message']
    )
    
    for attempt in range(max_retries):
        try:
            response = await client.generate(prompt)
            
            # Parse JSON response
            # Handle potential markdown wrapping
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response)
            
            return {
                'user_message': conversation['user_message'],
                'assistant_message': conversation['assistant_message'],
                'label': data['label'].upper(),
                'reasoning': data['reasoning'],
                'confidence': data['confidence'],
                'expected_label': conversation.get('expected_label'),
                'category': conversation.get('category'),
            }
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a rate limit error
            if "rate_limit_exceeded" in error_msg and attempt < max_retries - 1:
                # Extract wait time from error message if available  
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                await asyncio.sleep(wait_time)
                continue
            elif attempt < max_retries - 1:
                # Other errors, brief retry
                await asyncio.sleep(1)
                continue
            else:
                # Final attempt failed
                if "rate_limit_exceeded" not in error_msg:
                    print(f"  ✗ Labeling failed: {error_msg[:100]}")
                return None
    
    return None


def validate_example(example: Dict[str, Any]) -> bool:
    """Validate a labeled example."""
    if not example:
        return False
    
    # Check required fields
    if not all(key in example for key in ['user_message', 'assistant_message', 'label', 'confidence']):
        return False
    
    # Check label is valid
    if example['label'] not in ['DISCARD', 'STORE', 'IMPORTANT']:
        return False
    
    # Check confidence threshold
    if example['confidence'] < 0.7:
        return False
    
    # Check message length
    if len(example['user_message']) < 5 or len(example['user_message']) > 500:
        return False
    
    return True


async def generate_training_data(
    api_key: str,
    target_count: int = 1000,
    output_file: str = "data/training/training_data.jsonl"
):
    """Generate training data using GPT-4o."""
    
    print("=" * 60)
    print("Training Data Generation with GPT-4o")
    print("=" * 60)
    print(f"Target: {target_count} examples")
    print(f"Output: {output_file}")
    print()
    
    # Initialize client
    client = GPT4oClient(api_key)
    
    # Get template categories
    templates = get_all_templates()
    
    # Generate conversations from templates
    print("Generating conversations from templates...")
    conversations = []
    for category_name, (template_list, count) in templates.items():
        print(f"  {category_name}: {count} examples")
        for _ in range(count):
            template, expected = random.choice(template_list)
            conv = generate_from_template(template, expected, category_name)
            conversations.append(conv)
    
    print(f"\nGenerated {len(conversations)} conversations")
    print()
    
    # Label with GPT-4o
    print("Labeling with GPT-4o...")
    print("(This may take 5-10 minutes)")
    print()
    
    labeled_examples = []
    failed_count = 0
    
    # Process in batches to avoid rate limits
    batch_size = 3  # Smaller batches to avoid rate limits
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        
        # Label batch concurrently
        tasks = [label_conversation(client, conv) for conv in batch]
        results = await asyncio.gather(*tasks)
        
        # Filter and validate
        for result in results:
            if validate_example(result):
                labeled_examples.append(result)
            else:
                failed_count += 1
        
        # Progress update
        progress = min(i + batch_size, len(conversations))
        print(f"  Progress: {progress}/{len(conversations)} ({len(labeled_examples)} valid, {failed_count} failed)")
        
        # Delay to respect rate limits
        if i + batch_size < len(conversations):
            await asyncio.sleep(2)  # Longer delay between batches
    
    print()
    print(f"Labeling complete: {len(labeled_examples)} valid examples")
    print()
    
    # Check distribution
    distribution = {'DISCARD': 0, 'STORE': 0, 'IMPORTANT': 0}
    for ex in labeled_examples:
        distribution[ex['label']] += 1
    
    print("Label Distribution:")
    for label, count in distribution.items():
        pct = (count / len(labeled_examples)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    print()
    
    # Check agreement with expected labels
    agreement = sum(1 for ex in labeled_examples if ex['label'] == ex['expected_label'])
    agreement_pct = (agreement / len(labeled_examples)) * 100
    print(f"Agreement with templates: {agreement}/{len(labeled_examples)} ({agreement_pct:.1f}%)")
    print()
    
    # Export to JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create train/val split
    random.shuffle(labeled_examples)
    split_idx = int(len(labeled_examples) * 0.8)
    train_examples = labeled_examples[:split_idx]
    val_examples = labeled_examples[split_idx:]
    
    # Export training set
    train_file = output_path.with_stem(output_path.stem + "_train")
    with open(train_file, 'w') as f:
        for ex in train_examples:
            # Format for fine-tuning
            prompt = f"Rate this conversation:\nUSER: {ex['user_message']}\nASSISTANT: {ex['assistant_message']}"
            response = ex['label']
            
            entry = {
                "prompt": prompt,
                "response": response,
                "metadata": {
                    "reasoning": ex['reasoning'],
                    "confidence": ex['confidence'],
                    "category": ex['category'],
                }
            }
            f.write(json.dumps(entry) + '\n')
    
    # Export validation set
    val_file = output_path.with_stem(output_path.stem + "_val")
    with open(val_file, 'w') as f:
        for ex in val_examples:
            prompt = f"Rate this conversation:\nUSER: {ex['user_message']}\nASSISTANT: {ex['assistant_message']}"
            response = ex['label']
            
            entry = {
                "prompt": prompt,
                "response": response,
                "metadata": {
                    "reasoning": ex['reasoning'],
                    "confidence": ex['confidence'],
                    "category": ex['category'],
                }
            }
            f.write(json.dumps(entry) + '\n')
    
    print(f"✓ Training data exported:")
    print(f"  Train: {train_file} ({len(train_examples)} examples)")
    print(f"  Val:   {val_file} ({len(val_examples)} examples)")
    print()
    
    # Generate statistics report
    stats = {
        "generated_at": datetime.now().isoformat(),
        "total_examples": len(labeled_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "distribution": distribution,
        "agreement_rate": agreement_pct,
        "confidence_avg": sum(ex['confidence'] for ex in labeled_examples) / len(labeled_examples),
    }
    
    stats_file = output_path.with_suffix('.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Statistics: {stats_file}")
    print()
    print("=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Review a sample of examples to verify quality")
    print("2. Use training data to fine-tune qwen3:0.6b")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data for utility grading")
    parser.add_argument("--api-key", required=True, help="GPT-4o API key")
    parser.add_argument("--count", type=int, default=1000, help="Target number of examples")
    parser.add_argument("--output", default="data/training/training_data.jsonl", help="Output file")
    
    args = parser.parse_args()
    
    asyncio.run(generate_training_data(
        api_key=args.api_key,
        target_count=args.count,
        output_file=args.output
    ))
