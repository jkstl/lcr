#!/usr/bin/env python3
"""
Comprehensive training data generator v2.

Generates training data for BOTH:
1. Utility grading (DISCARD/STORE/IMPORTANT)
2. Entity/relationship extraction

Uses gpt-oss:20b locally or GPT-4o API as fallback.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.training.conversation_templates_v2 import get_all_templates, fill_template
from src.models.llm import OllamaClient


class LabelingClient:
    """Client for labeling with gpt-oss:20b or GPT-4o."""
    
    def __init__(self, use_local: bool = True):
        self.use_local = use_local
        if use_local:
            self.client = OllamaClient()
            self.model = "gpt-oss:20b"
        else:
            import aiohttp
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.base_url = "https://api.openai.com/v1/chat/completions"
    
    async def generate(self, prompt: str, system: str = None) -> str:
        """Generate completion."""
        if self.use_local:
            # Use Ollama with gpt-oss:20b
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            return await self.client.generate(self.model, full_prompt)
        else:
            # Use GPT-4o API
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
                "temperature": 0.3,
                "max_tokens": 800,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error: {error_text}")
                    
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]


# ============================================================================
# PROMPTS
# ============================================================================

UTILITY_GRADING_PROMPT = """You are labeling conversations for a MEMORY SYSTEM.
Rate what the USER shared (ignore assistant's response quality).

CONVERSATION:
USER: {user_message}
ASSISTANT: {assistant_message}

**ONLY evaluate the USER'S message:**

DISCARD - User said nothing worth remembering:
- Pure greetings/acknowledgments: "thanks", "ok", "hi"
- No facts, no preferences, nothing concrete

STORE - User shared preferences, opinions, or general facts:
- "I prefer X over Y"
- "I like/enjoy/think..."
- Non-critical information

IMPORTANT - User revealed core identity or life facts:
- Job, company, role
- Relationships (family, friends)
- Where they live
- Major life events (moving, job change)
- Possessions they own

The assistant's response quality is IRRELEVANT. Focus ONLY on what the USER shared.

Respond in JSON format:
{{
    "label": "DISCARD|STORE|IMPORTANT",
    "reasoning": "one sentence explanation focusing on USER's message",
    "confidence": 0.95
}}

Respond with ONLY the JSON, no other text."""

ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from this conversation.

CONVERSATION:
USER: {user_message}
ASSISTANT: {assistant_message}

Extract:
1. All entities (people, places, organizations, technologies, concepts)
2. All relationships between entities
3. Temporal information (dates, state changes)
4. Attributes for each entity

Respond in JSON format:
{{
    "entities": [
        {{"name": "entity_name", "type": "Person|Place|Organization|Technology|Concept", "attributes": {{"key": "value"}}}}
    ],
    "relationships": [
        {{"subject": "entity1", "predicate": "RELATIONSHIP_TYPE", "object": "entity2", "temporal": "ongoing|completed|future"}}
    ]
}}

Important:
- Use "User" for the user
- Use actual names for other people mentioned
- Include temporal states: VISITING vs RETURNED_HOME, WORKS_AT vs RESIGNED_FROM
- Capture all attributes (age, role, location, specs, etc.)

Respond with ONLY the JSON, no other text."""


async def label_utility(client: LabelingClient, user_msg: str, assistant_msg: str, max_retries: int = 3) -> Dict[str, Any]:
    """Label utility grade with retry."""
    prompt = UTILITY_GRADING_PROMPT.format(
        user_message=user_msg,
        assistant_message=assistant_msg
    )
    
    for attempt in range(max_retries):
        try:
            response = await client.generate(prompt)
            
            # Parse JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response)
            
            return {
                'user_message': user_msg,
                'assistant_message': assistant_msg,
                'label': data['label'].upper(),
                'reasoning': data['reasoning'],
                'confidence': data['confidence'],
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                print(f"  ✗ Utility labeling failed: {e}")
                return None
    
    return None


async def extract_entities(client: LabelingClient, user_msg: str, assistant_msg: str, max_retries: int = 3) -> Dict[str, Any]:
    """Extract entities and relationships with retry."""
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        user_message=user_msg,
        assistant_message=assistant_msg
    )
    
    for attempt in range(max_retries):
        try:
            response = await client.generate(prompt)
            
            # Parse JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response)
            
            return {
                'user_message': user_msg,
                'assistant_message': assistant_msg,
                'entities': data.get('entities', []),
                'relationships': data.get('relationships', []),
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                print(f"  ✗ Entity extraction failed: {e}")
                return None
    
    return None


async def generate_training_data(
    use_local: bool = True,
    target_count: int = 2000,
    output_dir: str = "fine_tuning/qwen3_0.6b_v2"
):
    """Generate comprehensive training data."""
    
    print("=" * 70)
    print("Comprehensive Training Data Generation v2")
    print("=" * 70)
    print(f"Target: {target_count} examples")
    print(f"Labeler: {'gpt-oss:20b (local)' if use_local else 'GPT-4o (API)'}")
    print(f"Output: {output_dir}")
    print()
    
    # Initialize client
    client = LabelingClient(use_local=use_local)
    
    # Get templates
    templates = get_all_templates()
    
    # Generate conversations
    print("Generating conversations from templates...")
    conversations = []
    for category_name, (template_list, count) in templates.items():
        print(f"  {category_name}: {count} examples")
        for _ in range(count):
            if isinstance(template_list[0], tuple) and len(template_list[0]) >= 3:
                template_data = random.choice(template_list)
                if len(template_data) == 3:
                    template, expected, cat = template_data
                    metadata = {}
                else:
                    template, expected, cat, metadata = template_data
                
                user_msg = fill_template(template, metadata)
                assistant_msg = random.choice([
                    "That's interesting!",
                    "Got it.",
                    "Noted!",
                    "I see.",
                    "Thanks for sharing!",
                    "Cool!",
                ])
                
                conversations.append({
                    'user_message': user_msg,
                    'assistant_message': assistant_msg,
                    'expected_label': expected if expected in ['DISCARD', 'STORE', 'IMPORTANT'] else None,
                    'category': category_name,
                    'metadata': metadata,
                })
    
    print(f"\nGenerated {len(conversations)} conversations")
    print()
    
    # Label conversations
    print("Labeling conversations...")
    print("(This may take 10-20 minutes)")
    print()
    
    utility_examples = []
    extraction_examples = []
    failed_count = 0
    
    # Process in small batches
    batch_size = 5 if use_local else 3
    
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        
        # Label batch
        tasks = []
        for conv in batch:
            # Get both utility and extraction labels
            tasks.append(label_utility(client, conv['user_message'], conv['assistant_message']))
            # Only extract entities for STORE and IMPORTANT
            if conv['expected_label'] in ['STORE', 'IMPORTANT']:
                tasks.append(extract_entities(client, conv['user_message'], conv['assistant_message']))
        
        results = await asyncio.gather(*tasks)
        
        # Process results
        for j, result in enumerate(results):
            if result:
                if 'label' in result:
                    # Utility grading result
                    if result['confidence'] >= 0.7:
                        utility_examples.append(result)
                elif 'entities' in result:
                    # Extraction result
                    if result['entities'] or result['relationships']:
                        extraction_examples.append(result)
            else:
                failed_count += 1
        
        # Progress
        progress = min(i + batch_size, len(conversations))
        print(f"  Progress: {progress}/{len(conversations)} ({len(utility_examples)} utility, {len(extraction_examples)} extraction, {failed_count} failed)")
        
        # Delay
        if i + batch_size < len(conversations):
            await asyncio.sleep(1 if use_local else 2)
    
    print()
    print(f"Labeling complete!")
    print(f"  Utility examples: {len(utility_examples)}")
    print(f"  Extraction examples: {len(extraction_examples)}")
    print()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save utility grading data
    utility_file = output_path / "utility_grading.jsonl"
    with open(utility_file, 'w') as f:
        for ex in utility_examples:
            f.write(json.dumps(ex) + '\n')
    
    # Save extraction data
    extraction_file = output_path / "entity_extraction.jsonl"
    with open(extraction_file, 'w') as f:
        for ex in extraction_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"✓ Data saved:")
    print(f"  Utility: {utility_file}")
    print(f"  Extraction: {extraction_file}")
    print()
    print("=" * 70)
    print("Next: Combine and format for fine-tuning")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive training data")
    parser.add_argument("--use-api", action="store_true", help="Use GPT-4o API instead of local gpt-oss:20b")
    parser.add_argument("--count", type=int, default=2000, help="Target number of examples")
    parser.add_argument("--api-key", help="OpenAI API key (if using API)")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    asyncio.run(generate_training_data(
        use_local=not args.use_api,
        target_count=args.count,
    ))
