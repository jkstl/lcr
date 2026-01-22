#!/usr/bin/env python3
"""Interactive model testing tool for LCR."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.llm import OllamaClient
from src.models.embedder import Embedder
from src.observer.prompts import EXTRACTION_PROMPT, UTILITY_PROMPT, SEMANTIC_CONTRADICTION_PROMPT


def get_available_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')[1:]
        return [line.split()[0] for line in lines if line.strip()]
    except:
        return []


async def test_observer_extraction(model: str, test_case: dict):
    """Test observer extraction on a test case."""
    llm = OllamaClient()

    print(f"\n{'='*80}")
    print(f"Testing Observer Model: {model}")
    print(f"{'='*80}")
    print(f"\nTest: {test_case['name']}")
    print(f"Input: {test_case['text'][:100]}...")
    print()

    # Test extraction
    prompt = EXTRACTION_PROMPT.format(text=test_case['text'])

    try:
        response = await llm.generate(model, prompt)

        # Try to parse JSON
        try:
            data = json.loads(response)

            print("✓ Valid JSON Response")
            print(f"\nFact Type: {data.get('fact_type', 'N/A')}")

            entities = data.get('entities', [])
            print(f"\nEntities ({len(entities)}):")
            for e in entities[:5]:  # Show first 5
                print(f"  • {e.get('name')} ({e.get('type')})")

            relationships = data.get('relationships', [])
            print(f"\nRelationships ({len(relationships)}):")
            for r in relationships[:5]:  # Show first 5
                print(f"  • {r.get('subject')} {r.get('predicate')} {r.get('object')}")

            return {"success": True, "entities": len(entities), "relationships": len(relationships)}

        except json.JSONDecodeError as e:
            print(f"✗ JSON Parse Error: {e}")
            print(f"\nRaw response (first 300 chars):")
            print(response[:300])
            return {"success": False, "error": "JSON parse error"}

    except Exception as e:
        print(f"✗ LLM Error: {e}")
        return {"success": False, "error": str(e)}


async def test_utility_grading(model: str, test_case: dict):
    """Test utility grading."""
    llm = OllamaClient()

    print(f"\n{'='*80}")
    print(f"Testing Utility Grading: {model}")
    print(f"{'='*80}")
    print(f"\nInput: {test_case['text'][:100]}...")
    print(f"Expected: {test_case.get('expected_utility', 'N/A')}")
    print()

    prompt = UTILITY_PROMPT.format(text=test_case['text'])

    try:
        response = await llm.generate(model, prompt)
        grade = response.strip().upper()

        expected = test_case.get('expected_utility', '').upper()
        correct = grade == expected if expected else None

        status = "✓" if correct else "✗" if correct is not None else "?"
        print(f"{status} Result: {grade}")

        return {"success": True, "grade": grade, "correct": correct}

    except Exception as e:
        print(f"✗ Error: {e}")
        return {"success": False, "error": str(e)}


async def test_main_llm(model: str, prompt: str):
    """Test main LLM with a simple prompt."""
    llm = OllamaClient()

    print(f"\n{'='*80}")
    print(f"Testing Main LLM: {model}")
    print(f"{'='*80}")
    print(f"\nPrompt: {prompt}")
    print()

    try:
        response = await llm.generate(model, prompt)
        print("Response:")
        print(response)
        print()
        return {"success": True}

    except Exception as e:
        print(f"✗ Error: {e}")
        return {"success": False, "error": str(e)}


async def test_embedder(model: str, text: str):
    """Test embedding model."""
    print(f"\n{'='*80}")
    print(f"Testing Embedding Model: {model}")
    print(f"{'='*80}")
    print(f"\nText: {text[:100]}...")
    print()

    try:
        embedder = Embedder(model_name=model)
        embedding = await embedder.embed(text)

        print(f"✓ Generated embedding")
        print(f"  Dimensions: {len(embedding)}")
        print(f"  Sample values: {embedding[:5]}")

        return {"success": True, "dimensions": len(embedding)}

    except Exception as e:
        print(f"✗ Error: {e}")
        return {"success": False, "error": str(e)}


async def run_interactive_tests():
    """Run interactive model testing."""
    print("=" * 80)
    print("LCR MODEL TESTING TOOL")
    print("=" * 80)
    print()

    models = get_available_models()

    if not models:
        print("No Ollama models found!")
        return

    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print()

    # Predefined test cases
    observer_tests = [
        {
            "name": "Work Relationship",
            "text": "User: I work at Acme Corp as a software engineer\nAssistant: Interesting!",
            "expected_entities": ["User", "Acme Corp"],
            "expected_relationships": ["WORKS_AT"],
        },
        {
            "name": "Temporal State",
            "text": "User: My mom and sister Justine left yesterday and went back home to Massachusetts\nAssistant: Safe travels!",
            "expected_entities": ["Mom", "Justine", "Massachusetts"],
        },
        {
            "name": "Family Relationship",
            "text": "User: My sister Justine lives in Worcester Massachusetts\nAssistant: That's nice!",
            "expected_entities": ["Justine", "Worcester Massachusetts"],
        },
    ]

    utility_tests = [
        {
            "text": "User: I've been thinking about Giana a lot lately and it's making me sad\nAssistant: I'm sorry to hear that.",
            "expected_utility": "HIGH",
        },
        {
            "text": "User: thanks\nAssistant: You're welcome!",
            "expected_utility": "DISCARD",
        },
        {
            "text": "User: I prefer Python over JavaScript\nAssistant: That's common for backend work",
            "expected_utility": "MEDIUM",
        },
    ]

    print("\nTest Categories:")
    print("  1. Observer Extraction (entity & relationship extraction)")
    print("  2. Utility Grading (memory worthiness)")
    print("  3. Main LLM (conversation quality)")
    print("  4. Embedder (vector generation)")
    print("  5. Quick Comparison (run all tests on multiple models)")
    print()

    choice = input("Select test category (1-5): ").strip()

    if choice == "1":
        # Observer extraction test
        model_idx = input(f"Select observer model (1-{len(models)}): ").strip()
        try:
            model = models[int(model_idx) - 1]
        except:
            print("Invalid selection")
            return

        for test_case in observer_tests:
            await test_observer_extraction(model, test_case)
            print()

    elif choice == "2":
        # Utility grading test
        model_idx = input(f"Select observer model (1-{len(models)}): ").strip()
        try:
            model = models[int(model_idx) - 1]
        except:
            print("Invalid selection")
            return

        for test_case in utility_tests:
            await test_utility_grading(model, test_case)
            print()

    elif choice == "3":
        # Main LLM test
        model_idx = input(f"Select main model (1-{len(models)}): ").strip()
        try:
            model = models[int(model_idx) - 1]
        except:
            print("Invalid selection")
            return

        prompt = input("\nEnter test prompt: ").strip()
        await test_main_llm(model, prompt)

    elif choice == "4":
        # Embedder test
        embed_models = [m for m in models if 'embed' in m.lower() or 'nomic' in m.lower()]

        if not embed_models:
            print("No embedding models found!")
            return

        print("\nEmbedding models:")
        for i, model in enumerate(embed_models, 1):
            print(f"  {i}. {model}")

        model_idx = input(f"Select embedding model (1-{len(embed_models)}): ").strip()
        try:
            model = embed_models[int(model_idx) - 1]
        except:
            print("Invalid selection")
            return

        text = "I work at Acme Corp as a software engineer"
        await test_embedder(model, text)

    elif choice == "5":
        # Quick comparison
        print("\nSelect models to compare (comma-separated, e.g., 1,3,5):")
        indices = input("Models: ").strip()

        try:
            selected_indices = [int(i.strip()) - 1 for i in indices.split(',')]
            selected_models = [models[i] for i in selected_indices]
        except:
            print("Invalid selection")
            return

        print(f"\n{'='*80}")
        print(f"QUICK COMPARISON: {', '.join(selected_models)}")
        print(f"{'='*80}")

        # Run one test case per model
        test_case = observer_tests[0]  # Work relationship test

        results = {}
        for model in selected_models:
            result = await test_observer_extraction(model, test_case)
            results[model] = result
            await asyncio.sleep(0.5)  # Brief pause between tests

        # Summary
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        for model, result in results.items():
            if result.get("success"):
                print(f"\n{model}:")
                print(f"  Entities: {result.get('entities', 0)}")
                print(f"  Relationships: {result.get('relationships', 0)}")
            else:
                print(f"\n{model}: FAILED - {result.get('error')}")

    else:
        print("Invalid choice")


async def main():
    await run_interactive_tests()


if __name__ == "__main__":
    asyncio.run(main())
