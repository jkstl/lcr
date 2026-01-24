"""
Compare extraction accuracy between LFM2.5 fine-tuned model and NuExtract.

Tests on actual conversation data from the user to evaluate:
- Entity attribution accuracy
- Relationship extraction accuracy
- Hallucination prevention
- Complex multi-entity scenarios
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformers_client import TransformersClient
from src.models.nuextract_client import NuExtractClient
from src.observer.prompts import EXTRACTION_SYSTEM_MESSAGE
from src.observer.nuextract_templates import get_extraction_template, get_extraction_examples
from src.models.llm import parse_json_response
from src.config import settings


# Test cases from user's actual conversation
TEST_CASES = [
    {
        "name": "User identity and work",
        "input": "USER: My name is Jeffrey Kistler, most people call me Jeff. I live in Philadelphia, PA. I work for Airgas in Radnor, PA.",
        "expected": {
            "entities": ["User/Jeffrey Kistler", "Jeff", "Philadelphia", "Airgas"],
            "key_relationships": [
                "User HAS_NAME Jeffrey Kistler",
                "User GOES_BY Jeff",
                "User LIVES_IN Philadelphia",
                "User WORKS_AT Airgas",
            ]
        }
    },
    {
        "name": "Named family member with location",
        "input": "USER: My sister Justine lives in Worcester Massachusetts",
        "expected": {
            "entities": ["User", "Justine", "Worcester"],
            "key_relationships": [
                "User SIBLING_OF Justine",
                "Justine LIVES_IN Worcester",
            ]
        }
    },
    {
        "name": "Complex family info (multiple entities)",
        "input": "USER: My brother Sam lives in Falmouth, MA - he's 35 years old. Justine is 24 and lives with my mother in West Boylston, MA.",
        "expected": {
            "entities": ["User", "Sam", "Justine", "Mom/mother", "Falmouth", "West Boylston"],
            "key_relationships": [
                "User SIBLING_OF Sam",
                "Sam LIVES_IN Falmouth",
                "Justine LIVES_IN West Boylston",
            ]
        }
    },
    {
        "name": "User project (WORKS_ON)",
        "input": "USER: Im working on basketcall, an app for NBA statistics",
        "expected": {
            "entities": ["User", "basketcall", "NBA"],
            "key_relationships": [
                "User WORKS_ON basketcall",
            ]
        }
    },
    {
        "name": "Family siblings (attribution test)",
        "input": "USER: I have a brother Sam, a half sister Justine, and a half brother Michael.",
        "expected": {
            "entities": ["User", "Sam", "Justine", "Michael"],
            "key_relationships": [
                "User SIBLING_OF Sam",
                "User SIBLING_OF Justine",
                "User SIBLING_OF Michael",
            ]
        }
    },
]


async def test_lfm_model(test_case):
    """Test extraction with current LFM2.5 fine-tuned model."""
    model_path = settings.observer_model.replace('transformers:', '')
    client = TransformersClient(model_path)

    # Use training format (just the turn)
    response = await client.generate(
        settings.observer_model,
        test_case["input"],
        system=EXTRACTION_SYSTEM_MESSAGE
    )

    try:
        parsed = parse_json_response(response)
        return parsed
    except Exception as e:
        return {"error": str(e), "raw": response}


async def test_nuextract_model(test_case):
    """Test extraction with NuExtract 2.0-4B."""
    client = NuExtractClient('numind/NuExtract-2.0-4B')
    template = get_extraction_template()
    examples = get_extraction_examples()

    result = await client.extract(
        document=test_case["input"],
        template=template,
        examples=examples
    )

    try:
        parsed = json.loads(result)
        return parsed
    except Exception as e:
        return {"error": str(e), "raw": result}


def analyze_results(model_name, result, expected):
    """Analyze extraction results against expected output."""
    print(f"\n{model_name} Results:")
    print("-" * 70)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return 0, 0

    # Extract entity names
    entity_names = [e["name"] for e in result.get("entities", [])]
    relationships = result.get("relationships", [])

    print(f"Entities extracted: {entity_names}")
    print(f"Relationships: {len(relationships)}")

    # Check for common errors
    errors = []

    # Check entity attribution errors
    for rel in relationships:
        subj = rel.get("subject", "")
        pred = rel.get("predicate", "")
        obj = rel.get("object", "")

        # Error: Named person doing user's actions
        if subj in ["Justine", "Sam", "Michael"] and pred in ["WORKS_AT", "WORKS_ON", "LIVES_IN"]:
            # Check if this is about the family member or wrongly attributed
            if "Airgas" in obj or "basketcall" in obj.lower():
                errors.append(f"❌ Wrong attribution: {subj} {pred} {obj}")

    # Check for hallucinated entities (from training examples)
    hallucinated = []
    suspicious_names = ["Giana", "Python", "JavaScript", "Massachusetts"]
    for entity in entity_names:
        if entity in suspicious_names and entity not in expected.get("entities", []):
            hallucinated.append(entity)

    if hallucinated:
        errors.append(f"❌ Hallucinated entities: {hallucinated}")

    # Print relationships
    print("\nKey Relationships:")
    for rel in relationships[:10]:
        marker = "✓" if not any(rel["subject"] in e for e in errors) else "❌"
        print(f"  {marker} {rel['subject']} --[{rel['predicate']}]--> {rel['object']}")

    if errors:
        print("\nErrors detected:")
        for error in errors:
            print(f"  {error}")

    # Simple scoring: count errors
    score = max(0, 10 - len(errors) - len(hallucinated))
    return score, len(errors)


async def main():
    """Run comparison tests."""
    print("=" * 70)
    print("EXTRACTION MODEL COMPARISON")
    print("=" * 70)
    print()
    print("Comparing LFM2.5-1.2B (fine-tuned) vs NuExtract 2.0-4B")
    print()

    # Load models once
    print("Loading models...")
    lfm_model_path = settings.observer_model.replace('transformers:', '')
    lfm_client = TransformersClient(lfm_model_path)
    nuextract_client = NuExtractClient('numind/NuExtract-2.0-4B')
    print("✓ Models loaded")
    print()

    total_lfm_score = 0
    total_nuextract_score = 0
    total_lfm_errors = 0
    total_nuextract_errors = 0

    for i, test_case in enumerate(TEST_CASES, 1):
        print("=" * 70)
        print(f"TEST {i}: {test_case['name']}")
        print("=" * 70)
        print(f"Input: {test_case['input']}")
        print()

        # Test both models
        lfm_result = await test_lfm_model(test_case)
        nuextract_result = await test_nuextract_model(test_case)

        # Analyze results
        lfm_score, lfm_errors = analyze_results("LFM2.5-1.2B Fine-tuned", lfm_result, test_case["expected"])
        nuextract_score, nuextract_errors = analyze_results("NuExtract 2.0-4B", nuextract_result, test_case["expected"])

        total_lfm_score += lfm_score
        total_nuextract_score += nuextract_score
        total_lfm_errors += lfm_errors
        total_nuextract_errors += nuextract_errors

        print()

    # Final summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"LFM2.5-1.2B Fine-tuned:")
    print(f"  Total Score: {total_lfm_score}/{len(TEST_CASES) * 10}")
    print(f"  Total Errors: {total_lfm_errors}")
    print()
    print(f"NuExtract 2.0-4B:")
    print(f"  Total Score: {total_nuextract_score}/{len(TEST_CASES) * 10}")
    print(f"  Total Errors: {total_nuextract_errors}")
    print()

    if total_nuextract_score > total_lfm_score:
        improvement = ((total_nuextract_score - total_lfm_score) / (len(TEST_CASES) * 10)) * 100
        print(f"✓ NuExtract is {improvement:.1f}% better")
    elif total_lfm_score > total_nuextract_score:
        decline = ((total_lfm_score - total_nuextract_score) / (len(TEST_CASES) * 10)) * 100
        print(f"❌ NuExtract is {decline:.1f}% worse")
    else:
        print("= Models perform equally")


if __name__ == "__main__":
    asyncio.run(main())
