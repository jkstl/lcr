#!/usr/bin/env python
"""Test script for fact_type classification changes.

This script tests the Observer's ability to classify facts as:
- core: Work schedules, family relationships, tech inventory
- episodic: One-time events, meetings, trips
- preference: Opinions, likes/dislikes

Run with: python scripts/test_fact_type.py
"""
import asyncio
import sys
sys.path.insert(0, ".")

from src.config import settings
from src.models.llm import OllamaClient


# Test prompts designed to elicit specific fact_type classifications
TEST_CASES = [
    {
        "name": "Core Fact - Work Schedule",
        "user": "I work at TechCorp from 9 to 5 on weekdays. My manager is Sarah.",
        "assistant": "Got it! That's a standard schedule.",
        "expected_fact_type": "core",
        "reason": "Work schedules and manager relationships are persistent core facts"
    },
    {
        "name": "Core Fact - Family/Tech Inventory",
        "user": "I have a 65-inch Samsung TV in my living room and my sister lives in Boston.",
        "assistant": "Nice setup!",
        "expected_fact_type": "core",
        "reason": "Tech inventory and family relationships are core facts"
    },
    {
        "name": "Episodic - One-time Event",
        "user": "I'm meeting Jake for coffee tomorrow at Starbucks at 3pm.",
        "assistant": "Sounds like fun!",
        "expected_fact_type": "episodic",
        "reason": "One-time meetings are episodic events"
    },
    {
        "name": "Preference - Opinion",
        "user": "I really prefer Python over JavaScript for backend work. JavaScript is too messy.",
        "assistant": "That's a popular opinion among developers!",
        "expected_fact_type": "preference",
        "reason": "Opinions and preferences about technology"
    },
]


async def run_extraction_test(llm: OllamaClient, test_case: dict) -> dict:
    """Run a single extraction test and return results."""
    from src.observer.prompts import EXTRACTION_PROMPT
    from json import loads, JSONDecodeError
    
    combined = f"USER: {test_case['user']}\nASSISTANT: {test_case['assistant']}"
    prompt = EXTRACTION_PROMPT.format(text=combined)
    
    print(f"\n{'='*60}")
    print(f"TEST: {test_case['name']}")
    print(f"{'='*60}")
    print(f"Input: {test_case['user']}")
    print(f"Expected: {test_case['expected_fact_type']}")
    print(f"Reason: {test_case['reason']}")
    print("-" * 40)
    
    try:
        response = await llm.generate(settings.observer_model, prompt)
        print(f"Raw response:\n{response[:500]}...")
        
        # Parse JSON
        data = loads(response)
        actual_fact_type = data.get("fact_type", "MISSING")
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        
        passed = actual_fact_type == test_case["expected_fact_type"]
        
        print(f"\nExtracted fact_type: {actual_fact_type}")
        print(f"Entities: {len(entities)}")
        print(f"Relationships: {len(relationships)}")
        print(f"RESULT: {'✅ PASS' if passed else '❌ FAIL'}")
        
        return {
            "name": test_case["name"],
            "expected": test_case["expected_fact_type"],
            "actual": actual_fact_type,
            "passed": passed,
            "entities": entities,
            "relationships": relationships,
        }
        
    except JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        return {
            "name": test_case["name"],
            "expected": test_case["expected_fact_type"],
            "actual": "PARSE_ERROR",
            "passed": False,
            "error": str(e),
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "name": test_case["name"],
            "expected": test_case["expected_fact_type"],
            "actual": "ERROR",
            "passed": False,
            "error": str(e),
        }


async def main():
    print("=" * 60)
    print("FACT TYPE CLASSIFICATION TEST")
    print(f"Observer Model: {settings.observer_model}")
    print("=" * 60)
    
    llm = OllamaClient()
    
    results = []
    for test_case in TEST_CASES:
        result = await run_extraction_test(llm, test_case)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"{status} {r['name']}: expected={r['expected']}, actual={r['actual']}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed < total:
        print("\n⚠️ Some tests failed. Review the prompt or model output.")
    else:
        print("\n✅ All tests passed!")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
