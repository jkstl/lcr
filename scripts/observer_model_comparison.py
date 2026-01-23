#!/usr/bin/env python3
"""
Observer Model Comparison Test: qwen3:0.6b vs qwen3:1.7b

This script performs comprehensive evaluation of different observer models:
- Entity/relationship extraction accuracy
- JSON formatting reliability
- Utility grading consistency
- Performance/speed benchmarks
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.llm import OllamaClient, parse_json_response
from src.observer.prompts import EXTRACTION_PROMPT, UTILITY_PROMPT


# Test cases covering various scenarios
TEST_CASES = [
    {
        "name": "Work Relationship",
        "user_message": "I work at Acme Corp as a software engineer on the ML team",
        "assistant_response": "That's interesting! What kind of ML projects do you work on?",
        "expected": {
            "entities": ["User", "Acme Corp", "software engineer", "ML team"],
            "relationships": ["WORKS_AT"],
            "utility": "HIGH",
        },
    },
    {
        "name": "Family Relationship",
        "user_message": "My sister Justine lives in Worcester Massachusetts with her husband Tom",
        "assistant_response": "That's nice! Do you visit them often?",
        "expected": {
            "entities": ["Justine", "Worcester", "Massachusetts", "Tom"],
            "relationships": ["SIBLING_OF", "LIVES_IN", "MARRIED_TO"],
            "utility": "HIGH",
        },
    },
    {
        "name": "Temporal State Change",
        "user_message": "My mom and sister left yesterday and went back home to Massachusetts",
        "assistant_response": "Safe travels to them!",
        "expected": {
            "entities": ["Mom", "sister", "Massachusetts"],
            "relationships": ["RETURNED_TO", "LIVES_IN"],
            "utility": "MEDIUM",
        },
    },
    {
        "name": "Technology/Device",
        "user_message": "I bought a Dell Latitude 5520 laptop with 32GB RAM for my home server project",
        "assistant_response": "Nice specs! That should handle server tasks well.",
        "expected": {
            "entities": ["User", "Dell Latitude 5520", "32GB RAM"],
            "relationships": ["OWNS", "USES_FOR"],
            "utility": "MEDIUM",
        },
    },
    {
        "name": "Preference",
        "user_message": "I prefer Python over JavaScript because it's more readable",
        "assistant_response": "Python is definitely popular for readability.",
        "expected": {
            "entities": ["User", "Python", "JavaScript"],
            "relationships": ["PREFERS", "DISLIKES"],
            "utility": "MEDIUM",
        },
    },
    {
        "name": "Small Talk (Should be DISCARD)",
        "user_message": "Thanks!",
        "assistant_response": "You're welcome!",
        "expected": {
            "entities": [],
            "relationships": [],
            "utility": "DISCARD",
        },
    },
    {
        "name": "Complex Event",
        "user_message": "After the car repair, Mercedes and I realized the brakes still squeal, so the mechanic scheduled a follow-up on Monday with Laguna Auto",
        "assistant_response": "Laguna Auto on Monday is on the calendar for follow-up brake work.",
        "expected": {
            "entities": ["Mercedes", "mechanic", "Laguna Auto", "Monday"],
            "relationships": ["OWNS", "SCHEDULED_FOR"],
            "utility": "HIGH",
        },
    },
    {
        "name": "Emotional State",
        "user_message": "I've been thinking about Giana a lot lately and it's making me sad",
        "assistant_response": "I'm sorry to hear that. Would you like to talk about it?",
        "expected": {
            "entities": ["User", "Giana"],
            "relationships": ["KNOWS", "FEELS_SAD_ABOUT"],
            "utility": "HIGH",
        },
    },
]


class ModelTestResults:
    """Track results for a single model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.test_results = []
        self.total_time = 0.0
        self.json_errors = 0
        self.llm_errors = 0
        self.successful_tests = 0

    def add_result(self, test_name: str, result: Dict[str, Any]):
        """Add a test result."""
        self.test_results.append({"test_name": test_name, **result})
        self.total_time += result.get("time", 0)

        if result.get("json_valid"):
            self.successful_tests += 1
        elif result.get("json_error"):
            self.json_errors += 1
        elif result.get("llm_error"):
            self.llm_errors += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_tests = len(self.test_results)
        avg_time = self.total_time / total_tests if total_tests > 0 else 0

        return {
            "model": self.model_name,
            "total_tests": total_tests,
            "successful": self.successful_tests,
            "json_errors": self.json_errors,
            "llm_errors": self.llm_errors,
            "success_rate": f"{(self.successful_tests / total_tests * 100):.1f}%"
            if total_tests > 0
            else "0%",
            "total_time": f"{self.total_time:.2f}s",
            "avg_time_per_test": f"{avg_time:.2f}s",
        }


async def test_extraction(
    llm: OllamaClient, model: str, test_case: Dict[str, Any]
) -> Dict[str, Any]:
    """Test entity/relationship extraction on a test case."""
    text = f"User: {test_case['user_message']}\nAssistant: {test_case['assistant_response']}"
    prompt = EXTRACTION_PROMPT.format(text=text)

    start_time = time.time()

    try:
        response = await llm.generate(model, prompt)
        elapsed_time = time.time() - start_time

        # Try to parse JSON
        try:
            data = parse_json_response(response)

            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            fact_type = data.get("fact_type", "N/A")

            return {
                "json_valid": True,
                "time": elapsed_time,
                "fact_type": fact_type,
                "entities_count": len(entities),
                "relationships_count": len(relationships),
                "entities": [e.get("name") for e in entities],
                "relationships": [r.get("predicate") for r in relationships],
                "raw_response": response,
            }

        except (json.JSONDecodeError, ValueError) as e:
            return {
                "json_error": True,
                "time": elapsed_time,
                "error": str(e),
                "raw_response": response[:500],  # First 500 chars
            }

    except Exception as e:
        return {
            "llm_error": True,
            "time": time.time() - start_time,
            "error": str(e),
        }


async def test_utility_grading(
    llm: OllamaClient, model: str, test_case: Dict[str, Any]
) -> Dict[str, Any]:
    """Test utility grading."""
    text = f"User: {test_case['user_message']}\nAssistant: {test_case['assistant_response']}"
    prompt = UTILITY_PROMPT.format(text=text)

    start_time = time.time()

    try:
        response = await llm.generate(model, prompt)
        elapsed_time = time.time() - start_time

        grade = response.strip().upper()
        expected = test_case["expected"]["utility"]
        matches = grade == expected

        return {
            "success": True,
            "time": elapsed_time,
            "grade": grade,
            "expected": expected,
            "matches": matches,
        }

    except Exception as e:
        return {
            "error": True,
            "time": time.time() - start_time,
            "error_msg": str(e),
        }


async def run_model_tests(model: str) -> ModelTestResults:
    """Run all tests for a single model."""
    print(f"\n{'=' * 80}")
    print(f"Testing Model: {model}")
    print(f"{'=' * 80}\n")

    llm = OllamaClient()
    results = ModelTestResults(model)

    for i, test_case in enumerate(TEST_CASES):
        print(f"Test {i + 1}/{len(TEST_CASES)}: {test_case['name']}")

        # Test extraction
        extraction_result = await test_extraction(llm, model, test_case)

        # Test utility grading
        utility_result = await test_utility_grading(llm, model, test_case)

        # Combine results
        combined_result = {
            "extraction": extraction_result,
            "utility": utility_result,
            "time": extraction_result.get("time", 0) + utility_result.get("time", 0),
            "json_valid": extraction_result.get("json_valid", False),
            "json_error": extraction_result.get("json_error", False),
            "llm_error": extraction_result.get("llm_error", False),
        }

        results.add_result(test_case["name"], combined_result)

        # Display result summary
        if extraction_result.get("json_valid"):
            print(
                f"  ✓ Entities: {extraction_result['entities_count']}, "
                f"Relationships: {extraction_result['relationships_count']}, "
                f"Utility: {utility_result.get('grade', 'N/A')} "
                f"({'✓' if utility_result.get('matches') else '✗'}), "
                f"Time: {combined_result['time']:.2f}s"
            )
        else:
            error_type = (
                "JSON Error" if extraction_result.get("json_error") else "LLM Error"
            )
            print(f"  ✗ {error_type}: {extraction_result.get('error', 'Unknown')}")

        # Brief pause between tests
        await asyncio.sleep(0.2)

    return results


def print_comparison_table(results_list: List[ModelTestResults]):
    """Print a comparison table of results."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Model summaries
    for results in results_list:
        summary = results.get_summary()
        print(f"\n{summary['model']}:")
        print(f"  Tests:        {summary['total_tests']}")
        print(f"  Successful:   {summary['successful']}")
        print(f"  JSON Errors:  {summary['json_errors']}")
        print(f"  LLM Errors:   {summary['llm_errors']}")
        print(f"  Success Rate: {summary['success_rate']}")
        print(f"  Total Time:   {summary['total_time']}")
        print(f"  Avg Time:     {summary['avg_time_per_test']}")

    # Per-test accuracy comparison
    print("\n" + "=" * 80)
    print("PER-TEST ACCURACY COMPARISON")
    print("=" * 80)

    test_names = [test["name"] for test in TEST_CASES]

    # Header
    print(f"\n{'Test Name':<30} ", end="")
    for results in results_list:
        print(f"{results.model_name:<20}", end="")
    print()
    print("-" * 80)

    # Each test
    for i, test_name in enumerate(test_names):
        print(f"{test_name:<30} ", end="")
        for results in results_list:
            test_result = results.test_results[i]
            if test_result["extraction"].get("json_valid"):
                ent_count = test_result["extraction"]["entities_count"]
                rel_count = test_result["extraction"]["relationships_count"]
                utility_match = "✓" if test_result["utility"].get("matches") else "✗"
                print(f"E:{ent_count} R:{rel_count} U:{utility_match}  ", end="")
            else:
                print(f"{'FAILED':<20}", end="")
        print()

    # Performance comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON (Time per test)")
    print("=" * 80)

    print(f"\n{'Test Name':<30} ", end="")
    for results in results_list:
        print(f"{results.model_name:<15}", end="")
    print()
    print("-" * 80)

    for i, test_name in enumerate(test_names):
        print(f"{test_name:<30} ", end="")
        for results in results_list:
            test_result = results.test_results[i]
            time_val = test_result.get("time", 0)
            print(f"{time_val:.2f}s{'':<10}", end="")
        print()


def print_detailed_results(results_list: List[ModelTestResults]):
    """Print detailed results for each model and test."""
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for results in results_list:
        print(f"\n{results.model_name}")
        print("-" * 80)

        for test_result in results.test_results:
            test_name = test_result["test_name"]
            extraction = test_result["extraction"]
            utility = test_result["utility"]

            print(f"\n  {test_name}:")

            if extraction.get("json_valid"):
                print(f"    Fact Type: {extraction.get('fact_type')}")
                print(f"    Entities: {extraction.get('entities')}")
                print(f"    Relationships: {extraction.get('relationships')}")
                print(
                    f"    Utility: {utility.get('grade')} (expected: {utility.get('expected')}) "
                    f"{'✓' if utility.get('matches') else '✗'}"
                )
                print(f"    Time: {test_result.get('time', 0):.2f}s")
            else:
                error_type = "JSON Error" if extraction.get("json_error") else "LLM Error"
                print(f"    {error_type}: {extraction.get('error')}")
                print(f"    Raw response: {extraction.get('raw_response', 'N/A')[:200]}...")


async def main():
    """Main test execution."""
    print("=" * 80)
    print("OBSERVER MODEL BASELINE TEST")
    print("qwen3:1.7b (before improvements)")
    print("=" * 80)
    print(f"\nTest Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of Test Cases: {len(TEST_CASES)}")
    print()

    # Test only 1.7b for baseline
    models = ["qwen3:1.7b"]

    # Run tests for each model
    all_results = []
    for model in models:
        results = await run_model_tests(model)
        all_results.append(results)

    # Print comparisons
    print_comparison_table(all_results)
    print_detailed_results(all_results)

    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Determine winner based on success rate
    success_rates = [r.successful_tests / len(TEST_CASES) for r in all_results]
    avg_times = [r.total_time / len(TEST_CASES) for r in all_results]

    print("\nBased on the test results:")
    for i, results in enumerate(all_results):
        print(f"\n{results.model_name}:")
        print(f"  - Success Rate: {success_rates[i] * 100:.1f}%")
        print(f"  - Average Time: {avg_times[i]:.2f}s per test")
        print(f"  - JSON Errors: {results.json_errors}")

    # Determine recommendation
    if success_rates[0] > success_rates[1]:
        winner = all_results[0].model_name
    elif success_rates[1] > success_rates[0]:
        winner = all_results[1].model_name
    else:
        # Tie, choose faster one
        winner = all_results[0].model_name if avg_times[0] < avg_times[1] else all_results[1].model_name

    print(f"\n✓ Recommended Model: {winner}")

    if all_results[0].json_errors > 0 or all_results[1].json_errors > 0:
        print("\n⚠ WARNING: Some models produced JSON formatting errors.")
        print("  This may cause reliability issues in production.")


if __name__ == "__main__":
    asyncio.run(main())
