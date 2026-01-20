"""Test that project descriptions are not incorrectly graded as DISCARD."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.observer.observer import Observer, UtilityGrade
from src.models.llm import OllamaClient
from src.memory.vector_store import init_vector_store
from src.memory.graph_store import create_graph_store


async def test_project_description_grading():
    """Test that detailed project descriptions get HIGH or MEDIUM utility, not DISCARD."""
    
    # Initialize Observer
    llm = OllamaClient()
    vector_table = init_vector_store()
    graph_store = create_graph_store()
    observer = Observer(llm, vector_table, graph_store)
    
    # Test case: The exact turn that was lost in the review_chat
    test_input = """USER: I have 3 that I'm currently working on: 1. LCR - A local, privacy-first conversational AI system with persistent episodic memory. LCR runs entirely offline—no external API calls, no cloud dependencies—while maintaining rich contextual awareness across sessions through a dual-memory architecture combining semantic vector search with a structured knowledge graph, 2. Tenant Shield - Tenant Shield analyzes commercial lease agreements and operating expense invoices to automatically detect violations, calculate tenant savings, and return citation-backed explanations. It combines a deterministic audit core with RAG-powered evidence retrieval for high-precision results. and 3. Email Parser Pro - a Cloudflare Workers + static frontend app that turns messy emails, invoices, and receipts into structured JSON and CSV. It supports manual paste, batch parsing, file upload, and Gmail OAuth imports with receipt-focused extraction
ASSISTANT: Jeff, these sound like fascinating projects—especially the blend of technical depth and practical use cases! 😊"""
    
    print("Testing utility grading for detailed project description...")
    print(f"Input length: {len(test_input)} chars\n")
    
    utility = await observer._grade_utility(test_input)
    
    print(f"Result: {utility.value.upper()}")
    
    if utility == UtilityGrade.DISCARD:
        print("❌ FAIL: Project description was graded as DISCARD (this was the bug!)")
        return False
    elif utility == UtilityGrade.LOW:
        print("⚠️  WARN: Project description was graded as LOW (should be higher)")
        return False
    elif utility in [UtilityGrade.MEDIUM, UtilityGrade.HIGH]:
        print(f"✅ PASS: Project description was graded as {utility.value.upper()}")
        return True
    
    return False


async def test_simple_greeting():
    """Test that simple greetings are still graded as DISCARD."""
    
    llm = OllamaClient()
    vector_table = init_vector_store()
    graph_store = create_graph_store()
    observer = Observer(llm, vector_table, graph_store)
    
    test_input = """USER: thanks
ASSISTANT: You're welcome!"""
    
    print("\nTesting utility grading for simple greeting...")
    
    utility = await observer._grade_utility(test_input)
    
    print(f"Result: {utility.value.upper()}")
    
    if utility == UtilityGrade.DISCARD:
        print("✅ PASS: Simple greeting was correctly graded as DISCARD")
        return True
    else:
        print(f"⚠️  WARN: Simple greeting was graded as {utility.value.upper()} (should be DISCARD)")
        return False


async def main():
    print("=" * 60)
    print("UTILITY GRADING FIX VERIFICATION")
    print("=" * 60)
    print()
    
    test1 = await test_project_description_grading()
    test2 = await test_simple_greeting()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
