"""Run a suite of real-world prompts through the observer to validate extraction."""

import asyncio
import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.llm import OllamaClient
from src.memory.vector_store import init_vector_store
from src.memory.graph_store import create_graph_store
from src.observer.observer import Observer


STRESS_PROMPTS = [
    (
        "My neighbor asked me to take her dog Max to the vet next Friday morning.",
        "Sure, I can pick up Max and bring him to the vet. What time is the appointment and do you need me to walk him afterwards?"
    ),
    (
        "I finally got the promotion at CoSphere! I start the new role on March 1st and my team is called Nimbus.",
        "Congrats on the promotion! Nimbus sounds like a cool team name, and March 1st gives you just over two weeks to wrap things up."
    ),
    (
        "We are planning a weekend getaway to Asheville for my partner's birthday. She's a chef so she wants to cook in a rental with great natural light.",
        "That sounds amazing—Asheville has lovely homes. I'll note the birthday trip, the chef preferences, and the need for good lighting."
    ),
    (
        "After the car repair, Mercedes and I realized the brakes still squeal, so the mechanic scheduled a follow-up on Monday with Laguna Auto.",
        "Laguna Auto on Monday is on the calendar for follow-up brake work."
    ),
    (
        "My doctor just recommended I switch to a Mediterranean diet, so I'm shopping for tomatoes, chickpeas, and extra-virgin olive oil.",
        "Noted: doctor’s advice to go Mediterranean plus the shopping list."
    ),
]


async def main():
    print("Initializing stores...")
    vector_table = init_vector_store()
    graph_store = create_graph_store()
    print(f"✓ Using graph store: {type(graph_store).__name__}\n")

    observer = Observer(OllamaClient(), vector_table, graph_store)

    for idx, (user_text, assistant_text) in enumerate(STRESS_PROMPTS):
        output = await observer.process_turn(
            user_message=user_text,
            assistant_response=assistant_text,
            conversation_id="stress-test",
            turn_index=idx,
        )
        print(f"\n--- Prompt {idx + 1} ---")
        print("User:", user_text)
        print("Observer utility:", output.utility_grade.value)
        print("Summary:", output.summary)
        print("Entities:", output.entities)
        print("Relationships:", output.relationships)
        print("Queries:", output.retrieval_queries)
        print("Contradictions:", output.contradictions)

    # Show summary of persisted data
    print("\n" + "="*80)
    print("SUMMARY OF PERSISTED DATA")
    print("="*80)

    if hasattr(graph_store, 'entities'):
        print(f"\nTotal entities stored: {len(graph_store.entities)}")
        for name, data in list(graph_store.entities.items())[:10]:
            print(f"  • {name} ({data.get('type', 'Unknown')})")

        print(f"\nTotal relationships stored: {len(graph_store.relationships)}")
        for rel in graph_store.relationships[:15]:
            print(f"  [{rel.predicate}] {rel.subject} → {rel.object}")
    else:
        print("\n✓ Data persisted to FalkorDB")
        print("  (Use FalkorDB client to query persisted relationships)")


if __name__ == "__main__":
    asyncio.run(main())
