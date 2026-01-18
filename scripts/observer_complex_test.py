"""Run a complex prompt sequence, including contradictions, to stress the observer."""

import asyncio
import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.llm import OllamaClient
from src.memory.vector_store import init_vector_store
from src.memory.graph_store import create_graph_store
from src.observer.observer import Observer


COMPLEX_PROMPTS = [
    (
        "I am heading to Denver next week to meet my manager; the meeting is at noon on Thursday, please remind me to prepare the report.",
        "Denver can be a great place for that; I'll remind you with the report prep steps and any logistics."
    ),
    (
        "Actually the meeting got pulled forward to Wednesday morning, so I canceled the airfare and booked a later flight.",
        "Understood—Wednesday morning is the new slot, and the travel change is noted along with the canceled airfare."
    ),
    (
        "By the way, I told my contractor to start the patio expansion after I get back, but we just agreed to shift it to next month because of the rain.",
        "I'll keep the patio expansion scheduled for next month, consistent with the rain delay."
    ),
    (
        "My partner Taylor prefers the patio to be wood, not stone; we had discussed stone earlier, so make sure the materials list reflects that.",
        "Wood material for the patio is the latest preference and will override the earlier stone idea."
    ),
    (
        "Reminder: the birthday dinner for Taylor happens on Saturday at the waterfront restaurant, but cancel if the weather forecast shows heavy rain.",
        "Saturday at the waterfront is highlighted, and I’ll monitor the forecast for rain."
    ),
    (
        "Actually the dinner moved to Sunday because the rain stayed away all day, and we're adding Samantha from the design team as a guest.",
        "Sunday is now confirmed, and Samantha from design is also invited."
    ),
    (
        "I also need to confirm whether my sister Justine's birthday this weekend gets combined with the work dinner—she's 24 and will be coming in from Hartford's new train line.",
        "Noted that Justine (24, coming from Hartford's new train line) might join the celebration if it aligns with the work dinner."
    ),
]


async def run_observer():
    print("Initializing stores...")
    vector_table = init_vector_store()
    graph_store = create_graph_store()
    print(f"✓ Using graph store: {type(graph_store).__name__}\n")

    observer = Observer(OllamaClient(), vector_table, graph_store)

    for idx, (user, assistant) in enumerate(COMPLEX_PROMPTS, 1):
        output = await observer.process_turn(
            user_message=user,
            assistant_response=assistant,
            conversation_id="complex-test",
            turn_index=idx,
        )
        print(f"\n=== Turn {idx} ===")
        print("Utility:", output.utility_grade.value)
        print("Summary:", output.summary)
        print("Entities:", output.entities)
        print("Relationships:", output.relationships)
        print("Queries:", output.retrieval_queries)
        print("Contradictions:", output.contradictions)

    # Show summary of contradictions
    print("\n" + "="*80)
    print("CONTRADICTION SUMMARY")
    print("="*80)

    if hasattr(graph_store, 'relationships'):
        superseded = [r for r in graph_store.relationships if not r.metadata.get("still_valid", True)]
        if superseded:
            print(f"\nFound {len(superseded)} superseded relationships:")
            for rel in superseded:
                print(f"  ✗ [{rel.predicate}] {rel.subject} → {rel.object}")
                print(f"      Superseded by: {rel.metadata.get('superseded_by')}")
        else:
            print("\nNo contradictions detected.")
    else:
        print("\n✓ Data persisted to FalkorDB (use inspect_memory.py to view)")


if __name__ == "__main__":
    asyncio.run(run_observer())
