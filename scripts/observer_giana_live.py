"""Run the live observer (qwen3:1.7b) on the provided Giana/family prompts."""

import asyncio
import sys
from pathlib import Path

from rich import print

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.vector_store import init_vector_store
from src.memory.graph_store import create_graph_store
from src.models.llm import OllamaClient
from src.observer.observer import Observer


async def main():
    print("[bold cyan]Initializing stores...[/bold cyan]")
    vector_table = init_vector_store()
    graph_store = create_graph_store()
    observer = Observer(OllamaClient(), vector_table, graph_store)
    print(f"[green]✓[/green] Using graph store: {type(graph_store).__name__}\n")

    prompts = [
        (
            "I broke up with my girlfriend Giana last week, and am feeling kind of sad about it.",
            "I'm really sorry you're going through this. Breakups are never easy, especially when you care about someone. It's okay to feel sad — grief is a natural part of letting go. Would you like to talk about what's been on your mind? Sometimes just processing it out loud helps, even if it's messy. I'm here whether you want to vent, reflect, or just sit with the sadness for a while.",
        ),
        (
            "It's ok. My mom and sister are coming to see me in Philadelphia today, we are going out to dinner. They are traveling from West Boylston, Massachusetts. My sister's name is Justine, she's 24.",
            "That sounds like a lovely way to spend the day — your mom and Justine traveling all that way to see you must mean a lot. It’s sweet that you’re sharing this time with them, especially after everything. I hope dinner is a nice distraction, but also that you feel okay letting yourself feel sad in the moments between conversations. Family can be a great anchor, but it’s also okay if your heart still feels heavy sometimes. Would you like to talk about how the breakup feels now, or maybe what you’re looking forward to with your mom and sister?",
        ),
    ]

    for idx, (user_text, assistant_text) in enumerate(prompts):
        output = await observer.process_turn(
            user_message=user_text,
            assistant_response=assistant_text,
            conversation_id="giana-live",
            turn_index=idx,
        )
        print(f"[bold]Turn {idx} observer output:[/bold]")
        print("  Utility:", output.utility_grade.value)
        print("  Summary:", output.summary)
        print("  Entities:", output.entities)
        print("  Relationships:", output.relationships)
        print("  Queries:", output.retrieval_queries)
        print("  Contradictions:", output.contradictions)

    # Query persisted data
    print("\n[bold cyan]Querying persisted graph data...[/bold cyan]")
    relationships = await graph_store.search_relationships(["User", "Giana", "Mom", "Justine", "Philadelphia", "West Boylston"], limit=20)

    print("\n[bold green]Graph relationships persisted:[/bold green]")
    if relationships:
        for rel in relationships:
            valid = "✓" if rel.metadata.get("still_valid", True) else "✗"
            print(f"  {valid} [{rel.predicate}] {rel.subject} → {rel.object}")
            if rel.metadata:
                print(f"      metadata: {rel.metadata}")
    else:
        print("  [red]⚠ No relationships found![/red]")

    # Show entities if using InMemoryGraphStore
    if hasattr(graph_store, 'entities'):
        print("\n[bold green]Entities stored:[/bold green]")
        for name, data in graph_store.entities.items():
            print(f"  • {name} ({data.get('type', 'Unknown')})")
            if data.get('attributes'):
                print(f"      attributes: {data['attributes']}")


if __name__ == "__main__":
    asyncio.run(main())
