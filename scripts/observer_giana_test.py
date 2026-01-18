"""Run the AGENTS-style observer on the supplied Giana/family prompts."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.vector_store import init_vector_store
from src.memory.graph_store import InMemoryGraphStore  # Using in-memory for mocked LLM test
from src.observer.observer import Observer


EMBED_DIM = 768


class DummyEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.0] * EMBED_DIM


class FixedLLM:
    async def generate(self, model: str, prompt: str, **kwargs) -> str:
        low = prompt.lower()
        if "rate the memory-worthiness" in low:
            return "HIGH"

        if "summarize this conversation turn" in low:
            if "mom" in low:
                return "User is hosting their mom and sister in Philadelphia after a breakup."
            return "User is still sad about breaking up with Giana."

        if "list 2-3 questions" in low:
            return json.dumps(
                ["How is the breakup with Giana affecting the user?", "Who is visiting from Massachusetts?"]
            )

        if "extract entities and relationships" in low:
            relationships = [
                {
                    "subject": "User",
                    "predicate": "FEELS_ABOUT",
                    "object": "Giana",
                    "metadata": {"sentiment": "sad"},
                },
                {
                    "subject": "User",
                    "predicate": "TRAVELS_WITH",
                    "object": "Mom and Justine",
                    "metadata": {"origin": "West Boylston, Massachusetts"},
                },
            ]
            return json.dumps(
                {
                    "entities": [
                        {"name": "Giana", "type": "Person", "attributes": {}},
                        {"name": "Mom", "type": "Person", "attributes": {"role": "mother"}},
                        {"name": "Justine", "type": "Person", "attributes": {"age": 24}},
                        {"name": "Philadelphia", "type": "Place", "attributes": {}},
                    ],
                    "relationships": relationships,
                }
            )

        return ""


async def main():
    vector_table = init_vector_store("./data/lancedb_observer_giana")
    graph_store = InMemoryGraphStore()
    observer = Observer(FixedLLM(), vector_table, graph_store, embedder=DummyEmbedder())

    prompts = [
        (
            "I broke up with my girlfriend Giana last week, and am feeling kind of sad about it.",
            "I'm really sorry you're going through this. Breakups are never easy, especially when you care about someone...",
        ),
        (
            "It's ok. My mom and sister are coming to see me in Philadelphia today, we are going out to dinner. They are traveling from West Boylston Massachusetts. My sister's name is Justine, she's 24.",
            "That sounds like a lovely way to spend the day — your mom and Justine traveling all that way ...",
        ),
    ]

    for idx, (user_text, assistant_text) in enumerate(prompts):
        output = await observer.process_turn(
            user_message=user_text,
            assistant_response=assistant_text,
            conversation_id="giana-session",
            turn_index=idx,
        )
        print(f"Turn {idx} observer output:")
        print("  Utility:", output.utility_grade)
        print("  Summary:", output.summary)
        print("  Entities:", output.entities)
        print("  Relationships:", output.relationships)
        print("  Queries:", output.retrieval_queries)
        print("  Contradictions:", output.contradictions)

    relationships = await graph_store.search_relationships(
        ["User", "Giana", "Mom", "Justine"], limit=10
    )
    print("\nGraph relationships persisted:")
    for rel in relationships:
        print(f"  {rel.predicate}: {rel.subject} -> {rel.object} (metadata={rel.metadata})")


if __name__ == "__main__":
    asyncio.run(main())
