"""Exercise the memory stack with the requested RAG/memory prompts."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.context_assembler import ContextAssembler
from src.memory.graph_store import create_graph_store
from src.memory.vector_store import init_vector_store
from src.observer.observer import Observer


EMBED_DIM = 768


class DummyEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.0] * EMBED_DIM


class DummyReranker:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [1.0 for _ in pairs]


class DummyLLM:
    async def generate(self, model: str, prompt: str, **kwargs):
        normalized = prompt.lower()
        if "rate the memory-worthiness" in normalized:
            if "home server" in normalized or "changed my mind" in normalized:
                return "HIGH"
            if "sell" in normalized:
                return "HIGH"
            return "LOW"

        if "summarize this conversation turn" in normalized:
            if "home server" in normalized:
                return "User will repurpose the Latitude 5520 as a home server."
            return "User wants to sell the Latitude 5520 laptop."

        if "list 2-3 questions" in normalized:
            return json.dumps(["What is going on with my Latitude 5520 laptop?"])

        if "extract entities and relationships" in normalized:
            if "home server" in normalized:
                relationships = [
                    {
                        "subject": "User",
                        "predicate": "USES_AS",
                        "object": "Latitude 5520",
                        "metadata": {"context": "home server"},
                    }
                ]
            else:
                relationships = [
                    {
                        "subject": "User",
                        "predicate": "SELLS",
                        "object": "Latitude 5520",
                        "metadata": {},
                    }
                ]
            return json.dumps(
                {
                    "entities": [
                        {"name": "Latitude 5520", "type": "Technology", "attributes": {}}
                    ],
                    "relationships": relationships,
                }
            )

        return ""


async def run_sequence():
    vector_table = init_vector_store("./data/lancedb_sequence")
    graph_store = create_graph_store()
    observer = Observer(DummyLLM(), vector_table, graph_store, embedder=DummyEmbedder())

    prompts = [
        (
            "I want to sell my Latitude 5520 laptop.",
            "Let me know the details and I can help you list it.",
        ),
        (
            "I've changed my mind, going to use the 5520 as a home server.",
            "That sounds like a fun project!",
        ),
    ]

    history: list[dict[str, str]] = []

    for idx, (user_text, assistant_text) in enumerate(prompts):
        await observer.process_turn(
            user_message=user_text,
            assistant_response=assistant_text,
            conversation_id="sequence-run",
            turn_index=idx,
        )
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})

    assembler = ContextAssembler(
        vector_table,
        graph_store,
        reranker=DummyReranker(),
        embedder=DummyEmbedder(),
    )

    question = "What do you know about my laptop?"
    context = await assembler.assemble(question, history)
    print("Final context after RAG cycle:\n", context)

    relationships = await graph_store.search_relationships(
        ["User", "Latitude 5520", "OpenWRT-capable router"], limit=10
    )
    print("Graph relationships:", relationships)


if __name__ == "__main__":
    asyncio.run(run_sequence())
