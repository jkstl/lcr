"""Small demo that exercises the observer + context assembler with heuristic inputs."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.context_assembler import ContextAssembler
from src.memory.graph_store import create_graph_store
from src.memory.vector_store import init_vector_store
from src.observer.observer import Observer
from src.models.llm import OllamaClient


class DummyEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.0] * 768


class DummyReranker:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [1.0 for _ in pairs]


async def main():
    vector_table = init_vector_store("./data/lancedb_demo")
    graph_store = create_graph_store()
    observer = Observer(
        OllamaClient(),
        vector_table,
        graph_store,
        embedder=DummyEmbedder(),
    )

    prompts = [
        (
            "I broke up with my girlfriend Giana last week, and am feeling a little down today.",
            "I'm here to listen whenever you feel ready to talk.",
        ),
        (
            "I am looking to buy a new router that can run OpenWRT.",
            "Do you want something compact or with many LAN ports?",
        ),
    ]

    for index, (user_text, assistant_text) in enumerate(prompts):
        await observer.process_turn(
            user_message=user_text,
            assistant_response=assistant_text,
            conversation_id="demo-session",
            turn_index=index,
        )

    history = [
        {"role": "user", "content": prompts[0][0]},
        {"role": "assistant", "content": prompts[0][1]},
        {"role": "user", "content": prompts[1][0]},
        {"role": "assistant", "content": prompts[1][1]},
    ]

    assembler = ContextAssembler(
        vector_table,
        graph_store,
        reranker=DummyReranker(),
        embedder=DummyEmbedder(),
    )

    for user_query in [
        "What should I remember about Giana?",
        "Which router can run OpenWRT?",
    ]:
        context = await assembler.assemble(user_query, history)
        print(f"Context for '{user_query}':\n{context}\n")

    relationships = await graph_store.search_relationships(["User", "Giana", "OpenWRT-capable router"], limit=10)
    print("Persisted relationships:", relationships)


if __name__ == "__main__":
    asyncio.run(main())
