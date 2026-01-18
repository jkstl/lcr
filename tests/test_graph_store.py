import asyncio

from src.memory.graph_store import InMemoryGraphStore


def test_graph_store_can_persist_and_query():
    store = InMemoryGraphStore()

    async def run():
        await store.persist_entities([{"name": "User", "type": "Person", "attributes": {}}])
        await store.persist_relationships(
            [
                {"subject": "User", "predicate": "OWNS", "object": "Dell Latitude 5520", "metadata": {}},
            ]
        )
        relationships = await store.search_relationships(["User"], limit=5)
        assert relationships
        assert relationships[0].predicate == "OWNS"
        await store.mark_contradiction(relationships[0].id, "User OWNS Dell Latitude 5520")
        assert relationships[0].metadata.get("still_valid") is False

    asyncio.run(run())
