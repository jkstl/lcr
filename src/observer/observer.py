from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from json import JSONDecodeError, loads
from typing import Any

from ..config import settings
from ..models.embedder import Embedder
from ..models.llm import OllamaClient
from ..memory.vector_store import MemoryChunk, persist_chunks
from ..memory.graph_store import GraphRelationship, GraphStore
from .prompts import EXTRACTION_PROMPT, QUERIES_PROMPT, SUMMARY_PROMPT, UTILITY_PROMPT


class UtilityGrade(Enum):
    DISCARD = "discard"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ObserverOutput:
    utility_grade: UtilityGrade
    summary: str | None
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    contradictions: list[dict[str, Any]]
    retrieval_queries: list[str]


class Observer:
    def __init__(
        self,
        llm_client: OllamaClient,
        vector_table: Any,
        graph_store: GraphStore,
        embedder: Embedder | None = None,
        model: str | None = None,
    ) -> None:
        self.llm = llm_client
        self.vector_table = vector_table
        self.graph_store = graph_store
        self.embedder = embedder or Embedder()
        self.model = model or settings.observer_model

    async def process_turn(
        self,
        user_message: str,
        assistant_response: str,
        conversation_id: str,
        turn_index: int,
    ) -> ObserverOutput:
        combined_input = f"USER: {user_message}\nASSISTANT: {assistant_response}"

        utility_grade = await self._grade_utility(combined_input)
        structured_data = await self._extract_structured_data(combined_input)
        entities = structured_data.get("entities", [])
        relationships = structured_data.get("relationships", [])
        fact_type = structured_data.get("fact_type", "episodic")  # Default to episodic
        summary = await self._generate_summary(combined_input)
        queries = await self._generate_retrieval_queries(combined_input)

        if utility_grade == UtilityGrade.DISCARD:
            return ObserverOutput(
                utility_grade=utility_grade,
                summary=None,
                entities=[],
                relationships=[],
                contradictions=[],
                retrieval_queries=[],
            )

        contradictions = await self._check_contradictions(relationships)

        await self._persist_to_vector_store(
            combined_input,
            summary,
            queries,
            conversation_id,
            turn_index,
            utility_grade,
            fact_type,
        )

        await self._persist_to_graph_store(entities, relationships, contradictions)

        return ObserverOutput(
            utility_grade=utility_grade,
            summary=summary,
            entities=entities,
            relationships=relationships,
            contradictions=contradictions,
            retrieval_queries=queries,
        )

    async def _grade_utility(self, text: str) -> UtilityGrade:
        prompt = UTILITY_PROMPT.format(text=text)
        response = await self.llm.generate(self.model, prompt)
        cleaned = response.strip().upper()
        try:
            return UtilityGrade(cleaned.lower())
        except ValueError:
            return UtilityGrade.LOW

    async def _generate_summary(self, text: str) -> str:
        prompt = SUMMARY_PROMPT.format(text=text)
        return (await self.llm.generate(self.model, prompt)).strip()

    async def _generate_retrieval_queries(self, text: str) -> list[str]:
        prompt = QUERIES_PROMPT.format(text=text)
        try:
            data = loads(await self.llm.generate(self.model, prompt))
            if isinstance(data, list):
                return [str(item) for item in data]
        except JSONDecodeError:
            ...
        return []

    async def _extract_structured_data(self, text: str) -> dict[str, list[dict]]:
        prompt = EXTRACTION_PROMPT.format(text=text)
        try:
            return loads(await self.llm.generate(self.model, prompt))
        except JSONDecodeError:
            return {"entities": [], "relationships": []}

    async def _check_contradictions(self, new_relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contradictions = []
        for rel in new_relationships:
            existing_relationships = await self.graph_store.query(rel["subject"], rel["predicate"])
            for existing_rel in existing_relationships:
                if existing_rel.object != rel["object"]:
                    contradiction = {
                        "existing_fact_id": existing_rel.id,
                        "existing_statement": f"{existing_rel.subject} {existing_rel.predicate} {existing_rel.object}",
                        "new_statement": f"{rel['subject']} {rel['predicate']} {rel['object']}",
                        "resolution_needed": True,
                    }
                    contradictions.append(contradiction)
                    await self.graph_store.mark_contradiction(existing_rel.id, contradiction["new_statement"])
        return contradictions

    async def _persist_to_vector_store(
        self,
        content: str,
        summary: str,
        queries: list[str],
        conversation_id: str,
        turn_index: int,
        utility_grade: UtilityGrade,
        fact_type: str = "episodic",
    ) -> None:
        embedding = await self.embedder.embed(content)
        chunk = MemoryChunk(
            id=str(datetime.utcnow().timestamp()),
            content=content,
            summary=summary,
            embedding=embedding,
            chunk_type="conversation",
            source_conversation_id=conversation_id,
            turn_index=turn_index,
            created_at=datetime.utcnow(),
            last_accessed_at=datetime.utcnow(),
            access_count=0,
            retrieval_queries=queries,
            utility_score=self._utility_to_score(utility_grade),
            fact_type=fact_type,
        )
        persist_chunks(self.vector_table, [chunk])

    async def _persist_to_graph_store(
        self,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        contradictions: list[dict[str, Any]],
    ) -> None:
        await self.graph_store.persist_entities(entities)
        await self.graph_store.persist_relationships(relationships)
        # Track contradictions as superseded facts
        for contradiction in contradictions:
            await self.graph_store.persist_relationships(
                [
                    {
                        "subject": contradiction["existing_statement"].split()[0],
                        "predicate": contradiction["existing_statement"].split()[1],
                        "object": contradiction["existing_statement"].split()[-1],
                        "metadata": {"superseded": True},
                    }
                ]
            )

    def _utility_to_score(self, utility_grade: UtilityGrade) -> float:
        mapping = {
            UtilityGrade.DISCARD: 0.0,
            UtilityGrade.LOW: 0.3,
            UtilityGrade.MEDIUM: 0.6,
            UtilityGrade.HIGH: 1.0,
        }
        return mapping.get(utility_grade, 0.0)
