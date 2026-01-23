from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from json import JSONDecodeError, loads
from typing import Any, Callable, TypeVar

from ..config import settings
from ..models.embedder import Embedder
from ..models.llm import OllamaClient, parse_json_response
from ..memory.vector_store import MemoryChunk, persist_chunks
from ..memory.graph_store import GraphRelationship, GraphStore
from .prompts import (
    EXTRACTION_PROMPT,
    QUERIES_PROMPT,
    SEMANTIC_CONTRADICTION_PROMPT,
    SUMMARY_PROMPT,
    UTILITY_PROMPT,
)

LOGGER = logging.getLogger(__name__)

T = TypeVar('T')


async def retry_on_timeout(func: Callable[[], Any], max_retries: int = 3, delay: float = 2.0) -> Any:
    """Retry a function on httpx.ReadTimeout with exponential backoff."""
    import httpx

    for attempt in range(max_retries):
        try:
            return await func()
        except httpx.ReadTimeout as e:
            if attempt == max_retries - 1:
                LOGGER.error(f"Max retries ({max_retries}) exceeded for LLM call")
                raise
            wait_time = delay * (2 ** attempt)
            LOGGER.warning(f"LLM call timed out (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
        except Exception as e:
            # Don't retry on other exceptions
            raise



class UtilityGrade(Enum):
    DISCARD = "discard"
    # New 3-level system
    STORE = "store"
    IMPORTANT = "important"
    # Legacy 4-level system (backward compatibility)
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
        user_only = f"USER: {user_message}"
        assistant_only = f"ASSISTANT: {assistant_response}"

        # Grade utility first (gatekeeper)
        utility_grade = await self._grade_utility(combined_input)

        # Early exit for DISCARD - skip all other LLM calls
        if utility_grade == UtilityGrade.DISCARD:
            return ObserverOutput(
                utility_grade=utility_grade,
                summary=None,
                entities=[],
                relationships=[],
                contradictions=[],
                retrieval_queries=[],
            )

        # Extract from BOTH user and assistant content separately
        # Summary and queries use combined for full context
        user_data, assistant_data, summary, queries = await asyncio.gather(
            self._extract_structured_data(user_only),
            self._extract_structured_data(assistant_only),
            self._generate_summary(combined_input),
            self._generate_retrieval_queries(combined_input),
        )

        # Merge entities from both sources
        entities = user_data.get("entities", []) + assistant_data.get("entities", [])
        
        # Tag user relationships as high confidence (ground truth)
        user_relationships = user_data.get("relationships", [])
        for rel in user_relationships:
            rel["source"] = "user_stated"
            rel["confidence"] = 1.0
        
        # Tag assistant relationships as low confidence (may be hallucinated)
        assistant_relationships = assistant_data.get("relationships", [])
        for rel in assistant_relationships:
            rel["source"] = "assistant_inferred"
            rel["confidence"] = 0.3
        
        # Merge relationships, but user facts supersede assistant inferences
        # If same subject+predicate exists from both, only keep user version
        user_keys = {(r["subject"], r["predicate"]) for r in user_relationships}
        filtered_assistant = [
            r for r in assistant_relationships 
            if (r["subject"], r["predicate"]) not in user_keys
        ]
        
        relationships = user_relationships + filtered_assistant
        fact_type = user_data.get("fact_type", "episodic")

        contradictions = await self._check_contradictions(relationships)

        # Persist to both stores in parallel for better performance
        await asyncio.gather(
            self._persist_to_vector_store(
                combined_input,
                summary,
                queries,
                conversation_id,
                turn_index,
                utility_grade,
                fact_type,
            ),
            self._persist_to_graph_store(entities, relationships, contradictions),
        )

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
        response = await retry_on_timeout(lambda: self.llm.generate(self.model, prompt))
        cleaned = response.strip().upper()
        try:
            grade = UtilityGrade(cleaned.lower())
            # Defensive logging to track utility grading decisions
            preview = text.replace("\n", " ").replace("\r", "")[:100]
            LOGGER.info(f"Utility grading: {grade.value.upper()} | Preview: {preview}...")
            return grade
        except ValueError:
            LOGGER.warning(f"Invalid utility grade response: '{cleaned}', defaulting to LOW")
            return UtilityGrade.LOW

    async def _generate_summary(self, text: str) -> str:
        prompt = SUMMARY_PROMPT.format(text=text)
        response = await retry_on_timeout(lambda: self.llm.generate(self.model, prompt))
        return response.strip()

    async def _generate_retrieval_queries(self, text: str) -> list[str]:
        prompt = QUERIES_PROMPT.format(text=text)
        try:
            response = await retry_on_timeout(lambda: self.llm.generate(self.model, prompt))
            data = loads(response)
            if isinstance(data, list):
                return [str(item) for item in data]
        except JSONDecodeError:
            ...
        return []

    async def _extract_structured_data(self, text: str) -> dict[str, list[dict]]:
        from ..models.llm import parse_json_response
        
        prompt = EXTRACTION_PROMPT.format(text=text)
        try:
            response = await retry_on_timeout(lambda: self.llm.generate(self.model, prompt))
            return parse_json_response(response)
        except (JSONDecodeError, ValueError) as e:
            LOGGER.warning(f"JSON extraction failed: {e}")
            return {"entities": [], "relationships": []}

    async def _check_contradictions(self, new_relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Check for semantic contradictions using LLM reasoning.
        Handles temporal state transitions and mutually exclusive predicates.
        """
        contradictions = []

        for rel in new_relationships:
            # Get ALL existing relationships involving these entities (not just same predicate)
            existing_rels = await self._get_related_facts(rel["subject"], rel.get("object"))

            if not existing_rels:
                continue

            # Use LLM to detect semantic contradictions
            semantic_contradictions = await self._detect_semantic_contradictions(rel, existing_rels)

            for contradiction in semantic_contradictions:
                # Only mark high-confidence contradictions
                if contradiction.get("confidence") == "high":
                    contradictions.append({
                        "existing_fact_id": contradiction["existing_id"],
                        "existing_statement": contradiction["existing_statement"],
                        "new_statement": f"{rel['subject']} {rel['predicate']} {rel['object']}",
                        "reason": contradiction["reason"],
                        "temporal_type": contradiction.get("temporal_type"),
                        "resolution_needed": True,
                    })

                    # Mark the old fact as superseded
                    # Convert ID to int for FalkorDB (string IDs don't match)
                    existing_id = contradiction["existing_id"]
                    if isinstance(existing_id, str) and existing_id.isdigit():
                        existing_id = int(existing_id)

                    await self.graph_store.mark_contradiction(
                        existing_id,
                        f"{rel['subject']} {rel['predicate']} {rel['object']}"
                    )

        return contradictions

    async def _get_related_facts(self, subject: str, obj: str | None = None) -> list[GraphRelationship]:
        """
        Get all existing relationships involving the subject (and optionally object).
        This allows us to find contradictions across different predicates.
        """
        # Get all relationships where subject is involved
        subject_rels = await self.graph_store.query(subject, predicate=None)

        # If object is provided, also get relationships involving the object
        if obj:
            object_rels = await self.graph_store.query(obj, predicate=None)
            # Combine and deduplicate
            all_rels = {rel.id: rel for rel in subject_rels + object_rels}
            return list(all_rels.values())

        return subject_rels

    async def _detect_semantic_contradictions(
        self, new_rel: dict[str, Any], existing_rels: list[GraphRelationship]
    ) -> list[dict[str, Any]]:
        """
        Use LLM to detect semantic contradictions between new and existing relationships.
        """
        if not existing_rels:
            return []

        # Format new relationship
        new_rel_str = f"{new_rel['subject']} {new_rel['predicate']} {new_rel['object']}"

        # Format existing relationships with IDs
        existing_rels_str = "[\n"
        for rel in existing_rels[:10]:  # Limit to prevent context overflow
            existing_rels_str += f'  {{"id": "{rel.id}", "subject": "{rel.subject}", "predicate": "{rel.predicate}", "object": "{rel.object}"}},\n'
        existing_rels_str += "]"

        # Ask LLM to detect contradictions
        prompt = SEMANTIC_CONTRADICTION_PROMPT.format(
            new_relationship=new_rel_str,
            existing_relationships=existing_rels_str
        )

        try:
            response = await self.llm.generate(self.model, prompt)
            from ..models.llm import parse_json_response
            result = parse_json_response(response)
            return result.get("contradictions", [])
        except (JSONDecodeError, KeyError, ValueError) as e:
            LOGGER.warning(f"Contradiction detection JSON parse failed: {e}")
            # Fallback to simple contradiction detection if LLM fails
            return await self._simple_contradiction_check(new_rel, existing_rels)

    async def _simple_contradiction_check(
        self, new_rel: dict[str, Any], existing_rels: list[GraphRelationship]
    ) -> list[dict[str, Any]]:
        """
        Fallback: Simple predicate-based contradiction detection.
        """
        contradictions = []
        for existing_rel in existing_rels:
            if (existing_rel.subject == new_rel["subject"] and
                existing_rel.predicate == new_rel["predicate"] and
                existing_rel.object != new_rel["object"]):
                contradictions.append({
                    "existing_id": existing_rel.id,
                    "existing_statement": f"{existing_rel.subject} {existing_rel.predicate} {existing_rel.object}",
                    "reason": "Same predicate with different object",
                    "temporal_type": None,
                    "confidence": "high"
                })
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
        """Map utility grade to numeric score for temporal decay."""
        mapping = {
            # 3-level system (simplified)
            UtilityGrade.DISCARD: 0.0,
            UtilityGrade.STORE: 0.6,
            UtilityGrade.IMPORTANT: 1.0,
            # 4-level system (legacy backward compatibility)
            UtilityGrade.LOW: 0.3,
            UtilityGrade.MEDIUM: 0.6,
            UtilityGrade.HIGH: 1.0,
        }
        return mapping.get(utility_grade, 0.6)  # Default to STORE/MEDIUM level
