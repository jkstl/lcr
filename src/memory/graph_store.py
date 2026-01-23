from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Any, Iterable, List, Optional

from ..config import settings

try:
    from falkordb import FalkorDB
except ImportError:  # pragma: no cover
    FalkorDB = None

LOGGER = logging.getLogger(__name__)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.utcnow()


@dataclass
class GraphRelationship:
    id: str
    subject: str
    predicate: str
    object: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    # Temporal tracking fields
    status: str | None = None  # "ongoing" | "completed" | "planned" | None
    valid_until: datetime | None = None  # When this fact expires (for episodic events)
    superseded_by: str | None = None  # ID of relationship that replaced this
    # Source tracking fields
    source: str = "user_stated"  # "user_stated" | "assistant_inferred"
    confidence: float = 1.0  # 1.0 for user_stated, 0.5 for assistant_inferred


class GraphStore(ABC):
    """Abstract graph store interface matching the AGENTS schema."""

    @abstractmethod
    async def persist_entities(self, entities: Iterable[dict[str, Any]]) -> None:
        ...

    @abstractmethod
    async def persist_relationships(self, relationships: Iterable[dict[str, Any]]) -> None:
        ...

    @abstractmethod
    async def query(self, subject: str, predicate: str | None = None) -> list[GraphRelationship]:
        ...

    @abstractmethod
    async def query_by_object(self, obj: str, predicate: str | None = None) -> list[GraphRelationship]:
        """Query relationships where the given entity appears as the object."""
        ...

    @abstractmethod
    async def search_relationships(self, entity_names: Iterable[str], limit: int = 10) -> list[GraphRelationship]:
        ...

    @abstractmethod
    async def mark_contradiction(self, existing_id: str, superseded_by: str) -> None:
        ...


class InMemoryGraphStore(GraphStore):
    def __init__(self):
        self.entities: dict[str, dict[str, Any]] = {}
        self.relationships: list[GraphRelationship] = []

    async def persist_entities(self, entities: Iterable[dict[str, Any]]) -> None:
        now = datetime.utcnow()
        for entity in entities:
            name = entity["name"]
            entry = self.entities.get(name, {})
            entry.setdefault("first_mentioned", now)
            entry["last_mentioned"] = now
            entry["type"] = entity.get("type")
            entry["attributes"] = {**entry.get("attributes", {}), **entity.get("attributes", {})}
            self.entities[name] = entry

    async def persist_relationships(self, relationships: Iterable[dict[str, Any]]) -> None:
        for rel in relationships:
            record = GraphRelationship(
                id=rel.get("id") or str(uuid.uuid4()),
                subject=rel["subject"],
                predicate=rel["predicate"],
                object=rel["object"],
                metadata=rel.get("metadata", {}),
                created_at=datetime.utcnow(),
                status=rel.get("status"),
                valid_until=rel.get("valid_until"),
                superseded_by=rel.get("superseded_by"),
                source=rel.get("source", "user_stated"),
                confidence=rel.get("confidence", 1.0),
            )
            self.relationships.append(record)

    async def query(self, subject: str, predicate: str | None = None) -> list[GraphRelationship]:
        matches: list[GraphRelationship] = []
        for rel in self.relationships:
            if rel.subject != subject:
                continue
            if predicate and rel.predicate != predicate:
                continue
            matches.append(rel)
        return matches

    async def query_by_object(self, obj: str, predicate: str | None = None) -> list[GraphRelationship]:
        """Query relationships where the given entity appears as the object."""
        matches: list[GraphRelationship] = []
        for rel in self.relationships:
            if rel.object != obj:
                continue
            if predicate and rel.predicate != predicate:
                continue
            matches.append(rel)
        return matches

    async def search_relationships(self, entity_names: Iterable[str], limit: int = 10) -> list[GraphRelationship]:
        found: list[GraphRelationship] = []
        seen: set[tuple[str, str, str]] = set()
        names = set(entity_names)
        for rel in self.relationships:
            if rel.subject in names or rel.object in names:
                key = (rel.subject, rel.predicate, rel.object)
                if key in seen:
                    continue
                seen.add(key)
                found.append(rel)
            if len(found) >= limit:
                break
        return found

    async def mark_contradiction(self, existing_id: str | int, superseded_by: str) -> None:
        # Handle both string and int IDs
        existing_id_str = str(existing_id)
        for rel in self.relationships:
            if str(rel.id) == existing_id_str:
                rel.metadata["still_valid"] = False
                rel.metadata["superseded_at"] = datetime.utcnow()
                rel.superseded_by = superseded_by
                rel.status = "completed"  # Mark temporal state as completed
                break


class FalkorGraphStore(GraphStore):
    def __init__(self, host: str, port: int, graph_id: str):
        if FalkorDB is None:
            raise ImportError("FalkorDB is not available")
        self.client = FalkorDB(host=host, port=port)
        self.graph = self.client.select_graph(graph_id)

    async def persist_entities(self, entities: Iterable[dict[str, Any]]) -> None:
        for entity in entities:
            current_ts = datetime.utcnow().isoformat()
            attributes_json = json.dumps(entity.get("attributes", {}))
            params = {
                "name": entity["name"],
                "category": entity.get("type", "Entity"),
                "attributes": attributes_json,
                "current_ts": current_ts,
            }
            query = """
MERGE (memo:Entity {name:$name})
SET memo.category = $category,
    memo.attributes = $attributes,
    memo.last_mentioned = $current_ts
"""
            await self._run(self.graph.query, query, params=params)

    async def persist_relationships(self, relationships: Iterable[dict[str, Any]]) -> None:
        for rel in relationships:
            metadata = rel.get("metadata") or {}
            current_ts = datetime.utcnow().isoformat()
            metadata_json = json.dumps(metadata)

            # Handle temporal fields
            status = rel.get("status") or "null"
            valid_until = rel.get("valid_until").isoformat() if rel.get("valid_until") else "null"
            superseded_by = rel.get("superseded_by") or "null"
            
            # Handle source tracking fields
            source = rel.get("source", "user_stated")
            confidence = rel.get("confidence", 1.0)

            params = {
                "subject": rel["subject"],
                "object": rel["object"],
                "metadata": metadata_json,
                "current_ts": current_ts,
                "status": status,
                "valid_until": valid_until,
                "superseded_by": superseded_by,
                "source": source,
                "confidence": confidence,
            }
            query = f"""
MERGE (subject:Person {{name:$subject}})
MERGE (object:Entity {{name:$object}})
MERGE (subject)-[relation:{rel['predicate']}]->(object)
SET relation.metadata = $metadata,
    relation.created_at = $current_ts,
    relation.still_valid = true,
    relation.status = $status,
    relation.valid_until = $valid_until,
    relation.superseded_by = $superseded_by,
    relation.source = $source,
    relation.confidence = $confidence
"""
            await self._run(self.graph.query, query, params=params)

    async def query(self, subject: str, predicate: str | None = None) -> list[GraphRelationship]:
        pred = f"-[relation:{predicate}]->" if predicate else "-[relation]->"
        query = f"""
MATCH (subject {{name:$subject}}){pred}(object)
RETURN subject.name AS subject, type(relation) AS predicate, object.name AS object,
       relation.metadata AS metadata, id(relation) AS rel_id, relation.created_at AS created_at,
       relation.status AS status, relation.valid_until AS valid_until, relation.superseded_by AS superseded_by,
       relation.source AS source, relation.confidence AS confidence
"""
        result = await self._run(self.graph.query, query, params={"subject": subject})
        return [self._row_to_relationship(row) for row in result.result_set]

    async def query_by_object(self, obj: str, predicate: str | None = None) -> list[GraphRelationship]:
        """Query relationships where the given entity appears as the object."""
        pred = f"-[relation:{predicate}]->" if predicate else "-[relation]->"
        query = f"""
MATCH (subject){pred}(object {{name:$object}})
RETURN subject.name AS subject, type(relation) AS predicate, object.name AS object,
       relation.metadata AS metadata, id(relation) AS rel_id, relation.created_at AS created_at,
       relation.status AS status, relation.valid_until AS valid_until, relation.superseded_by AS superseded_by,
       relation.source AS source, relation.confidence AS confidence
"""
        result = await self._run(self.graph.query, query, params={"object": obj})
        return [self._row_to_relationship(row) for row in result.result_set]

    async def search_relationships(self, entity_names: Iterable[str], limit: int = 10) -> list[GraphRelationship]:
        names = list(entity_names)
        query = """
MATCH (subject)-[relation]->(object)
WHERE subject.name IN $names OR object.name IN $names
RETURN subject.name AS subject, type(relation) AS predicate, object.name AS object,
       relation.metadata AS metadata, id(relation) AS rel_id, relation.created_at AS created_at,
       relation.status AS status, relation.valid_until AS valid_until, relation.superseded_by AS superseded_by,
       relation.source AS source, relation.confidence AS confidence
ORDER BY relation.created_at DESC
LIMIT $limit
"""
        result = await self._run(self.graph.query, query, params={"names": names, "limit": limit})
        return [self._row_to_relationship(row) for row in result.result_set]

    async def mark_contradiction(self, existing_id: str | int, superseded_by: str) -> None:
        # FalkorDB expects integer IDs
        if isinstance(existing_id, str):
            existing_id = int(existing_id) if existing_id.isdigit() else existing_id

        current_ts = datetime.utcnow().isoformat()
        query = """
MATCH ()-[relation]->()
WHERE id(relation) = $rel_id
SET relation.still_valid = false,
    relation.superseded_by = $superseded_by,
    relation.superseded_at = $current_ts
"""
        params = {"rel_id": existing_id, "superseded_by": superseded_by, "current_ts": current_ts}
        await self._run(self.graph.query, query, params=params)

    @staticmethod
    def _row_to_relationship(row: List[Any]) -> GraphRelationship:
        metadata_raw = row[3]
        if isinstance(metadata_raw, (str, bytes)) and metadata_raw:
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                metadata = {"value": metadata_raw}
        else:
            metadata = metadata_raw or {}
        created_at = _parse_datetime(row[5])

        # Parse temporal fields (indices 6, 7, 8)
        status = row[6] if len(row) > 6 and row[6] != "null" else None
        valid_until = _parse_datetime(row[7]) if len(row) > 7 and row[7] and row[7] != "null" else None
        superseded_by = row[8] if len(row) > 8 and row[8] and row[8] != "null" else None

        # Parse source tracking fields (indices 9, 10)
        source = row[9] if len(row) > 9 and row[9] else "user_stated"
        confidence = float(row[10]) if len(row) > 10 and row[10] is not None else 1.0

        return GraphRelationship(
            id=str(row[4]),
            subject=str(row[0]),
            predicate=str(row[1]),
            object=str(row[2]),
            metadata=dict(metadata),
            created_at=created_at,
            status=status,
            valid_until=valid_until,
            superseded_by=superseded_by,
            source=source,
            confidence=confidence,
        )

    async def _run(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)


def create_graph_store() -> GraphStore:
    if FalkorDB is None:
        LOGGER.info("FalkorDB dependency missing; using in-memory graph store")
        return InMemoryGraphStore()

    try:
        return FalkorGraphStore(settings.falkordb_host, settings.falkordb_port, settings.falkordb_graph_id)
    except Exception as exc:
        LOGGER.warning("FalkorDB unavailable (%s); falling back to in-memory store", exc)
        return InMemoryGraphStore()
