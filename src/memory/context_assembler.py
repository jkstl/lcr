from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from ..config import settings
from ..models.embedder import Embedder
from .vector_store import vector_search


@dataclass
class RetrievedContext:
    content: str
    source: str
    relevance_score: float
    temporal_score: float
    final_score: float
    created_at: datetime
    fact_type: str = "episodic"  # "core" | "episodic" | "preference"
    utility_score: float = 0.5


class ContextAssembler:
    """Assembles memories from vector and graph sources into an LLM-friendly context."""

    def __init__(
        self,
        vector_table: Any,
        graph_store: Any,
        reranker: Any,
        embedder: Embedder | None = None,
    ):
        self.vector_table = vector_table
        self.graph_store = graph_store
        self.reranker = reranker
        self.embedder = embedder or Embedder()
        self.max_context_tokens = settings.max_context_tokens
        self.sliding_window_tokens = settings.sliding_window_tokens
        # Tiered decay rates by utility/fact type
        self.decay_rates = {
            "core": settings.temporal_decay_core,      # 0 = no decay
            "high": settings.temporal_decay_high,      # 180 days
            "medium": settings.temporal_decay_medium,  # 60 days
            "low": settings.temporal_decay_low,        # 14 days
        }

    async def assemble(
        self,
        query: str,
        conversation_history: list[dict[str, Any]],
        top_k_vector: int | None = None,
        top_k_graph: int | None = None,
        final_k: int | None = None,
    ) -> str:
        top_k_vector = top_k_vector or settings.vector_search_top_k
        top_k_graph = top_k_graph or settings.graph_search_top_k
        final_k = final_k or settings.rerank_top_k

        sliding_context = self._get_sliding_window(conversation_history)
        remaining_tokens = max(0, self.max_context_tokens - self._count_tokens(sliding_context))

        vector_results = await self._vector_search(query, top_k_vector)
        graph_results = await self._graph_search(query, top_k_graph)

        all_candidates = self._merge_results(vector_results, graph_results)

        for candidate in all_candidates:
            candidate.temporal_score = self._calculate_temporal_decay(
                candidate.created_at, candidate.fact_type, candidate.utility_score
            )
            candidate.final_score = candidate.relevance_score * candidate.temporal_score

        last_user_text = self._extract_last_user_message(conversation_history)
        reranked = self._rerank(query, all_candidates, final_k, last_user_text)
        memory_context = self._format_memories(reranked, remaining_tokens)
        return self._build_final_context(sliding_context, memory_context)

    async def _vector_search(self, query: str, top_k: int) -> list[RetrievedContext]:
        embedding = await self.embedder.embed(query)
        hits = vector_search(self.vector_table, embedding, top_k)
        results: list[RetrievedContext] = []
        for hit in hits:
            created_at = hit.get("created_at") or datetime.utcnow()
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = datetime.utcnow()
            results.append(
                RetrievedContext(
                    content=hit.get("content", ""),
                    source="vector",
                    relevance_score=hit.get("utility_score", 0.5),
                    temporal_score=1.0,
                    final_score=hit.get("utility_score", 0.5),
                    created_at=created_at,
                    fact_type=hit.get("fact_type", "episodic"),
                    utility_score=hit.get("utility_score", 0.5),
                )
            )
        return results

    async def _graph_search(self, query: str, top_k: int) -> list[RetrievedContext]:
        entity_names = self._extract_entities_from_query(query)
        relationships = await self.graph_store.search_relationships(entity_names, limit=top_k)
        results: list[RetrievedContext] = []
        for rel in relationships:
            content = f"{rel.subject} {rel.predicate} {rel.object}"
            created_at = rel.created_at
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = datetime.utcnow()
            results.append(
                RetrievedContext(
                    content=content,
                    source="graph",
                    relevance_score=0.4,
                    temporal_score=1.0,
                    final_score=0.4,
                    created_at=created_at,
                )
            )
        return results

    def _merge_results(
        self, vector_results: Iterable[RetrievedContext], graph_results: Iterable[RetrievedContext]
    ) -> list[RetrievedContext]:
        merged: dict[tuple[str, str], RetrievedContext] = {}
        for candidate in (*vector_results, *graph_results):
            key = (candidate.content, candidate.source)
            existing = merged.get(key)
            if existing is None or candidate.final_score > existing.final_score:
                merged[key] = candidate
        return list(merged.values())

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievedContext],
        top_k: int,
        last_user_message: str | None,
    ) -> list[RetrievedContext]:
        pairs = [(query, candidate.content) for candidate in candidates]
        scores = self.reranker.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate.final_score *= score if score else 1.0
            if last_user_message:
                lowered = last_user_message.lower().strip()
                if lowered and lowered in candidate.content.lower():
                    candidate.final_score *= 1.4
        return sorted(candidates, key=lambda candidate: candidate.final_score, reverse=True)[:top_k]

    def _format_memories(self, memories: list[RetrievedContext], max_tokens: int) -> str:
        entries: list[str] = []
        tokens = 0
        for mem in memories:
            entry_tokens = self._count_tokens(mem.content)
            if tokens + entry_tokens > max_tokens:
                break
            entries.append(f"- {mem.content}")
            tokens += entry_tokens
        return "\n".join(entries)

    def _build_final_context(self, sliding: str, memories: str) -> str:
        return f"""## Recent Conversation
{sliding}

## Relevant Memories
{memories}"""

    def _calculate_temporal_decay(self, created_at: datetime, fact_type: str, utility_score: float) -> float:
        """Calculate temporal decay based on fact_type and utility_score.
        
        Core facts have no decay (always return 1.0).
        Other facts decay based on utility: HIGH=180d, MEDIUM=60d, LOW=14d half-life.
        """
        # Core facts never decay
        if fact_type == "core":
            return 1.0
        
        # Determine decay rate based on utility score
        if utility_score >= 0.9:  # HIGH utility
            decay_days = self.decay_rates["high"]
        elif utility_score >= 0.5:  # MEDIUM utility
            decay_days = self.decay_rates["medium"]
        else:  # LOW utility
            decay_days = self.decay_rates["low"]
        
        # If decay_days is 0, no decay
        if decay_days == 0:
            return 1.0
        
        age_days = (datetime.now() - created_at).days
        return 0.5 ** (age_days / decay_days)

    def _get_sliding_window(self, history: list[dict[str, Any]]) -> str:
        result = []
        tokens = 0
        for msg in reversed(history):
            msg_tokens = self._count_tokens(msg.get("content", ""))
            if tokens + msg_tokens > self.sliding_window_tokens:
                break
            result.insert(0, f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}")
            tokens += msg_tokens
        return "\n".join(result)

    def _extract_entities_from_query(self, query: str) -> list[str]:
        return list({match for match in re.findall(r"\b[A-Z][a-zA-Z0-9\-']+\b", query)})

    def _count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _extract_last_user_message(self, history: list[dict[str, Any]]) -> str | None:
        for message in reversed(history):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
        return None
