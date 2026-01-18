from __future__ import annotations

from datetime import datetime
from typing import Iterable
import uuid

from ..models.embedder import Embedder
from ..memory.vector_store import MemoryChunk, persist_chunks
from .chunker import SemanticChunker


class IngestionPipeline:
    """Semantically chunk and persist text into the vector store."""

    def __init__(
        self,
        chunker: SemanticChunker | None = None,
        embedder: Embedder | None = None,
    ):
        self.chunker = chunker or SemanticChunker()
        self.embedder = embedder or Embedder()

    async def ingest(
        self,
        text: str,
        vector_table,
        conversation_id: str,
        chunk_type: str = "document",
    ) -> None:
        chunks = []
        for turn_index, chunk in enumerate(self.chunker.chunk(text)):
            embedding = await self.embedder.embed(chunk)
            memory = MemoryChunk(
                id=str(uuid.uuid4()),
                content=chunk,
                summary=chunk[:250],
                embedding=embedding,
                chunk_type=chunk_type,
                source_conversation_id=conversation_id,
                turn_index=turn_index,
                created_at=datetime.utcnow(),
                last_accessed_at=datetime.utcnow(),
                access_count=0,
                retrieval_queries=[],
                utility_score=0.0,
            )
            chunks.append(memory)

        persist_chunks(vector_table, chunks)
