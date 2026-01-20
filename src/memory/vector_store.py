from __future__ import annotations

from datetime import datetime
from typing import Iterable

import lancedb
import numpy as np
import pyarrow as pa
from pydantic import BaseModel


EMBED_DIM = 4096  # llama3 embedding dimension

class MemoryChunk(BaseModel):
    id: str
    content: str
    summary: str
    embedding: list[float]
    chunk_type: str
    source_conversation_id: str
    turn_index: int
    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    retrieval_queries: list[str]
    utility_score: float
    fact_type: str = "episodic"  # "core" | "episodic" | "preference"


MEMORY_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("summary", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), EMBED_DIM)),
        pa.field("chunk_type", pa.string()),
        pa.field("source_conversation_id", pa.string()),
        pa.field("turn_index", pa.int64()),
        pa.field("created_at", pa.timestamp("us")),
        pa.field("last_accessed_at", pa.timestamp("us")),
        pa.field("access_count", pa.int64()),
        pa.field("retrieval_queries", pa.list_(pa.string())),
        pa.field("utility_score", pa.float64()),
        pa.field("fact_type", pa.string()),
    ]
)


def init_vector_store(db_path: str = "./data/lancedb"):
    db = lancedb.connect(db_path)
    if "memories" not in db.table_names():
        db.create_table("memories", schema=MEMORY_SCHEMA)
    return db.open_table("memories")


def persist_chunks(table, chunks: Iterable[MemoryChunk]) -> None:
    if table is None or not chunks:
        return
    prepared = []
    for chunk in chunks:
        record = chunk.dict()
        embedding = [float(np.float32(val)) for val in record["embedding"]]
        if len(embedding) != EMBED_DIM:
            raise ValueError("Embedding length mismatch")
        record["embedding"] = embedding
        prepared.append(record)
    table.add(prepared)


def vector_search(table, embedding: list[float], top_k: int = 10):
    vector = np.array(embedding, dtype=np.float32).tolist()
    try:
        result = (
            table.search(vector, "embedding")
            .limit(top_k * 2)  # Fetch extra to re-rank with combined score
            .to_list()
        )
    except RuntimeError:
        result = _naive_vector_search(table, vector, top_k * 2)

    # Combine similarity (from search order) and utility score
    # Weight: 70% similarity rank, 30% utility
    for i, item in enumerate(result):
        # Normalize rank to 0-1 (first result = 1.0, last = lower)
        rank_score = 1.0 - (i / max(len(result), 1))
        utility = item.get("utility_score", 0.5)
        item["_combined_score"] = 0.7 * rank_score + 0.3 * utility
    
    result.sort(key=lambda item: item.get("_combined_score", 0.0), reverse=True)
    return result[:top_k]


def _naive_vector_search(table, vector, top_k):
    try:
        rows = table.to_arrow().to_pylist()
    except AttributeError:
        rows = []

    scored: list[tuple[float, dict]] = []
    for record in rows:
        stored = record.get("embedding") or []
        score = _cosine_similarity(vector, stored)
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for score, record in scored[:top_k]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)
