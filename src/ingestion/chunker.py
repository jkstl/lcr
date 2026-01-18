from __future__ import annotations

import re
from typing import Generator

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticChunker:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        min_chunk_sentences: int = 3,
        max_chunk_tokens: int = 512,
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_tokens = max_chunk_tokens

    def chunk(self, text: str) -> Generator[str, None, None]:
        sentences = self._split_sentences(text)
        if not sentences:
            return

        embeddings = self.embedder.encode(sentences)
        current_chunk = [sentences[0]]
        current_tokens = self._count_tokens(sentences[0])

        for idx in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[idx - 1], embeddings[idx])
            sentence_tokens = self._count_tokens(sentences[idx])
            should_break = (
                similarity < self.similarity_threshold and len(current_chunk) >= self.min_chunk_sentences
            )
            would_exceed_max = (current_tokens + sentence_tokens) > self.max_chunk_tokens

            if should_break or would_exceed_max:
                yield " ".join(current_chunk)
                current_chunk = [sentences[idx]]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentences[idx])
                current_tokens += sentence_tokens

        if current_chunk:
            yield " ".join(current_chunk)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def _split_sentences(self, text: str) -> list[str]:
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]

    def _count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
