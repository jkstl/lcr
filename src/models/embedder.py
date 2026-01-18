from __future__ import annotations

from ..config import settings
from .llm import OllamaClient


class Embedder:
    """Embedding helper that delegates to Ollama for nomic-embed-text."""

    def __init__(self, client: OllamaClient | None = None, model: str | None = None) -> None:
        self.client = client or OllamaClient()
        self.model = model or settings.embedding_model

    async def embed(self, text: str) -> list[float]:
        return await self.client.embed(self.model, text)
