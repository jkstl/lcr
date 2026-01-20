from __future__ import annotations

import httpx
import json
import logging
from typing import Any

from ..config import settings


class OllamaClient:
    """Simple Ollama HTTP wrapper for generation and embedding."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or settings.ollama_host
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        text = response.text
        try:
            data = response.json()
            response_text = data.get("response", "")
        except ValueError:
            response_text = ""
            data = {}
            lines = [line for line in text.splitlines() if line.strip()]
            for line in lines:
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                chunk_response = chunk.get("response")
                if chunk_response:
                    response_text += chunk_response
                data = chunk
        return response_text.strip()

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        """Async generator that yields response tokens as they stream from Ollama."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system:
            payload["system"] = system
        
        async with self._client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

    async def embed(self, model: str, text: str) -> list[float]:
        payload = {"model": model, "prompt": text}
        try:
            return await self._call_embedding(payload)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                fallback_model = "llama3"
                logger = logging.getLogger(__name__)
                logger.info("Embedding model %s not available; retrying with %s", model, fallback_model)
                payload["model"] = fallback_model
                return await self._call_embedding(payload)
            raise

    async def _call_embedding(self, payload: dict[str, Any]) -> list[float]:
        response = await self._client.post("/api/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()
        if "embedding" in data:
            return data["embedding"]
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            return embeddings[0]
        raise ValueError("Unexpected embedding response")

    async def close(self) -> None:
        await self._client.aclose()
