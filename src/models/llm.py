from __future__ import annotations

import httpx
import json
import logging
import re
from typing import Any

from ..config import settings


def parse_json_response(response: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response with multiple fallback strategies.
    
    LLMs often wrap JSON in markdown blocks or add explanatory text.
    This function tries multiple strategies to extract valid JSON.
    
    Strategies (in order):
    1. Direct parse (fast path for well-formed responses)
    2. Extract from markdown code blocks (```json ... ```)
    3. Find first JSON object in text ({...})
    4. Strip common preambles and try again
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If no valid JSON can be extracted
    """
    # Strategy 1: Direct parse (current approach - fast path)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks
    # Matches: ```json\n{...}\n``` or ```\n{...}\n```
    json_match = re.search(
        r'```(?:json)?\s*(.*?)```',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find first JSON object in text
    # Look for {...} pattern (greedy to capture full object)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Strip common preambles and postambles
    # Remove everything before first { and after last }
    cleaned = response.strip()
    if '{' in cleaned and '}' in cleaned:
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass
    
    # All strategies failed
    raise ValueError(
        f"No valid JSON found in response. "
        f"First 200 chars: {response[:200]}"
    )



class OllamaClient:
    """Simple Ollama HTTP wrapper for generation and embedding."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or settings.ollama_host
        # Increased timeout to handle parallel observer tasks
        # When multiple observers run concurrently, Ollama may take longer to respond
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=180.0)

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
