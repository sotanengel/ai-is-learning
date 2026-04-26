"""OpenAI-compatible adapter for any LLM backend."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx


class LLMAdapter:
    """Thin async HTTP client for any OpenAI-compatible backend.

    Handles retries on 401/429/5xx with exponential backoff.
    """

    _RETRY_STATUS = {401, 429, 500, 502, 503, 504}
    _MAX_RETRIES = 3
    _BASE_BACKOFF = 1.0

    def __init__(self, base_url: str, api_key: str = "sk-no-key-required") -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=120.0)

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/chat/completions"
        for attempt in range(self._MAX_RETRIES):
            resp = await self._client.post(url, json=payload, headers=self._headers)
            if resp.status_code not in self._RETRY_STATUS:
                resp.raise_for_status()
                result: dict[str, Any] = resp.json()
                return result
            if attempt < self._MAX_RETRIES - 1:
                await asyncio.sleep(self._BASE_BACKOFF * (2**attempt))
        resp.raise_for_status()
        return {}

    async def chat_completions_stream(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        url = f"{self._base_url}/chat/completions"
        stream_payload = {**payload, "stream": True}
        async with self._client.stream(
            "POST", url, json=stream_payload, headers=self._headers
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def embeddings(self, texts: list[str], model: str) -> list[list[float]]:
        url = f"{self._base_url}/embeddings"
        payload = {"input": texts, "model": model}
        resp = await self._client.post(url, json=payload, headers=self._headers)
        resp.raise_for_status()
        data: list[dict[str, Any]] = resp.json()["data"]
        return [item["embedding"] for item in data]

    async def aclose(self) -> None:
        await self._client.aclose()
