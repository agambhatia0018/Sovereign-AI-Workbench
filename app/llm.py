"""Thin wrapper around the local Ollama server.

Everything runs on localhost, so no data leaves the machine. This is the
'sovereign' part of the project.
"""
import json
from typing import AsyncIterator

import httpx

from app.config import settings


async def chat(messages: list[dict], temperature: float | None = None) -> str:
    """Send the full conversation and return the complete answer."""
    payload = {"model": settings.llm_model, "messages": messages, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{settings.ollama_url}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]


async def chat_stream(messages: list[dict]) -> AsyncIterator[str]:
    """Yield the answer token by token (typing effect in the UI)."""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_url}/api/chat",
            json={"model": settings.llm_model, "messages": messages, "stream": True},
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if not chunk.get("done"):
                    yield chunk["message"]["content"]


async def is_up() -> bool:
    """Health check: is Ollama reachable?"""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_url}/api/tags")
            return r.status_code == 200
    except httpx.HTTPError:
        return False
