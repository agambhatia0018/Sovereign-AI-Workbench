"""Turn text into vectors (lists of numbers) using a local embedding model.

Texts with similar meaning get vectors that are close together. That is what
makes semantic search possible.
"""
import httpx

from app.config import settings


async def embed(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f"{settings.ollama_url}/api/embed",
            json={"model": settings.embed_model, "input": texts},
        )
        r.raise_for_status()
        return r.json()["embeddings"]


async def embed_in_batches(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(await embed(texts[i : i + batch_size]))
    return out
