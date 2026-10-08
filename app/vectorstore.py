"""ChromaDB wrapper. Stores chunks + their vectors on local disk."""
from pathlib import Path

import chromadb

from app.config import settings

_path = Path(settings.data_dir) / "chroma"
_path.mkdir(parents=True, exist_ok=True)
_client = chromadb.PersistentClient(path=str(_path))

# cosine distance: 0 = identical meaning, larger = less related
_collection = _client.get_or_create_collection(
    "documents", metadata={"hnsw:space": "cosine"}
)


def add_chunks(doc_id: str, filename: str, chunks: list[dict], vectors: list[list[float]]):
    _collection.add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[
            {"doc_id": doc_id, "filename": filename, "page": c["page"], "chunk": i}
            for i, c in enumerate(chunks)
        ],
    )


def search(vector: list[float], k: int) -> list[dict]:
    if _collection.count() == 0:
        return []
    res = _collection.query(query_embeddings=[vector], n_results=k)
    return [
        {"text": t, "distance": d, **m}
        for t, d, m in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
    ]


def list_documents() -> list[dict]:
    metas = _collection.get(include=["metadatas"])["metadatas"]
    docs: dict[str, dict] = {}
    for m in metas:
        d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "filename": m["filename"], "chunks": 0})
        d["chunks"] += 1
    return list(docs.values())


def delete_document(doc_id: str):
    _collection.delete(where={"doc_id": doc_id})
