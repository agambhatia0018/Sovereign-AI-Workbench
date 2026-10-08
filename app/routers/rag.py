from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import llm, vectorstore
from app.config import settings
from app.embeddings import embed

router = APIRouter(prefix="/rag", tags=["rag"])

SYSTEM_PROMPT = (
    "You are a secure on-premise assistant for an organization. "
    "Answer ONLY using the numbered context below. "
    "Cite the sources you used like [1] or [2]. "
    "If the answer is not in the context, say exactly: "
    "'This information was not found in the uploaded documents.' "
    "Reply in the same language as the question."
)


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


@router.post("/ask")
async def ask(req: AskRequest):
    try:
        [q_vec] = await embed([req.question])
    except Exception as e:
        raise HTTPException(503, f"Embedding model unavailable: {e}")

    hits = vectorstore.search(q_vec, req.top_k or settings.top_k)
    if not hits:
        return {"answer": "No documents uploaded yet.", "sources": []}

    context = "\n\n".join(
        f"[{i}] (file: {h['filename']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"},
    ]
    try:
        answer = await llm.chat(messages, temperature=0.2)
    except Exception as e:
        raise HTTPException(503, f"LLM unavailable: {e}")

    return {
        "answer": answer,
        "sources": [
            {
                "ref": i,
                "filename": h["filename"],
                "page": h["page"],
                "distance": round(h["distance"], 3),
                "snippet": h["text"][:200],
            }
            for i, h in enumerate(hits, start=1)
        ],
    }
