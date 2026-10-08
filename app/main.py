from fastapi import FastAPI

from app import llm
from app.config import settings
from app.routers import chat, documents, rag

app = FastAPI(title="Sovereign On-Premise AI Workbench", version="0.2.0")
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(rag.router)


@app.get("/health")
async def health():
    return {
        "api": "ok",
        "llm_server": await llm.is_up(),
        "model": settings.llm_model,
        "embed_model": settings.embed_model,
    }
