from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import llm

router = APIRouter(prefix="/chat", tags=["chat"])

SYSTEM_PROMPT = (
    "You are a secure on-premise assistant for an organization. "
    "Answer clearly and concisely. If you do not know, say so."
)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]


def _build_messages(req: ChatRequest) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *req.history,
        {"role": "user", "content": req.message},
    ]


@router.post("")
async def chat(req: ChatRequest):
    try:
        answer = await llm.chat(_build_messages(req))
    except Exception as e:
        raise HTTPException(503, f"LLM unavailable: {e}")
    return {"answer": answer}


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        llm.chat_stream(_build_messages(req)), media_type="text/plain"
    )
