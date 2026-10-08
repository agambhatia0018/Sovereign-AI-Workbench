import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app import vectorstore
from app.chunking import split_text
from app.config import settings
from app.embeddings import embed_in_batches
from app.parsers import parse_file

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path(settings.data_dir) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload(file: UploadFile):
    doc_id = uuid.uuid4().hex[:12]
    safe_name = Path(file.filename or "file").name
    path = UPLOAD_DIR / f"{doc_id}_{safe_name}"
    path.write_bytes(await file.read())

    try:
        pages = await run_in_threadpool(parse_file, path)
    except ValueError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(400, str(e))

    chunks = [
        {"page": page_no, "text": piece}
        for page_no, text in pages
        for piece in split_text(text, settings.chunk_size, settings.chunk_overlap)
    ]
    try:
        vectors = await embed_in_batches([c["text"] for c in chunks])
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(503, f"Embedding model unavailable: {e}")

    await run_in_threadpool(vectorstore.add_chunks, doc_id, safe_name, chunks, vectors)
    return {"doc_id": doc_id, "filename": safe_name, "pages": len(pages), "chunks": len(chunks)}


@router.get("")
async def list_docs():
    return vectorstore.list_documents()


@router.delete("/{doc_id}")
async def delete(doc_id: str):
    vectorstore.delete_document(doc_id)
    for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
        f.unlink(missing_ok=True)
    return {"deleted": doc_id}
