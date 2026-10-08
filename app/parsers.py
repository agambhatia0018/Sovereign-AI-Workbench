"""Read uploaded files and return a list of (page_number, text)."""
from pathlib import Path


def parse_file(path: Path) -> list[tuple[int, str]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [(i + 1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    elif ext == ".docx":
        from docx import Document

        doc = Document(str(path))
        pages = [(1, "\n".join(p.text for p in doc.paragraphs))]
    elif ext in (".txt", ".md"):
        pages = [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    else:
        raise ValueError(f"Unsupported file type: {ext} (use PDF, DOCX, TXT or MD)")

    pages = [(n, t) for n, t in pages if t.strip()]
    if not pages:
        raise ValueError("No text found. Scanned PDFs need OCR (not supported yet).")
    return pages
